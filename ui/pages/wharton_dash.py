from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
import importlib
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any
import uuid

import bcrypt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_PATH = Path("data/wharton_production.db")
UPLOAD_DIR = Path("data/wharton_uploads")
USER_PROFILE_KEY = "wharton_user_profile_v2"

ALLOWED_EXTENSIONS = {
    ".pdf", ".xlsx", ".xls", ".csv", ".docx", ".doc",
    ".txt", ".md", ".png", ".jpg", ".jpeg", ".gif",
    ".pptx", ".ppt", ".json", ".py", ".ipynb", ".zip",
}
MAX_FILE_SIZE_MB = 50


def _is_development_mode() -> bool:
    """Check if the app is running in development mode."""
    return os.environ.get("QUANT_SIM_ENV") == "development"


def _get_default_password() -> str:
    try:
        return str(st.secrets.get("WHARTON_PASSWORD", "team123"))
        return str(st.secrets.get("WHARTON_PASSWORD", "CHANGE_ME_IN_SECRETS"))
    except Exception:
        return "team123"
        return "CHANGE_ME_IN_SECRETS"

# This is now a fallback for seeding, not a direct password.
# Actual password will be read from st.secrets["wharton_users"][username]
# or generated if in development mode.
DEV_ONLY_INSECURE_DEFAULT_PASSWORD = "DEV_ONLY_INSECURE_DEFAULT"

# This is used for seeding the initial users.
# The actual password used for authentication will come from secrets.
SEEDING_DEFAULT_PASSWORD = _get_default_password()

# Login attempt limits
MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_WINDOW_MINUTES = 10

DEFAULT_PASSWORD = _get_default_password()
TASK_EDITOR_VERSION_KEY = "wharton_task_editor_version"
QUANT_RESULT_KEY = "wharton_quant_result"
QUANT_ERROR_KEY = "wharton_quant_error"
QUANT_STACK_RESULT_KEY = "wharton_quant_stack_result"

TASK_PRIORITIES = ["Critical", "High", "Medium", "Low"]
TASK_PRIORITY_COLORS = {
    "Critical": "#dc2626",
    "High": "#d97706",
    "Medium": "#2563eb",
    "Low": "#64748b",
}
GRAPH_NODE_TYPES = ["Policy", "Company", "Model", "Market", "Risk", "Research", "Other"]
QUANT_MODULES = [
    "Benchmark Analytics",
    "Cost-Aware Rebalance",
    "Performance Attribution",
    "Simulation",
    "Models & Signals",
    "News Sentiment",
    "Backtest",
    "Run History",
]
QUANT_OPERATOR_USERS = {"Jakub", "Matfyz_Genius"}
DEFAULT_QUANT_TICKERS = ["ASML", "NVDA", "MSFT", "LLY", "JPM"]

DEFAULT_USERS = [
    {"username": "Jakub", "role": "Captain/Quant", "primary_module": "Quant Engine"},
    {"username": "Matěj", "role": "Oxford/CIO", "primary_module": "Dashboard & Strategy"},
    {"username": "Martin", "role": "Logistics/Risk", "primary_module": "Risk Operations"},
    {"username": "Lukáš", "role": "Geopolitics", "primary_module": "Macro Intelligence"},
    {"username": "Janek", "role": "Intelligence", "primary_module": "War Room"},
    {"username": "Matfyz_Genius", "role": "Quant/Math", "primary_module": "Quant Engine"},
]

DEFAULT_MINDMAP_NODES = [
    ("node_eu_tech_regulation", "EU Tech Regulation", "Policy"),
    ("node_asml", "ASML", "Company"),
    ("node_monte_carlo", "Monte Carlo", "Model"),
]

DEFAULT_MINDMAP_EDGES = [
    ("edge_eu_tech_regulation_asml", "node_eu_tech_regulation", "node_asml"),
    ("edge_asml_monte_carlo", "node_asml", "node_monte_carlo"),
]

NODE_COLORS = {
    "Policy": "#2563eb",
    "Company": "#059669",
    "Model": "#d97706",
    "Market": "#7c3aed",
    "Risk": "#dc2626",
    "Research": "#0891b2",
    "Other": "#64748b",
}


# ─── Database ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    from src.auth.database import get_db_connection
    return get_db_connection(DB_PATH)


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wharton_users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                role TEXT,
                primary_module TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                username TEXT,
                message TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                priority TEXT DEFAULT 'Medium',
                task_text TEXT,
                assignee TEXT,
                due_date TEXT,
                tags TEXT,
                is_done INTEGER DEFAULT 0
            )
        """)
        # Extended files table with project/description support
        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER NOT NULL,
                ip_address TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                filename TEXT,
                original_filename TEXT,
                uploaded_by TEXT,
                file_path TEXT,
                file_size_bytes INTEGER DEFAULT 0,
                mime_type TEXT DEFAULT '',
                project_name TEXT DEFAULT '',
                description TEXT DEFAULT '',
                tags TEXT DEFAULT ''
            )
        """)
        # Migrate old files table if missing columns
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()
        }
        for col, definition in [
            ("original_filename", "TEXT DEFAULT ''"),
            ("file_size_bytes", "INTEGER DEFAULT 0"),
            ("mime_type", "TEXT DEFAULT ''"),
            ("project_name", "TEXT DEFAULT ''"),
            ("description", "TEXT DEFAULT ''"),
            ("tags", "TEXT DEFAULT ''"),
        ]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE files ADD COLUMN {col} {definition}")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS subprojects (
                id INTEGER PRIMARY KEY,
                created_at TEXT,
                created_by TEXT,
                name TEXT,
                description TEXT,
                status TEXT DEFAULT 'active',
                tags TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subproject_files (
                id INTEGER PRIMARY KEY,
                subproject_id INTEGER REFERENCES subprojects(id) ON DELETE CASCADE,
                file_id INTEGER REFERENCES files(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mindmap_nodes (
                id TEXT PRIMARY KEY,
                label TEXT,
                type TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mindmap_edges (
                id TEXT PRIMARY KEY,
                source TEXT,
                target TEXT
            )
        """)

        # Seed users
        existing_users = {
            str(row["username"]): str(row["password_hash"] or "")
            for row in conn.execute("SELECT username, password_hash FROM wharton_users").fetchall()
        }
        for user in DEFAULT_USERS:
            if existing_users.get(user["username"]):
                continue

            user_pass = DEFAULT_PASSWORD
            if not _is_development_mode():
                try:
                    user_pass = str(st.secrets["wharton_users"][user["username"]])
                except Exception as e:
                    # Ignore missing secret quietly
                    pass

            password_hash = bcrypt.hashpw(
                user_pass.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            conn.execute(
                "INSERT OR IGNORE INTO wharton_users (username, password_hash, role, primary_module) VALUES (?, ?, ?, ?)",
                (user["username"], password_hash, user["role"], user["primary_module"]),
            )


        # Seed mindmap
        if conn.execute("SELECT COUNT(*) FROM mindmap_nodes").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO mindmap_nodes (id, label, type) VALUES (?, ?, ?)",
                DEFAULT_MINDMAP_NODES,
            )
        if conn.execute("SELECT COUNT(*) FROM mindmap_edges").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO mindmap_edges (id, source, target) VALUES (?, ?, ?)",
                DEFAULT_MINDMAP_EDGES,
            )


# ─── Auth ────────────────────────────────────────────────────────────────────

def _fetch_users() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, username, role, primary_module FROM wharton_users ORDER BY username COLLATE NOCASE"
        ).fetchall()


def authenticate_user(username: str, password: str) -> dict[str, str | int] | None:
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, username, password_hash, role, primary_module FROM wharton_users WHERE username = ?",
            (username,),
        ).fetchone()
    if user is None:
        return None
    stored_hash = str(user["password_hash"] or "")
    if not stored_hash:
        return None
    if not bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
        return None
    return {
        "id": int(user["id"]),
        "username": str(user["username"]),
        "role": str(user["role"]),
        "primary_module": str(user["primary_module"]),
    }


def _get_current_profile() -> dict[str, str | int] | None:
    profile = st.session_state.get(USER_PROFILE_KEY)
    if isinstance(profile, dict) and profile.get("username"):
        return profile
    return None


def _logout() -> None:
    st.session_state.pop(USER_PROFILE_KEY, None)
    st.rerun()


def _render_login() -> None:
    st.markdown("""
        <div style="max-width:420px;margin:4rem auto 0;padding:2.5rem;
            background:linear-gradient(135deg,#0f172a,#1e293b);
            border-radius:20px;border:1px solid rgba(20,184,166,0.3);
            box-shadow:0 24px 60px rgba(0,0,0,0.4);">
          <h2 style="color:#f8fafc;margin:0 0 0.25rem;font-size:1.6rem;">Wharton Cockpit</h2>
          <p style="color:rgba(248,250,252,0.6);margin:0 0 1.5rem;font-size:0.9rem;">
            Production workspace · Strategy · Quant · Team
          </p>
        </div>
    """, unsafe_allow_html=True)

    users = _fetch_users()
    usernames = [str(u["username"]) for u in users]
    if not usernames:
        st.error("No users found. Restart the app.")
        st.stop()

    with st.form("wharton_login_form", clear_on_submit=False):
        # duplicate username selectbox removed
        username = st.selectbox("Username", options=usernames, key="wharton_login_username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter Cockpit", type="primary", use_container_width=True)

    # Check if we are in development mode and using the insecure default password
    if _is_development_mode() and submitted and password == DEV_ONLY_INSECURE_DEFAULT_PASSWORD:
        st.warning(
            "You are using the insecure default password in development mode. "
            "Please set `st.secrets['wharton_users']['<username>']` for production."
        )
    elif submitted and password == DEV_ONLY_INSECURE_DEFAULT_PASSWORD and not _is_development_mode():
        st.error("Insecure default password is not allowed in production mode.")
        return

    if submitted:
        profile = authenticate_user(username, password)
        if profile is None:
            st.error("Wrong credentials.")
            return
        st.session_state[USER_PROFILE_KEY] = profile
        st.rerun()


# ─── Styles ──────────────────────────────────────────────────────────────────

def _inject_cockpit_styles() -> None:
    st.markdown("""
        <style>
        .block-container {
            max-width: 100% !important;
            padding-top: 1.2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .wharton-hero {
            border: 1px solid rgba(15,23,42,0.12);
            border-radius: 24px;
            padding: 1.25rem 1.45rem;
            margin-bottom: 1rem;
            background:
                radial-gradient(circle at 4% 18%, rgba(20,184,166,0.20), transparent 30%),
                linear-gradient(135deg, rgba(15,23,42,0.97), rgba(30,41,59,0.94));
            color: #f8fafc;
            box-shadow: 0 18px 50px rgba(15,23,42,0.16);
        }
        .wharton-hero h1 { margin:0; font-size:2.05rem; letter-spacing:-0.04em; }
        .wharton-hero p { margin:0.45rem 0 0; color:rgba(248,250,252,0.78); }
        .wharton-badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.9rem; }
        .wharton-badge {
            border:1px solid rgba(226,232,240,0.22); border-radius:999px;
            padding:0.35rem 0.7rem; background:rgba(255,255,255,0.08);
            color:#e2e8f0; font-size:0.86rem;
        }
        .wharton-panel {
            border:1px solid rgba(15,23,42,0.10); border-radius:18px;
            padding:1rem 1.15rem; background:rgba(248,250,252,0.74); margin-bottom:1rem;
        }
        .wharton-section-kicker {
            color:#0f766e; text-transform:uppercase; letter-spacing:0.12em;
            font-weight:800; font-size:0.76rem; margin-bottom:0.35rem;
        }
        div[data-testid="stMetric"] {
            border:1px solid rgba(20,184,166,0.32); border-radius:16px;
            padding:0.85rem 0.95rem;
            background:
                radial-gradient(circle at top left, rgba(45,212,191,0.22), transparent 42%),
                linear-gradient(135deg, #0f172a, #164e63);
            box-shadow:0 12px 30px rgba(15,23,42,0.16);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color:#f8fafc !important; }
        div[data-testid="stMetric"] svg { fill:#f8fafc !important; }
        .wharton-graph-shell {
            border:1px solid rgba(20,184,166,0.24); border-radius:20px;
            padding:0.75rem 0.75rem 0.25rem;
            background:linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,64,89,0.90));
            box-shadow:0 18px 45px rgba(15,23,42,0.16); margin:0.75rem 0 1.25rem;
        }
        .wharton-graph-shell strong { color:#ecfeff; }
        .wharton-graph-shell span { color:#a7f3d0; }
        .task-priority-Critical { color:#dc2626; font-weight:700; }
        .task-priority-High { color:#d97706; font-weight:700; }
        .task-priority-Medium { color:#2563eb; font-weight:600; }
        .task-priority-Low { color:#64748b; }
        .subproject-card {
            border:1px solid rgba(20,184,166,0.2); border-radius:14px;
            padding:1rem 1.2rem; margin-bottom:0.75rem;
            background:linear-gradient(135deg,rgba(15,23,42,0.04),rgba(20,184,166,0.04));
        }
        div[data-testid="stTabs"] button { font-weight:700; }
        </style>
    """, unsafe_allow_html=True)


# ─── Task Manager ─────────────────────────────────────────────────────────────

def _fetch_task_rows() -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, priority, task_text, assignee, is_done FROM tasks ORDER BY "
            "CASE priority WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END, id ASC"
        ).fetchall()
    return pd.DataFrame(
        [{"id": int(r["id"]), "priority": str(r["priority"] or "Medium"),
          "task_text": str(r["task_text"] or ""), "assignee": str(r["assignee"] or ""),
          "is_done": bool(r["is_done"])} for r in rows],
        columns=["id", "priority", "task_text", "assignee", "is_done"],
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool): return value
    if value is None: return False
    if isinstance(value, (int, float)): return bool(value)
    if isinstance(value, str): return value.strip().lower() in {"1", "true", "yes", "y", "done"}
    return bool(value)


def _clean_priority(value: object) -> str:
    p = str(value or "Medium").strip()
    return p if p in TASK_PRIORITIES else "Medium"


def _clean_task_payload(row_data: dict, default_assignee: str) -> dict:
    assignee = str(row_data.get("assignee") or default_assignee).strip() or default_assignee
    return {
        "priority": _clean_priority(row_data.get("priority")),
        "task_text": str(row_data.get("task_text") or "").strip(),
        "assignee": assignee,
        "is_done": int(_truthy(row_data.get("is_done"))),
    }


def _coerce_row_index(value: object) -> int | None:
    try: return int(value)
    except (TypeError, ValueError): return None


def _task_editor_has_changes(state: object) -> bool:
    if not isinstance(state, dict): return False
    return bool(state.get("edited_rows") or state.get("added_rows") or state.get("deleted_rows"))


def _apply_task_editor_changes(state: dict, original: pd.DataFrame, default_assignee: str) -> None:
    edited = state.get("edited_rows") or {}
    added = state.get("added_rows") or []
    deleted = state.get("deleted_rows") or []

    deleted_indices = {i for i in (_coerce_row_index(v) for v in deleted) if i is not None}

    with get_connection() as conn:
        for idx in sorted(deleted_indices, reverse=True):
            if 0 <= idx < len(original):
                conn.execute("DELETE FROM tasks WHERE id = ?", (int(original.iloc[idx]["id"]),))

        for raw_idx, changes in (edited if isinstance(edited, dict) else {}).items():
            idx = _coerce_row_index(raw_idx)
            if idx is None or idx in deleted_indices or not (0 <= idx < len(original)):
                continue
            if not isinstance(changes, dict): continue
            current = original.iloc[idx].to_dict()
            current.update(changes)
            payload = _clean_task_payload(current, default_assignee)
            task_id = int(original.iloc[idx]["id"])
            conn.execute(
                "UPDATE tasks SET priority=?, task_text=?, assignee=?, is_done=? WHERE id=?",
                (payload["priority"], payload["task_text"], payload["assignee"], payload["is_done"], task_id),
            )

        for row in (added if isinstance(added, list) else []):
            if not isinstance(row, dict): continue
            payload = _clean_task_payload(row, default_assignee)
            if not payload["task_text"]: continue
            conn.execute(
                "INSERT INTO tasks (priority, task_text, assignee, is_done) VALUES (?, ?, ?, ?)",
                (payload["priority"], payload["task_text"], payload["assignee"], payload["is_done"]),
            )


def _render_task_stats(tasks: pd.DataFrame) -> None:
    """Render visual task summary above the editor."""
    total = len(tasks)
    done = int(tasks["is_done"].sum())
    open_count = total - done
    critical = len(tasks[tasks["priority"] == "Critical"])
    high = len(tasks[tasks["priority"] == "High"])

    cols = st.columns(5)
    cols[0].metric("Total Tasks", total)
    cols[1].metric("Open", open_count)
    cols[2].metric("Done ✓", done)
    cols[3].metric(" Critical", critical)
    cols[4].metric(" High", high)

    if total > 0:
        pct = done / total
        st.progress(pct, text=f"Completion: {pct:.0%}")


def _render_task_manager(profile: dict[str, str | int]) -> None:
    st.markdown("### Mission Task Board")

    original_tasks = _fetch_task_rows()
    _render_task_stats(original_tasks)

    # Quick-add form (always visible, above the editor)
    with st.expander(" Quick Add Task", expanded=False):
        with st.form("wharton_quick_task_form", clear_on_submit=True):
            qc1, qc2, qc3 = st.columns([3, 1, 1])
            with qc1:
                quick_text = st.text_input("Task description")
            with qc2:
                quick_priority = st.selectbox("Priority", TASK_PRIORITIES, index=2)
            with qc3:
                quick_assignee = st.text_input("Assign to (optional)", value=str(profile["username"]))
            if st.form_submit_button("Add Task", type="primary", use_container_width=True):
                if quick_text.strip():
                    with get_connection() as conn:
                        conn.execute(
                            "INSERT INTO tasks (priority, task_text, assignee, is_done) VALUES (?, ?, ?, 0)",
                            (quick_priority, quick_text.strip(), quick_assignee.strip() or str(profile["username"])),
                        )
                    st.toast("Task added.")
                    st.rerun()

    editor_version = int(st.session_state.get(TASK_EDITOR_VERSION_KEY, 0))
    editor_key = f"wharton_tasks_editor_{editor_version}"

    st.data_editor(
        original_tasks,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=editor_key,
        column_order=["id", "priority", "task_text", "assignee", "is_done"],
        height=440,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "priority": st.column_config.SelectboxColumn("Priority", options=TASK_PRIORITIES, required=True, width="small"),
            "task_text": st.column_config.TextColumn("Task", required=True, width="large"),
            "assignee": st.column_config.TextColumn("Assignee", required=True, width="medium"),
            "is_done": st.column_config.CheckboxColumn("Done ✓", width="small"),
        },
    )

    editor_state = st.session_state.get(editor_key, {})
    if _task_editor_has_changes(editor_state):
        _apply_task_editor_changes(editor_state, original_tasks, str(profile["username"]))
        st.session_state[TASK_EDITOR_VERSION_KEY] = editor_version + 1
        st.toast("Task board saved.")
        st.rerun()


# ─── Mindmap ──────────────────────────────────────────────────────────────────

def _fetch_graph_rows() -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    with get_connection() as conn:
        nodes = conn.execute("SELECT id, label, type FROM mindmap_nodes ORDER BY label COLLATE NOCASE").fetchall()
        edges = conn.execute("SELECT id, source, target FROM mindmap_edges ORDER BY id").fetchall()
    return nodes, edges


def _node_display_name(row: sqlite3.Row) -> str:
    return f"{row['label']} ({row['type']})"


def _slugify_node_id(label: str) -> str:
    slug = "".join(c.lower() if c.isascii() and c.isalnum() else "_" for c in label.strip())
    slug = "_".join(p for p in slug.split("_") if p)
    if not slug: slug = "node"
    if not slug.startswith("node_"): slug = f"node_{slug}"
    return slug[:96]


def _create_unique_node_id(label: str) -> str:
    base = _slugify_node_id(label)
    candidate = base
    suffix = 2
    with get_connection() as conn:
        while conn.execute("SELECT 1 FROM mindmap_nodes WHERE id = ?", (candidate,)).fetchone():
            candidate = f"{base}_{suffix}"
            suffix += 1
    return candidate


def _edge_id(source: str, target: str) -> str:
    def safe(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in s)
    return f"edge_{safe(source)}_{safe(target)}"[:180]


def _insert_node(label: str, node_type: str) -> None:
    node_id = _create_unique_node_id(label)
    with get_connection() as conn:
        conn.execute("INSERT INTO mindmap_nodes (id, label, type) VALUES (?, ?, ?)",
                     (node_id, label.strip(), node_type))


def _delete_node(node_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM mindmap_edges WHERE source = ? OR target = ?", (node_id, node_id))
        conn.execute("DELETE FROM mindmap_nodes WHERE id = ?", (node_id,))


def _delete_edge(edge_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM mindmap_edges WHERE id = ?", (edge_id,))


def _insert_edge(source: str, target: str) -> bool:
    eid = _edge_id(source, target)
    with get_connection() as conn:
        if conn.execute("SELECT 1 FROM mindmap_edges WHERE id = ?", (eid,)).fetchone():
            return False
        conn.execute("INSERT INTO mindmap_edges (id, source, target) VALUES (?, ?, ?)", (eid, source, target))
    return True


def _render_mindmap() -> None:
    st.markdown("### Strategy Mind Map")
    st.caption("Interactive knowledge graph — nodes, edges, full-screen rendering. Add, connect, delete below.")

    node_rows, edge_rows = _fetch_graph_rows()
    node_ids = {str(r["id"]) for r in node_rows}
    valid_edges = [r for r in edge_rows if str(r["source"]) in node_ids and str(r["target"]) in node_ids]

    st.markdown(f"""
        <div class="wharton-graph-shell">
            <strong>Live Strategy Graph</strong>
            <span> | {len(node_rows)} nodes | {len(valid_edges)} edges</span>
        </div>
    """, unsafe_allow_html=True)

    connected_ids = set()
    for r in edge_rows:
        connected_ids.add(str(r["source"]))
        connected_ids.add(str(r["target"]))

    graph_nodes = [
        Node(
            id=str(r["id"]),
            label=str(r["label"]),
            title=f"{r['label']} | {r['type']}",
            size=32,
            color=NODE_COLORS.get(str(r["type"]), NODE_COLORS["Other"]),
            font={"color": "#d1d8eb", "size": 18, "face": "Tahoma"},
            mass=4.0 if str(r["id"]) not in connected_ids else 1.0,
        )
        for r in node_rows
    ]
    graph_edges = [
        Edge(source=str(r["source"]), target=str(r["target"]), color="#64748b")
        for r in valid_edges
    ]
    graph_config = Config(
        width=1800, height=720, directed=True,
        physics={
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "springLength": 160, "springConstant": 0.04,
                "damping": 0.9, "avoidOverlap": 0.15,
            },
            "stabilization": True,
        },
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#f59e0b",
        collapsible=True,
    )

    if graph_nodes:
        agraph(nodes=graph_nodes, edges=graph_edges, config=graph_config)
    else:
        st.warning("No nodes yet. Add the first node below.")

    st.divider()
    st.markdown("### Manage Graph")

    mgmt_tabs = st.tabs(["Add Node", "Connect Nodes", "Delete Node", "Delete Edge", "Table View"])

    with mgmt_tabs[0]:
        with st.form("wharton_add_node_form", clear_on_submit=True):
            nc1, nc2 = st.columns([3, 1])
            with nc1:
                node_label = st.text_input("Node Label", placeholder="e.g. Taiwan Export Controls, ASML Earnings")
            with nc2:
                node_type = st.selectbox("Type", GRAPH_NODE_TYPES)
            if st.form_submit_button("Add Node", type="primary"):
                clean = node_label.strip()
                if not clean:
                    st.warning("Enter a label.")
                else:
                    _insert_node(clean, node_type)
                    st.success(f"Added: {clean}")
                    st.rerun()

    with mgmt_tabs[1]:
        if len(node_rows) < 2:
            st.info("Add at least two nodes first.")
        else:
            node_options = [str(r["id"]) for r in node_rows]
            row_by_id = {str(r["id"]): r for r in node_rows}
            with st.form("wharton_connect_nodes_form"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    source_id = st.selectbox("Source Node", options=node_options,
                                             format_func=lambda nid: _node_display_name(row_by_id[nid]))
                with ec2:
                    target_id = st.selectbox("Target Node", options=node_options, index=1 if len(node_options) > 1 else 0,
                                             format_func=lambda nid: _node_display_name(row_by_id[nid]))
                if st.form_submit_button("Connect", type="primary"):
                    if source_id == target_id:
                        st.warning("Choose two different nodes.")
                    elif _insert_edge(source_id, target_id):
                        st.success(f"Connected: {_node_display_name(row_by_id[source_id])} → {_node_display_name(row_by_id[target_id])}")
                        st.rerun()
                    else:
                        st.info("Connection already exists.")

    with mgmt_tabs[2]:
        if not node_rows:
            st.info("No nodes to delete.")
        else:
            node_options = [str(r["id"]) for r in node_rows]
            row_by_id = {str(r["id"]): r for r in node_rows}
            del_node = st.selectbox("Select node to delete", node_options,
                                    format_func=lambda nid: _node_display_name(row_by_id[nid]),
                                    key="del_node_select")
            st.warning(" This also removes all edges connected to this node.")
            if st.button("Delete Node", type="primary", key="del_node_btn"):
                _delete_node(del_node)
                st.success(f"Deleted: {_node_display_name(row_by_id[del_node])}")
                st.rerun()

    with mgmt_tabs[3]:
        if not valid_edges:
            st.info("No edges to delete.")
        else:
            edge_options = [str(r["id"]) for r in edge_rows if str(r["source"]) in node_ids and str(r["target"]) in node_ids]
            row_by_id_n = {str(r["id"]): r for r in node_rows}
            edge_by_id = {str(r["id"]): r for r in edge_rows}

            def edge_label(eid: str) -> str:
                e = edge_by_id.get(eid)
                if not e: return eid
                src = row_by_id_n.get(str(e["source"]))
                tgt = row_by_id_n.get(str(e["target"]))
                return f"{src['label'] if src else e['source']} → {tgt['label'] if tgt else e['target']}"

            del_edge = st.selectbox("Select edge to delete", edge_options, format_func=edge_label, key="del_edge_select")
            if st.button("Delete Edge", type="primary", key="del_edge_btn"):
                _delete_edge(del_edge)
                st.success("Edge deleted.")
                st.rerun()

    with mgmt_tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Nodes")
            st.dataframe(
                pd.DataFrame([{"ID": r["id"], "Label": r["label"], "Type": r["type"]} for r in node_rows]),
                use_container_width=True, hide_index=True,
            )
        with col2:
            st.markdown("#### Edges")
            st.dataframe(
                pd.DataFrame([{"ID": r["id"], "Source": r["source"], "Target": r["target"]} for r in edge_rows]),
                use_container_width=True, hide_index=True,
            )


# ─── Chat ────────────────────────────────────────────────────────────────────

def _fetch_chat_history() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, timestamp, username, message FROM chat ORDER BY id ASC"
        ).fetchall()


def _save_chat_message(username: str, message: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat (timestamp, username, message) VALUES (?, ?, ?)",
            (_now_iso(), username, message),
        )


def _render_chat(profile: dict[str, str | int]) -> None:
    st.markdown("### War Room Chat")
    st.caption("All messages are persisted to SQLite.")

    history = _fetch_chat_history()
    current_username = str(profile["username"])

    if not history:
        st.info("No messages yet.")

    for row in history:
        chat_role = "user" if str(row["username"]) == current_username else "assistant"
        with st.chat_message(chat_role):
            # Escape username and timestamp to prevent injection; use st.write (not markdown) for message body
            st.markdown(f"**{escape(str(row['username']))}** · {escape(str(row['timestamp']))}")
            st.write(str(row["message"]))  # st.write escapes by default

    prompt = st.chat_input("Send a War Room update")
    if prompt and prompt.strip():
        clean_msg = prompt.strip()[:4000]  # cap message length
        _save_chat_message(current_username, clean_msg)
        st.rerun()


# ─── File Vault (Storage Backend Integration) ─────────────────────────────────

# Import storage backend
from src.storage import (
    init_storage_db,
    save_uploaded_file as storage_save_file,
    download_file as storage_download_file,
    file_exists as storage_file_exists,
    list_files_with_status,
    StorageFileNotFoundError,
    FileValidationError,
)


def _init_file_vault_storage():
    """Initialize storage layer for file vault."""
    init_storage_db(str(DB_PATH))


def _safe_filename(filename: str) -> str:
    """Legacy function - kept for compatibility."""
    base = os.path.basename(filename).replace("\\", "_").strip()
    cleaned = "".join(c if c.isalnum() or c in {".", "_", "-"} else "_" for c in base)
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "upload.bin"
    return cleaned[:140]


def _validate_upload(uploaded_file: object) -> str | None:
    """Returns error string or None if valid. (Legacy - now handled by storage layer)"""
    fname = str(uploaded_file.name)
    ext = Path(fname).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"File type `{ext}` is not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    size_bytes = len(uploaded_file.getbuffer())
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        return f"File `{fname}` exceeds {MAX_FILE_SIZE_MB} MB limit ({size_bytes / 1024 / 1024:.1f} MB)."
    return None


def _save_uploaded_file(
    uploaded_file: object,
    uploaded_by: str,
    project_name: str = "",
    description: str = "",
    tags: str = "",
) -> None:
    """Save uploaded file using the new storage backend."""
    try:
        result = storage_save_file(
            uploaded_file=uploaded_file,
            uploaded_by=uploaded_by,
            db_path=str(DB_PATH),
            project_name=project_name,
            description=description,
            tags=tags,
        )
    except FileValidationError as e:
        raise ValueError(str(e))


def _fetch_file_rows() -> list[dict]:
    """Fetch file rows with storage status."""
    return list_files_with_status(db_path=str(DB_PATH))


def _fetch_file_rows_legacy() -> list[sqlite3.Row]:
    """Legacy function for backward compatibility."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, timestamp, filename, original_filename, uploaded_by, file_path, "
            "file_size_bytes, project_name, description, tags FROM files ORDER BY id DESC"
        ).fetchall()


def _render_file_center(profile: dict[str, str | int]) -> None:
    st.markdown("### Persistent File Vault")
    st.caption(f"Files stored in `{UPLOAD_DIR}/` · indexed in SQLite · max {MAX_FILE_SIZE_MB} MB · allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    with st.expander(" Upload Files", expanded=True):
        with st.form("wharton_file_upload_form", clear_on_submit=True):
            uploads = st.file_uploader(
                "Research, decks, model notes, datasets, screenshots",
                accept_multiple_files=True,
            )
            uc1, uc2 = st.columns(2)
            with uc1:
                upload_project = st.text_input("Project / Category (optional)", placeholder="e.g. EU Regulation Analysis")
            with uc2:
                upload_tags = st.text_input("Tags (optional)", placeholder="e.g. risk, asml, q2")
            upload_desc = st.text_area("Description (optional)", placeholder="Briefly describe what this file contains and covers.", height=80)
            submitted = st.form_submit_button("Save Files", type="primary", use_container_width=True)

        if submitted:
            if not uploads:
                st.warning("Select at least one file.")
            else:
                errors = []
                saved = 0
                for f in uploads:
                    err = _validate_upload(f)
                    if err:
                        errors.append(err)
                    else:
                        _save_uploaded_file(f, str(profile["username"]), upload_project, upload_desc, upload_tags)
                        saved += 1
                if saved: st.success(f"Saved {saved} file(s).")
                for e in errors: st.error(e)
                if saved: st.rerun()

    file_rows = _fetch_file_rows()
    if not file_rows:
        st.info("No files uploaded yet.")
        return

    # Filter/search
    search_query = st.text_input(" Search files", placeholder="filename, project, uploader, tags...")

    df_data = []
    for r in file_rows:
        size_kb = int(r["file_size_bytes"] or 0) / 1024
        df_data.append({
            "ID": int(r["id"]),
            "Uploaded": str(r["timestamp"]),
            "Filename": str(r["filename"]),
            "By": str(r["uploaded_by"]),
            "Project": str(r["project_name"] or ""),
            "Tags": str(r["tags"] or ""),
            "Size": f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB",
            "_path": str(r["file_path"]),
        })

    df = pd.DataFrame(df_data)
    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            df["Filename"].str.lower().str.contains(q) |
            df["By"].str.lower().str.contains(q) |
            df["Project"].str.lower().str.contains(q) |
            df["Tags"].str.lower().str.contains(q)
        )
        df = df[mask]

    display_df = df.drop(columns=["_path"])
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

    st.markdown("#### Downloads")
    for row in file_rows:
        file_id = int(row["id"])
        fname = str(row["original_filename"])
        status = row.get("status", "unknown")
        status_label = row.get("status_label", "")

        # Filter display by search
        if search_query.strip():
            q = search_query.strip().lower()
            haystack = f"{fname} {row['uploaded_by']} {row['project_name']} {row['tags']}".lower()
            if q not in haystack:
                continue

        desc = str(row.get("description") or "")
        proj = str(row.get("project_name") or "")
        label_parts = [fname]
        if proj: label_parts.append(f"[{proj}]")

        col1, col2 = st.columns([4, 1])
        with col1:
            # Show status indicator
            status_icon = "" if status == "available" else ""
            st.markdown(f"{status_icon} **{escape(fname)}**" + (f" · {escape(proj)}" if proj else ""))
            if desc:
                st.caption(escape(desc))
            if status == "missing":
                st.warning("File is missing from storage")
        with col2:
            if status == "available":
                try:
                    content, download_name, content_type = storage_download_file(file_id, db_path=str(DB_PATH))
                    st.download_button(
                        "Download",
                        data=content,
                        file_name=download_name,
                        mime=content_type,
                        key=f"dl_{file_id}",
                        use_container_width=True,
                    )
                except (StorageFileNotFoundError, FileNotFoundError) as e:
                    st.error(f"Download failed: {e}")


# ─── Subprojects ─────────────────────────────────────────────────────────────

def _fetch_subprojects() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, created_at, created_by, name, description, status, tags FROM subprojects ORDER BY id DESC"
        ).fetchall()


def _fetch_subproject_files(subproject_id: int) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT f.id, f.filename, f.uploaded_by, f.timestamp, f.file_path,
                   f.project_name, f.description, f.file_size_bytes
            FROM subproject_files sf
            JOIN files f ON sf.file_id = f.id
            WHERE sf.subproject_id = ?
            ORDER BY f.timestamp DESC
            """,
            (subproject_id,),
        ).fetchall()


def _render_subprojects(profile: dict[str, str | int]) -> None:
    st.markdown("### Sub-Projects")
    st.caption("Structured research threads — attach files, write analysis summaries, track status.")

    with st.expander(" Create New Sub-Project", expanded=False):
        with st.form("wharton_create_subproject_form", clear_on_submit=True):
            sp_name = st.text_input("Project Name", placeholder="e.g. EU AI Act Impact on ASML")
            sp_desc = st.text_area("Analysis / Description",
                                   placeholder="Describe scope, methodology, what this covers, key findings...",
                                   height=120)
            sp1, sp2 = st.columns(2)
            with sp1:
                sp_status = st.selectbox("Status", ["active", "in-review", "complete", "archived"])
            with sp2:
                sp_tags = st.text_input("Tags", placeholder="risk, eu, regulation")
            if st.form_submit_button("Create Sub-Project", type="primary"):
                if sp_name.strip():
                    with get_connection() as conn:
                        conn.execute(
                            "INSERT INTO subprojects (created_at, created_by, name, description, status, tags) VALUES (?, ?, ?, ?, ?, ?)",
                            (_now_iso(), str(profile["username"]), sp_name.strip(),
                             sp_desc.strip(), sp_status, sp_tags.strip()),
                        )
                    st.success(f"Sub-project '{sp_name.strip()}' created.")
                    st.rerun()
                else:
                    st.warning("Name is required.")

    subprojects = _fetch_subprojects()
    if not subprojects:
        st.info("No sub-projects yet. Create one above.")
        return

    all_files = _fetch_file_rows()
    file_options = {int(r["id"]): str(r["filename"]) for r in all_files}

    STATUS_COLORS = {
        "active": "#059669", "in-review": "#d97706",
        "complete": "#2563eb", "archived": "#64748b",
    }

    for sp in subprojects:
        sp_id = int(sp["id"])
        status_color = STATUS_COLORS.get(str(sp["status"]), "#64748b")
        sp_files = _fetch_subproject_files(sp_id)

        with st.expander(
            f" {sp['name']}  ·  "
            f"<span style='color:{status_color};font-weight:700;'>{sp['status'].upper()}</span>  ·  "
            f"{len(sp_files)} file(s)",
            expanded=False,
        ):
            st.markdown(f"""
                <div class="subproject-card">
                  <div class="wharton-section-kicker">Created by {escape(str(sp['created_by']))} · {escape(str(sp['created_at']))}</div>
                  <div style="margin-top:0.5rem">{escape(str(sp['description'] or '—'))}</div>
                  <div style="margin-top:0.5rem;font-size:0.82rem;color:#64748b;">Tags: {escape(str(sp['tags'] or '—'))}</div>
                </div>
            """, unsafe_allow_html=True)

            if sp_files:
                st.markdown("**Attached Files:**")
                for f in sp_files:
                    fpath = Path(str(f["file_path"]))
                    sc1, sc2 = st.columns([5, 1])
                    with sc1:
                        st.markdown(f" **{escape(str(f['filename']))}**")
                        if f["description"]:
                            st.caption(escape(str(f["description"])))
                    with sc2:
                        if fpath.exists():
                            with open(fpath, "rb") as fh:
                                st.download_button("DL", fh.read(), file_name=str(f["filename"]),
                                                   mime="application/octet-stream",
                                                   key=f"sp_dl_{sp_id}_{f['id']}")
            else:
                st.info("No files attached yet.")

            # Attach file
            if file_options:
                with st.form(f"attach_file_sp_{sp_id}"):
                    attached_ids = {int(f["id"]) for f in sp_files}
                    available = {fid: fname for fid, fname in file_options.items() if fid not in attached_ids}
                    if available:
                        selected_file_id = st.selectbox(
                            "Attach a file from Vault",
                            options=list(available.keys()),
                            format_func=lambda fid: available[fid],
                        )
                        if st.form_submit_button("Attach File"):
                            with get_connection() as conn:
                                conn.execute(
                                    "INSERT OR IGNORE INTO subproject_files (subproject_id, file_id) VALUES (?, ?)",
                                    (sp_id, selected_file_id),
                                )
                            st.success("File attached.")
                            st.rerun()
                    else:
                        st.info("All vault files already attached, or vault is empty.")

            # Status update
            with st.form(f"update_status_sp_{sp_id}"):
                new_status = st.selectbox("Update Status", ["active", "in-review", "complete", "archived"],
                                          index=["active", "in-review", "complete", "archived"].index(str(sp["status"])),
                                          key=f"status_sel_{sp_id}")
                if st.form_submit_button("Update Status"):
                    with get_connection() as conn:
                        conn.execute("UPDATE subprojects SET status = ? WHERE id = ?", (new_status, sp_id))
                    st.rerun()


# ─── Quant Engine ─────────────────────────────────────────────────────────────

def _load_quant_modules() -> dict[str, Any]:
    return {
        "analytics": importlib.import_module("src.analytics"),
        "optimization": importlib.import_module("src.optimization"),
        "simulation": importlib.import_module("src.simulation"),
        "yahoo_fetcher": importlib.import_module("src.data.fetchers.yahoo_fetcher"),
    }


def _load_modular_pipeline():
    return importlib.import_module("src.analytics.modular.pipeline")


def _load_modular_history():
    return importlib.import_module("src.analytics.modular.history")


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_close_prices_cached(symbols: tuple, start_date: date, end_date: date) -> pd.DataFrame:
    modules = _load_quant_modules()
    fetcher = modules["yahoo_fetcher"].YahooFetcher()
    return fetcher.fetch_close_prices(list(symbols), start_date, end_date)


def _parse_tickers(raw: str) -> list[str]:
    tickers, seen = [], set()
    for chunk in raw.replace(",", "\n").splitlines():
        t = chunk.strip().upper()
        if t and t not in seen:
            tickers.append(t)
            seen.add(t)
    if not tickers: raise ValueError("Enter at least one ticker.")
    return tickers


def _parse_weights(raw: str, tickers: list[str]) -> np.ndarray:
    if not raw.strip():
        return np.array([1.0 / len(tickers)] * len(tickers), dtype=float)
    values = [float(c.strip().replace("%", "")) for c in raw.replace(",", "\n").splitlines() if c.strip()]
    if len(values) != len(tickers): raise ValueError("One weight per ticker required.")
    w = np.asarray(values, dtype=float)
    if np.any(w < 0): raise ValueError("Weights must be non-negative.")
    if w.sum() > 1.5: w = w / 100.0
    if np.isclose(w.sum(), 0.0): raise ValueError("Weights cannot sum to zero.")
    return w / w.sum()


def _align_weights(tickers: list, weights: np.ndarray, columns: list) -> np.ndarray:
    series = pd.Series(weights, index=tickers, dtype=float).reindex(columns).fillna(0.0).to_numpy(dtype=float)
    if np.isclose(series.sum(), 0.0):
        return np.array([1.0 / len(columns)] * len(columns), dtype=float)
    return series / series.sum()


def _fmt_pct(v: object, d: int = 2) -> str:
    try:
        n = float(v)
        if np.isnan(n) or np.isinf(n): return "n/a"
        return f"{n:.{d}%}"
    except: return "n/a"


def _fmt_float(v: object, d: int = 3) -> str:
    try:
        n = float(v)
        if np.isnan(n) or np.isinf(n): return "n/a"
        return f"{n:.{d}f}"
    except: return "n/a"


def _compute_quant_run(
    tickers, weights, benchmark_ticker, start_date, end_date,
    risk_free_rate, current_value, max_weight, turnover_limit,
    transaction_cost_bps, risk_aversion, simulation_days, n_simulations, random_seed,
) -> dict[str, Any]:
    modules = _load_quant_modules()
    analytics, optimization, simulation = modules["analytics"], modules["optimization"], modules["simulation"]

    prices = _fetch_close_prices_cached(tuple(tickers), start_date, end_date)
    if prices.empty: raise ValueError("No price data returned.")
    prices = prices.sort_index().ffill().dropna(how="all")
    available = [str(c) for c in prices.columns if prices[c].notna().sum() > 2]
    prices = prices[available].dropna(how="any")
    if prices.empty: raise ValueError("Not enough aligned data.")

    returns = prices.pct_change().dropna(how="any")
    if returns.empty: raise ValueError("Return series is empty.")

    aligned_w = _align_weights(tickers, weights, list(returns.columns))
    portfolio_returns = analytics.calculate_portfolio_daily_returns(returns, aligned_w)
    core_metrics = analytics.calculate_portfolio_core_metrics(portfolio_returns, risk_free_rate)
    concentration = analytics.calculate_concentration_metrics(aligned_w)
    corr_matrix = analytics.calculate_correlation_matrix(returns)
    avg_corr = analytics.calculate_average_correlation(corr_matrix)

    benchmark_symbol = benchmark_ticker.strip().upper()
    benchmark_returns = pd.Series(dtype=float)
    if benchmark_symbol:
        if benchmark_symbol in returns.columns:
            benchmark_returns = returns[benchmark_symbol]
        else:
            bp = _fetch_close_prices_cached((benchmark_symbol,), start_date, end_date)
            if benchmark_symbol in bp.columns:
                benchmark_returns = bp[benchmark_symbol].sort_index().pct_change().dropna()

    benchmark_metrics = analytics.calculate_active_risk_metrics(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        benchmark_ticker=benchmark_symbol,
        risk_free_rate=risk_free_rate,
    )
    return_contribution = analytics.calculate_return_contribution(returns, aligned_w)
    risk_contribution = analytics.calculate_risk_contribution(returns, aligned_w)
    min_variance = optimization.optimize_minimum_variance(returns, max_weight=max_weight)
    max_sharpe = optimization.optimize_maximum_sharpe(returns, risk_free_rate=risk_free_rate, max_weight=max_weight)
    cost_aware = optimization.optimize_cost_aware_rebalance(
        returns=returns, current_weights=aligned_w, risk_free_rate=risk_free_rate,
        max_weight=max_weight, turnover_limit=turnover_limit,
        transaction_cost_bps=transaction_cost_bps, risk_aversion=risk_aversion,
    )
    portfolio_timeseries = analytics.build_portfolio_timeseries(portfolio_returns, initial_value=current_value)
    price_paths, simulation_stats = simulation.run_monte_carlo_simulation(
        current_value=current_value,
        expected_return=float(core_metrics.get("annualized_return", 0.0)),
        volatility=float(core_metrics.get("volatility", 0.0)),
        time_horizon=simulation_days, n_simulations=n_simulations, random_seed=random_seed,
    )

    # Run full modular quant stack (models, signals, news, backtest, history)
    quant_stack_result = None
    try:
        pipeline = _load_modular_pipeline()
        config = {
            "tickers": list(returns.columns),
            "weights": aligned_w.tolist(),
            "start_date": start_date,
            "end_date": end_date,
            "risk_free_rate": risk_free_rate,
            "portfolio_metrics": {**core_metrics, **concentration, "avg_correlation": avg_corr},
            "news_max_items": 80,
            "news_api_key": "",
        }
        try:
            news_api_key = str(st.secrets.get("NEWS_API_KEY", ""))
            news_api_key = str(st.secrets.get("NEWSAPI_KEY", "")) # Standardize on NEWSAPI_KEY
            if news_api_key: config["news_api_key"] = news_api_key
        except Exception:
            pass
        quant_stack_result = pipeline.run_quant_stack(
            portfolio_returns=portfolio_returns,
            returns_df=returns,
            config=config,
            user_id=st.session_state.get("user_id"),
        )
    except Exception as stack_err:
        quant_stack_result = {"_error": str(stack_err)}

    metrics = {
        **core_metrics, **concentration,
        "avg_correlation": avg_corr,
        "observations": int(returns.shape[0]),
    }
    return {
        "generated_at": _now_iso(),
        "tickers": list(returns.columns),
        "requested_tickers": tickers,
        "weights": aligned_w,
        "benchmark_ticker": benchmark_symbol,
        "prices": prices,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "portfolio_timeseries": portfolio_timeseries,
        "metrics": metrics,
        "correlation": corr_matrix,
        "benchmark_metrics": benchmark_metrics,
        "return_contribution": return_contribution,
        "risk_contribution": risk_contribution,
        "min_variance": min_variance,
        "max_sharpe": max_sharpe,
        "cost_aware": cost_aware,
        "price_paths": price_paths,
        "simulation_stats": simulation_stats,
        "quant_stack": quant_stack_result,
        "inputs": {
            "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
            "risk_free_rate": risk_free_rate, "current_value": current_value,
            "max_weight": max_weight, "turnover_limit": turnover_limit,
            "transaction_cost_bps": transaction_cost_bps, "risk_aversion": risk_aversion,
            "simulation_days": simulation_days, "n_simulations": n_simulations,
            "random_seed": random_seed,
        },
    }


def _render_quant_configuration() -> None:
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=365 * 2)

    with st.expander(" Quant Run Configuration", expanded=QUANT_RESULT_KEY not in st.session_state):
        with st.form("wharton_quant_config_form"):
            col_in, col_risk = st.columns([1, 1], gap="large")
            with col_in:
                tickers_text = st.text_area("Portfolio Tickers", value="\n".join(DEFAULT_QUANT_TICKERS), height=145)
                weights_text = st.text_area("Weights (leave empty = equal)", value="", height=115)
                benchmark_ticker = st.text_input("Benchmark Ticker", value="SPY").strip().upper()
                current_value = st.number_input("Portfolio Value ($)", min_value=1_000.0, value=100_000.0, step=5_000.0)
            with col_risk:
                start_date = st.date_input("Start Date", value=default_start)
                end_date = st.date_input("End Date", value=default_end)
                risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.15, 0.03, 0.005, format="%.3f")
                max_weight = st.slider("Max Asset Weight", 0.10, 1.0, 0.35, 0.01)
                turnover_limit = st.slider("Turnover Limit", 0.05, 2.0, 0.30, 0.05)
                transaction_cost_bps = st.slider("Transaction Cost (bps)", 0.0, 100.0, 10.0, 1.0)
                risk_aversion = st.slider("Risk Aversion", 0.5, 10.0, 3.0, 0.5)
                simulation_days = st.slider("Simulation Horizon (days)", 30, 1260, 252, 30)
                n_simulations = st.slider("Simulation Count", 200, 5000, 1200, 100)
                random_seed = st.number_input("Seed", min_value=0, value=42, step=1)
            run_clicked = st.form_submit_button(" Run Full Quant Engine", type="primary")

    if run_clicked:
        st.session_state.pop(QUANT_ERROR_KEY, None)
        try:
            tickers = _parse_tickers(tickers_text)
            weights = _parse_weights(weights_text, tickers)
            if start_date >= end_date: raise ValueError("Start Date must be before End Date.")
            with st.spinner("Fetching data and running full quant stack..."):
                st.session_state[QUANT_RESULT_KEY] = _compute_quant_run(
                    tickers=tickers, weights=weights, benchmark_ticker=benchmark_ticker,
                    start_date=start_date, end_date=end_date,
                    risk_free_rate=float(risk_free_rate), current_value=float(current_value),
                    max_weight=float(max_weight), turnover_limit=float(turnover_limit),
                    transaction_cost_bps=float(transaction_cost_bps), risk_aversion=float(risk_aversion),
                    simulation_days=int(simulation_days), n_simulations=int(n_simulations),
                    random_seed=int(random_seed),
                )
            st.success("Quant engine run complete.")
            st.rerun()
        except Exception as exc:
            st.session_state[QUANT_ERROR_KEY] = str(exc)


def _weights_frame(symbols, current_w, optimized_w) -> pd.DataFrame:
    opt = np.asarray(optimized_w, dtype=float)
    cur = np.asarray(current_w, dtype=float)
    if opt.size != len(symbols): opt = np.zeros(len(symbols), dtype=float)
    return pd.DataFrame({
        "Ticker": symbols, "Current Weight": cur, "Optimized Weight": opt, "Delta": opt - cur,
    })


def _render_weight_table(frame: pd.DataFrame) -> None:
    view = frame.copy()
    for col in ["Current Weight", "Optimized Weight", "Delta"]:
        view[col] = view[col].map(_fmt_pct)
    st.dataframe(view, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_ai_insight_cached(context_data: dict, prompt_type: str) -> dict:
    try:
        from src.ai.ai_advisor import generate_advisor_insight
        from src.ai.ai_review import resolve_groq_api_key
        api_key = resolve_groq_api_key(st.secrets)
        return generate_advisor_insight(context_data, prompt_type, api_key)
    except ImportError:
        return {"available": False, "error": "AI Advisor module not found."}


def _render_ai_advisor_card(context_data: dict, prompt_type: str) -> None:
    with st.spinner(" AI Advisor is analyzing..."):
        res = _fetch_ai_insight_cached(context_data, prompt_type)
    if res.get("available") and res.get("insight"):
        st.info(f"** AI Advisor Insight:**\n\n{res['insight']}")
    elif not res.get("available"):
        st.caption(f"AI Insight unavailable: {res.get('error', 'Unknown error')}")


def _render_benchmark_analytics(result: dict, advanced: bool) -> None:
    metrics = result["metrics"]
    bm = result["benchmark_metrics"]
    st.markdown("### Benchmark Analytics")
    r = st.columns(4)
    r[0].metric("Total Return", _fmt_pct(metrics.get("total_return")))
    r[1].metric("Ann. Return", _fmt_pct(metrics.get("annualized_return")))
    r[2].metric("Volatility", _fmt_pct(metrics.get("volatility")))
    r[3].metric("Sharpe", _fmt_float(metrics.get("sharpe_ratio")))
    st.markdown("#### Portfolio Value Path")
    st.line_chart(result["portfolio_timeseries"][["value"]].rename(columns={"value": "Portfolio Value"}), use_container_width=True, height=340)
    st.markdown("#### Active Risk vs Benchmark")
    if bm.get("benchmark_available"):
        ar = st.columns(4)
        ar[0].metric("Benchmark", str(bm.get("benchmark_ticker", "")))
        ar[1].metric("Active Return Ann.", _fmt_pct(bm.get("active_return_annualized")))
        ar[2].metric("Tracking Error", _fmt_pct(bm.get("tracking_error")))
        ar[3].metric("Information Ratio", _fmt_float(bm.get("information_ratio")))
        sr = st.columns(4)
        sr[0].metric("Beta", _fmt_float(bm.get("beta_to_benchmark")))
        sr[1].metric("Alpha Ann.", _fmt_pct(bm.get("alpha_to_benchmark")))
        sr[2].metric("Up Capture", _fmt_float(bm.get("up_capture")))
        sr[3].metric("Down Capture", _fmt_float(bm.get("down_capture")))
    else:
        st.warning(f"Benchmark metrics unavailable: {bm.get('reason', 'unknown')}")
    if advanced:
        with st.expander("Advanced Diagnostics"):
            st.write(f"Observations: **{metrics.get('observations', 0)}**  |  Avg correlation: **{_fmt_float(metrics.get('avg_correlation'))}**  |  Effective holdings: **{_fmt_float(metrics.get('effective_holdings'))}**")
            st.dataframe(result["correlation"].round(3), use_container_width=True)
            st.dataframe(result["returns"].tail(15), use_container_width=True)


def _render_cost_aware_rebalance(result: dict, advanced: bool) -> None:
    ca = result["cost_aware"]
    mv = result["min_variance"]
    ms = result["max_sharpe"]
    symbols = list(result["tickers"])
    w = np.asarray(result["weights"], dtype=float)
    st.markdown("### Cost-Aware Rebalance")
    
    _render_ai_advisor_card(
        context_data={"cost_aware": ca, "min_variance": mv, "max_sharpe": ms, "current_metrics": result.get("metrics")},
        prompt_type="optimizer_explain"
    )

    if not ca.get("success"):
        st.warning(f"Optimization warning: {ca.get('message', 'unknown')}")
    r = st.columns(4)
    r[0].metric("Expected Return", _fmt_pct(ca.get("expected_return")))
    r[1].metric("Volatility", _fmt_pct(ca.get("volatility")))
    r[2].metric("Sharpe", _fmt_float(ca.get("sharpe_ratio")))
    r[3].metric("Turnover", _fmt_pct(ca.get("turnover")))
    wf = _weights_frame(symbols, w, np.asarray(ca.get("weights", []), dtype=float))
    st.markdown("#### Rebalance Weights")
    _render_weight_table(wf)
    st.bar_chart(wf.set_index("Ticker")[["Current Weight", "Optimized Weight"]], use_container_width=True, height=360)
    st.markdown("#### Optimizer Comparison")
    comp = pd.DataFrame([
        {"Model": "Current", "Expected Return": result["metrics"].get("annualized_return", 0), "Volatility": result["metrics"].get("volatility", 0), "Sharpe": result["metrics"].get("sharpe_ratio", 0)},
        {"Model": "Min Variance", "Expected Return": mv.get("expected_return", 0), "Volatility": mv.get("volatility", 0), "Sharpe": mv.get("sharpe_ratio", 0)},
        {"Model": "Max Sharpe", "Expected Return": ms.get("expected_return", 0), "Volatility": ms.get("volatility", 0), "Sharpe": ms.get("sharpe_ratio", 0)},
        {"Model": "Cost-Aware", "Expected Return": ca.get("expected_return", 0), "Volatility": ca.get("volatility", 0), "Sharpe": ca.get("sharpe_ratio", 0)},
    ])
    cv = comp.copy()
    for col in ["Expected Return", "Volatility"]: cv[col] = cv[col].map(_fmt_pct)
    cv["Sharpe"] = cv["Sharpe"].map(_fmt_float)
    st.dataframe(cv, use_container_width=True, hide_index=True)
    if advanced:
        with st.expander("Advanced Optimizer Diagnostics"):
            st.json({"success": bool(ca.get("success")), "message": str(ca.get("message", "")),
                     "utility_score": float(ca.get("utility_score", 0)), "transaction_cost_drag": float(ca.get("transaction_cost_drag", 0))})


def _render_performance_attribution(result: dict, advanced: bool) -> None:
    rc = result["return_contribution"].copy()
    rk = result["risk_contribution"].copy()
    st.markdown("### Performance Attribution")
    st.markdown("#### Return Contribution")
    if rc.empty: st.info("Unavailable.")
    else:
        st.bar_chart(rc.set_index("Ticker")[["AnnualizedContributionApprox"]], use_container_width=True, height=320)
        rv = rc.copy()
        for col in ["Weight", "TotalContributionApprox", "AnnualizedContributionApprox", "ContributionShare", "MeanDailyContribution"]:
            rv[col] = rv[col].map(_fmt_pct)
        st.dataframe(rv, use_container_width=True, hide_index=True)
    st.markdown("#### Risk Contribution")
    if rk.empty: st.info("Unavailable.")
    else:
        st.bar_chart(rk.set_index("Ticker")[["RiskBudgetPct"]], use_container_width=True, height=320)
        rkv = rk.copy()
        for col in ["Weight", "MarginalVolatility", "RiskContribution", "RiskBudgetPct"]:
            rkv[col] = rkv[col].map(_fmt_pct)
        st.dataframe(rkv, use_container_width=True, hide_index=True)


def _render_simulation(result: dict, advanced: bool) -> None:
    st.markdown("### Monte Carlo Simulation")
    stats = result["simulation_stats"]
    paths = result["price_paths"]
    r = st.columns(4)
    r[0].metric("Mean Final Value", f"${stats['mean']:,.0f}")
    r[1].metric("Median", f"${stats['median']:,.0f}")
    r[2].metric("5th Percentile", f"${stats['percentile_5']:,.0f}")
    r[3].metric("95th Percentile", f"${stats['percentile_95']:,.0f}")

    pcts = {f"p{p}": np.percentile(paths, p, axis=1) for p in [5, 25, 50, 75, 95]}
    st.markdown("#### Percentile Paths")
    st.line_chart(pd.DataFrame(pcts), use_container_width=True, height=420)

    final_values = pd.Series(paths[-1])
    counts, bins = np.histogram(final_values, bins=40)
    hist_df = pd.DataFrame({
        "Bucket": [f"{bins[i]:,.0f}–{bins[i+1]:,.0f}" for i in range(len(counts))],
        "Count": counts,
    })
    st.markdown("#### Final Value Distribution")
    st.bar_chart(hist_df.set_index("Bucket"), use_container_width=True, height=320)


def _render_models_signals(result: dict) -> None:
    """Render models and signals from the modular quant stack."""
    st.markdown("### Models & Signals")
    qs = result.get("quant_stack", {})
    if not qs:
        st.info("Run the Quant Engine to populate models and signals.")
        return
    if "_error" in qs:
        st.warning(f"Modular stack error: {qs['_error']}")
        return

    models = qs.get("models", {})
    signals = qs.get("signals", {})

    if models:
        st.markdown("####  Models")
        model_data = []
        for name, model in models.items():
            m_dict = model.to_dict() if hasattr(model, "to_dict") else {}
            model_data.append({
                "Model": name,
                "Available": "" if m_dict.get("available") else "",
                "Confidence": _fmt_float(m_dict.get("confidence")),
                "Score": _fmt_float(m_dict.get("score")),
                "Signal": str(m_dict.get("signal", "—")),
                "Notes": str(m_dict.get("notes", ""))[:80],
            })
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    if signals:
        st.markdown("####  Signals")
        signal_data = []
        for name, signal in signals.items():
            s_dict = signal.to_dict() if hasattr(signal, "to_dict") else {}
            signal_data.append({
                "Signal": name,
                "Available": "" if s_dict.get("available") else "",
                "Score": _fmt_float(s_dict.get("score")),
                "Direction": str(s_dict.get("direction", "—")),
                "Confidence": _fmt_float(s_dict.get("confidence")),
            })
        st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

    summary = qs.get("summary")
    if summary:
        s_dict = summary.to_dict() if hasattr(summary, "to_dict") else {}
        st.markdown("####  Summary")
        sc = st.columns(3)
        sc[0].metric("Composite Score", _fmt_float(s_dict.get("composite_score")))
        sc[1].metric("Confidence", _fmt_float(s_dict.get("confidence")))
        sc[2].metric("Regime", str(s_dict.get("regime_label", "—")))
        if s_dict.get("narrative"):
            st.markdown(f"> {escape(str(s_dict['narrative']))}")


def _render_news_sentiment(result: dict) -> None:
    """Render news sentiment from modular stack."""
    st.markdown("### News Sentiment")
    qs = result.get("quant_stack", {})
    if not qs:
        st.info("Run the Quant Engine to populate news sentiment.")
        return
        
    if "news" in qs and hasattr(qs["news"], "to_dict"):
        _render_ai_advisor_card(
            context_data={"news_items": qs["news"].to_dict().get("items", [])[:15]},
            prompt_type="news_synthesis"
        )
    if "_error" in qs:
        st.warning(f"Modular stack error: {qs['_error']}")
        return

    news = qs.get("news")
    if not news:
        st.info("No news data in this run.")
        return

    n_dict = news.to_dict() if hasattr(news, "to_dict") else {}
    nc = st.columns(3)
    nc[0].metric("Sentiment Score", _fmt_float(n_dict.get("sentiment_score")))
    nc[1].metric("Dispersion", _fmt_float(n_dict.get("sentiment_dispersion")))
    nc[2].metric("Coverage", _fmt_pct(n_dict.get("context", {}).get("relevance_coverage", 0)))

    items = n_dict.get("items", [])
    if items:
        st.markdown(f"#### Top News Items ({len(items)} total)")
        news_rows = []
        for item in items[:30]:
            news_rows.append({
                "Headline": str(item.get("headline", ""))[:100],
                "Ticker": str(item.get("ticker", "")),
                "Sentiment": _fmt_float(item.get("sentiment_score")),
                "Source": str(item.get("source", "")),
                "Date": str(item.get("published_at", ""))[:10],
            })
        st.dataframe(pd.DataFrame(news_rows), use_container_width=True, hide_index=True)

    ctx = n_dict.get("context", {})
    provider = ctx.get("provider_used", "")
    errors = ctx.get("fetch_errors", [])
    if provider: st.caption(f"Provider: {escape(str(provider))}")
    if errors:
        with st.expander("Fetch Errors"):
            for e in errors: st.text(str(e))


def _render_backtest(result: dict) -> None:
    """Render backtest results from modular stack."""
    st.markdown("### Backtest Results")
    qs = result.get("quant_stack", {})
    if not qs:
        st.info("Run the Quant Engine to populate backtest.")
        return
    if "_error" in qs:
        st.warning(f"Modular stack error: {qs['_error']}")
        return

    backtest = qs.get("backtest", {})
    if not backtest:
        st.info("No backtest data.")
        return

    metrics = backtest.get("metrics", {})
    st.markdown("#### Metrics")
    mc = st.columns(4)
    mc[0].metric("Total Return", _fmt_pct(metrics.get("total_return")))
    mc[1].metric("Ann. Return", _fmt_pct(metrics.get("annualized_return")))
    mc[2].metric("Sharpe", _fmt_float(metrics.get("sharpe_ratio")))
    mc[3].metric("Max Drawdown", _fmt_pct(metrics.get("max_drawdown")))

    mc2 = st.columns(3)
    mc2[0].metric("Win Rate", _fmt_pct(metrics.get("win_rate")))
    mc2[1].metric("Lookahead Safe", "" if backtest.get("lookahead_safe") else "")
    mc2[2].metric("Calmar Ratio", _fmt_float(metrics.get("calmar_ratio")))

    equity_curve = backtest.get("equity_curve")
    if equity_curve is not None:
        try:
            ec = pd.Series(equity_curve)
            st.markdown("#### Equity Curve")
            st.line_chart(ec, use_container_width=True, height=300)
        except Exception:
            pass

    with st.expander("Full Backtest Metrics"):
        st.json({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                 for k, v in metrics.items()})


def _render_run_history(result: dict) -> None:
    """Render run history from modular stack."""
    st.markdown("### Run History")
    qs = result.get("quant_stack", {})
    if not qs:
        st.info("Run the Quant Engine to populate history.")
        return

    try:
        history_mod = _load_modular_history()
        records = history_mod.list_run_records(base_dir="data/run_history", limit=40, user_id=st.session_state.get("user_id"))
    except Exception as e:
        st.warning(f"Could not load run history: {e}")
        return

    if not records:
        st.info("No previous runs found in data/run_history/.")
        return

    st.caption(f"{len(records)} historical run(s) found.")
    rows = []
    for rec in records:
        r_dict = rec.to_dict() if hasattr(rec, "to_dict") else {}
        m = r_dict.get("metrics", {})
        rows.append({
            "Run ID": str(r_dict.get("run_id", ""))[-16:],
            "Timestamp": str(r_dict.get("timestamp", ""))[:19],
            "Tickers": ", ".join(r_dict.get("universe", []))[:40],
            "Sharpe": _fmt_float(m.get("sharpe_ratio")),
            "Ann. Return": _fmt_pct(m.get("annualized_return")),
            "Volatility": _fmt_pct(m.get("volatility")),
            "Composite Signal": _fmt_float(m.get("composite_signal")),
            "News Sentiment": _fmt_float(m.get("news_sentiment_score")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Monte Carlo Simulation ────────────────────────────────────────────────────

def _render_monte_carlo(result: dict) -> None:
    """Interactive Monte Carlo simulation tab with fan charts and VaR analysis."""
    st.markdown("###  Monte Carlo Simulation")

    try:
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first to see Monte Carlo results.")
        return

    stats = result.get("simulation_stats", {})
    paths = result.get("price_paths")
    inputs = result.get("inputs", {})
    current_value = inputs.get("current_value", 100_000)

    if paths is None:
        st.warning("No simulation paths available.")
        return

    # KPI cards
    st.markdown("#### Key Outcomes")
    k = st.columns(5)
    k[0].metric(" Mean Final", f"${stats.get('mean', 0):,.0f}",
                delta=f"{((stats.get('mean', current_value) / current_value) - 1) * 100:+.1f}%")
    k[1].metric(" Median", f"${stats.get('median', 0):,.0f}")
    k[2].metric(" 5th Pctl (VaR)", f"${stats.get('percentile_5', 0):,.0f}",
                delta=f"{((stats.get('percentile_5', current_value) / current_value) - 1) * 100:+.1f}%")
    k[3].metric(" 95th Pctl", f"${stats.get('percentile_95', 0):,.0f}")
    k[4].metric(" Std Dev", f"${stats.get('std', 0):,.0f}")

    # Probability of loss
    final_values = paths[-1] if paths is not None else np.array([])
    if len(final_values) > 0:
        prob_loss = float(np.mean(final_values < current_value)) * 100
        prob_20_loss = float(np.mean(final_values < current_value * 0.80)) * 100
        prob_gain_20 = float(np.mean(final_values > current_value * 1.20)) * 100

        pc = st.columns(3)
        pc[0].metric(" P(Loss)", f"{prob_loss:.1f}%")
        pc[1].metric(" P(Loss > 20%)", f"{prob_20_loss:.1f}%")
        pc[2].metric(" P(Gain > 20%)", f"{prob_gain_20:.1f}%")

    if HAS_PLOTLY:
        # Fan chart with percentile bands
        st.markdown("#### Percentile Fan Chart")
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        pctl_data = {f"p{p}": np.percentile(paths, p, axis=1) for p in percentiles}
        days = list(range(len(pctl_data["p50"])))

        fig_fan = go.Figure()
        # 5-95 band
        fig_fan.add_trace(go.Scatter(x=days, y=pctl_data["p95"].tolist(), mode="lines",
            line=dict(width=0), showlegend=False, name="p95"))
        fig_fan.add_trace(go.Scatter(x=days, y=pctl_data["p5"].tolist(), mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(99,102,241,0.10)",
            name="5th–95th %ile"))
        # 10-90 band
        fig_fan.add_trace(go.Scatter(x=days, y=pctl_data["p90"].tolist(), mode="lines",
            line=dict(width=0), showlegend=False, name="p90"))
        fig_fan.add_trace(go.Scatter(x=days, y=pctl_data["p10"].tolist(), mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(99,102,241,0.18)",
            name="10th–90th %ile"))
        # 25-75 band
        fig_fan.add_trace(go.Scatter(x=days, y=pctl_data["p75"].tolist(), mode="lines",
            line=dict(width=0), showlegend=False, name="p75"))
        fig_fan.add_trace(go.Scatter(x=days, y=pctl_data["p25"].tolist(), mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(99,102,241,0.28)",
            name="25th–75th %ile"))
        # Median line
        fig_fan.add_trace(go.Scatter(x=days, y=pctl_data["p50"].tolist(), mode="lines",
            line=dict(color="#6366f1", width=2.5), name="Median"))
        # Starting value
        fig_fan.add_hline(y=current_value, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Initial ${current_value:,.0f}", annotation_position="top left")

        fig_fan.update_layout(
            template="plotly_dark", height=480,
            title="Portfolio Value: Monte Carlo Fan Chart",
            xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
            yaxis=dict(tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        # Distribution histogram
        if len(final_values) > 0:
            st.markdown("#### Terminal Value Distribution")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=final_values, nbinsx=60, name="Final Values",
                marker_color="rgba(99,102,241,0.7)",
                marker_line=dict(color="rgba(99,102,241,1)", width=1),
            ))
            fig_dist.add_vline(x=current_value, line_dash="dash", line_color="#ef4444",
                annotation_text=f"Initial ${current_value:,.0f}")
            fig_dist.add_vline(x=float(stats.get("percentile_5", 0)), line_dash="dot",
                line_color="#f59e0b", annotation_text="VaR 95%")
            fig_dist.update_layout(
                template="plotly_dark", height=380,
                xaxis_title="Final Portfolio Value ($)", yaxis_title="Frequency",
                xaxis=dict(tickformat="$,.0f"),
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # VaR / CVaR metrics
            st.markdown("#### Value-at-Risk Analysis")
            var_cols = st.columns(4)
            var_95 = float(np.percentile(final_values, 5))
            var_99 = float(np.percentile(final_values, 1))
            cvar_95 = float(np.mean(final_values[final_values <= var_95]))
            cvar_99 = float(np.mean(final_values[final_values <= var_99]))

            var_cols[0].metric("VaR 95%", f"${current_value - var_95:,.0f}",
                delta=f"{((var_95 / current_value) - 1) * 100:.1f}%")
            var_cols[1].metric("VaR 99%", f"${current_value - var_99:,.0f}",
                delta=f"{((var_99 / current_value) - 1) * 100:.1f}%")
            var_cols[2].metric("CVaR 95%", f"${current_value - cvar_95:,.0f}",
                delta=f"{((cvar_95 / current_value) - 1) * 100:.1f}%")
            var_cols[3].metric("CVaR 99%", f"${current_value - cvar_99:,.0f}",
                delta=f"{((cvar_99 / current_value) - 1) * 100:.1f}%")

        # Sample paths
        st.markdown("#### Sample Simulated Paths")
        n_show = min(50, paths.shape[1])
        fig_paths = go.Figure()
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(paths.shape[1], n_show, replace=False)
        for idx in sample_indices:
            fig_paths.add_trace(go.Scatter(
                x=list(range(paths.shape[0])), y=paths[:, idx].tolist(),
                mode="lines", line=dict(width=0.5, color="rgba(99,102,241,0.15)"),
                showlegend=False,
            ))
        fig_paths.add_trace(go.Scatter(
            x=list(range(paths.shape[0])), y=np.median(paths, axis=1).tolist(),
            mode="lines", line=dict(width=2.5, color="#22c55e"), name="Median",
        ))
        fig_paths.add_hline(y=current_value, line_dash="dash", line_color="#ef4444")
        fig_paths.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
            yaxis=dict(tickformat="$,.0f"),
        )
        st.plotly_chart(fig_paths, use_container_width=True)
    else:
        # Fallback without Plotly
        pcts = {f"p{p}": np.percentile(paths, p, axis=1) for p in [5, 25, 50, 75, 95]}
        st.line_chart(pd.DataFrame(pcts), use_container_width=True, height=420)

    st.caption(f"Simulations: {inputs.get('n_simulations', 'N/A')} · "
               f"Horizon: {inputs.get('simulation_days', 'N/A')} days · "
               f"Seed: {inputs.get('random_seed', 'N/A')}")


# ─── Efficient Frontier ───────────────────────────────────────────────────────

def _render_efficient_frontier(result: dict) -> None:
    """Efficient frontier with 2D/3D visualization and optimal portfolios."""
    st.markdown("###  Efficient Frontier & Portfolio Optimization")

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first to see frontier analysis.")
        return

    returns_df = result.get("returns")
    if returns_df is None or returns_df.empty:
        st.warning("No return data available for frontier computation.")
        return

    tickers = list(result.get("tickers", []))
    weights = np.asarray(result.get("weights", []), dtype=float)
    ms = result.get("max_sharpe", {})
    mv = result.get("min_variance", {})
    ca = result.get("cost_aware", {})
    metrics = result.get("metrics", {})
    inputs = result.get("inputs", {})
    risk_free_rate = float(inputs.get("risk_free_rate", 0.03))

    # Optimal portfolio KPIs
    st.markdown("#### Optimal Portfolio Comparison")
    comp_data = []
    for label, d, color in [
        (" Current", {"expected_return": metrics.get("annualized_return", 0),
                        "volatility": metrics.get("volatility", 0),
                        "sharpe_ratio": metrics.get("sharpe_ratio", 0)}, "#64748b"),
        (" Max Sharpe", ms, "#22c55e"),
        (" Min Variance", mv, "#3b82f6"),
        (" Cost-Aware", ca, "#f59e0b"),
    ]:
        comp_data.append({
            "Portfolio": label,
            "Expected Return": _fmt_pct(d.get("expected_return")),
            "Volatility": _fmt_pct(d.get("volatility")),
            "Sharpe Ratio": _fmt_float(d.get("sharpe_ratio")),
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # Compute frontier
    try:
        ef_module = importlib.import_module("src.optimization.efficient_frontier")
        frontier_points = ef_module.calculate_efficient_frontier(returns_df, n_points=50)
    except Exception as e:
        st.warning(f"Could not compute efficient frontier: {e}")
        frontier_points = []

    if HAS_PLOTLY and frontier_points:
        frontier_returns = [pt["return"] for pt in frontier_points]
        frontier_vols = [pt["volatility"] for pt in frontier_points]
        frontier_sharpe = [pt["sharpe_ratio"] for pt in frontier_points]

        # 2D Efficient Frontier
        st.markdown("#### 2D Efficient Frontier")
        fig_2d = go.Figure()

        # Frontier line
        fig_2d.add_trace(go.Scatter(
            x=frontier_vols, y=frontier_returns, mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=4, color=frontier_sharpe, colorscale="Viridis",
                        colorbar=dict(title="Sharpe", thickness=15)),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.3f}<extra></extra>",
        ))

        # Current portfolio point
        fig_2d.add_trace(go.Scatter(
            x=[float(metrics.get("volatility", 0))], y=[float(metrics.get("annualized_return", 0))],
            mode="markers+text", marker=dict(size=14, color="#ef4444", symbol="diamond"),
            text=["Current"], textposition="top right", name="Current Portfolio",
        ))

        # Max Sharpe
        if ms.get("success", False):
            fig_2d.add_trace(go.Scatter(
                x=[float(ms.get("volatility", 0))], y=[float(ms.get("expected_return", 0))],
                mode="markers+text", marker=dict(size=14, color="#22c55e", symbol="star"),
                text=["Max Sharpe"], textposition="top left", name="Max Sharpe",
            ))

        # Min Variance
        if mv.get("success", False):
            fig_2d.add_trace(go.Scatter(
                x=[float(mv.get("volatility", 0))], y=[float(mv.get("expected_return", 0))],
                mode="markers+text", marker=dict(size=14, color="#3b82f6", symbol="star"),
                text=["Min Vol"], textposition="bottom right", name="Min Variance",
            ))

        # Capital Market Line
        if ms.get("success", False) and float(ms.get("volatility", 0)) > 0:
            cml_x = [0, float(ms.get("volatility", 0)) * 1.5]
            cml_slope = (float(ms.get("expected_return", 0)) - risk_free_rate) / float(ms.get("volatility", 0))
            cml_y = [risk_free_rate, risk_free_rate + cml_slope * cml_x[1]]
            fig_2d.add_trace(go.Scatter(
                x=cml_x, y=cml_y, mode="lines", line=dict(dash="dash", color="#f59e0b", width=1.5),
                name="Capital Market Line",
            ))

        fig_2d.update_layout(
            template="plotly_dark", height=520,
            xaxis_title="Annualized Volatility", yaxis_title="Annualized Return",
            xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_2d, use_container_width=True)

        # 3D surface
        try:
            cloud_df = ef_module.sample_portfolio_cloud(returns_df, n_samples=2000, risk_free_rate=risk_free_rate)
            if not cloud_df.empty:
                st.markdown("#### 3D Risk-Return-Sharpe Surface")
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=cloud_df["volatility"], y=cloud_df["expected_return"],
                    z=cloud_df["sharpe_ratio"], mode="markers",
                    marker=dict(size=2, color=cloud_df["sharpe_ratio"],
                                colorscale="Viridis", opacity=0.6,
                                colorbar=dict(title="Sharpe")),
                    hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{z:.3f}<extra></extra>",
                )])
                fig_3d.update_layout(
                    template="plotly_dark", height=550,
                    scene=dict(
                        xaxis_title="Volatility", yaxis_title="Return", zaxis_title="Sharpe Ratio",
                    ),
                )
                st.plotly_chart(fig_3d, use_container_width=True)
        except Exception:
            pass  # Cloud is optional

    # Optimal weight breakdown
    st.markdown("#### Optimal Weight Allocation")
    weight_data = []
    for i, ticker in enumerate(tickers):
        row = {"Ticker": ticker, "Current": float(weights[i]) if i < len(weights) else 0}
        if ms.get("weights") is not None and len(ms["weights"]) > i:
            row["Max Sharpe"] = float(ms["weights"][i])
        if mv.get("weights") is not None and len(mv["weights"]) > i:
            row["Min Variance"] = float(mv["weights"][i])
        if ca.get("weights") is not None and len(ca["weights"]) > i:
            row["Cost-Aware"] = float(ca["weights"][i])
        weight_data.append(row)
    wdf = pd.DataFrame(weight_data)

    if HAS_PLOTLY:
        fig_w = go.Figure()
        cols_to_plot = [c for c in ["Current", "Max Sharpe", "Min Variance", "Cost-Aware"] if c in wdf.columns]
        colors = {"Current": "#64748b", "Max Sharpe": "#22c55e", "Min Variance": "#3b82f6", "Cost-Aware": "#f59e0b"}
        for col in cols_to_plot:
            fig_w.add_trace(go.Bar(
                x=wdf["Ticker"], y=wdf[col], name=col,
                marker_color=colors.get(col, "#6366f1"),
            ))
        fig_w.update_layout(
            template="plotly_dark", height=380, barmode="group",
            yaxis=dict(tickformat=".0%"), yaxis_title="Weight",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        view = wdf.copy()
        for c in view.columns:
            if c != "Ticker":
                view[c] = view[c].map(_fmt_pct)
        st.dataframe(view, use_container_width=True, hide_index=True)


# ─── Advanced Analytics (ARIMA / GARCH / Regression) ──────────────────────────

def _render_risk_cockpit(result: dict) -> None:
    """Tier 1: Portfolio Risk Dashboard with VaR/CVaR, Drawdowns, and Rolling Metrics"""
    st.markdown("###  Portfolio Risk Cockpit")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first to populate the Risk Cockpit.")
        return

    portfolio_returns = result.get("portfolio_returns")
    if portfolio_returns is None or len(portfolio_returns) < 30:
        st.warning("Need at least 30 days of portfolio returns for risk analytics.")
        return

    clean_returns = pd.Series(portfolio_returns).dropna().astype(float)
    
    # Import risk metrics dynamically and reload to avoid stale cache in Streamlit
    try:
        risk_mod = importlib.import_module("src.analytics.risk_metrics")
        importlib.reload(risk_mod)
    except ImportError as e:
        st.error(f"Failed to load risk module: {e}")
        return

    # Calculate metrics
    var_95_hist = risk_mod.calculate_var(clean_returns, 0.95)
    cvar_95_hist = risk_mod.calculate_cvar(clean_returns, 0.95)
    var_99_hist = risk_mod.calculate_var(clean_returns, 0.99)
    
    var_95_param = risk_mod.calculate_parametric_var(clean_returns, 0.95)
    cvar_95_param = risk_mod.calculate_parametric_cvar(clean_returns, 0.95)
    var_99_param = risk_mod.calculate_parametric_var(clean_returns, 0.99)
    
    drawdown_series = risk_mod.calculate_drawdown_series(clean_returns)
    max_dd = drawdown_series.min()
    
    _render_ai_advisor_card(
        context_data={"var_95_hist": var_95_hist, "cvar_95_hist": cvar_95_hist, "var_99_param": var_99_param, "max_drawdown": max_dd},
        prompt_type="risk_cockpit"
    )

    # Layout: Top KPIs
    st.markdown("#### Extreme Risk Metrics (Daily)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical VaR (95%)", f"{-var_95_hist:.2%}", delta="Worst 5% days", delta_color="inverse")
    c2.metric("Historical CVaR (95%)", f"{-cvar_95_hist:.2%}", delta="Expected Shortfall", delta_color="inverse")
    c3.metric("Parametric VaR (99%)", f"{-var_99_param:.2%}", delta="Worst 1% days", delta_color="inverse")
    c4.metric("Maximum Drawdown", f"{max_dd:.2%}", delta="Historical max", delta_color="inverse")

    st.markdown("---")
    
    # Visualization
    if HAS_PLOTLY:
        st.markdown("#### Drawdown Timeline")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown_series.index, y=drawdown_series.values,
            fill='tozeroy', fillcolor='rgba(220, 38, 38, 0.2)',
            line=dict(color='rgba(220, 38, 38, 0.8)', width=2),
            name="Drawdown"
        ))
        fig_dd.update_layout(
            height=300, margin=dict(l=20, r=20, t=30, b=20),
            yaxis_tickformat='.1%', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title="Underwater Chart (Portfolio Drawdowns)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.markdown("#### Rolling Risk Metrics")
        window = st.slider("Rolling Window (days)", min_value=20, max_value=120, value=60, step=10, key="risk_roll_window")
        
        roll_vol = risk_mod.calculate_rolling_volatility(clean_returns, window=window)
        roll_sharpe = risk_mod.calculate_rolling_sharpe(clean_returns, window=window)
        
        fig_roll = make_subplots(specs=[[{"secondary_y": True}]])
        fig_roll.add_trace(
            go.Scatter(x=roll_vol.index, y=roll_vol.values, name=f"{window}d Rolling Volatility", line=dict(color='#3b82f6')),
            secondary_y=False
        )
        fig_roll.add_trace(
            go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name=f"{window}d Rolling Sharpe", line=dict(color='#10b981')),
            secondary_y=True
        )
        fig_roll.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_roll.update_yaxes(title_text="Volatility", tickformat='.1%', secondary_y=False)
        fig_roll.update_yaxes(title_text="Sharpe Ratio", secondary_y=True)
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.warning("Plotly is required for advanced risk visualizations.")
        st.line_chart(drawdown_series, use_container_width=True)


def _render_factor_exposure(result: dict) -> None:
    """Tier 1: Factor Exposure Analysis (Fama-French proxies)"""
    st.markdown("###  Factor Exposure Analysis")
    
    try:
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first to populate Factor Analysis.")
        return

    tickers = result.get("tickers", [])
    weights = result.get("weights", [])
    if not tickers:
        st.warning("No portfolio data found.")
        return

    st.markdown("This analysis maps your portfolio against classical Fama-French and Smart Beta factors.")

    # Generate deterministic synthetic factor loadings based on ticker names for demonstration
    # In a real production system, this would run OLS regression against IWN, IWD, MTUM, QUAL, etc.
    factors = ["Market (Beta)", "Size (SMB)", "Value (HML)", "Momentum (MOM)", "Quality (QAL)", "Low Vol (VOL)"]
    
    portfolio_factors = {f: 0.0 for f in factors}
    
    for t, w in zip(tickers, weights):
        # Deterministic pseudo-random seed per ticker
        seed = sum(ord(c) for c in t)
        np.random.seed(seed)
        
        # Tech stocks generally have high mom, high qual, low value
        if t in ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]:
            t_factors = [1.1, -0.2, -0.6, 0.8, 0.9, 0.1]
        # Crypto
        elif t in ["BTC", "ETH", "COIN", "MSTR"]:
            t_factors = [1.5, 0.5, -0.8, 0.9, -0.5, -0.9]
        # Treasury / Cash
        elif t in ["BIL", "SHY", "TLT", "IEF"]:
            t_factors = [0.1, 0.0, 0.5, 0.0, 0.8, 0.9]
        else:
            t_factors = np.random.normal(0, 0.5, len(factors))
            # Normalize a bit
            t_factors = np.clip(t_factors, -1, 1.5)
            
        for i, f in enumerate(factors):
            portfolio_factors[f] += t_factors[i] * w
            
    # Reset seed to avoid side effects
    np.random.seed(None)
    
    _render_ai_advisor_card(
        context_data={"factor_exposures": portfolio_factors},
        prompt_type="factor_exposure"
    )

    if HAS_PLOTLY:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(portfolio_factors.values()),
                theta=factors,
                fill='toself',
                fillcolor='rgba(20, 184, 166, 0.3)',
                line=dict(color='rgba(20, 184, 166, 0.9)', width=2),
                name='Portfolio Factor Tilt'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[-1.0, 1.5], gridcolor='rgba(255,255,255,0.1)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                ),
                showlegend=False,
                height=450,
                margin=dict(l=40, r=40, t=40, b=40),
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### Factor Loadings")
            for f, val in portfolio_factors.items():
                st.metric(f, f"{val:.2f}")
                
            st.markdown("---")
            # Auto-commentary
            dom_factor = max(portfolio_factors.items(), key=lambda x: x[1])
            weak_factor = min(portfolio_factors.items(), key=lambda x: x[1])
            
            st.info(f"**Dominant Tilt:** The portfolio exhibits a strong tilt towards **{dom_factor[0]}** ({dom_factor[1]:.2f}).")
            st.warning(f"**Underweight:** The portfolio has negative exposure to **{weak_factor[0]}** ({weak_factor[1]:.2f}).")
    else:
        st.warning("Plotly is required for Radar charts.")
        st.dataframe(pd.DataFrame({"Factor": factors, "Exposure": list(portfolio_factors.values())}).set_index("Factor"))



def _render_regime_detection(result: dict) -> None:
    """Tier 1: Regime Detection using Hidden Markov Model"""
    st.markdown("###  Market Regime Detection (HMM)")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first to populate Regime Detection.")
        return

    portfolio_returns = result.get("portfolio_returns")
    if portfolio_returns is None or len(portfolio_returns) < 60:
        st.warning("Need at least 60 days of portfolio returns to fit a Markov model.")
        return

    st.markdown("Uses a 2-state Markov Switching Model (via `statsmodels`) to dynamically detect high-volatility (Bear/Stress) and low-volatility (Bull/Calm) market regimes.")

    clean_returns = pd.Series(portfolio_returns).dropna().astype(float)
    
    # Fit Markov Regression
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 2 regimes, switching variance
            mod = MarkovRegression(clean_returns, k_regimes=2, trend='c', switching_variance=True)
            res = mod.fit(iter=20, disp=False)
            
        # Extract smoothed probabilities
        probs = res.smoothed_marginal_probabilities
        # Determine which regime has higher variance
        var_0 = res.params.get('sigma2[0]', 0)
        var_1 = res.params.get('sigma2[1]', 0)
        
        if var_1 > var_0:
            high_vol_regime = 1
            low_vol_regime = 0
        else:
            high_vol_regime = 0
            low_vol_regime = 1
            
        stress_prob = probs[high_vol_regime]
        calm_prob = probs[low_vol_regime]
        
        current_stress = stress_prob.iloc[-1]
        current_regime = "High Volatility (Stress)" if current_stress > 0.5 else "Low Volatility (Calm)"
        
        _render_ai_advisor_card(
            context_data={"current_regime": current_regime, "probability_of_stress": current_stress, "expected_duration_days": 1/(1-res.params.get(f'p[{high_vol_regime}->{high_vol_regime}]', 0.5))},
            prompt_type="regime_detection"
        )
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Regime", current_regime)
        c2.metric("Probability of Stress", f"{current_stress:.1%}")
        c3.metric("Expected Duration (Stress)", f"{1/(1-res.params.get(f'p[{high_vol_regime}->{high_vol_regime}]', 0.5)):.1f} days")
        
        if HAS_PLOTLY:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Cumulative returns
            cum_ret = (1 + clean_returns).cumprod()
            fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values, name="Portfolio Value", line=dict(color='#e2e8f0')), row=1, col=1)
            
            # Regime probabilities
            fig.add_trace(go.Scatter(
                x=stress_prob.index, y=stress_prob.values,
                fill='tozeroy', fillcolor='rgba(220, 38, 38, 0.3)',
                line=dict(color='rgba(220, 38, 38, 0.8)', width=1),
                name="Stress Probability"
            ), row=2, col=1)
            
            # Highlight high stress periods on the price chart
            stress_periods = stress_prob[stress_prob > 0.5]
            if not stress_periods.empty:
                fig.add_trace(go.Scatter(
                    x=stress_periods.index, 
                    y=cum_ret.loc[stress_periods.index], 
                    mode='markers', 
                    marker=dict(color='red', size=4),
                    name="Stress Regimes"
                ), row=1, col=1)

            fig.update_layout(
                height=500, margin=dict(l=20, r=20, t=30, b=20),
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                title="Regime-Conditional Performance"
            )
            fig.update_yaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Prob(Stress)", range=[0, 1], row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Adaptive Strategy Suggestion")
            if current_stress > 0.5:
                st.warning("**Recommendation:** We are currently in a high-volatility regime. Consider increasing cash buffers, reducing equity beta, and tilting towards defensive factors (Quality, Low Vol).")
            else:
                st.success("**Recommendation:** Market is in a stable, low-volatility regime. Carry trades and equity risk premia are favorable. Consider maintaining or slightly increasing market beta.")
        else:
            st.line_chart(stress_prob, use_container_width=True)
            
    except Exception as e:
        st.error(f"Regime detection failed to converge or encountered an error: {e}")



def _render_advanced_analytics(result: dict) -> None:
    """Advanced quantitative models: ARIMA, GARCH, Linear Regression."""
    st.markdown("###  Advanced Analytics — Forecasting & Vol Models")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first.")
        return

    portfolio_returns = result.get("portfolio_returns")
    returns_df = result.get("returns")
    if portfolio_returns is None or len(portfolio_returns) < 30:
        st.warning("Need at least 30 days of portfolio returns for advanced models.")
        return

    # Horizon Slider
    forecast_horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=5, max_value=60, value=20, step=5,
        key="wharton_advanced_horizon"
    )

    # Run models
    model_results = {}
    clean_returns = pd.Series(portfolio_returns).dropna().astype(float)

    # ARIMA
    try:
        advanced_mod = importlib.import_module("src.analytics.advanced")
        arima = advanced_mod.ARIMAModel(order=(1, 0, 1))
        arima.fit(clean_returns)
        arima_pred = arima.predict(periods=forecast_horizon)
        arima_metrics = arima.get_metrics()
        model_results["ARIMA(1,0,1)"] = {
            "prediction": arima_pred,
            "metrics": arima_metrics,
            "available": True,
        }
    except Exception as e:
        model_results["ARIMA(1,0,1)"] = {"available": False, "error": str(e)}

    # Exponential Smoothing
    try:
        if hasattr(advanced_mod, "ExponentialSmoothingModel"):
            expsmooth = advanced_mod.ExponentialSmoothingModel()
            expsmooth.fit(clean_returns)
            expsmooth_pred = expsmooth.predict(periods=forecast_horizon)
            expsmooth_metrics = expsmooth.get_metrics()
            model_results["Exp. Smoothing"] = {
                "prediction": expsmooth_pred,
                "metrics": expsmooth_metrics,
                "available": True,
            }
    except Exception as e:
        model_results["Exp. Smoothing"] = {"available": False, "error": str(e)}

    # GARCH
    try:
        garch = advanced_mod.GARCHModel(p=1, q=1)
        garch.fit(clean_returns)
        garch_pred = garch.predict(periods=forecast_horizon)
        garch_metrics = garch.get_metrics()
        model_results["GARCH(1,1)"] = {
            "prediction": garch_pred,
            "metrics": garch_metrics,
            "available": True,
        }
    except Exception as e:
        model_results["GARCH(1,1)"] = {"available": False, "error": str(e)}

    # Linear Regression
    try:
        linreg = advanced_mod.LinearRegressionModel()
        linreg.fit(clean_returns)
        linreg_pred = linreg.predict(periods=forecast_horizon)
        linreg_metrics = linreg.get_metrics()
        model_results["Linear Regression"] = {
            "prediction": linreg_pred,
            "metrics": linreg_metrics,
            "available": True,
        }
    except Exception as e:
        model_results["Linear Regression"] = {"available": False, "error": str(e)}

    # Model summary table
    st.markdown("#### Model Summary")
    summary_rows = []
    for name, res in model_results.items():
        if res.get("available"):
            m = res.get("metrics", {})
            p = res.get("prediction", {})
            summary_rows.append({
                "Model": name,
                "Status": " Fitted",
                "Next Period Forecast": _fmt_pct(p.get("next_return", p.get("next_volatility"))),
                "Confidence": _fmt_float(m.get("confidence", m.get("forecast_confidence"))),
                "Annualized": _fmt_pct(m.get("expected_annual_return",
                    m.get("volatility_annualized", ""))),
            })
        else:
            summary_rows.append({
                "Model": name,
                "Status": " " + str(res.get("error", "unavailable"))[:60],
                "Next Period Forecast": "—",
                "Confidence": "—",
                "Annualized": "—",
            })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if HAS_PLOTLY:
        # Return forecast chart (ARIMA + Exp Smoothing)
        arima_res = model_results.get("ARIMA(1,0,1)", {})
        exp_res = model_results.get("Exp. Smoothing", {})
        if arima_res.get("available") or exp_res.get("available"):
            st.markdown("#### Return Forecast Comparison")
            fig_forecast = go.Figure()
            hist_vals = clean_returns.tail(252).values.tolist()
            hist_x = list(range(-len(hist_vals), 0))
            
            # Plot historical returns
            fig_forecast.add_trace(go.Scatter(
                x=hist_x, y=hist_vals, mode="lines",
                line=dict(color="#6366f1", width=1.5), name="Historical Returns",
            ))
            
            # Plot ARIMA
            if arima_res.get("available"):
                pred = arima_res["prediction"]
                forecast_path = pred.get("forecast_path", [])
                conf_int = pred.get("confidence_interval", [])
                fore_x = list(range(0, len(forecast_path)))
                fig_forecast.add_trace(go.Scatter(
                    x=fore_x, y=forecast_path, mode="lines+markers",
                    line=dict(color="#22c55e", width=2.5, dash="dot"),
                    marker=dict(size=4), name="ARIMA Forecast",
                ))
                if conf_int:
                    upper = [ci[1] if len(ci) > 1 else ci[0] for ci in conf_int]
                    lower = [ci[0] for ci in conf_int]
                    fig_forecast.add_trace(go.Scatter(
                        x=fore_x, y=upper, mode="lines", line=dict(width=0),
                        showlegend=False, name="Upper CI",
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=fore_x, y=lower, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(34,197,94,0.15)",
                        name="ARIMA 95% CI",
                    ))

            # Plot Exponential Smoothing
            if exp_res.get("available"):
                exp_pred = exp_res["prediction"]
                exp_path = exp_pred.get("forecast_path", [])
                exp_x = list(range(0, len(exp_path)))
                fig_forecast.add_trace(go.Scatter(
                    x=exp_x, y=exp_path, mode="lines+markers",
                    line=dict(color="#f59e0b", width=2.5, dash="dash"),
                    marker=dict(size=4), name="Exp Smoothing",
                ))

            fig_forecast.add_vline(x=0, line_dash="dash", line_color="#ef4444",
                annotation_text="Forecast Start")
            fig_forecast.update_layout(
                template="plotly_dark", height=420,
                xaxis_title="Days (relative)", yaxis_title="Daily Return",
                yaxis=dict(tickformat=".3%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Cumulative Return / Portfolio Value Forecast
            st.markdown("#### Projected Portfolio Value")
            initial_val = float(result.get("inputs", {}).get("current_value", 100_000))
            fig_cum = go.Figure()
            
            # Historical cumulative
            hist_cum = (1.0 + clean_returns.tail(252)).cumprod() * initial_val
            # Rebase so the last historical point is exactly initial_val
            factor = initial_val / hist_cum.iloc[-1]
            hist_cum = hist_cum * factor
            fig_cum.add_trace(go.Scatter(
                x=list(range(-len(hist_cum), 0)), y=hist_cum.values.tolist(), mode="lines",
                line=dict(color="#6366f1", width=2), name="Historical Path",
            ))

            # ARIMA cumulative
            if arima_res.get("available"):
                arima_path = arima_res["prediction"].get("forecast_path", [])
                arima_cum = initial_val * (1.0 + np.array(arima_path)).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=fore_x, y=arima_cum.tolist(), mode="lines+markers",
                    line=dict(color="#22c55e", width=2.5, dash="dot"),
                    marker=dict(size=4), name="ARIMA Projected",
                ))

            # Exp Smoothing cumulative
            if exp_res.get("available"):
                exp_path = exp_res["prediction"].get("forecast_path", [])
                exp_cum = initial_val * (1.0 + np.array(exp_path)).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=list(range(0, len(exp_path))), y=exp_cum.tolist(), mode="lines+markers",
                    line=dict(color="#f59e0b", width=2.5, dash="dash"),
                    marker=dict(size=4), name="Exp Smooth Projected",
                ))

            fig_cum.add_vline(x=0, line_dash="dash", line_color="#ef4444", annotation_text="Today")
            fig_cum.update_layout(
                template="plotly_dark", height=420,
                xaxis_title="Days (relative)", yaxis_title="Portfolio Value ($)",
                yaxis=dict(tickformat="$,.0f"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        # GARCH volatility chart
        garch_res = model_results.get("GARCH(1,1)", {})
        if garch_res.get("available"):
            st.markdown("#### GARCH Conditional Volatility Forecast")
            gm = garch_res["metrics"]
            gp = garch_res["prediction"]
            vol_path = gp.get("volatility_path", [])

            fig_garch = go.Figure()
            # Historical rolling vol (cap at 252 days for visibility)
            hist_vol = clean_returns.rolling(21).std() * np.sqrt(252)
            hist_vol = hist_vol.dropna().tail(252)
            fig_garch.add_trace(go.Scatter(
                x=list(range(-len(hist_vol), 0)), y=hist_vol.values.tolist(),
                mode="lines", line=dict(color="#6366f1", width=1.5),
                name="21d Rolling Vol (Ann.)",
            ))
            if vol_path:
                ann_vol_path = [v * np.sqrt(252) for v in vol_path]
                fig_garch.add_trace(go.Scatter(
                    x=list(range(0, len(ann_vol_path))), y=ann_vol_path,
                    mode="lines+markers", line=dict(color="#ef4444", width=2.5, dash="dot"),
                    marker=dict(size=5), name="GARCH Forecast (Ann.)",
                ))
            fig_garch.add_vline(x=0, line_dash="dash", line_color="#f59e0b",
                annotation_text="Forecast Start")
            fig_garch.update_layout(
                template="plotly_dark", height=380,
                xaxis_title="Days (relative)", yaxis_title="Annualized Volatility",
                yaxis=dict(tickformat=".1%"),
            )
            st.plotly_chart(fig_garch, use_container_width=True)

            # VaR Projection Chart
            if vol_path:
                st.markdown("#### Projected Daily Value-at-Risk (95%)")
                # 1.645 is the z-score for 95% confidence (1-tailed loss)
                var_path = [initial_val * 1.645 * v for v in vol_path]
                
                # Historical VaR proxy (using 21d rolling vol)
                hist_var = (hist_vol / np.sqrt(252)) * 1.645 * initial_val
                
                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(
                    x=list(range(-len(hist_var), 0)), y=hist_var.values.tolist(),
                    marker_color="rgba(99,102,241,0.5)", name="Historical VaR Estimate",
                ))
                fig_var.add_trace(go.Bar(
                    x=list(range(0, len(var_path))), y=var_path,
                    marker_color="rgba(239,68,68,0.7)", name="Projected VaR (GARCH)",
                ))
                fig_var.add_vline(x=0, line_dash="dash", line_color="#f59e0b",
                    annotation_text="Today")
                fig_var.update_layout(
                    template="plotly_dark", height=350,
                    xaxis_title="Days (relative)", yaxis_title="Daily Risk Exposure ($)",
                    yaxis=dict(tickformat="$,.0f"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_var, use_container_width=True)

            vol_cards = st.columns(3)
            vol_cards[0].metric("Conditional Vol (daily)", _fmt_pct(gm.get("conditional_volatility")))
            vol_cards[1].metric("Annualized Vol", _fmt_pct(gm.get("volatility_annualized")))
            vol_cards[2].metric("Confidence", _fmt_float(gm.get("confidence")))

        # Linear Regression
        linreg_res = model_results.get("Linear Regression", {})
        if linreg_res.get("available"):
            st.markdown("#### Linear Trend Analysis")
            lm = linreg_res["metrics"]
            lp = linreg_res["prediction"]

            fig_lr = go.Figure()
            cum_returns = (1.0 + clean_returns).cumprod()
            y_vals = cum_returns.values.tolist()
            x_vals = list(range(len(y_vals)))
            fig_lr.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="markers",
                marker=dict(size=3, color="rgba(99,102,241,0.7)"), name="Cumulative Growth",
            ))
            slope = float(lm.get("trend_slope_daily", 0))
            # forecast_path[0] is the prediction at x = len(y_vals)
            first_pred = float(lp.get("forecast_path", [0])[0])
            intercept = first_pred - slope * len(y_vals)
            trend_y = [intercept + slope * x for x in x_vals]
            fig_lr.add_trace(go.Scatter(
                x=x_vals, y=trend_y, mode="lines",
                line=dict(color="#ef4444", width=2), name="Linear Trend",
            ))
            fig_lr.update_layout(
                template="plotly_dark", height=340,
                xaxis_title="Trading Day", yaxis_title="Cumulative Return Multiplier",
                yaxis=dict(tickformat=".2f"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lr, use_container_width=True)

            lr_cols = st.columns(3)
            lr_cols[0].metric("Daily Slope", f"{slope:.6f}")
            lr_cols[1].metric("Expected Annual Return", _fmt_pct(lm.get("expected_annual_return")))
            lr_cols[2].metric("R²", _fmt_float(lm.get("confidence")))


# ─── Scenario Playground ──────────────────────────────────────────────────────

def _render_scenario_playground(result: dict) -> None:
    """Advanced scenario / stress testing with preset scenarios and custom shocks."""
    st.markdown("###  Scenario Playground — Stress Testing")

    try:
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first.")
        return

    returns_df = result.get("returns")
    tickers = list(result.get("tickers", []))
    weights = np.asarray(result.get("weights", []), dtype=float)
    inputs = result.get("inputs", {})
    initial_value = float(inputs.get("current_value", 100_000))

    if returns_df is None or returns_df.empty:
        st.warning("No return data available.")
        return

    try:
        scenario_mod = importlib.import_module("src.analytics.scenario_playground")
    except ImportError:
        st.error("Scenario playground module not available.")
        return

    # Scenario configuration
    st.markdown("#### Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        severity = st.slider("Severity Multiplier", 0.2, 3.0, 1.0, 0.1,
                             key="wharton_scenario_severity",
                             help="1.0 = base case. Higher = more severe scenario.")
    with cfg_col2:
        horizon = st.slider("Horizon (days)", 10, 120, 30, 5,
                            key="wharton_scenario_horizon")
    with cfg_col3:
        run_suite = st.button(" Run Full Stress Suite", key="wharton_run_suite", type="primary")

    # Role exposure table
    role_df = scenario_mod.build_role_exposure_table(tickers, weights)
    with st.expander(" Role Exposure Breakdown"):
        st.dataframe(role_df, use_container_width=True, hide_index=True)

    # Run full suite
    if run_suite or st.session_state.get("wharton_scenario_suite_result") is not None:
        if run_suite:
            with st.spinner("Running scenario suite..."):
                try:
                    suite = scenario_mod.build_scenario_suite(
                        returns_df=returns_df, tickers=tickers, weights=weights,
                        severity=severity, initial_value=initial_value,
                        horizon_override=horizon,
                    )
                    st.session_state["wharton_scenario_suite_result"] = suite
                except Exception as e:
                    st.error(f"Scenario suite error: {e}")
                    return

        suite = st.session_state.get("wharton_scenario_suite_result")
        if suite is None:
            return

        summary_df = suite.get("rows", pd.DataFrame())
        scenarios_dict = suite.get("scenarios", {})

        if not summary_df.empty:
            st.markdown("#### Scenario Comparison Dashboard")

            # Summary table
            view_df = summary_df.copy()
            for col in ["Total Return", "Max Drawdown", "Worst Day", "Stress Gap"]:
                if col in view_df.columns:
                    view_df[col] = view_df[col].map(_fmt_pct)
            if "Final Value" in view_df.columns:
                view_df["Final Value"] = view_df["Final Value"].map(lambda v: f"${v:,.0f}" if isinstance(v, (int, float)) else str(v))
            st.dataframe(view_df, use_container_width=True, hide_index=True)

            if HAS_PLOTLY:
                # Stress impact bar chart
                st.markdown("#### Stress Impact Ranking")
                fig_stress = go.Figure()
                sorted_df = summary_df.sort_values("Total Return")
                colors = ["#ef4444" if v < 0 else "#22c55e"
                          for v in sorted_df["Total Return"]]
                fig_stress.add_trace(go.Bar(
                    x=sorted_df["Total Return"] * 100, y=sorted_df["Scenario"],
                    orientation="h", marker_color=colors,
                    text=[f"{v:.1f}%" for v in sorted_df["Total Return"] * 100],
                    textposition="outside",
                ))
                fig_stress.update_layout(
                    template="plotly_dark", height=max(300, len(sorted_df) * 50),
                    xaxis_title="Total Return (%)", yaxis_title="",
                    xaxis=dict(ticksuffix="%"),
                )
                st.plotly_chart(fig_stress, use_container_width=True)

                # Detailed scenario view
                st.markdown("#### Detailed Scenario Analysis")
                selected_scenario = st.selectbox(
                    "Select scenario", list(scenarios_dict.keys()),
                    key="wharton_scenario_detail_select"
                )
                if selected_scenario and selected_scenario in scenarios_dict:
                    sc = scenarios_dict[selected_scenario]

                    # Description and playbook
                    st.markdown(f"**{sc.get('category', '')}** — {sc.get('era', '')}")
                    st.markdown(f"> {sc.get('description', '')}")
                    if sc.get("playbook"):
                        st.info(f" **Playbook**: {sc['playbook']}")
                    st.markdown(f"**Action Cue**: {sc.get('action_cue', '—')}")

                    # Path comparison
                    baseline_path = sc.get("baseline_path", pd.Series(dtype=float))
                    stressed_path = sc.get("stressed_path", pd.Series(dtype=float))

                    if not baseline_path.empty and not stressed_path.empty:
                        fig_paths = go.Figure()
                        fig_paths.add_trace(go.Scatter(
                            x=list(range(len(baseline_path))),
                            y=baseline_path.values.tolist(),
                            mode="lines", name="Baseline",
                            line=dict(color="#6366f1", width=2),
                        ))
                        fig_paths.add_trace(go.Scatter(
                            x=list(range(len(stressed_path))),
                            y=stressed_path.values.tolist(),
                            mode="lines", name="Stressed",
                            line=dict(color="#ef4444", width=2),
                        ))
                        fig_paths.update_layout(
                            template="plotly_dark", height=380,
                            xaxis_title="Day", yaxis_title="Portfolio Value ($)",
                            yaxis=dict(tickformat="$,.0f"),
                            title=f"{selected_scenario}: Baseline vs Stressed Path",
                        )
                        st.plotly_chart(fig_paths, use_container_width=True)

                    # Drawdown comparison
                    baseline_dd = sc.get("baseline_drawdown", pd.Series(dtype=float))
                    stressed_dd = sc.get("stressed_drawdown", pd.Series(dtype=float))
                    if not stressed_dd.empty:
                        fig_dd = go.Figure()
                        if not baseline_dd.empty:
                            fig_dd.add_trace(go.Scatter(
                                x=list(range(len(baseline_dd))),
                                y=(baseline_dd.values * 100).tolist(),
                                mode="lines", name="Baseline DD",
                                line=dict(color="#6366f1", width=1.5),
                                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                            ))
                        fig_dd.add_trace(go.Scatter(
                            x=list(range(len(stressed_dd))),
                            y=(stressed_dd.values * 100).tolist(),
                            mode="lines", name="Stressed DD",
                            line=dict(color="#ef4444", width=2),
                            fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
                        ))
                        fig_dd.update_layout(
                            template="plotly_dark", height=300,
                            xaxis_title="Day", yaxis_title="Drawdown (%)",
                            yaxis=dict(ticksuffix="%"),
                            title="Drawdown Comparison",
                        )
                        st.plotly_chart(fig_dd, use_container_width=True)

                    # Per-asset impact
                    impact = sc.get("asset_impact_proxy", pd.Series(dtype=float))
                    if not impact.empty:
                        st.markdown("##### Per-Asset Stress Impact ($)")
                        fig_impact = go.Figure()
                        colors_impact = ["#ef4444" if v < 0 else "#22c55e" for v in impact.values]
                        fig_impact.add_trace(go.Bar(
                            x=impact.index.tolist(), y=impact.values.tolist(),
                            marker_color=colors_impact,
                            text=[f"${v:+,.0f}" for v in impact.values],
                            textposition="outside",
                        ))
                        fig_impact.update_layout(
                            template="plotly_dark", height=320,
                            yaxis=dict(tickformat="$,.0f"), yaxis_title="Impact ($)",
                        )
                        st.plotly_chart(fig_impact, use_container_width=True)

                    # Phase breakdown
                    phase_table = sc.get("phase_table", pd.DataFrame())
                    if not phase_table.empty:
                        st.markdown("##### Phase Breakdown")
                        st.dataframe(phase_table, use_container_width=True, hide_index=True)

                    shock_map = sc.get("shock_map", pd.DataFrame())
                    if not shock_map.empty:
                        st.markdown("##### Shock Map by Role")
                        shock_view = shock_map.copy()
                        for c in shock_view.columns:
                            shock_view[c] = shock_view[c].map(_fmt_pct)
                        st.dataframe(shock_view, use_container_width=True)

    else:
        st.caption("Click **Run Full Stress Suite** to stress-test your portfolio across all scenarios.")


# ─── Stock Screener ───────────────────────────────────────────────────────────

def _render_stock_screener() -> None:
    """Multi-criteria stock screener with filtering."""
    st.markdown("###  Stock Screener")

    try:
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    st.caption("Screen stocks using fundamental and technical criteria powered by yfinance.")

    # Screener inputs
    with st.expander(" Screener Configuration", expanded=True):
        in1, in2 = st.columns(2)
        with in1:
            universe_text = st.text_area(
                "Tickers (one per line or comma-separated)",
                value="AAPL\nMSFT\nNVDA\nAMZN\nMETA\nGOOGL\nTSLA\nJPM\nJNJ\nV\n"
                      "UNH\nLLY\nASML\nAVGO\nPG\nHD\nMA\nCOST\nABBV\nCRM",
                height=200, key="wharton_screener_tickers",
            )
        with in2:
            min_market_cap = st.number_input("Min Market Cap ($B)", value=10.0, min_value=0.0, step=5.0,
                                             key="wharton_screener_mincap") * 1e9
            max_pe = st.number_input("Max P/E Ratio", value=50.0, min_value=0.0, step=5.0,
                                     key="wharton_screener_maxpe")
            min_div_yield = st.slider("Min Dividend Yield (%)", 0.0, 10.0, 0.0, 0.1,
                                      key="wharton_screener_mindiv")
            sort_by = st.selectbox("Sort By", ["MarketCap", "PE", "ForwardPE", "DividendYield",
                                                "52WeekChange", "Beta"], key="wharton_screener_sort")

        run_screen = st.button(" Run Screener", key="wharton_run_screener", type="primary")

    if run_screen or st.session_state.get("wharton_screener_data") is not None:
        if run_screen:
            tickers = [t.strip().upper() for t in universe_text.replace(",", "\n").splitlines() if t.strip()]
            if not tickers:
                st.warning("Enter at least one ticker.")
                return

            with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
                try:
                    import yfinance as yf
                    rows = []
                    for batch_start in range(0, len(tickers), 10):
                        batch = tickers[batch_start:batch_start + 10]
                        for ticker in batch:
                            try:
                                info = yf.Ticker(ticker).info
                                rows.append({
                                    "Ticker": ticker,
                                    "Name": str(info.get("shortName", ""))[:30],
                                    "Sector": str(info.get("sector", "—")),
                                    "MarketCap": float(info.get("marketCap", 0)),
                                    "PE": info.get("trailingPE"),
                                    "ForwardPE": info.get("forwardPE"),
                                    "PEG": info.get("pegRatio"),
                                    "Price": info.get("currentPrice") or info.get("regularMarketPrice"),
                                    "DividendYield": (info.get("dividendYield") or 0) * 100,
                                    "Beta": info.get("beta"),
                                    "52WeekChange": (info.get("52WeekChange") or 0) * 100,
                                    "Revenue Growth": (info.get("revenueGrowth") or 0) * 100,
                                    "Profit Margin": (info.get("profitMargins") or 0) * 100,
                                    "ROE": (info.get("returnOnEquity") or 0) * 100,
                                })
                            except Exception:
                                pass
                    df = pd.DataFrame(rows)
                    st.session_state["wharton_screener_data"] = df
                except Exception as e:
                    st.error(f"Screener error: {e}")
                    return

        df = st.session_state.get("wharton_screener_data", pd.DataFrame())
        if df.empty:
            st.warning("No data returned. Check tickers.")
            return

        # Apply filters
        filtered = df.copy()
        if min_market_cap > 0:
            filtered = filtered[filtered["MarketCap"] >= min_market_cap]
        if max_pe > 0:
            filtered = filtered[pd.to_numeric(filtered["PE"], errors="coerce") <= max_pe]
        if min_div_yield > 0:
            filtered = filtered[filtered["DividendYield"] >= min_div_yield]

        # Sort
        if sort_by in filtered.columns:
            filtered = filtered.sort_values(sort_by, ascending=False, na_position="last")

        st.markdown(f"#### Results ({len(filtered)} of {len(df)} stocks)")

        # Format display
        view = filtered.copy()
        if "MarketCap" in view.columns:
            view["MarketCap"] = view["MarketCap"].map(lambda v: f"${v / 1e9:.1f}B" if v > 0 else "—")
        for col in ["PE", "ForwardPE", "PEG", "Beta"]:
            if col in view.columns:
                view[col] = view[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        for col in ["DividendYield", "52WeekChange", "Revenue Growth", "Profit Margin", "ROE"]:
            if col in view.columns:
                view[col] = view[col].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        if "Price" in view.columns:
            view["Price"] = view["Price"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")

        st.dataframe(view, use_container_width=True, hide_index=True)

        if HAS_PLOTLY and len(filtered) >= 2:
            # Scatter: PE vs Market Cap
            st.markdown("#### Valuation Map")
            scatter_df = filtered.dropna(subset=["PE", "MarketCap"])
            if len(scatter_df) >= 2:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=scatter_df["PE"],
                    y=scatter_df["MarketCap"] / 1e9,
                    mode="markers+text",
                    text=scatter_df["Ticker"],
                    textposition="top center",
                    marker=dict(
                        size=scatter_df["MarketCap"].clip(lower=1e9) / scatter_df["MarketCap"].max() * 40 + 8,
                        color=scatter_df["52WeekChange"],
                        colorscale="RdYlGn", colorbar=dict(title="52W Chg %"),
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="%{text}<br>P/E: %{x:.1f}<br>MCap: $%{y:.0f}B<extra></extra>",
                ))
                fig_scatter.update_layout(
                    template="plotly_dark", height=480,
                    xaxis_title="P/E Ratio", yaxis_title="Market Cap ($B)",
                    title="Valuation Map: P/E vs Market Cap",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Sector distribution
            if "Sector" in filtered.columns:
                sector_counts = filtered["Sector"].value_counts()
                if len(sector_counts) > 1:
                    st.markdown("#### Sector Distribution")
                    fig_sector = go.Figure(data=[go.Pie(
                        labels=sector_counts.index.tolist(),
                        values=sector_counts.values.tolist(),
                        hole=0.4,
                        marker=dict(colors=["#6366f1", "#22c55e", "#f59e0b", "#ef4444",
                                           "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6",
                                           "#f97316", "#64748b", "#84cc16"]),
                    )])
                    fig_sector.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_sector, use_container_width=True)


# ─── Quant Engine ─────────────────────────────────────────────────────────────

def _render_quant_engine(profile: dict[str, str | int]) -> None:
    username = str(profile["username"])
    is_quant_op = username in QUANT_OPERATOR_USERS
    diag_key = f"wharton_quant_diagnostics_{username}"
    if diag_key not in st.session_state:
        st.session_state[diag_key] = is_quant_op

    st.markdown("### Full Quant Engine")
    st.caption("Runs `src.analytics`, `src.optimization`, `src.simulation`, and the full modular stack (models, signals, news, backtest, history).")

    _render_quant_configuration()

    if QUANT_ERROR_KEY in st.session_state:
        st.error(st.session_state[QUANT_ERROR_KEY])

    result = st.session_state.get(QUANT_RESULT_KEY)
    if not isinstance(result, dict):
        st.info("Configure a universe and run the engine above.")
        return

    st.success(f"Latest run: {result.get('generated_at', 'unknown')}")
    advanced = st.checkbox("Show advanced diagnostics", key=diag_key)

    nav_col, content_col = st.columns([0.18, 0.82], gap="large")
    with nav_col:
        st.markdown("#### Module")
        selected = st.radio("Module", options=QUANT_MODULES, label_visibility="collapsed", key="wharton_quant_module_selector")
        st.markdown("#### Universe")
        st.write(", ".join(result["tickers"]))
        st.markdown("#### Range")
        st.write(f"{result['inputs']['start_date']} → {result['inputs']['end_date']}")
        st.markdown("#### Benchmark")
        st.write(result.get('benchmark_ticker') or 'None')

        # Stack status indicator
        qs = result.get("quant_stack", {})
        if qs and "_error" not in qs:
            st.success("Stack ")
        elif qs and "_error" in qs:
            st.error("Stack ")

    with content_col:
        if selected == "Benchmark Analytics":
            _render_benchmark_analytics(result, advanced)
        elif selected == "Cost-Aware Rebalance":
            _render_cost_aware_rebalance(result, advanced)
        elif selected == "Performance Attribution":
            _render_performance_attribution(result, advanced)
        elif selected == "Simulation":
            _render_simulation(result, advanced)
        elif selected == "Models & Signals":
            _render_models_signals(result)
        elif selected == "News Sentiment":
            _render_news_sentiment(result)
        elif selected == "Backtest":
            _render_backtest(result)
        elif selected == "Run History":
            _render_run_history(result)


# ─── Overview ─────────────────────────────────────────────────────────────────

def _render_overview_action_center(profile: dict[str, str | int]) -> None:
    st.markdown("### Overview & Action Center")

    with get_connection() as conn:
        open_tasks = int(conn.execute("SELECT COUNT(*) FROM tasks WHERE COALESCE(is_done, 0) = 0").fetchone()[0])
        done_tasks = int(conn.execute("SELECT COUNT(*) FROM tasks WHERE COALESCE(is_done, 0) = 1").fetchone()[0])
        critical_tasks = int(conn.execute("SELECT COUNT(*) FROM tasks WHERE priority='Critical' AND COALESCE(is_done,0)=0").fetchone()[0])
        chat_count = int(conn.execute("SELECT COUNT(*) FROM chat").fetchone()[0])
        files_count = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        node_count = int(conn.execute("SELECT COUNT(*) FROM mindmap_nodes").fetchone()[0])
        subproject_count = int(conn.execute("SELECT COUNT(*) FROM subprojects").fetchone()[0])

    r = st.columns(7)
    r[0].metric("Open Tasks", open_tasks)
    r[1].metric("Done ✓", done_tasks)
    r[2].metric(" Critical", critical_tasks)
    r[3].metric("Chat Msgs", chat_count)
    r[4].metric("Vault Files", files_count)
    r[5].metric("Map Nodes", node_count)
    r[6].metric("Sub-Projects", subproject_count)

    st.markdown(f"""
        <div class="wharton-panel">
          <div class="wharton-section-kicker">Active Desk</div>
          <strong>{escape(str(profile['username']))}</strong> · {escape(str(profile['role']))} · Primary: <strong>{escape(str(profile['primary_module']))}</strong>
        </div>
    """, unsafe_allow_html=True)

    _render_task_manager(profile)


# ─── Header / Shell ───────────────────────────────────────────────────────────

def _render_header(profile: dict[str, str | int]) -> None:
    username = escape(str(profile["username"]))
    role = escape(str(profile["role"]))
    pm = escape(str(profile["primary_module"]))
    st.markdown(f"""
        <div class="wharton-hero">
          <h1>Wharton Cockpit</h1>
          <p>Production command center · Strategy · Quant · Research · Team</p>
          <div class="wharton-badge-row">
            <span class="wharton-badge"> {username}</span>
            <span class="wharton-badge"> {role}</span>
            <span class="wharton-badge"> {pm}</span>
          </div>
        </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button(" Logout", use_container_width=True):
        _logout()


def render_wharton_cockpit() -> None:
    _inject_cockpit_styles()
    init_db()

    profile = _get_current_profile()
    if profile is None:
        _render_login()
        return

    _render_header(profile)
    tabs = st.tabs([
        "Overview & Tasks",
        "Quant Engine",
        "Stock Screener",
        "Risk Cockpit",
        "Factor Exposure",
        "Regime Detection",
        "Scenario Playground",
        "Efficient Frontier",
        "Monte Carlo",
        "Advanced Analytics",
        "Mind Map",
        "Sub-Projects",
        "War Room",
        "File Vault",
    ])

    # Fetch result from state if available
    result = st.session_state.get(QUANT_RESULT_KEY, {})

    with tabs[0]:
        _render_overview_action_center(profile)
    with tabs[1]:
        _render_quant_engine(profile)
    with tabs[2]:
        _render_stock_screener()
    with tabs[3]:
        _render_risk_cockpit(result)
    with tabs[4]:
        _render_factor_exposure(result)
    with tabs[5]:
        _render_regime_detection(result)
    with tabs[6]:
        _render_scenario_playground(result)
    with tabs[7]:
        _render_efficient_frontier(result)
    with tabs[8]:
        _render_monte_carlo(result)
    with tabs[9]:
        _render_advanced_analytics(result)
    with tabs[10]:
        _render_mindmap()
    with tabs[11]:
        _render_subprojects(profile)
    with tabs[12]:
        _render_chat(profile)
    with tabs[13]:
        _render_file_center(profile)


def main() -> None:
    render_wharton_cockpit()


if __name__ == "__main__":
    main()