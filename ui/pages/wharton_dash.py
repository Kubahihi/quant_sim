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
    connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
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
                priority TEXT,
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
                timestamp TEXT NOT NOT NULL,
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
            for row in conn.execute("SELECT username, password_hash FROM users").fetchall()
        }
        for user in DEFAULT_USERS:
            if existing_users.get(user["username"]):
                continue

            password_hash = bcrypt.hashpw(
                DEFAULT_PASSWORD.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

                # If not in development mode, and no specific secret is set,
                # this will fail authentication later.
            if not _is_development_mode():
                    try:
                        _ = st.secrets["wharton_users"][user["username"]]
                    except (KeyError, AttributeError):
                        st.warning(f"Wharton user {user['username']} seeded with default password. Ensure st.secrets['wharton_users']['{user['username']}'] is set for production.")

            conn.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, role, primary_module) VALUES (?, ?, ?, ?)",
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
            "SELECT id, username, role, primary_module FROM users ORDER BY username COLLATE NOCASE"
        ).fetchall()


def authenticate_user(username: str, password: str) -> dict[str, str | int] | None:
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, username, password_hash, role, primary_module FROM users WHERE username = ?",
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
        username = st.selectbox("Username", options=usernames)
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
    cols[3].metric("🔴 Critical", critical)
    cols[4].metric("🟠 High", high)

    if total > 0:
        pct = done / total
        st.progress(pct, text=f"Completion: {pct:.0%}")


def _render_task_manager(profile: dict[str, str | int]) -> None:
    st.markdown("### Mission Task Board")

    original_tasks = _fetch_task_rows()
    _render_task_stats(original_tasks)

    # Quick-add form (always visible, above the editor)
    with st.expander("➕ Quick Add Task", expanded=False):
        with st.form("wharton_quick_task_form", clear_on_submit=True):
            qc1, qc2, qc3 = st.columns([3, 1, 1])
            with qc1:
                quick_text = st.text_input("Task description")
            with qc2:
                quick_priority = st.selectbox("Priority", TASK_PRIORITIES, index=2)
            with qc3:
                quick_assignee = st.text_input("Assignee", value=str(profile["username"]))
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
            st.warning("⚠️ This also removes all edges connected to this node.")
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

    with st.expander("📤 Upload Files", expanded=True):
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
    search_query = st.text_input("🔍 Search files", placeholder="filename, project, uploader, tags...")

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
            status_icon = "✅" if status == "available" else "❌"
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

    with st.expander("➕ Create New Sub-Project", expanded=False):
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
            f"📁 {sp['name']}  ·  "
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
                        st.markdown(f"📄 **{escape(str(f['filename']))}**")
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

    with st.expander("⚙️ Quant Run Configuration", expanded=QUANT_RESULT_KEY not in st.session_state):
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
            run_clicked = st.form_submit_button("▶ Run Full Quant Engine", type="primary")

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
        st.markdown("#### 📊 Models")
        model_data = []
        for name, model in models.items():
            m_dict = model.to_dict() if hasattr(model, "to_dict") else {}
            model_data.append({
                "Model": name,
                "Available": "✅" if m_dict.get("available") else "❌",
                "Confidence": _fmt_float(m_dict.get("confidence")),
                "Score": _fmt_float(m_dict.get("score")),
                "Signal": str(m_dict.get("signal", "—")),
                "Notes": str(m_dict.get("notes", ""))[:80],
            })
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    if signals:
        st.markdown("#### 📡 Signals")
        signal_data = []
        for name, signal in signals.items():
            s_dict = signal.to_dict() if hasattr(signal, "to_dict") else {}
            signal_data.append({
                "Signal": name,
                "Available": "✅" if s_dict.get("available") else "❌",
                "Score": _fmt_float(s_dict.get("score")),
                "Direction": str(s_dict.get("direction", "—")),
                "Confidence": _fmt_float(s_dict.get("confidence")),
            })
        st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

    summary = qs.get("summary")
    if summary:
        s_dict = summary.to_dict() if hasattr(summary, "to_dict") else {}
        st.markdown("#### 🎯 Summary")
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
    mc2[1].metric("Lookahead Safe", "✅" if backtest.get("lookahead_safe") else "⚠️")
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
        records = history_mod.list_run_records(base_dir="data/run_history", limit=40)
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
            st.success("Stack ✅")
        elif qs and "_error" in qs:
            st.error("Stack ⚠️")

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
    r[2].metric("🔴 Critical", critical_tasks)
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
            <span class="wharton-badge">👤 {username}</span>
            <span class="wharton-badge">🎯 {role}</span>
            <span class="wharton-badge">⚡ {pm}</span>
          </div>
        </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("🚪 Logout", use_container_width=True):
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
        "Mind Map",
        "Sub-Projects",
        "Quant Engine",
        "War Room",
        "File Vault",
    ])

    with tabs[0]:
        _render_overview_action_center(profile)
    with tabs[1]:
        _render_mindmap()
    with tabs[2]:
        _render_subprojects(profile)
    with tabs[3]:
        _render_quant_engine(profile)
    with tabs[4]:
        _render_chat(profile)
    with tabs[5]:
        _render_file_center(profile)


def main() -> None:
    render_wharton_cockpit()


if __name__ == "__main__":
    main()