from __future__ import annotations

from datetime import date, datetime, timedelta
from copy import deepcopy
from html import escape
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any
import uuid
import secrets

import bcrypt
import numpy as np
import pandas as pd
import streamlit as st


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
# Keep the UI aligned with the hard limit enforced by the storage backend.
MAX_FILE_SIZE_MB = 20


def _is_development_mode() -> bool:
    """Check if the app is running in development mode."""
    return os.environ.get("QUANT_SIM_ENV") == "development"


def _should_sync_seeded_passwords() -> bool:
    """Avoid expensive bcrypt checks on every production start."""
    return _is_development_mode() or os.environ.get("QUANT_SIM_SYNC_DEFAULT_PASSWORDS") == "1"


def _get_default_password() -> str:
    try:
        pw = st.secrets.get("WHARTON_PASSWORD")
        if pw:
            return str(pw)
    except Exception:
        pass
    if _is_development_mode():
        return "CHANGE_ME_IN_SECRETS"
    return secrets.token_urlsafe(32)

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
COMPANY_ANALYSIS_KEY = "wharton_company_analysis_v1"
LIVE_PORTFOLIO_ANALYTICS_KEY = "wharton_live_portfolio_analytics_v1"
HIDDEN_COCKPIT_TABS = {"Mind Map", "War Room", "File Vault"}

TASK_PRIORITIES = ["Critical", "High", "Medium", "Low"]
TASK_PRIORITY_COLORS = {
    "Critical": "#dc2626",
    "High": "#d97706",
    "Medium": "#2563eb",
    "Low": "#64748b",
}
GRAPH_NODE_TYPES = ["Policy", "Company", "Model", "Market", "Risk", "Research", "Other"]
QUANT_MODULES = [
    "Methodology & Validation",
    "Benchmark Analytics",
    "Cost-Aware Rebalance",
    "Performance Attribution",
    "Simulation",
    "Models & Signals",
    "News Sentiment",
    "Robustness Check",
    "Backtest",
    "Run History",
]
QUANT_OPERATOR_USERS = {"Jakub", "Matfyz_Genius"}
DEFAULT_QUANT_TICKERS = ["ASML", "NVDA", "MSFT", "LLY", "JPM"]

DEFAULT_USERS = [
    {"username": "Jakub", "role": "Co-Captain / Quant", "primary_module": "Quant Engine"},
    {"username": "Matěj", "role": "Co-Captain / Strategy", "primary_module": "Dashboard & Strategy"},
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
        from src.analytics.macro_snapshot_store import init_macro_snapshot_table
        from src.portfolio_tracker.governance_store import init_governance_tables
        from src.portfolio_tracker.strategy_store import init_strategy_tables

        # Shared, versioned macro cache. With Turso configured this table is
        # available to every app instance; otherwise it remains a local cache.
        init_macro_snapshot_table(conn)
        init_strategy_tables(conn)
        init_governance_tables(conn)
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
        # login_attempts table is managed by src.auth.database.py, so we don't recreate it here.
        # Decision log stores the analytical context captured with each action.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_log (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                action TEXT,
                date TEXT,
                thesis TEXT,
                client_goal_tags TEXT,
                team_member TEXT,
                quant_snapshot_json TEXT,
                updated_at TEXT,
                updated_by TEXT,
                horizon_days INTEGER,
                benchmark_ticker TEXT,
                expected_return_min REAL,
                expected_return_max REAL,
                decision_confidence INTEGER,
                target_condition TEXT,
                invalidation_condition TEXT,
                planned_weight REAL
            )
        """)
        # Existing installations may already have the original decision_log table.
        # Keep its author and decision snapshot intact while adding collaboration data.
        decision_log_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(decision_log)").fetchall()
        }
        for col, definition in [
            ("updated_at", "TEXT"),
            ("updated_by", "TEXT"),
            ("horizon_days", "INTEGER"),
            ("benchmark_ticker", "TEXT"),
            ("expected_return_min", "REAL"),
            ("expected_return_max", "REAL"),
            ("decision_confidence", "INTEGER"),
            ("target_condition", "TEXT"),
            ("invalidation_condition", "TEXT"),
            ("planned_weight", "REAL"),
        ]:
            if col not in decision_log_cols:
                conn.execute(f"ALTER TABLE decision_log ADD COLUMN {col} {definition}")
        # Preserve every collaboration event; decision_log only holds the latest editor.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_edit_log (
                id INTEGER PRIMARY KEY,
                decision_id INTEGER NOT NULL,
                ticker TEXT,
                edited_at TEXT NOT NULL,
                edited_by TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS competition_compliance (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                team_size INTEGER DEFAULT 0,
                leader_age INTEGER DEFAULT 0,
                same_school INTEGER DEFAULT 0,
                eligible_students INTEGER DEFAULT 0,
                leader_designated INTEGER DEFAULT 0,
                advisor_is_teacher INTEGER DEFAULT 0,
                advisor_team_count INTEGER DEFAULT 0,
                one_wins_account INTEGER DEFAULT 0,
                members_single_team INTEGER DEFAULT 0,
                no_client_contact INTEGER DEFAULT 0,
                no_paid_advisor INTEGER DEFAULT 0,
                student_owned_work INTEGER DEFAULT 0,
                ai_cited INTEGER DEFAULT 0,
                sources_cited INTEGER DEFAULT 0,
                school_permission INTEGER DEFAULT 0,
                updated_at TEXT,
                updated_by TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS competition_positions (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                security_type TEXT NOT NULL DEFAULT 'Stock',
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_date TEXT NOT NULL,
                opened_by TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                last_price REAL,
                notes TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                exit_price REAL,
                exit_date TEXT,
                closed_by TEXT
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
            user_pass = DEFAULT_PASSWORD
            if not _is_development_mode():
                try:
                    user_pass = str(st.secrets["wharton_users"][user["username"]])
                except Exception as e:
                    # Ignore missing secret quietly
                    pass

            if existing_users.get(user["username"]):
                if not _should_sync_seeded_passwords():
                    continue
                stored_hash = existing_users[user["username"]]
                if stored_hash and bcrypt.checkpw(user_pass.encode("utf-8"), stored_hash.encode("utf-8")):
                    conn.execute(
                        "UPDATE wharton_users SET role = ?, primary_module = ? WHERE username = ?",
                        (user["role"], user["primary_module"], user["username"]),
                    )
                    continue
                # Password changed, update it
                password_hash = bcrypt.hashpw(
                    user_pass.encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8")
                conn.execute(
                    "UPDATE wharton_users SET password_hash = ?, role = ?, primary_module = ? WHERE username = ?",
                    (password_hash, user["role"], user["primary_module"], user["username"])
                )
            else:
                password_hash = bcrypt.hashpw(
                    user_pass.encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8")

                conn.execute(
                    "INSERT INTO wharton_users (username, password_hash, role, primary_module) VALUES (?, ?, ?, ?)",
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
        # Commit schema migrations before pushing the local libSQL replica to Turso.
        conn.commit()
        if hasattr(conn, "sync"):
            conn.sync()


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
        from src.auth.database import log_login_attempt, get_recent_failed_attempts
        
        failed_attempts = get_recent_failed_attempts(username, minutes=LOGIN_ATTEMPT_WINDOW_MINUTES)
        if failed_attempts >= MAX_LOGIN_ATTEMPTS:
            st.error(f"Too many failed attempts. Try again in {LOGIN_ATTEMPT_WINDOW_MINUTES} minutes.")
            return

        profile = authenticate_user(username, password)
        if profile is None:
            log_login_attempt(username, success=False)
            st.error("Wrong credentials.")
            return
            
        log_login_attempt(username, success=True)
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
        conn.commit()
        if hasattr(conn, 'sync'): conn.sync()


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
    cols[3].metric("Critical", critical)
    cols[4].metric("High", high)

    if total > 0:
        pct = done / total
        st.progress(pct, text=f"Completion: {pct:.0%}")


def _render_task_manager(profile: dict[str, str | int]) -> None:
    st.markdown("### Mission Task Board")

    original_tasks = _fetch_task_rows()
    _render_task_stats(original_tasks)

    # Quick-add form (always visible, above the editor)
    with st.expander("Quick Add Task", expanded=False):
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
                        conn.commit()
                        if hasattr(conn, 'sync'): conn.sync()
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
        conn.commit()
        if hasattr(conn, 'sync'): conn.sync()


def _delete_node(node_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM mindmap_edges WHERE source = ? OR target = ?", (node_id, node_id))
        conn.execute("DELETE FROM mindmap_nodes WHERE id = ?", (node_id,))
        conn.commit()
        if hasattr(conn, 'sync'): conn.sync()


def _delete_edge(edge_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM mindmap_edges WHERE id = ?", (edge_id,))
        conn.commit()
        if hasattr(conn, 'sync'): conn.sync()


def _insert_edge(source: str, target: str) -> bool:
    eid = _edge_id(source, target)
    with get_connection() as conn:
        if conn.execute("SELECT 1 FROM mindmap_edges WHERE id = ?", (eid,)).fetchone():
            return False
        conn.execute("INSERT INTO mindmap_edges (id, source, target) VALUES (?, ?, ?)", (eid, source, target))
        conn.commit()
        if hasattr(conn, 'sync'): conn.sync()
    return True


def _render_mindmap() -> None:
    from streamlit_agraph import Config, Edge, Node, agraph

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
        conn.commit()
        if hasattr(conn, 'sync'): conn.sync()


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

    from src.storage.wharton_adapter import get_storage_backend
    is_local_backend = get_storage_backend().backend_name == "local"
    if is_local_backend:
        st.error("**VAROVÁNÍ: File Vault používá LOKÁLNÍ úložiště!**\n\nCloudflare R2 není správně nakonfigurováno. Všechny soubory, které teď nahrajete, po restartu aplikace zmizí (přestože jejich názvy zůstanou v databázi). Zkontrolujte, zda máte v nastavení Streamlit Cloud Secrets přidanou sekci `[storage]` s údaji pro R2.")

    if is_local_backend and not _is_development_mode():
        st.error("Uploads are disabled in production when running on local storage to prevent data loss.")
    else:
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
            sp_name = st.text_input("Project Name", placeholder="e.g. EU regulation impact on ASML")
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
                        conn.commit()
                        if hasattr(conn, 'sync'): conn.sync()
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
            f"{sp['status'].upper()}  ·  "
            f"{len(sp_files)} file(s)",
            expanded=False,
        ):
            st.markdown(f"<span style='color:{status_color};font-weight:700;padding: 0.2rem 0.5rem; border: 1px solid {status_color}; border-radius: 4px; display: inline-block; margin-bottom: 0.5rem;'>{sp['status'].upper()}</span>", unsafe_allow_html=True)
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
    import sys
    if "src.analytics.model_validation" in sys.modules:
        importlib.reload(sys.modules["src.analytics.model_validation"])
    if "src.analytics" in sys.modules:
        importlib.reload(sys.modules["src.analytics"])

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
    jump_intensity, jump_mean, jump_volatility,
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
        # GBM expects arithmetic drift; CAGR remains the headline realized
        # return metric but is not the correct drift estimator here.
        expected_return=float(portfolio_returns.mean() * 252.0),
        volatility=float(core_metrics.get("volatility", 0.0)),
        time_horizon=simulation_days, n_simulations=n_simulations, random_seed=random_seed,
    )
    
    adv_price_paths, adv_simulation_stats = simulation.run_advanced_monte_carlo_simulation(
        current_value=current_value,
        expected_return=float(core_metrics.get("annualized_return", 0.0)),
        volatility=float(core_metrics.get("volatility", 0.0)),
        time_horizon=simulation_days, n_simulations=n_simulations, random_seed=random_seed,
        jump_intensity=jump_intensity, jump_mean=jump_mean, jump_volatility=jump_volatility,
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
            "transaction_cost_bps": transaction_cost_bps,
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

    backtest_for_validation = (
        quant_stack_result.get("backtest", {})
        if isinstance(quant_stack_result, dict) and "_error" not in quant_stack_result
        else {}
    )
    model_validation = analytics.build_model_validation_report(
        portfolio_returns=portfolio_returns,
        simulation_stats=simulation_stats,
        backtest=backtest_for_validation,
        risk_free_rate=risk_free_rate,
        random_seed=random_seed,
    )

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
        "model_validation": model_validation,
        "adv_price_paths": adv_price_paths,
        "adv_simulation_stats": adv_simulation_stats,
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
                n_simulations = st.slider("Simulation Count", 200, 15000, 1200, 100)
                random_seed = st.number_input("Seed", min_value=0, value=42, step=1)
                st.markdown("#### Jump Diffusion (Advanced MC)")
                jump_intensity = st.slider("Jump Intensity (λ)", 0.0, 5.0, 1.5, 0.1)
                jump_mean = st.slider("Mean Jump Size (μ_J)", -0.5, 0.0, -0.05, 0.01)
                jump_volatility = st.slider("Jump Volatility (σ_J)", 0.0, 0.3, 0.08, 0.01)
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
                    jump_intensity=float(jump_intensity), jump_mean=float(jump_mean), jump_volatility=float(jump_volatility),
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
        return {"available": False, "error": "Supplementary analysis module not found."}


def _render_ai_advisor_card(context_data: dict, prompt_type: str) -> None:
    with st.spinner("Preparing supplementary analysis..."):
        res = _fetch_ai_insight_cached(context_data, prompt_type)
    if res.get("available") and res.get("insight"):
        st.info(f"**Supplementary insight:**\n\n{res['insight']}")
    elif not res.get("available"):
        st.caption(f"Supplementary insight unavailable: {res.get('error', 'Unknown error')}")


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


def _render_methodology_validation(result: dict) -> None:
    """Explain what QuantSim can and cannot support with current evidence."""
    st.markdown("### Methodology & Validation")
    report = result.get("model_validation", {})
    if not report:
        st.info("Run the Quant Engine to generate the validation report.")
        return

    score = float(report.get("methodology_score", 0.0))
    distribution = report.get("distribution", {})
    backtest = result.get("quant_stack", {}).get("backtest", {}) if isinstance(result.get("quant_stack"), dict) else {}
    metrics = backtest.get("metrics", {}) if isinstance(backtest, dict) else {}
    top = st.columns(4)
    top[0].metric("Methodology score", f"{score:.0f}/100")
    top[1].metric("Evidence band", str(report.get("band", "unknown")).replace("_", " ").title())
    top[2].metric("History", f"{float(distribution.get('observations', 0)) / 252.0:.1f} years")
    top[3].metric("Causal baseline hit rate", _fmt_pct(metrics.get("directional_hit_rate")))

    st.info(
        "This score measures evidence quality and reproducibility — not future predictive accuracy, "
        "and it is not an official Wharton rating or endorsement."
    )
    st.markdown(f"**Assessment:** {escape(str(report.get('verdict', '')))}")

    st.markdown("#### Validation gates")
    gate_rows = []
    status_labels = {"pass": "Pass", "warning": "Partial", "fail": "Gap"}
    for gate in report.get("gates", []):
        gate_rows.append({
            "Gate": gate.get("gate", ""),
            "Status": status_labels.get(str(gate.get("status", "")), str(gate.get("status", ""))),
            "Score": f"{float(gate.get('points', 0)):.0f}/{float(gate.get('maximum', 0)):.0f}",
            "Evidence": gate.get("evidence", ""),
        })
    st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

    intervals = report.get("metric_intervals", {})
    if intervals:
        labels = {
            "annualized_return": "Annualized return",
            "volatility": "Volatility",
            "sharpe_ratio": "Sharpe ratio",
            "var_95": "Daily VaR 95%",
            "cvar_95": "Daily CVaR 95%",
        }
        interval_rows = []
        for key, values in intervals.items():
            is_ratio = key == "sharpe_ratio"
            formatter = _fmt_float if is_ratio else _fmt_pct
            interval_rows.append({
                "Metric": labels.get(key, key),
                "Estimate": formatter(values.get("estimate")),
                "95% low": formatter(values.get("ci_low")),
                "95% high": formatter(values.get("ci_high")),
                "Method": values.get("method", ""),
            })
        st.markdown("#### Estimates with uncertainty")
        st.dataframe(pd.DataFrame(interval_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Model-risk diagnostics")
    diag = st.columns(4)
    diag[0].metric("Skewness", _fmt_float(distribution.get("skewness")))
    diag[1].metric("Excess kurtosis", _fmt_float(distribution.get("excess_kurtosis")))
    diag[2].metric("Normality p-value", _fmt_float(distribution.get("normality_p_value")))
    diag[3].metric("Lag-1 autocorrelation", _fmt_float(distribution.get("lag1_autocorrelation")))

    with st.expander("Limitations that must accompany a Wharton-level presentation", expanded=True):
        for limitation in report.get("limitations", []):
            st.markdown(f"- {escape(str(limitation))}")


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


def _render_robustness_check(result: dict) -> None:
    """Render Robustness Validation."""
    st.markdown("### Robustness Check (Walk-Forward Validation)")
    qs = result.get("quant_stack", {})
    if not qs:
        st.info("Run the Quant Engine to populate data.")
        return
        
    portfolio_timeseries = result.get("portfolio_timeseries", pd.DataFrame())
    if portfolio_timeseries.empty or "cumulative_return" not in portfolio_timeseries.columns:
        st.warning("Portfolio returns not available for robustness validation.")
        return
        
    returns = portfolio_timeseries["cumulative_return"].diff().fillna(0.0)
    
    st.markdown("Configure Walk-Forward Out-of-Sample Validation and run it to compute Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).")
    
    col1, col2, col3, col4 = st.columns(4)
    train_days = col1.number_input("Train Window (Days)", min_value=30, max_value=3650, value=252, step=30)
    test_days = col2.number_input("Test Window (Days)", min_value=10, max_value=1095, value=60, step=10)
    step_days = col3.number_input("Step (Days)", min_value=10, max_value=1095, value=30, step=10)
    num_trials = col4.number_input("Number of Trials", min_value=1, max_value=10000, value=100, step=10, help="Number of strategy variations tested. Used for DSR.")
    
    if st.button("Run Validation", key="btn_run_robustness"):
        with st.spinner("Running Walk-Forward validation..."):
            from src.analytics.modular.robustness_validation import run_walk_forward_validation
            
            try:
                res = run_walk_forward_validation(
                    portfolio_returns=returns,
                    train_days=train_days,
                    test_days=test_days,
                    step_days=step_days,
                    num_trials=num_trials
                )
                
                metrics = res["metrics"]
                
                r1, r2, r3 = st.columns(3)
                r1.metric("PSR", f"{metrics['psr']:.2%}")
                r2.metric("DSR", f"{metrics['dsr']:.2%}")
                r3.metric("OOS Sharpe", f"{metrics['oos_sharpe']:.3f}")
                
                st.info(f"**PSR Interpretation:** {metrics['psr_interpretation']}")
                st.info(f"**DSR Interpretation:** {metrics['dsr_interpretation']}")
                
                if res["windows"]:
                    st.markdown("#### Walk-Forward Windows")
                    df_windows = pd.DataFrame(res["windows"])
                    st.dataframe(df_windows, use_container_width=True)
                    
                    st.markdown("#### Rolling Out-of-Sample Performance")
                    agg_oos = res["aggregate_oos_returns"]
                    if not agg_oos.empty:
                        cum_oos = (1 + agg_oos).cumprod()
                        import plotly.express as px
                        fig = px.line(cum_oos, title="Cumulative Out-of-Sample Returns", labels={"value": "Cumulative Growth", "index": "Date"})
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error running validation: {e}")


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


def _render_advanced_monte_carlo(result: dict) -> None:
    """Interactive Monte Carlo simulation tab with fan charts and VaR analysis."""
    st.markdown("### Advanced Monte Carlo (Merton Jump Diffusion)")

    try:
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not result:
        st.info("Run the Quant Engine first to see Monte Carlo results.")
        return

    stats = result.get("adv_simulation_stats", {})
    paths = result.get("adv_price_paths")
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
            title="Portfolio Value: Advanced Monte Carlo Fan Chart",
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

    factors = ["Market (Beta)", "Size (SMB)", "Value (HML)", "Momentum (MOM)", "Quality (QAL)", "Low Vol (VOL)"]
    portfolio_factors = {f: 0.0 for f in factors}
    is_synthetic = False

    try:
        start_date_str = result.get("inputs", {}).get("start_date")
        end_date_str = result.get("inputs", {}).get("end_date")
        if not start_date_str or not end_date_str:
            raise ValueError("Missing start/end dates for factor regression.")
        
        from datetime import date
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        etf_symbols = ("SPY", "IWN", "IWD", "MTUM", "QUAL", "USMV")
        
        with st.spinner("Fetching Factor ETFs for OLS Regression..."):
            factor_prices = _fetch_close_prices_cached(etf_symbols, start_date, end_date)

        if factor_prices.empty:
            raise ValueError("Could not fetch factor ETF price data.")

        factor_returns = factor_prices.pct_change().dropna()

        # Retrieve portfolio returns — may be a pd.Series with a DatetimeIndex
        port_ret = result.get("portfolio_returns")
        if port_ret is None or (hasattr(port_ret, "empty") and port_ret.empty):
            raise ValueError("Portfolio returns are empty.")
        port_ret = pd.Series(port_ret)

        # Normalize both indexes to timezone-naive dates so pd.concat can align them
        # regardless of whether yfinance returns UTC-aware or naive timestamps.
        def _to_date_index(s: pd.Series) -> pd.Series:
            idx = s.index
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert("UTC").normalize().tz_localize(None)
            else:
                idx = pd.DatetimeIndex(idx).normalize()
            return s.set_axis(idx)

        port_ret = _to_date_index(port_ret)
        factor_returns = factor_returns.apply(_to_date_index)

        aligned = pd.concat([port_ret, factor_returns], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            raise ValueError(f"Not enough aligned data points for OLS (only {len(aligned)} days).")

        import statsmodels.api as sm
        available_etfs = [s for s in etf_symbols if s in aligned.columns]
        if not available_etfs:
            raise ValueError("None of the factor ETFs are present in the aligned dataset.")
        X = aligned[available_etfs]
        X = sm.add_constant(X)
        y = aligned.iloc[:, 0]
        model = sm.OLS(y, X).fit()

        portfolio_factors = {
            "Market (Beta)": float(model.params.get("SPY", 0.0)),
            "Size (SMB)": float(model.params.get("IWN", 0.0)),
            "Value (HML)": float(model.params.get("IWD", 0.0)),
            "Momentum (MOM)": float(model.params.get("MTUM", 0.0)),
            "Quality (QAL)": float(model.params.get("QUAL", 0.0)),
            "Low Vol (VOL)": float(model.params.get("USMV", 0.0)),
        }
    except Exception as e:
        is_synthetic = True
        st.warning(f"**Illustrative / Placeholder — do not cite in report** (Real regression failed: {e})")
        # Generate deterministic synthetic factor loadings based on ticker names for demonstration
        for t, w in zip(tickers, weights):
            seed = sum(ord(c) for c in t)
            np.random.seed(seed)
            if t in ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]:
                t_factors = [1.1, -0.2, -0.6, 0.8, 0.9, 0.1]
            elif t in ["BTC", "ETH", "COIN", "MSTR"]:
                t_factors = [1.5, 0.5, -0.8, 0.9, -0.5, -0.9]
            elif t in ["BIL", "SHY", "TLT", "IEF"]:
                t_factors = [0.1, 0.0, 0.5, 0.0, 0.8, 0.9]
            else:
                t_factors = np.random.normal(0, 0.5, len(factors))
                t_factors = np.clip(t_factors, -1, 1.5)
            for i, f in enumerate(factors):
                portfolio_factors[f] += t_factors[i] * w
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

    st.caption("Screen stocks using fundamental and technical criteria from yfinance.")

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

    st.markdown("### Custom Quant Sandbox")
    st.caption(
        "Runs a manually configured universe through analytics, optimization, simulation, models, signals, news, "
        "backtests, and history. Live competition-portfolio analytics are kept separately in Portfolio Tracker."
    )

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
        if selected == "Methodology & Validation":
            _render_methodology_validation(result)
        elif selected == "Benchmark Analytics":
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
        elif selected == "Robustness Check":
            _render_robustness_check(result)
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


# ─── Strategy & Decisions ─────────────────────────────────────────────────────

_FUNDAMENTAL_FIELDS = {
    # label: (yfinance key, format_str, description)
    "Price":            ("currentPrice",          "{:.2f}",    "Current market price"),
    "Market Cap (B)": ("marketCap",              "{:.2f}B",   "Market capitalisation in billions"),
    "Sector":           ("sector",                 "{}",        "GICS sector"),
    "Trailing P/E":     ("trailingPE",             "{:.1f}x",   "Price / trailing 12M EPS"),
    "Forward P/E":      ("forwardPE",              "{:.1f}x",   "Price / next-12M EPS consensus"),
    "PEG Ratio":        ("pegRatio",               "{:.2f}",    "P/E to EPS Growth (5Y) ratio"),
    "EV/EBITDA":        ("enterpriseToEbitda",     "{:.1f}x",   "Enterprise Value / EBITDA"),
    "EV/Revenue":       ("enterpriseToRevenue",    "{:.2f}x",   "Enterprise Value / Revenue"),
    "Rev Growth (YoY)": ("revenueGrowth",          "{:.1%}",    "Annual revenue growth (YoY)"),
    "Earnings Growth":  ("earningsGrowth",         "{:.1%}",    "Annual earnings growth (YoY)"),
    "Gross Margin":     ("grossMargins",           "{:.1%}",    "Gross profit / revenue"),
    "Op Margin":        ("operatingMargins",       "{:.1%}",    "Operating income / revenue"),
    "Net Margin":       ("profitMargins",          "{:.1%}",    "Net income / revenue"),
    "ROE":              ("returnOnEquity",         "{:.1%}",    "Return on equity (trailing)"),
    "ROIC":             ("returnOnAssets",         "{:.1%}",    "Return on assets (proxy for ROIC)"),
    "Debt/Equity":      ("debtToEquity",           "{:.2f}",    "Total debt / shareholders equity"),
    "Free Cash Flow (B)":("freeCashflow",          "{:.2f}B",   "Trailing twelve-month free cash flow"),
    "52W High":         ("fiftyTwoWeekHigh",       "{:.2f}",    "52-week high price"),
    "52W Low":          ("fiftyTwoWeekLow",        "{:.2f}",    "52-week low price"),
    "Beta":             ("beta",                   "{:.2f}",    "Market beta (5Y monthly)"),
    "Dividend Yield":   ("dividendYield",          "{:.2%}",    "Trailing annual dividend yield"),
    "Analyst Target":   ("targetMeanPrice",        "{:.2f}",    "Consensus analyst 12M price target"),
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ticker_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental metrics for a single ticker via yfinance.
    Returns a flat dict of human-readable label -> (raw_value, formatted_string).
    Calculates CAGR (3Y and 5Y) from historical price data.
    Falls back gracefully if the fetch fails.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}

        metrics: dict[str, Any] = {}
        for label, (yf_key, fmt, desc) in _FUNDAMENTAL_FIELDS.items():
            raw = info.get(yf_key)
            if raw is None:
                continue
            try:
                if "Cap" in label or "Cash" in label:
                    formatted = fmt.format(float(raw) / 1e9)
                else:
                    formatted = fmt.format(raw)
                metrics[label] = {"value": raw, "formatted": formatted, "desc": desc}
            except Exception:
                continue

        # ── CAGR calculations ─────────────────────────────────────────────────
        try:
            hist = t.history(period="5y", interval="1mo")
            if hist is not None and len(hist) >= 12:
                close = hist["Close"].dropna()
                def _cagr(series: pd.Series, years: int) -> float | None:
                    n = years * 12
                    if len(series) < n + 1:
                        return None
                    return float((series.iloc[-1] / series.iloc[-(n + 1)]) ** (1 / years) - 1)
                for yrs in (3, 5):
                    c = _cagr(close, yrs)
                    if c is not None:
                        metrics[f"Price CAGR ({yrs}Y)"] = {
                            "value": c,
                            "formatted": f"{c:.1%}",
                            "desc": f"Annualised price return over {yrs} years"
                        }
        except Exception:
            pass

        # ── Upside to analyst target ───────────────────────────────────────────
        price = info.get("currentPrice")
        target = info.get("targetMeanPrice")
        if price and target and price > 0:
            upside = (target - price) / price
            metrics["Upside to Target"] = {
                "value": upside,
                "formatted": f"{upside:.1%}",
                "desc": "Implied upside from current price to analyst consensus target"
            }

        return metrics
    except Exception as e:
        return {"_error": {"value": str(e), "formatted": str(e), "desc": "Fetch failed"}}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_six_month_price_history(ticker: str) -> list[tuple[str, float]]:
    """Return daily closing prices for the latest six months, if available."""
    try:
        import yfinance as yf
        history = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if history is None or history.empty or "Close" not in history:
            return []
        return [
            (date.strftime("%Y-%m-%d"), float(close))
            for date, close in history["Close"].dropna().items()
        ]
    except Exception:
        return []


def _strategy_payload(record: Any) -> dict[str, Any]:
    """Return the JSON payload from a strategy-store record."""
    if not isinstance(record, dict):
        return {}
    payload = record.get("payload")
    return dict(payload) if isinstance(payload, dict) else dict(record)


def _finite_form_number(value: Any, default: float = 0.0) -> float:
    """Coerce editable-table cells without allowing NaN into JSON storage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _saved_number(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key)
    return _finite_form_number(value, default) if value is not None else float(default)


def _strategy_rows(value: Any, key_label: str) -> list[dict[str, Any]]:
    """Normalize mapping/list analytics output for display."""
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, dict):
                rows.append({key_label: key, **item})
            else:
                rows.append({key_label: key, "value": item})
        return rows
    return []


def _load_strategy_workspace_data() -> dict[str, Any]:
    from src.portfolio_tracker.strategy_store import (
        get_active_strategy_version,
        list_approved_securities,
        list_holding_theses,
        list_strategy_versions,
        load_client_mandate,
    )

    with get_connection() as conn:
        return {
            "mandate_record": load_client_mandate(conn),
            "strategy_record": get_active_strategy_version(conn),
            "strategy_versions": list_strategy_versions(conn),
            "theses": list_holding_theses(conn),
            "approved_securities": list_approved_securities(conn),
        }


def _render_client_mandate(profile: dict[str, str | int], record: dict[str, Any] | None) -> None:
    from src.portfolio_tracker.strategy_alignment import normalize_client_mandate
    from src.portfolio_tracker.strategy_store import save_client_mandate

    current = _strategy_payload(record)
    st.markdown("#### Client Mandate")
    st.caption(
        "Translate the case study into measurable goals, horizons, liquidity needs, risk tolerance, and constraints. "
        "These fields are analyst inputs until Wharton publishes the 2026–2027 client case."
    )
    st.info("The official 2026–2027 client case is still pending. Save assumptions explicitly and replace them when it is released.")

    current_goals = current.get("goals", []) if isinstance(current.get("goals"), list) else []
    current_constraints = current.get("values_constraints", {}) if isinstance(current.get("values_constraints"), dict) else {}
    goal_frame = pd.DataFrame([
        {
            "Goal": str(item.get("name") or ""),
            "Target %": float(item.get("target_weight") or 0.0) * 100.0,
            "Priority (1–5)": int(round(_finite_form_number(item.get("priority"), 3.0))),
            "Horizon": str(item.get("horizon") or "Long term"),
            "Description / success condition": str(item.get("description") or ""),
        }
        for item in current_goals if isinstance(item, dict)
    ])
    if goal_frame.empty:
        goal_frame = pd.DataFrame([
            {"Goal": "", "Target %": 0.0, "Priority (1–5)": 5, "Horizon": "Long term", "Description / success condition": ""},
            {"Goal": "", "Target %": 0.0, "Priority (1–5)": 3, "Horizon": "Short term", "Description / success condition": ""},
        ])

    risk_options = ["Not specified", "Conservative", "Moderate", "Growth", "Aggressive"]
    saved_risk = str(current.get("risk_tolerance") or "Not specified")
    risk_index = risk_options.index(saved_risk) if saved_risk in risk_options else 0
    with st.form("strategy_client_mandate_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            client_name = st.text_input("Client / mandate name", value=str(current.get("client_name") or ""))
            case_status = st.selectbox(
                "Case status",
                ["Pending official case", "Analyst assumptions", "Official case entered"],
                index=["Pending official case", "Analyst assumptions", "Official case entered"].index(
                    str(current.get("case_status") or "Pending official case")
                ) if str(current.get("case_status") or "Pending official case") in
                ["Pending official case", "Analyst assumptions", "Official case entered"] else 0,
            )
        with c2:
            risk_tolerance = st.selectbox("Risk tolerance", risk_options, index=risk_index)
            horizon_years = st.number_input(
                "Primary horizon (years)", min_value=0.0, max_value=100.0,
                value=float(current.get("horizon_years") or 0.0), step=0.5,
            )
        with c3:
            liquidity_need_pct = st.number_input(
                "Near-term liquidity need (%)", min_value=0.0, max_value=100.0,
                value=float(current.get("liquidity_need_pct") or 0.0) * 100.0, step=1.0,
            )
            base_currency = st.text_input("Base currency", value=str(current.get("base_currency") or "USD"))
        mandate_summary = st.text_area(
            "Mandate summary and explicit assumptions",
            value=str(current.get("mandate_summary") or ""),
            height=100,
        )
        values_constraints = st.text_area(
            "Values, exclusions, liquidity, legal, or ethical constraints",
            value=str(current.get("values_constraints_text") or ""),
            height=80,
        )
        v1, v2, v3 = st.columns(3)
        with v1:
            mandate_excluded_tickers = st.text_input(
                "Client-excluded tickers",
                value=", ".join(current_constraints.get("excluded_tickers", [])),
            )
        with v2:
            mandate_excluded_sectors = st.text_input(
                "Client-excluded sectors",
                value=", ".join(current_constraints.get("excluded_sectors", [])),
            )
        with v3:
            mandate_required_tags = st.text_input(
                "Required holding tags",
                value=", ".join(current_constraints.get("required_tags", [])),
                help="For example impact, climate, income. Add matching tags in Thesis Monitor.",
            )
        st.markdown("##### Goal and capital buckets")
        edited_goals = st.data_editor(
            goal_frame,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="strategy_client_goal_editor",
            column_config={
                "Target %": st.column_config.NumberColumn("Target %", min_value=0.0, max_value=100.0, format="%.1f"),
                "Priority (1–5)": st.column_config.NumberColumn("Priority (1–5)", min_value=1, max_value=5, step=1),
                "Horizon": st.column_config.SelectboxColumn("Horizon", options=["Short term", "Medium term", "Long term"]),
            },
        )
        save_mandate = st.form_submit_button("Save Client Mandate", type="primary", use_container_width=True)

    if save_mandate:
        goals = []
        for row in edited_goals.to_dict("records"):
            name = str(row.get("Goal") or "").strip()
            if not name:
                continue
            goals.append({
                "name": name,
                "target_weight": _finite_form_number(row.get("Target %"), 0.0) / 100.0,
                "priority": min(5, max(1, int(round(_finite_form_number(row.get("Priority (1–5)"), 3.0))))),
                "horizon": str(row.get("Horizon") or "Long term"),
                "description": str(row.get("Description / success condition") or "").strip(),
            })
        total_target = sum(float(item["target_weight"]) for item in goals)
        if not client_name.strip() or not goals:
            st.error("Enter a mandate name and at least one measurable client goal.")
        elif abs(total_target - 1.0) > 0.005:
            st.error(f"Client goal targets must total 100%; the current total is {total_target:.1%}.")
        else:
            payload = normalize_client_mandate({
                "client_name": client_name.strip(),
                "case_status": case_status,
                "risk_tolerance": risk_tolerance,
                "horizon_years": float(horizon_years),
                "liquidity_need_pct": float(liquidity_need_pct) / 100.0,
                "base_currency": base_currency.strip().upper() or "USD",
                "mandate_summary": mandate_summary.strip(),
                "values_constraints": {
                    "notes": [values_constraints.strip()] if values_constraints.strip() else [],
                    "excluded_tickers": [item.strip().upper() for item in mandate_excluded_tickers.replace(";", ",").split(",") if item.strip()],
                    "excluded_sectors": [item.strip() for item in mandate_excluded_sectors.replace(";", ",").split(",") if item.strip()],
                    "required_tags": [item.strip().lower() for item in mandate_required_tags.replace(";", ",").split(",") if item.strip()],
                },
                "goals": goals,
            })
            payload.update({
                "case_status": case_status,
                "risk_tolerance": risk_tolerance,
                "mandate_summary": mandate_summary.strip(),
                "values_constraints_text": values_constraints.strip(),
            })
            with get_connection() as conn:
                save_client_mandate(conn, payload, updated_by=str(profile["username"]))
            st.success("Client Mandate saved to the shared database.")
            st.rerun()

    if current:
        summary_cols = st.columns(4)
        summary_cols[0].metric("Goal buckets", len(current_goals))
        summary_cols[1].metric("Risk tolerance", str(current.get("risk_tolerance") or "Not specified"))
        summary_cols[2].metric("Horizon", f"{float(current.get('horizon_years') or 0):g} years")
        summary_cols[3].metric("Liquidity need", f"{float(current.get('liquidity_need_pct') or 0):.1%}")


def _render_strategy_rulebook(profile: dict[str, str | int], data: dict[str, Any]) -> None:
    from src.portfolio_tracker.strategy_alignment import normalize_strategy_rulebook
    from src.portfolio_tracker.strategy_store import append_strategy_version, set_active_strategy_version

    record = data.get("strategy_record")
    current = _strategy_payload(record)
    mandate = _strategy_payload(data.get("mandate_record"))
    st.markdown("#### Versioned Strategy Rulebook")
    st.caption(
        "Define the repeatable rules that turn the client mandate into security selection, sizing, diversification, "
        "cash, and sell discipline. Every save creates an auditable strategy version."
    )

    sector_targets = current.get("sector_targets", []) if isinstance(current.get("sector_targets"), list) else []
    sector_frame = pd.DataFrame([{
        "Sector": str(item.get("sector") or item.get("name") or ""),
        "Target %": float(item.get("target_weight") or 0.0) * 100.0,
        "Minimum %": float(item.get("min_weight") or 0.0) * 100.0,
        "Maximum %": float(item.get("max_weight") or 0.0) * 100.0,
    } for item in sector_targets if isinstance(item, dict)])
    if sector_frame.empty:
        sector_frame = pd.DataFrame([{"Sector": "", "Target %": 0.0, "Minimum %": 0.0, "Maximum %": 100.0}])

    factor_rows = current.get("selection_factors", []) if isinstance(current.get("selection_factors"), list) else []
    factor_frame = pd.DataFrame([{
        "Factor": str(item.get("factor") or ""),
        "Weight %": float(item.get("weight") or 0.0) * 100.0,
        "Minimum / decision rule": str(item.get("rule") or ""),
    } for item in factor_rows if isinstance(item, dict)])
    if factor_frame.empty:
        factor_frame = pd.DataFrame([
            {"Factor": "Quality", "Weight %": 0.0, "Minimum / decision rule": ""},
            {"Factor": "Valuation", "Weight %": 0.0, "Minimum / decision rule": ""},
            {"Factor": "Growth", "Weight %": 0.0, "Minimum / decision rule": ""},
        ])

    with st.form("strategy_rulebook_form"):
        name = st.text_input("Strategy name", value=str(current.get("name") or ""))
        thesis = st.text_area(
            "One-sentence strategy thesis",
            value=str(current.get("thesis") or ""),
            height=75,
            help="It should explain who you invest for, what you select, and why the process should work over the client's horizon.",
        )
        process = st.text_area(
            "Selection, review, and sell discipline",
            value=str(current.get("process") or ""),
            height=110,
        )
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            max_position = st.number_input("Maximum position (%)", 0.0, 100.0, _saved_number(current, "max_position_weight", 0.15) * 100.0, 1.0)
            max_sector = st.number_input("Maximum sector (%)", 0.0, 100.0, _saved_number(current, "max_sector_weight", 0.35) * 100.0, 1.0)
        with r2:
            min_cash = st.number_input("Minimum cash (%)", 0.0, 100.0, _saved_number(current, "min_cash_weight", 0.0) * 100.0, 1.0)
            max_cash = st.number_input("Maximum cash (%)", 0.0, 100.0, _saved_number(current, "max_cash_weight", 0.20) * 100.0, 1.0)
        with r3:
            target_holdings = st.number_input("Target holdings", 1, 100, int(current.get("target_holdings") or 12))
            max_goal_drift = st.number_input("Maximum goal drift (pp)", 0.0, 100.0, _saved_number(current, "max_goal_drift", 0.10) * 100.0, 1.0)
        with r4:
            max_sector_drift = st.number_input("Maximum sector drift (pp)", 0.0, 100.0, _saved_number(current, "max_sector_drift", 0.10) * 100.0, 1.0)
            max_turnover = st.number_input("Maximum cumulative turnover (%)", 0.0, 500.0, _saved_number(current, "max_turnover", 0.25) * 100.0, 5.0)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            min_holdings = st.number_input("Minimum holdings", 0, 100, int(current.get("min_holdings") or 0))
        with a2:
            max_holdings = st.number_input("Maximum holdings", 0, 100, int(current.get("max_holdings") or 0))
        with a3:
            enforce_beta = st.checkbox(
                "Enforce portfolio beta range",
                value=current.get("min_beta") is not None or current.get("max_beta") is not None,
            )
            min_beta = st.number_input("Minimum portfolio beta", -5.0, 10.0, float(current.get("min_beta") if current.get("min_beta") is not None else 0.0), 0.1)
        with a4:
            max_beta = st.number_input("Maximum portfolio beta", -5.0, 10.0, float(current.get("max_beta") if current.get("max_beta") is not None else 2.0), 0.1)
        require_approved = st.checkbox(
            "Require every holding to be on the loaded Approved Universe",
            value=bool(current.get("require_approved")),
        )
        allowed_type_options = ["Stock", "ETF", "Bond", "Other"]
        saved_allowed_types = {str(item).casefold() for item in current.get("allowed_asset_types", [])}
        default_allowed_types = [
            item for item in allowed_type_options if item.casefold() in saved_allowed_types
        ] or ["Stock", "ETF"]
        allowed_types = st.multiselect(
            "Allowed security types",
            allowed_type_options,
            default=default_allowed_types,
        )
        e1, e2 = st.columns(2)
        with e1:
            excluded_tickers = st.text_input(
                "Excluded tickers (comma separated)",
                value=", ".join(str(item) for item in current.get("prohibited_tickers", []) if item),
            )
        with e2:
            excluded_sectors = st.text_input(
                "Excluded sectors (comma separated)",
                value=", ".join(str(item) for item in current.get("excluded_sectors", []) if item),
            )
        st.markdown("##### Sector risk budgets")
        edited_sectors = st.data_editor(
            sector_frame, num_rows="dynamic", hide_index=True, use_container_width=True,
            key="strategy_sector_target_editor",
            column_config={name: st.column_config.NumberColumn(name, min_value=0.0, max_value=100.0, format="%.1f")
                           for name in ("Target %", "Minimum %", "Maximum %")},
        )
        st.markdown("##### Security-selection model")
        edited_factors = st.data_editor(
            factor_frame, num_rows="dynamic", hide_index=True, use_container_width=True,
            key="strategy_factor_editor",
            column_config={"Weight %": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, format="%.1f")},
        )
        save_strategy = st.form_submit_button("Save as New Active Strategy Version", type="primary", use_container_width=True)

    if save_strategy:
        parsed_sectors = []
        for row in edited_sectors.to_dict("records"):
            sector = str(row.get("Sector") or "").strip()
            if sector:
                parsed_sectors.append({
                    "sector": sector,
                    "target_weight": _finite_form_number(row.get("Target %"), 0.0) / 100.0,
                    "min_weight": _finite_form_number(row.get("Minimum %"), 0.0) / 100.0,
                    "max_weight": _finite_form_number(row.get("Maximum %"), 100.0) / 100.0,
                })
        parsed_factors = []
        for row in edited_factors.to_dict("records"):
            factor = str(row.get("Factor") or "").strip()
            if factor:
                parsed_factors.append({
                    "factor": factor,
                    "weight": _finite_form_number(row.get("Weight %"), 0.0) / 100.0,
                    "rule": str(row.get("Minimum / decision rule") or "").strip(),
                })
        factor_total = sum(float(item["weight"]) for item in parsed_factors)
        sector_total = sum(float(item["target_weight"]) for item in parsed_sectors)
        invalid_sector = any(
            not item["min_weight"] <= item["target_weight"] <= item["max_weight"]
            for item in parsed_sectors
        )
        if not name.strip() or not thesis.strip():
            st.error("Strategy name and one-sentence thesis are required.")
        elif min_cash > max_cash:
            st.error("Minimum cash cannot exceed maximum cash.")
        elif min_holdings and max_holdings and min_holdings > max_holdings:
            st.error("Minimum holdings cannot exceed maximum holdings.")
        elif enforce_beta and min_beta > max_beta:
            st.error("Minimum beta cannot exceed maximum beta.")
        elif invalid_sector:
            st.error("Every sector target must lie between its minimum and maximum.")
        elif parsed_sectors and abs(sector_total - 1.0) > 0.005:
            st.error(f"Sector target weights must total 100%; the current total is {sector_total:.1%}.")
        elif parsed_factors and abs(factor_total - 1.0) > 0.005:
            st.error(f"Selection-factor weights must total 100%; the current total is {factor_total:.1%}.")
        else:
            ticker_exclusions = [item.strip().upper() for item in excluded_tickers.replace(";", ",").split(",") if item.strip()]
            sector_exclusions = [item.strip() for item in excluded_sectors.replace(";", ",").split(",") if item.strip()]
            payload = normalize_strategy_rulebook({
                "name": name.strip(), "thesis": thesis.strip(), "process": process.strip(),
                "max_position_weight": max_position / 100.0,
                "max_sector_weight": max_sector / 100.0,
                "min_cash_weight": min_cash / 100.0,
                "max_cash_weight": max_cash / 100.0,
                "target_holdings": int(target_holdings),
                "min_holdings": int(min_holdings) if min_holdings else None,
                "max_holdings": int(max_holdings) if max_holdings else None,
                "max_goal_drift": max_goal_drift / 100.0,
                "max_sector_drift": max_sector_drift / 100.0,
                "max_turnover": max_turnover / 100.0,
                "min_beta": float(min_beta) if enforce_beta else None,
                "max_beta": float(max_beta) if enforce_beta else None,
                "require_approved": bool(require_approved),
                "allowed_asset_types": allowed_types,
                "prohibited_tickers": ticker_exclusions,
                "excluded_sectors": sector_exclusions,
                "sector_targets": parsed_sectors,
                "selection_factors": parsed_factors,
            }, mandate)
            payload.update({"process": process.strip(), "selection_factors": parsed_factors})
            with get_connection() as conn:
                append_strategy_version(conn, payload, created_by=str(profile["username"]), activate=True)
            st.success("A new active strategy version was saved.")
            st.rerun()

    versions = data.get("strategy_versions", [])
    if versions:
        st.markdown("##### Strategy version history")
        st.dataframe(pd.DataFrame([{
            "Version": item.get("version") or item.get("version_number") or item.get("id"),
            "Strategy": _strategy_payload(item).get("name") or "—",
            "Created": item.get("created_at") or "—",
            "Created by": item.get("created_by") or "—",
            "Active": bool(item.get("is_active") or item.get("active")),
        } for item in versions]), use_container_width=True, hide_index=True)
        version_options = [int(item.get("version") or item.get("version_number") or item.get("id")) for item in versions]
        selected_version = st.selectbox("Activate an earlier version", version_options, key="strategy_version_selector")
        if st.button("Set Selected Version Active", use_container_width=True):
            with get_connection() as conn:
                set_active_strategy_version(conn, int(selected_version))
            st.rerun()


def _render_approved_universe(profile: dict[str, str | int], data: dict[str, Any]) -> None:
    from src.portfolio_tracker.strategy_store import replace_approved_securities

    records = data.get("approved_securities", [])
    current_tickers = sorted({str(item.get("ticker") or "").upper() for item in records if item.get("ticker")})
    first_payload = _strategy_payload(records[0]) if records else {}
    try:
        saved_source_as_of = date.fromisoformat(str(first_payload.get("source_as_of") or ""))
    except ValueError:
        saved_source_as_of = date.today()
    st.markdown("#### Approved Security Universe")
    st.caption(
        "Paste the official approved list when Wharton releases it. Until then, this is an analyst-controlled universe, "
        "clearly separated from official competition rules."
    )
    with st.form("strategy_approved_universe_form"):
        raw_universe = st.text_area(
            "Approved tickers (comma, space, or newline separated)",
            value="\n".join(current_tickers), height=220,
        )
        u1, u2 = st.columns(2)
        with u1:
            source_name = st.text_input("List source", value=str(first_payload.get("source_name") or "Analyst-entered list"))
        with u2:
            source_url = st.text_input("Source URL", value=str(first_payload.get("source_url") or ""))
        source_as_of = st.date_input("List as-of date", value=saved_source_as_of)
        save_universe = st.form_submit_button("Replace Approved Universe", type="primary", use_container_width=True)
    if save_universe:
        import re

        tickers = sorted(set(item.upper() for item in re.split(r"[\s,;]+", raw_universe) if item.strip()))
        shared_payload = {
            "source_name": source_name.strip() or "Analyst-entered list",
            "source_url": source_url.strip(),
            "source_as_of": source_as_of.isoformat(),
        }
        try:
            with get_connection() as conn:
                replace_approved_securities(
                    conn,
                    [{"ticker": ticker, "payload": shared_payload, "approved": True} for ticker in tickers],
                    updated_by=str(profile["username"]),
                )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(f"Approved universe now contains {len(tickers)} securities.")
            st.rerun()
    if current_tickers:
        st.metric("Approved securities", len(current_tickers))
        st.dataframe(pd.DataFrame({"Ticker": current_tickers}), use_container_width=True, hide_index=True)
    else:
        st.warning("No approved list is loaded. Alignment analysis will label universe validation as not configured.")


def _render_thesis_monitor(profile: dict[str, str | int], data: dict[str, Any]) -> None:
    from src.portfolio_tracker.strategy_store import upsert_holding_thesis

    positions = _fetch_competition_positions()
    open_tickers = [
        str(row.get("ticker") or "").upper() for row in positions
        if row.get("ticker") and str(row.get("status") or "open").lower() == "open"
    ]
    closed_tickers = [
        str(row.get("ticker") or "").upper() for row in positions
        if row.get("ticker") and str(row.get("status") or "open").lower() != "open"
    ]
    portfolio_state = {
        str(row.get("ticker") or "").upper(): str(row.get("status") or "open").replace("_", " ").title()
        for row in positions if row.get("ticker")
    }
    thesis_records = data.get("theses", [])
    thesis_by_ticker = {str(item.get("ticker") or "").upper(): item for item in thesis_records if item.get("ticker")}
    tickers = list(dict.fromkeys([*open_tickers, *closed_tickers, *thesis_by_ticker]))
    mandate = _strategy_payload(data.get("mandate_record"))
    goal_names = [str(item.get("name")) for item in mandate.get("goals", []) if isinstance(item, dict) and item.get("name")]

    st.markdown("#### Holding Thesis Monitor")
    st.caption(
        "Track the forward-looking thesis, scenarios, catalysts, invalidation condition, conviction, and review date. "
        "A price move by itself does not validate or invalidate a thesis."
    )
    if not tickers:
        st.info("Add a position in Portfolio Tracker to start thesis monitoring.")
        return
    selected = st.selectbox("Holding", tickers, key="strategy_thesis_ticker")
    current_record = thesis_by_ticker.get(selected, {})
    current = _strategy_payload(current_record)
    current_goal = str(current.get("primary_goal") or "")
    goal_options = ["Not assigned", *goal_names]
    current_goal_label = current_goal if current_goal in goal_options else "Not assigned"
    status_options = ["Active", "Watch", "Under Review", "Invalidated", "Exited"]
    saved_status = str(current_record.get("status") or current.get("status") or "Active").replace("_", " ").title()
    status_index = status_options.index(saved_status) if saved_status in status_options else 0
    try:
        saved_review_date = date.fromisoformat(str(current.get("review_date") or ""))
    except ValueError:
        saved_review_date = date.today() + timedelta(days=30)

    with st.form(f"strategy_thesis_form_{selected}"):
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            sector = st.text_input("Sector", value=str(current.get("sector") or ""))
        with t2:
            primary_goal = st.selectbox("Primary client goal", goal_options, index=goal_options.index(current_goal_label))
        with t3:
            conviction = st.slider("Conviction", 1, 5, int(current.get("conviction") or 3))
        with t4:
            thesis_status = st.selectbox("Thesis status", status_options, index=status_index)
        detail1, detail2, detail3 = st.columns(3)
        with detail1:
            review_date = st.date_input("Next thesis review", value=saved_review_date)
        with detail2:
            beta_available = st.checkbox(
                "Beta has been verified",
                value=current.get("beta") is not None,
                key=f"thesis_beta_available_{selected}",
            )
            holding_beta = st.number_input(
                "Observed beta (optional)",
                -5.0, 10.0,
                float(current.get("beta") if current.get("beta") is not None else 0.0),
                0.1,
                disabled=not beta_available,
            )
        with detail3:
            holding_tags = st.text_input(
                "Mandate tags",
                value=", ".join(current.get("tags", [])),
                help="Tags are checked against required tags in the Client Mandate.",
            )
        investment_thesis = st.text_area("Core investment thesis", value=str(current.get("investment_thesis") or ""), height=100)
        s1, s2, s3 = st.columns(3)
        with s1:
            bear_case = st.text_area("Bear case", value=str(current.get("bear_case") or ""), height=100)
        with s2:
            base_case = st.text_area("Base case", value=str(current.get("base_case") or ""), height=100)
        with s3:
            bull_case = st.text_area("Bull case", value=str(current.get("bull_case") or ""), height=100)
        catalysts = st.text_area("Catalysts (one per line)", value="\n".join(current.get("catalysts", [])), height=80)
        risks = st.text_area("Key risks (one per line)", value="\n".join(current.get("risks", [])), height=80)
        invalidation = st.text_area(
            "Observable thesis-invalidation condition",
            value=str(current.get("invalidation") or ""), height=80,
        )
        save_thesis = st.form_submit_button("Save Thesis Monitor", type="primary", use_container_width=True)
    if save_thesis:
        payload = {
            "sector": sector.strip(),
            "primary_goal": "" if primary_goal == "Not assigned" else primary_goal,
            "goals": [] if primary_goal == "Not assigned" else [primary_goal],
            "conviction": int(conviction),
            "beta": float(holding_beta) if beta_available else None,
            "tags": [item.strip().lower() for item in holding_tags.replace(";", ",").split(",") if item.strip()],
            "review_date": review_date.isoformat(),
            "investment_thesis": investment_thesis.strip(),
            "bear_case": bear_case.strip(), "base_case": base_case.strip(), "bull_case": bull_case.strip(),
            "catalysts": [item.strip() for item in catalysts.splitlines() if item.strip()],
            "risks": [item.strip() for item in risks.splitlines() if item.strip()],
            "invalidation": invalidation.strip(),
            "updated_by": str(profile["username"]),
        }
        with get_connection() as conn:
            active_strategy = data.get("strategy_record") if isinstance(data.get("strategy_record"), dict) else {}
            upsert_holding_thesis(
                conn,
                selected,
                payload,
                status=thesis_status.lower().replace(" ", "_"),
                conviction=int(conviction),
                strategy_version=active_strategy.get("version"),
                next_review_at=review_date.isoformat(),
                updated_by=str(profile["username"]),
            )
        st.success(f"{selected} thesis monitor saved.")
        st.rerun()

    monitor_rows = []
    today = date.today()
    for ticker in tickers:
        item = thesis_by_ticker.get(ticker, {})
        payload = _strategy_payload(item)
        required = ["sector", "primary_goal", "investment_thesis", "bear_case", "base_case", "bull_case", "invalidation", "review_date"]
        completeness = sum(bool(payload.get(field)) for field in required) / len(required)
        review_text = str(payload.get("review_date") or "")
        try:
            overdue = date.fromisoformat(review_text) < today
        except ValueError:
            overdue = False
        monitor_rows.append({
            "Ticker": ticker,
            "Portfolio state": portfolio_state.get(ticker, "Not in tracker"),
            "Status": str(item.get("status") or "Not created").replace("_", " ").title(),
            "Primary goal": payload.get("primary_goal") or "Unassigned",
            "Sector": payload.get("sector") or "Unassigned",
            "Conviction": payload.get("conviction") or "—",
            "Next review": review_text or "Not scheduled",
            "Overdue": overdue,
            "Completeness": completeness,
        })
    st.dataframe(
        pd.DataFrame(monitor_rows), hide_index=True, use_container_width=True,
        column_config={"Completeness": st.column_config.ProgressColumn("Completeness", min_value=0.0, max_value=1.0, format="percent")},
    )


def _render_catalyst_calendar(profile: dict[str, str | int], data: dict[str, Any]) -> None:
    from src.portfolio_tracker.governance_store import (
        delete_catalyst_event,
        list_catalyst_events,
        list_research_sources,
        upsert_catalyst_event,
    )

    position_tickers = [
        str(row.get("ticker") or "").upper()
        for row in _fetch_competition_positions() if row.get("ticker")
    ]
    thesis_tickers = [
        str(item.get("ticker") or "").upper()
        for item in data.get("theses", []) if item.get("ticker")
    ]
    tickers = list(dict.fromkeys([*position_tickers, *thesis_tickers]))
    with get_connection() as conn:
        events = list_catalyst_events(conn)
        sources = list_research_sources(conn)

    st.markdown("#### Catalyst Calendar")
    st.caption(
        "Track dated thesis tests across holdings. Date confidence, expected effect, evidence source, and actual "
        "outcome remain separate so uncertain events are not presented as confirmed facts."
    )
    if not tickers:
        st.info("Add a tracked holding or thesis before creating catalyst events.")
        return

    source_options: dict[str, int | None] = {"No linked source": None}
    for item in sources:
        label = f"#{item['id']} · {item.get('ticker') or 'Global'} · {item.get('title') or 'Untitled'}"
        source_options[label] = int(item["id"])
    confidence_labels = {
        "Exact date": "exact",
        "Estimated date": "estimated",
        "Date window": "window",
        "Unknown timing": "unknown",
    }
    with st.form("catalyst_event_create_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            catalyst_ticker = st.selectbox("Ticker", tickers, key="catalyst_create_ticker")
            catalyst_title = st.text_input("Catalyst / thesis test")
        with c2:
            confidence_label = st.selectbox("Date confidence", list(confidence_labels))
            window_start = st.date_input("Window start", value=date.today() + timedelta(days=30))
        with c3:
            window_end = st.date_input("Window end", value=date.today() + timedelta(days=30))
            catalyst_status = st.selectbox("Event status", ["Expected", "Occurred", "Delayed", "Cancelled"])
        c4, c5, c6 = st.columns(3)
        with c4:
            probability = st.slider("Team probability rating", 1, 5, 3)
        with c5:
            impact = st.slider("Expected thesis impact", -5, 5, 0)
        with c6:
            source_label = st.selectbox("Evidence source", list(source_options))
        expected_effect = st.text_area("Expected observable effect", height=75)
        actual_result = st.text_area("Actual result (leave blank until observed)", height=75)
        create_event = st.form_submit_button("Add Catalyst Event", type="primary", use_container_width=True)
    if create_event:
        confidence = confidence_labels[confidence_label]
        start_value = None if confidence == "unknown" else window_start.isoformat()
        end_value = (
            None if confidence == "unknown"
            else start_value if confidence in {"exact", "estimated"}
            else window_end.isoformat()
        )
        try:
            with get_connection() as conn:
                upsert_catalyst_event(
                    conn,
                    catalyst_ticker,
                    catalyst_title,
                    {
                        "expected_effect": expected_effect.strip(),
                        "actual_result": actual_result.strip(),
                    },
                    window_start=start_value,
                    window_end=end_value,
                    date_confidence=confidence,
                    probability=int(probability),
                    impact=int(impact),
                    status=catalyst_status.lower(),
                    source_id=source_options[source_label],
                    updated_by=str(profile["username"]),
                )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success("Catalyst event added to the shared calendar.")
            st.rerun()

    today = date.today()
    rows = []
    for event in events:
        start = None
        end = None
        try:
            start = date.fromisoformat(str(event.get("window_start") or ""))
        except ValueError:
            pass
        try:
            end = date.fromisoformat(str(event.get("window_end") or ""))
        except ValueError:
            end = start
        payload = _strategy_payload(event)
        status = str(event.get("status") or "expected")
        rows.append({
            "ID": event.get("id"),
            "Ticker": event.get("ticker"),
            "Catalyst": event.get("title"),
            "Window start": event.get("window_start") or "Unknown",
            "Window end": event.get("window_end") or "Unknown",
            "Date confidence": str(event.get("date_confidence") or "unknown").replace("_", " ").title(),
            "Probability (1–5)": event.get("probability"),
            "Impact (-5 to +5)": event.get("impact"),
            "Status": status.title(),
            "Days to start": (start - today).days if start else None,
            "Outcome overdue": bool(status == "expected" and end and end < today),
            "Expected effect": payload.get("expected_effect") or "",
            "Actual result": payload.get("actual_result") or "",
            "Source ID": event.get("source_id"),
        })
    if rows:
        next_30 = sum(
            isinstance(row["Days to start"], int) and 0 <= row["Days to start"] <= 30
            and row["Status"] == "Expected" for row in rows
        )
        overdue = sum(bool(row["Outcome overdue"]) for row in rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Open catalyst events", sum(row["Status"] in {"Expected", "Delayed"} for row in rows))
        c2.metric("Expected in 30 days", next_30)
        c3.metric("Outcome overdue", overdue)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        event_by_label = {
            f"#{event['id']} · {event['ticker']} · {event['title']}": event for event in events
        }
        selected_label = st.selectbox("Update or remove event", list(event_by_label), key="catalyst_update_select")
        selected = event_by_label[selected_label]
        selected_payload = _strategy_payload(selected)
        status_options = ["Expected", "Occurred", "Delayed", "Cancelled"]
        saved_status = str(selected.get("status") or "expected").title()
        with st.form(f"catalyst_update_form_{selected['id']}"):
            new_status = st.selectbox(
                "Updated status", status_options,
                index=status_options.index(saved_status) if saved_status in status_options else 0,
            )
            updated_result = st.text_area(
                "Observed result / status evidence",
                value=str(selected_payload.get("actual_result") or ""),
                height=85,
            )
            update_event = st.form_submit_button("Save Event Update", type="primary", use_container_width=True)
        if update_event:
            payload = dict(selected_payload)
            payload["actual_result"] = updated_result.strip()
            with get_connection() as conn:
                upsert_catalyst_event(
                    conn,
                    selected["ticker"],
                    selected["title"],
                    payload,
                    event_id=int(selected["id"]),
                    window_start=selected.get("window_start"),
                    window_end=selected.get("window_end"),
                    date_confidence=selected.get("date_confidence") or "unknown",
                    probability=int(selected.get("probability") or 3),
                    impact=int(selected.get("impact") or 0),
                    status=new_status.lower(),
                    source_id=selected.get("source_id"),
                    updated_by=str(profile["username"]),
                )
            st.success("Catalyst event updated.")
            st.rerun()
        if st.button("Delete Selected Catalyst Event", key=f"delete_catalyst_{selected['id']}"):
            with get_connection() as conn:
                delete_catalyst_event(conn, int(selected["id"]))
            st.rerun()
    else:
        st.info("No structured catalyst events have been created yet.")


def _render_strategy_alignment(data: dict[str, Any]) -> None:
    from src.portfolio_tracker.strategy_alignment import analyze_strategy_alignment
    from src.portfolio_tracker.wharton_competition import INITIAL_CAPITAL_USD, calculate_portfolio_performance

    mandate = _strategy_payload(data.get("mandate_record"))
    strategy = _strategy_payload(data.get("strategy_record"))
    positions = _fetch_competition_positions()
    open_tickers = [str(row.get("ticker") or "").upper() for row in positions if str(row.get("status") or "open") == "open"]
    performance = calculate_portfolio_performance(positions, _competition_live_prices(open_tickers))
    thesis_by_ticker = {
        str(item.get("ticker") or "").upper(): item
        for item in data.get("theses", []) if item.get("ticker")
    }
    approved_tickers = {
        str(item.get("ticker") or "").upper()
        for item in data.get("approved_securities", [])
        if item.get("ticker") and bool(item.get("approved", True))
    }
    universe_configured = bool(approved_tickers)
    holdings = []
    for row in performance.get("positions", []):
        if str(row.get("status") or "open") != "open":
            continue
        ticker = str(row.get("ticker") or "").upper()
        thesis_record = thesis_by_ticker.get(ticker, {})
        thesis = _strategy_payload(thesis_record)
        holdings.append({
            "ticker": ticker,
            "market_value": float(row.get("current_value") or 0.0),
            "sector": thesis.get("sector") or "Unassigned",
            "primary_goal": thesis.get("primary_goal") or "",
            "goals": thesis.get("goals") or [],
            "thesis_status": thesis_record.get("status") or "missing",
            "beta": thesis.get("beta"),
            "approved": ticker in approved_tickers if universe_configured else None,
            "asset_type": row.get("security_type") or "Stock",
            "tags": thesis.get("tags") or [],
        })

    st.markdown("#### Strategy Alignment & Drift")
    st.caption(
        "Measures whether the live analytical portfolio follows the Client Mandate and active Strategy Rulebook. "
        "The score is a transparent process diagnostic, not a return forecast."
    )
    if not mandate or not strategy:
        st.warning("Save both a Client Mandate and an active Strategy Rulebook to calculate alignment.")
        return
    observed_turnover = sum(
        abs(float(row.get("quantity") or 0.0) * float(row.get("entry_price") or 0.0))
        for row in positions
        if str(row.get("status") or "open").lower() == "closed"
    ) / INITIAL_CAPITAL_USD
    strategy_with_observations = dict(strategy)
    strategy_with_observations["current_turnover"] = observed_turnover
    actual_cash = float(performance.get("cash_before_pnl") or 0.0) + float(performance.get("realized_pnl") or 0.0)
    analysis = analyze_strategy_alignment(
        holdings,
        mandate,
        strategy_with_observations,
        cash_value=actual_cash,
        portfolio_value=float(performance.get("equity") or INITIAL_CAPITAL_USD),
    )
    top = st.columns(5)
    top[0].metric("Strategy alignment", f"{float(analysis.get('alignment_score') or 0):.0f}/100")
    top[1].metric("Assessment", str(analysis.get("rating") or "Not rated"))
    summary = analysis.get("portfolio_summary", {}) if isinstance(analysis.get("portfolio_summary"), dict) else {}
    top[2].metric("Cash weight", f"{float(summary.get('cash_weight') or 0):.1%}")
    top[3].metric("Largest holding", f"{float(summary.get('largest_position_weight') or 0):.1%}")
    top[4].metric("Effective holdings", f"{float(summary.get('effective_holdings') or 0):.1f}")
    st.caption(
        f"Observed cumulative turnover: {observed_turnover:.1%} (entry cost of closed positions ÷ initial capital)."
    )

    components = _strategy_rows(analysis.get("components"), "Component")
    if components:
        st.markdown("##### Alignment components")
        st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)

    goal_rows = _strategy_rows(analysis.get("goal_allocation"), "Goal")
    sector_rows = _strategy_rows(analysis.get("sector_allocation"), "Sector")
    left, right = st.columns(2)
    with left:
        st.markdown("##### Client goal allocation drift")
        if goal_rows:
            goal_df = pd.DataFrame(goal_rows)
            st.dataframe(goal_df, use_container_width=True, hide_index=True)
            chart_columns = [column for column in ("actual_weight", "target_weight") if column in goal_df]
            if chart_columns:
                label_column = "goal_name" if "goal_name" in goal_df else goal_df.columns[0]
                st.bar_chart(goal_df.set_index(label_column)[chart_columns])
        else:
            st.info("Assign holdings to client goals in Thesis Monitor.")
    with right:
        st.markdown("##### Sector allocation drift")
        if sector_rows:
            sector_df = pd.DataFrame(sector_rows)
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
            chart_columns = [column for column in ("actual_weight", "target_weight") if column in sector_df]
            if chart_columns:
                label_column = "sector_name" if "sector_name" in sector_df else sector_df.columns[0]
                st.bar_chart(sector_df.set_index(label_column)[chart_columns])
        else:
            st.info("Enter sectors in Thesis Monitor and risk budgets in Strategy Rulebook.")

    violations = analysis.get("violations", [])
    warnings = analysis.get("warnings", [])
    if violations:
        st.markdown("##### Rule violations")
        for item in violations:
            st.error(str(item.get("message") or item) if isinstance(item, dict) else str(item))
    if warnings:
        st.markdown("##### Data and coverage warnings")
        for item in warnings:
            st.warning(str(item.get("message") or item) if isinstance(item, dict) else str(item))
    holding_rows = _strategy_rows(analysis.get("holdings"), "Ticker")
    if holding_rows:
        st.markdown("##### Holding-level strategy fit")
        st.dataframe(pd.DataFrame(holding_rows), use_container_width=True, hide_index=True)
    if not universe_configured:
        if bool(strategy.get("require_approved")):
            st.error("The active strategy requires an Approved Universe, but no list has been loaded.")
        else:
            st.caption("Approved-universe validation is not scored because the rule is disabled and no list is loaded.")


def _render_pretrade_lab(data: dict[str, Any]) -> None:
    from src.portfolio_tracker.pretrade_analysis import analyze_pretrade_impact

    mandate = _strategy_payload(data.get("mandate_record"))
    strategy = _strategy_payload(data.get("strategy_record"))
    positions = _fetch_competition_positions()
    open_tickers = [
        str(row.get("ticker") or "").upper()
        for row in positions
        if row.get("ticker") and str(row.get("status") or "open").lower() == "open"
    ]
    live_prices = _competition_live_prices(open_tickers)

    st.markdown("#### Pre-Trade / What-If Lab")
    st.caption(
        "Simulate an ordered basket of trades against cash, current holdings, the Client Mandate, and the active "
        "Strategy Rulebook. This analysis never places or stores a trade."
    )
    if not mandate or not strategy:
        st.warning("Save both a Client Mandate and an active Strategy Rulebook before testing trades.")
        return

    default_plan = pd.DataFrame([
        {"Ticker": "", "Action": "Buy", "Quantity": 0.0, "Execution price (optional)": 0.0, "Type": "Stock"},
    ])
    with st.form("strategy_pretrade_form"):
        edited_plan = st.data_editor(
            default_plan,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="strategy_pretrade_editor",
            column_config={
                "Action": st.column_config.SelectboxColumn("Action", options=["Buy", "Sell"], required=True),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, format="%.4f"),
                "Execution price (optional)": st.column_config.NumberColumn(
                    "Execution price (optional)", min_value=0.0, format="$%.2f",
                ),
                "Type": st.column_config.SelectboxColumn("Type", options=["Stock", "ETF", "Bond", "Other"]),
            },
        )
        analyze_plan = st.form_submit_button("Analyze Trade Plan", type="primary", use_container_width=True)
    if analyze_plan:
        trades: list[dict[str, Any]] = []
        for row in edited_plan.to_dict("records"):
            ticker = str(row.get("Ticker") or "").strip().upper()
            quantity = _finite_form_number(row.get("Quantity"), 0.0)
            if not ticker and quantity <= 0:
                continue
            trade = {
                "ticker": ticker,
                "action": str(row.get("Action") or "Buy").lower(),
                "quantity": quantity,
                "security_type": str(row.get("Type") or "Stock"),
            }
            price = _finite_form_number(row.get("Execution price (optional)"), 0.0)
            if price > 0:
                trade["price"] = price
            trades.append(trade)
        result = analyze_pretrade_impact(
            positions,
            trades,
            mandate,
            strategy,
            live_prices=live_prices,
            theses=data.get("theses", []),
            approved_securities=data.get("approved_securities", []),
        )
        st.session_state["wharton_pretrade_result"] = result

    result = st.session_state.get("wharton_pretrade_result")
    if not isinstance(result, dict):
        st.info("Enter one or more proposed trades to compare the portfolio before and after execution.")
        return

    status = str(result.get("status") or "review")
    if status == "blocked":
        st.error("The plan is technically infeasible and was not applied to the hypothetical portfolio.")
    elif status == "review":
        st.warning("The plan is executable, but it creates or retains strategy issues that require review.")
    else:
        st.success("The plan is executable and creates no new strategy violation.")

    after_alignment = result.get("after", {}).get("alignment", {})
    after_summary = after_alignment.get("portfolio_summary", {})
    metrics = st.columns(6)
    metrics[0].metric(
        "Alignment",
        f"{float(after_alignment.get('alignment_score') or 0):.0f}/100",
        f"{float(result.get('deltas', {}).get('alignment_score') or 0):+.1f}",
    )
    metrics[1].metric(
        "Cash",
        f"${float(after_summary.get('cash_value') or 0):,.0f}",
        f"${float(result.get('deltas', {}).get('cash_value') or 0):+,.0f}",
    )
    metrics[2].metric(
        "Cash weight",
        f"{float(after_summary.get('cash_weight') or 0):.1%}",
        f"{float(result.get('deltas', {}).get('cash_weight') or 0):+.1%}",
    )
    metrics[3].metric(
        "Largest position",
        f"{float(after_summary.get('largest_position_weight') or 0):.1%}",
        f"{float(result.get('deltas', {}).get('largest_position_weight') or 0):+.1%}",
    )
    metrics[4].metric("Gross proposed notional", f"${float(result.get('gross_proposed_notional') or 0):,.0f}")
    turnover = result.get("incremental_turnover")
    metrics[5].metric("Incremental turnover", f"{float(turnover):.1%}" if turnover is not None else "N/A")
    st.caption(str(result.get("turnover_definition") or ""))

    for blocker in result.get("blockers", []):
        st.error(str(blocker.get("message") or blocker))
    changes = result.get("violation_changes", {})
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("##### New violations")
        if changes.get("new"):
            for item in changes["new"]:
                st.error(str(item.get("message") or item))
        else:
            st.caption("None")
    with v2:
        st.markdown("##### Resolved violations")
        if changes.get("resolved"):
            for item in changes["resolved"]:
                st.success(str(item.get("message") or item))
        else:
            st.caption("None")
    with v3:
        st.markdown("##### Persistent violations")
        if changes.get("persistent"):
            for item in changes["persistent"]:
                issue = item.get("after") or item.get("before") or item
                st.warning(str(issue.get("message") or issue) if isinstance(issue, dict) else str(issue))
        else:
            st.caption("None")

    if result.get("holding_changes"):
        st.markdown("##### Position impact")
        st.dataframe(pd.DataFrame(result["holding_changes"]), use_container_width=True, hide_index=True)
    drift_left, drift_right = st.columns(2)
    with drift_left:
        st.markdown("##### Client-goal drift before / after")
        if result.get("goal_changes"):
            st.dataframe(pd.DataFrame(result["goal_changes"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No client-goal changes available.")
    with drift_right:
        st.markdown("##### Sector drift before / after")
        if result.get("sector_changes"):
            st.dataframe(pd.DataFrame(result["sector_changes"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No sector changes available.")


def _render_review_learning(profile: dict[str, str | int], data: dict[str, Any]) -> None:
    from src.portfolio_tracker.governance_store import (
        append_decision_review,
        append_thesis_review,
        list_decision_reviews,
        list_thesis_reviews,
    )
    from src.portfolio_tracker.strategy_store import upsert_holding_thesis

    thesis_records = data.get("theses", [])
    thesis_by_ticker = {
        str(item.get("ticker") or "").upper(): item
        for item in thesis_records if item.get("ticker")
    }
    with get_connection() as conn:
        thesis_reviews = list_thesis_reviews(conn)
        decision_reviews = list_decision_reviews(conn)
        decision_rows = [dict(row) for row in conn.execute("SELECT * FROM decision_log ORDER BY id DESC").fetchall()]

    st.markdown("#### Review & Learning")
    st.caption(
        "Review records are append-only. Process outcome and market outcome stay separate to reduce hindsight and "
        "outcome bias; a profitable decision can still have a poor process, and vice versa."
    )
    thesis_tab, decision_tab, learning_tab = st.tabs([
        "Thesis Reviews", "Decision Reviews", "Learning Diagnostics",
    ])
    with thesis_tab:
        if not thesis_by_ticker:
            st.info("Create a holding thesis before recording a thesis review.")
        else:
            selected_ticker = st.selectbox("Thesis to review", list(thesis_by_ticker), key="review_thesis_ticker")
            current_record = thesis_by_ticker[selected_ticker]
            current_payload = _strategy_payload(current_record)
            current_status = str(current_record.get("status") or "active").lower()
            status_labels = {
                "Active": "active",
                "Watch": "watch",
                "Under review": "under_review",
                "Invalidated": "invalidated",
                "Exited": "exited",
            }
            saved_label = next((label for label, value in status_labels.items() if value == current_status), "Active")
            with st.form(f"thesis_review_form_{selected_ticker}"):
                t1, t2, t3 = st.columns(3)
                with t1:
                    new_status_label = st.selectbox(
                        "Review conclusion", list(status_labels),
                        index=list(status_labels).index(saved_label),
                    )
                with t2:
                    prior_conviction = _finite_form_number(current_record.get("conviction"), 3.0)
                    new_conviction = st.slider("Updated conviction", 1, 5, int(round(prior_conviction)))
                with t3:
                    next_review = st.date_input("Next review date", value=date.today() + timedelta(days=30))
                what_changed = st.text_area("What changed since the prior thesis state?", height=90)
                evidence = st.text_area("New evidence and source references", height=90)
                review_decision = st.selectbox(
                    "Portfolio-process conclusion",
                    ["Keep and monitor", "Research further", "Review sizing", "Prepare exit", "Thesis invalidated"],
                )
                lesson = st.text_area("Lesson for the investment process", height=75)
                save_thesis_review = st.form_submit_button("Append Thesis Review", type="primary", use_container_width=True)
            if save_thesis_review:
                new_status = status_labels[new_status_label]
                reviewed_at = _now_iso()
                new_payload = dict(current_payload)
                new_payload.update({
                    "review_date": next_review.isoformat(),
                    "last_reviewed_at": reviewed_at,
                    "last_reviewed_by": str(profile["username"]),
                })
                with get_connection() as conn:
                    append_thesis_review(
                        conn,
                        selected_ticker,
                        {
                            "what_changed": what_changed.strip(),
                            "evidence": evidence.strip(),
                            "decision": review_decision,
                            "lesson": lesson.strip(),
                            "next_review_date": next_review.isoformat(),
                        },
                        prior_status=current_status,
                        new_status=new_status,
                        prior_conviction=prior_conviction,
                        new_conviction=int(new_conviction),
                        prior_snapshot=current_payload,
                        new_snapshot=new_payload,
                        reviewed_by=str(profile["username"]),
                    )
                    active_strategy = data.get("strategy_record") if isinstance(data.get("strategy_record"), dict) else {}
                    upsert_holding_thesis(
                        conn,
                        selected_ticker,
                        new_payload,
                        status=new_status,
                        conviction=int(new_conviction),
                        strategy_version=active_strategy.get("version"),
                        next_review_at=next_review.isoformat(),
                        updated_by=str(profile["username"]),
                    )
                st.success("Thesis review appended and the current thesis projection updated.")
                st.rerun()
        if thesis_reviews:
            st.markdown("##### Append-only thesis-review history")
            st.dataframe(pd.DataFrame([{
                "ID": item.get("id"),
                "Ticker": item.get("ticker"),
                "Reviewed": item.get("reviewed_at"),
                "Reviewer": item.get("reviewed_by"),
                "Prior status": str(item.get("prior_status") or "").replace("_", " ").title(),
                "New status": str(item.get("new_status") or "").replace("_", " ").title(),
                "Prior conviction": item.get("prior_conviction"),
                "New conviction": item.get("new_conviction"),
                "What changed": _strategy_payload(item).get("what_changed") or "",
                "Lesson": _strategy_payload(item).get("lesson") or "",
            } for item in thesis_reviews]), use_container_width=True, hide_index=True)

    with decision_tab:
        if not decision_rows:
            st.info("Log an investment decision before completing a post-mortem.")
        else:
            reviewed_ids = {int(item.get("decision_id")) for item in decision_reviews}
            due_rows = []
            for item in decision_rows:
                horizon = int(item.get("horizon_days") or 0)
                try:
                    decision_day = datetime.fromisoformat(str(item.get("date") or "")).date()
                except ValueError:
                    decision_day = None
                due_day = decision_day + timedelta(days=horizon) if decision_day and horizon else None
                due_rows.append({
                    "Decision ID": int(item["id"]),
                    "Date": item.get("date"),
                    "Ticker": item.get("ticker"),
                    "Action": item.get("action"),
                    "Horizon days": horizon or None,
                    "Review due": due_day.isoformat() if due_day else "Not scheduled",
                    "Due status": (
                        "Reviewed" if int(item["id"]) in reviewed_ids
                        else "Overdue" if due_day and due_day < date.today()
                        else "Upcoming" if due_day
                        else "Unscheduled"
                    ),
                })
            st.dataframe(pd.DataFrame(due_rows), use_container_width=True, hide_index=True)
            decision_by_label = {
                f"#{item['id']} · {str(item.get('date') or '')[:10]} · {item.get('action')} {item.get('ticker')}": item
                for item in decision_rows
            }
            selected_label = st.selectbox("Decision to review", list(decision_by_label), key="decision_review_select")
            selected = decision_by_label[selected_label]
            process_labels = {
                "Rules followed": "confirmed",
                "Partially followed": "mixed",
                "Rules not followed": "invalidated",
                "Not assessed": "not_assessed",
            }
            market_labels = {
                "Not assessed": "not_assessed",
                "Win": "win",
                "Flat": "flat",
                "Loss": "loss",
            }
            thesis_outcomes = ["Supported", "Mixed", "Invalidated", "Not assessed"]
            with st.form(f"decision_review_form_{selected['id']}"):
                d1, d2, d3 = st.columns(3)
                with d1:
                    process_label = st.selectbox("Process outcome", list(process_labels))
                with d2:
                    thesis_outcome = st.selectbox("Thesis outcome", thesis_outcomes)
                with d3:
                    market_label = st.selectbox("Market outcome", list(market_labels))
                d4, d5 = st.columns(2)
                with d4:
                    actual_return_known = st.checkbox("Actual return measured")
                    actual_return = st.number_input("Ticker return (%)", -100.0, 1000.0, 0.0, 1.0, disabled=not actual_return_known)
                with d5:
                    benchmark_return_known = st.checkbox("Benchmark return measured")
                    observed_benchmark = st.text_input(
                        "Observed benchmark", value=str(selected.get("benchmark_ticker") or "SPY")
                    ).strip().upper()
                    benchmark_return = st.number_input("Benchmark return (%)", -100.0, 1000.0, 0.0, 1.0, disabled=not benchmark_return_known)
                catalyst_outcome = st.text_area("Catalyst and evidence outcome", height=80)
                lessons = st.text_area("What should the team repeat or change?", height=90)
                next_action = st.text_input("Next analytical action")
                append_review = st.form_submit_button("Append Decision Review", type="primary", use_container_width=True)
            if append_review:
                actual_value = float(actual_return) / 100.0 if actual_return_known else None
                benchmark_value = float(benchmark_return) / 100.0 if benchmark_return_known else None
                active_return = (
                    actual_value - benchmark_value
                    if actual_value is not None and benchmark_value is not None else None
                )
                decision_tickers = [
                    item.strip().upper()
                    for item in str(selected.get("ticker") or "").replace(",", " ").split()
                    if item.strip()
                ]
                review_ticker = decision_tickers[0] if len(decision_tickers) == 1 else "MULTI"
                with get_connection() as conn:
                    append_decision_review(
                        conn,
                        int(selected["id"]),
                        review_ticker,
                        {
                            "thesis_outcome": thesis_outcome.lower().replace(" ", "_"),
                            "ticker_return": actual_value,
                            "benchmark_ticker": observed_benchmark,
                            "benchmark_return": benchmark_value,
                            "active_return": active_return,
                            "catalyst_outcome": catalyst_outcome.strip(),
                            "lessons": lessons.strip(),
                            "next_action": next_action.strip(),
                            "calculation_as_of": date.today().isoformat(),
                            "returns_are_user_verified": bool(actual_return_known or benchmark_return_known),
                        },
                        process_outcome=process_labels[process_label],
                        market_outcome=market_labels[market_label],
                        reviewed_by=str(profile["username"]),
                    )
                st.success("Decision review appended without changing the original decision record.")
                st.rerun()
        if decision_reviews:
            st.markdown("##### Append-only decision-review history")
            st.dataframe(pd.DataFrame([{
                "ID": item.get("id"),
                "Decision ID": item.get("decision_id"),
                "Ticker": item.get("ticker"),
                "Reviewed": item.get("reviewed_at"),
                "Reviewer": item.get("reviewed_by"),
                "Process outcome": str(item.get("process_outcome") or "").replace("_", " ").title(),
                "Market outcome": str(item.get("market_outcome") or "Not assessed").replace("_", " ").title(),
                "Thesis outcome": str(_strategy_payload(item).get("thesis_outcome") or "Not assessed").replace("_", " ").title(),
                "Active return": _strategy_payload(item).get("active_return"),
                "Lesson": _strategy_payload(item).get("lessons") or "",
            } for item in decision_reviews]), use_container_width=True, hide_index=True)

    with learning_tab:
        reviewed_decisions = {int(item.get("decision_id")) for item in decision_reviews}
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Thesis reviews", len(thesis_reviews))
        l2.metric("Decision reviews", len(decision_reviews))
        l3.metric(
            "Decision review coverage",
            f"{len(reviewed_decisions) / len(decision_rows):.0%}" if decision_rows else "N/A",
        )
        l4.metric(
            "Rules-followed reviews",
            sum(item.get("process_outcome") == "confirmed" for item in decision_reviews),
        )
        if decision_reviews:
            matrix = pd.crosstab(
                pd.Series([str(item.get("process_outcome") or "not_assessed") for item in decision_reviews], name="Process"),
                pd.Series([str(item.get("market_outcome") or "not_assessed") for item in decision_reviews], name="Market"),
            )
            st.markdown("##### Process outcome versus market outcome")
            st.dataframe(matrix, use_container_width=True)
            st.caption("This matrix is descriptive only; market wins do not validate a process and losses do not automatically invalidate it.")
        lessons = [
            str(_strategy_payload(item).get("lessons") or "").strip()
            for item in decision_reviews if str(_strategy_payload(item).get("lessons") or "").strip()
        ]
        if lessons:
            st.markdown("##### Recorded lessons")
            for lesson in lessons:
                st.write(f"• {lesson}")


def _render_strategy_workspace(profile: dict[str, str | int], result: dict) -> None:
    st.markdown("### Strategy Lab")
    st.caption(
        "Client mandate, repeatable investment rules, portfolio alignment, thesis lifecycle, and approved-universe control. "
        "This is an analytical workspace; it does not generate competition submissions."
    )
    data = _load_strategy_workspace_data()
    mandate = _strategy_payload(data.get("mandate_record"))
    goal_names = [
        str(item.get("name")) for item in mandate.get("goals", [])
        if isinstance(item, dict) and item.get("name")
    ]
    mandate_tab, rulebook_tab, alignment_tab, pretrade_tab, thesis_tab, catalysts_tab, universe_tab, decisions_tab, review_tab = st.tabs([
        "Client Mandate", "Strategy Rulebook", "Alignment & Drift", "Pre-Trade Lab",
        "Thesis Monitor", "Catalyst Calendar", "Approved Universe", "Decision Journal", "Review & Learning",
    ])
    with mandate_tab:
        _render_client_mandate(profile, data.get("mandate_record"))
    with rulebook_tab:
        _render_strategy_rulebook(profile, data)
    with alignment_tab:
        _render_strategy_alignment(data)
    with pretrade_tab:
        _render_pretrade_lab(data)
    with thesis_tab:
        _render_thesis_monitor(profile, data)
    with catalysts_tab:
        _render_catalyst_calendar(profile, data)
    with universe_tab:
        _render_approved_universe(profile, data)
    with decisions_tab:
        _render_decision_log(profile, result, goal_names=goal_names or None)
    with review_tab:
        _render_review_learning(profile, data)


def _render_decision_log(
    profile: dict[str, str | int],
    result: dict,
    goal_names: list[str] | None = None,
) -> None:
    import json
    available_goals = goal_names or ["Growth", "Income", "Risk Tolerance", "Community/Impact"]
    metric_groups = {
        "Valuation": ["Trailing P/E", "Forward P/E", "PEG Ratio", "EV/EBITDA", "EV/Revenue"],
        "Growth": ["Rev Growth (YoY)", "Earnings Growth", "Price CAGR (3Y)", "Price CAGR (5Y)"],
        "Profitability": ["Gross Margin", "Op Margin", "Net Margin", "ROE", "ROIC", "Free Cash Flow (B)"],
        "Risk": ["Beta", "Debt/Equity", "52W High", "52W Low"],
        "Analyst": ["Analyst Target", "Upside to Target"],
    }
    st.markdown("### Investment Decision Log")
    st.caption("Chronological record of strategy decisions, including the price captured at each decision. Every team edit is marked in the price chart with that member's colour.")

    # ── Form to log a new decision ────────────────────────────────────────────
    with st.expander("Log New Decision", expanded=False):
        with st.form("add_decision_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                ticker = st.text_input("Ticker(s)", placeholder="e.g. MSFT or MSFT, AAPL")
            with c2:
                action = st.selectbox("Action", ["Buy", "Sell", "Hold", "Rebalance", "Other"])

            c3, c4 = st.columns(2)
            with c3:
                if goal_names:
                    st.caption("Goals come from the active Client Mandate in Strategy Lab.")
                else:
                    st.caption("Default analytical buckets are shown until the Client Mandate is saved.")
                tags = st.multiselect("Client Goals Addressed", available_goals)
            with c4:
                fetch_fundamentals = st.checkbox("Fetch live fundamentals from Yahoo Finance", value=True)

            thesis = st.text_area("Investment Thesis / Rationale", height=120,
                                  placeholder="Why are we making this decision? What is the expected outcome?")
            st.markdown("##### Testable expectation")
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                horizon_days = st.number_input("Review horizon (days)", 1, 3650, 365, 30)
            with e2:
                decision_benchmark = st.text_input("Decision benchmark", value="SPY").strip().upper()
            with e3:
                expected_return_min = st.number_input("Expected return floor (%)", -100.0, 1000.0, -10.0, 1.0)
            with e4:
                expected_return_max = st.number_input("Expected return ceiling (%)", -100.0, 1000.0, 20.0, 1.0)
            e5, e6 = st.columns(2)
            with e5:
                decision_confidence = st.slider("Decision confidence", 1, 5, 3)
                planned_weight = st.number_input("Planned portfolio weight (%)", 0.0, 100.0, 0.0, 1.0)
            with e6:
                target_condition = st.text_area(
                    "Observable success condition",
                    placeholder="What must become true for the thesis to be supported?",
                    height=85,
                )
            invalidation_condition = st.text_area(
                "Observable invalidation condition",
                placeholder="What evidence would prove the decision thesis wrong?",
                height=75,
            )
            st.caption(
                "The range and confidence are team-authored expectations for later review, not an app forecast."
            )

            if st.form_submit_button("Log Decision", type="primary"):
                if not ticker or not thesis:
                    st.warning("Ticker and Thesis are required.")
                elif expected_return_min > expected_return_max:
                    st.warning("Expected return floor cannot exceed the ceiling.")
                else:
                    snap: dict[str, Any] = {}
                    decision_timestamp = _now_iso()

                    # Portfolio-level context from Quant Engine
                    if result and "metrics" in result:
                        m = result["metrics"]
                        portfolio_snap: dict[str, Any] = {
                            "Sharpe Ratio": m.get("sharpe_ratio"),
                            "Ann. Return": m.get("annualized_return"),
                            "Volatility": m.get("volatility"),
                            "Max Drawdown": m.get("max_drawdown"),
                        }
                        # Compute CVaR-95 directly from portfolio returns
                        port_rets = result.get("portfolio_returns")
                        if port_rets is not None and hasattr(port_rets, "__len__") and len(port_rets) >= 30:
                            try:
                                from src.analytics.risk_metrics import calculate_cvar
                                cvar_95 = calculate_cvar(pd.Series(port_rets).dropna(), confidence_level=0.95)
                                portfolio_snap["CVaR-95 (daily)"] = float(cvar_95)
                            except Exception:
                                pass
                        # Cast all values to plain Python float to avoid json.dumps
                        # failing on numpy.float64 / numpy.float32 scalars.
                        snap["_portfolio"] = {
                            k: (float(v) if v is not None else None)
                            for k, v in portfolio_snap.items()
                        }

                    # Per-ticker fundamentals
                    if fetch_fundamentals:
                        tickers_list = [t.strip().upper() for t in ticker.replace(",", " ").split() if t.strip()]
                        with st.spinner(f"Fetching fundamentals for {', '.join(tickers_list)}…"):
                            snap["_fundamentals"] = {}
                            for t in tickers_list:
                                snap["_fundamentals"][t] = _fetch_ticker_fundamentals(t)
                    snap["_metadata"] = {
                        "captured_at": decision_timestamp,
                        "fundamentals_source": "Yahoo Finance" if fetch_fundamentals else None,
                    }

                    snap_json = json.dumps(snap)
                    tags_str = ",".join(tags)
                    with get_connection() as conn:
                        conn.execute(
                            """
                            INSERT INTO decision_log (
                                ticker, action, date, thesis, client_goal_tags, team_member,
                                quant_snapshot_json, updated_at, updated_by, horizon_days,
                                benchmark_ticker, expected_return_min, expected_return_max,
                                decision_confidence, target_condition, invalidation_condition,
                                planned_weight
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ticker.upper(), action, decision_timestamp, thesis, tags_str,
                                str(profile["username"]), snap_json, None, None, int(horizon_days),
                                decision_benchmark or None, float(expected_return_min) / 100.0,
                                float(expected_return_max) / 100.0, int(decision_confidence),
                                target_condition.strip(), invalidation_condition.strip(),
                                float(planned_weight) / 100.0,
                            ),
                        )
                        conn.commit()
                        if hasattr(conn, 'sync'): conn.sync()
                    st.success(f"Decision logged with {len(snap.get('_fundamentals', {}))} ticker(s) snapshotted.")
                    st.rerun()

    # ── Load log ──────────────────────────────────────────────────────────────
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM decision_log ORDER BY id DESC").fetchall()
        edit_rows = conn.execute("SELECT * FROM decision_edit_log ORDER BY id ASC").fetchall()

    if not rows:
        st.info("No decisions logged yet. Use the form above to log your first investment decision.")
        return

    # -- Decision timeline table
    st.markdown('#### Decision Timeline')
    df_data = []
    for r in rows:
        snap = {}
        try:
            snap = json.loads(r['quant_snapshot_json'] or '{}')
        except Exception:
            pass
        funds = snap.get('_fundamentals', {})
        first_tkr_metrics = next(iter(funds.values()), {}) if funds else {}
        edited_by = r['updated_by'] if r['updated_by'] else ''
        row_dict = {
            'Date': r['date'][:16],
            'Action': r['action'],
            'Ticker(s)': r['ticker'],
            'Goals': r['client_goal_tags'],
            'Author': r['team_member'],
            'Horizon': f"{int(r['horizon_days'])}d" if r['horizon_days'] else '—',
            'Benchmark': r['benchmark_ticker'] or '—',
            'Expected range': (
                f"{float(r['expected_return_min']):+.1%} to {float(r['expected_return_max']):+.1%}"
                if r['expected_return_min'] is not None and r['expected_return_max'] is not None else '—'
            ),
            'Confidence': r['decision_confidence'] or '—',
            'Planned weight': (
                f"{float(r['planned_weight']):.1%}" if r['planned_weight'] is not None else '—'
            ),
            'Last edit': edited_by or '—',
        }
        for label in ('Price', 'Forward P/E', 'PEG Ratio', 'Price CAGR (3Y)', 'EV/EBITDA', 'Upside to Target'):
            row_dict[label] = first_tkr_metrics.get(label, {}).get('formatted', '—')
        preview = str(r['thesis'])[:80]
        row_dict['Thesis (preview)'] = preview + ('…' if len(str(r['thesis'])) > 80 else '')
        df_data.append(row_dict)
    st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)

    # -- Price at each decision, with a persistent colour for each team editor
    st.markdown('#### Price at Decision & Team Edit History')
    st.caption('The curve shows the latest six months of daily prices. Circles mark the price captured at each decision; diamonds show subsequent edits in the editor’s colour.')
    try:
        import plotly.graph_objects as go

        price_points: dict[str, list[dict[str, Any]]] = {}
        for r in reversed(rows):
            try:
                snapshot = json.loads(r['quant_snapshot_json'] or '{}')
            except Exception:
                snapshot = {}
            for tkr, metrics in snapshot.get('_fundamentals', {}).items():
                price_data = metrics.get('Price', {})
                try:
                    price = float(price_data['value'])
                except (KeyError, TypeError, ValueError):
                    continue
                price_points.setdefault(tkr, []).append({
                    'id': int(r['id']), 'date': str(r['date']), 'price': price,
                    'action': str(r['action']), 'author': str(r['team_member']),
                })

        # Older records only retained their latest editor. Show that edit too,
        # without duplicating an event already stored in the audit table.
        recorded_decision_ids = {int(event['decision_id']) for event in edit_rows}
        chart_edits = [dict(event) for event in edit_rows]
        for r in rows:
            if r['updated_by'] and int(r['id']) not in recorded_decision_ids:
                chart_edits.append({
                    'decision_id': int(r['id']), 'ticker': str(r['ticker']),
                    'edited_at': str(r['updated_at'] or r['date']),
                    'edited_by': str(r['updated_by']),
                })

        editor_names = sorted({str(event['edited_by']) for event in chart_edits if event['edited_by']})
        editor_palette = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#0891b2', '#db2777', '#4f46e5']
        editor_colours = {
            name: editor_palette[index % len(editor_palette)]
            for index, name in enumerate(editor_names)
        }

        chart_count = 0
        for tkr, points in price_points.items():
            fig = go.Figure()
            history = _fetch_six_month_price_history(tkr)
            if history:
                period_start = history[0][0]
                points = [point for point in points if point['date'][:10] >= period_start]
                fig.add_trace(go.Scatter(
                    x=[date for date, _ in history],
                    y=[close for _, close in history],
                    mode='lines',
                    name='6-month price curve',
                    line=dict(width=2, color='#64748b'),
                    hovertemplate='%{x}<br>Close: $%{y:,.2f}<extra></extra>',
                ))
            fig.add_trace(go.Scatter(
                x=[point['date'] for point in points],
                y=[point['price'] for point in points],
                mode='markers',
                name='Decision price',
                marker=dict(size=10, color='#f8fafc', line=dict(width=2, color='#475569')),
                customdata=[[point['action'], point['author']] for point in points],
                hovertemplate='%{x}<br>Price: $%{y:,.2f}<br>%{customdata[0]} by %{customdata[1]}<extra></extra>',
            ))
            points_by_id = {point['id']: point for point in points}
            for editor in editor_names:
                edited_points = [
                    (event, points_by_id.get(int(event['decision_id'])))
                    for event in chart_edits if str(event['edited_by']) == editor
                ]
                edited_points = [(event, point) for event, point in edited_points if point]
                if not edited_points:
                    continue
                fig.add_trace(go.Scatter(
                    x=[point['date'] for _, point in edited_points],
                    y=[point['price'] for _, point in edited_points],
                    mode='markers',
                    name=f'Edited by {editor}',
                    marker=dict(size=13, symbol='diamond', color=editor_colours[editor],
                                line=dict(width=1, color='#ffffff')),
                    customdata=[[str(event['edited_at']), point['action']] for event, point in edited_points],
                    hovertemplate=(
                        f'Edited by {editor}<br>%{{customdata[0]}}<br>'
                        'Decision: %{customdata[1]}<br>Price: $%{y:,.2f}<extra></extra>'
                    ),
                ))
            fig.update_layout(
                title=f'{tkr} — six-month price curve and decisions',
                height=310, margin=dict(l=20, r=20, t=45, b=25),
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', yaxis_title='Price (USD)',
                legend_title_text='Team edit markers',
            )
            st.plotly_chart(fig, use_container_width=True)
            chart_count += 1
        if not chart_count:
            st.info('Price charts appear once a decision is logged with live fundamentals enabled.')
    except ImportError:
        st.info('Install plotly to see the price and edit history charts.')

    # -- Per-decision metric detail cards
    st.markdown('#### Decision Detail & Metrics')
    for r in rows:
        snap = {}
        try:
            snap = json.loads(r['quant_snapshot_json'] or '{}')
        except Exception:
            pass
        funds = snap.get('_fundamentals', {})
        action_color = {'Buy': '#10b981', 'Sell': '#ef4444', 'Hold': '#f59e0b',
                        'Rebalance': '#6366f1', 'Other': '#64748b'}.get(str(r['action']), '#64748b')
        exp_label = f"{r['date'][:10]}  |  {r['action']} {r['ticker']}  |  {r['team_member']}"
        with st.expander(exp_label, expanded=False):
            updated_by = str(r['updated_by'] or '')
            was_edited_by_teammate = bool(updated_by and updated_by != str(r['team_member']))
            if was_edited_by_teammate:
                st.markdown(
                    f"<div style='background:#f3e8ff;color:#6b21a8;border-left:4px solid #9333ea;"
                    f"padding:0.55rem 0.75rem;border-radius:4px;margin-bottom:0.75rem;font-weight:600'>"
                    f"Edited by {escape(updated_by)} on {escape(str(r['updated_at'] or 'an unknown date'))}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif updated_by:
                st.caption(f"Last updated by {updated_by} on {r['updated_at']}")
            st.markdown(
                f"<span style='background:{action_color};color:#fff;padding:2px 10px;"
                f"border-radius:4px;font-weight:700'>{r['action']}</span>  "
                f"<span style='color:#94a3b8'>Goals: {r['client_goal_tags'] or '—'}</span>",
                unsafe_allow_html=True
            )
            thesis_style = "color:#6b21a8;background:#faf5ff;padding:0.75rem;border-radius:4px;" if was_edited_by_teammate else ""
            st.markdown(
                f"<p><strong>Thesis:</strong></p><div style='{thesis_style}'>{escape(str(r['thesis']))}</div>",
                unsafe_allow_html=True,
            )
            expectation_cols = st.columns(4)
            expectation_cols[0].metric("Review horizon", f"{int(r['horizon_days'])} days" if r['horizon_days'] else "Not set")
            expectation_cols[1].metric("Benchmark", str(r['benchmark_ticker'] or "Not set"))
            expectation_cols[2].metric(
                "Expected range",
                (
                    f"{float(r['expected_return_min']):+.1%} to {float(r['expected_return_max']):+.1%}"
                    if r['expected_return_min'] is not None and r['expected_return_max'] is not None
                    else "Not set"
                ),
            )
            expectation_cols[3].metric("Confidence", f"{int(r['decision_confidence'])}/5" if r['decision_confidence'] else "Not set")
            if r['target_condition']:
                st.markdown(f"**Success condition:** {escape(str(r['target_condition']))}")
            if r['invalidation_condition']:
                st.markdown(f"**Invalidation condition:** {escape(str(r['invalidation_condition']))}")
            st.caption("Original expectation fields are locked; later evaluation is appended in Review & Learning.")
            # This is intentionally a container rather than a nested expander:
            # Streamlit does not support expanders inside expanders.
            with st.container():
                st.markdown("**Edit this decision**")
                allowed_actions = ["Buy", "Sell", "Hold", "Rebalance", "Other"]
                current_action = str(r['action'])
                action_index = allowed_actions.index(current_action) if current_action in allowed_actions else len(allowed_actions) - 1
                current_tags = [tag for tag in str(r['client_goal_tags'] or '').split(',') if tag]
                with st.form(f"edit_decision_{r['id']}"):
                    edit_ticker = st.text_input("Ticker(s)", value=str(r['ticker']), key=f"decision_ticker_{r['id']}")
                    edit_action = st.selectbox("Action", allowed_actions, index=action_index, key=f"decision_action_{r['id']}")
                    edit_tags = st.multiselect(
                        "Client Goals Addressed",
                        ["Growth", "Income", "Risk Tolerance", "Community/Impact"],
                        default=[tag for tag in current_tags if tag in {"Growth", "Income", "Risk Tolerance", "Community/Impact"}],
                        key=f"decision_tags_{r['id']}",
                    )
                    edit_thesis = st.text_area("Investment Thesis / Rationale", value=str(r['thesis']), height=150, key=f"decision_thesis_{r['id']}")
                    if st.form_submit_button("Save team edit", type="primary"):
                        if not edit_ticker.strip() or not edit_thesis.strip():
                            st.warning("Ticker and Thesis are required.")
                        else:
                            edited_at = _now_iso()
                            with get_connection() as conn:
                                conn.execute(
                                    "UPDATE decision_log SET ticker = ?, action = ?, thesis = ?, client_goal_tags = ?, updated_at = ?, updated_by = ? WHERE id = ?",
                                    (
                                        edit_ticker.strip().upper(), edit_action, edit_thesis.strip(), ",".join(edit_tags),
                                        edited_at, str(profile["username"]), r['id'],
                                    ),
                                )
                                conn.execute(
                                    "INSERT INTO decision_edit_log (decision_id, ticker, edited_at, edited_by) VALUES (?, ?, ?, ?)",
                                    (r['id'], edit_ticker.strip().upper(), edited_at, str(profile["username"])),
                                )
                                conn.commit()
                                if hasattr(conn, 'sync'):
                                    conn.sync()
                            st.success("Decision updated. Its edit marker now appears in the price chart using this team member's colour.")
                            st.rerun()
            if funds:
                for tkr, mets in funds.items():
                    st.markdown(f'**{tkr} — Fundamentals at Decision Date**')
                    if '_error' in mets:
                        st.warning(f"Could not fetch data: {mets['_error']['formatted']}")
                        continue
                    for group_name, labels in metric_groups.items():
                        group_mets = {lbl: mets[lbl] for lbl in labels if lbl in mets}
                        if not group_mets:
                            continue
                        st.caption(group_name)
                        cols = st.columns(min(len(group_mets), 5))
                        for col, (lbl, m_data) in zip(cols, group_mets.items()):
                            col.metric(lbl, m_data['formatted'], help=m_data.get('desc', ''))
            if '_portfolio' in snap:
                st.caption('Portfolio Context')
                p = snap['_portfolio']
                pcols = st.columns(3)
                for col, (k, v) in zip(pcols, {kk: vv for kk, vv in p.items() if vv is not None}.items()):
                    try:
                        col.metric(k, f'{float(v):.3f}')
                    except Exception:
                        col.metric(k, str(v))

    # -- Time-series: Key metrics evolution
    st.markdown('#### Metric Evolution Over Decisions')
    st.caption('Tracks how key valuation metrics changed across logged decisions for the same ticker.')
    try:
        import plotly.graph_objects as go
        TRACKED = ['Forward P/E', 'PEG Ratio', 'Price CAGR (3Y)', 'EV/EBITDA', 'Upside to Target']
        ticker_series: dict[str, dict] = {}
        for r in reversed(rows):
            snap = {}
            try:
                snap = json.loads(r['quant_snapshot_json'] or '{}')
            except Exception:
                pass
            for tkr, mets in snap.get('_fundamentals', {}).items():
                if tkr not in ticker_series:
                    ticker_series[tkr] = {m: ([], []) for m in TRACKED}
                for m in TRACKED:
                    if m in mets and '_error' not in mets:
                        try:
                            ticker_series[tkr][m][0].append(r['date'][:10])
                            ticker_series[tkr][m][1].append(float(mets[m]['value']))
                        except Exception:
                            pass
        any_chart = False
        for tkr, metrics_data in ticker_series.items():
            for metric, (dates, values) in metrics_data.items():
                if len(dates) >= 2:
                    if not any_chart:
                        st.info('Showing metric history for tickers with ≥2 logged decisions.')
                    any_chart = True
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers',
                                             line=dict(width=2, color='#6366f1'),
                                             marker=dict(size=8)))
                    fig.update_layout(title=f'{tkr} — {metric} over time', height=280,
                                      margin=dict(l=20, r=20, t=40, b=20), template='plotly_dark',
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        if not any_chart:
            st.info('Log the same ticker at least twice to see metric evolution charts.')
    except ImportError:
        st.info('Install plotly to see metric evolution charts.')

    # -- Client-Goal Alignment Matrix
    st.markdown('#### Client-Goal Alignment Matrix')
    st.caption(
        'Uses the active Client Mandate.' if goal_names
        else 'Default analytical buckets are used until the Client Mandate is saved.'
    )
    matrix_data: dict[str, dict] = {}
    for r in rows:
        t = r['ticker']
        if t not in matrix_data:
            matrix_data[t] = {goal: '' for goal in available_goals}
        for tag in str(r['client_goal_tags']).split(','):
            tag = tag.strip()
            if tag in matrix_data[t]:
                matrix_data[t][tag] = '✓'
    if matrix_data:
        matrix_df = pd.DataFrame.from_dict(matrix_data, orient='index').reset_index()
        matrix_df = matrix_df.rename(columns={'index': 'Ticker'})
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    else:
        st.info('Add client goals to decisions to populate the matrix.')

# ─── Header / Shell ───────────────────────────────────────────────────────────

def _fetch_competition_settings() -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM competition_compliance WHERE id = 1").fetchone()
    return dict(row) if row else {}


def _fetch_competition_positions() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM competition_positions ORDER BY entry_date, id").fetchall()
    return [dict(row) for row in rows]


def _render_competition_rules(profile: dict[str, str | int]) -> None:
    from src.portfolio_tracker.wharton_competition import COMPETITION_URL, OFFICIAL_RULES_URL, evaluate_compliance

    st.markdown("### Assignment & Rules — Wharton 2026–2027")
    st.caption("The compliance check uses only currently published official rules and gives the exact reason for every failed check.")
    source_col, overview_col = st.columns(2)
    source_col.link_button("Official 2026–2027 Rules", OFFICIAL_RULES_URL, use_container_width=True)
    overview_col.link_button("Official Competition Overview", COMPETITION_URL, use_container_width=True)
    st.info(
        "**Currently published assignment:** build a long-term investment strategy for the client and "
        "manage USD 500,000 of virtual capital in WInS over 10 weeks. The competition evaluates the "
        "quality and explanation of the strategy, not simply the highest return."
    )
    st.warning(
        "Wharton currently marks the 2026–2027 trading rules as ‘More information coming soon.’ "
        "The new client case and detailed deliverables have not been published either. These checks "
        "therefore remain yellow instead of relying on outdated requirements."
    )

    current = _fetch_competition_settings()
    st.markdown("#### Team Declaration")
    with st.form("competition_compliance_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            team_size = st.number_input("Number of active students", 0, 20, int(current.get("team_size") or 0))
            leader_age = st.number_input("Team leader's age at the start", 0, 25, int(current.get("leader_age") or 0))
            advisor_team_count = st.number_input("Number of teams led by the primary advisor", 0, 50, int(current.get("advisor_team_count") or 0))
        with c2:
            same_school = st.checkbox("All members attend the same school and campus", value=bool(current.get("same_school")))
            eligible_students = st.checkbox("All members meet the age and student-status rules and have not graduated from high school", value=bool(current.get("eligible_students")))
            leader_designated = st.checkbox("Exactly one student leader has been designated", value=bool(current.get("leader_designated")))
            advisor_is_teacher = st.checkbox("The primary advisor is a teacher at the team's school", value=bool(current.get("advisor_is_teacher")))
            one_wins_account = st.checkbox("The team uses one shared WInS account", value=bool(current.get("one_wins_account")))
            members_single_team = st.checkbox("No student participates on another competition team", value=bool(current.get("members_single_team")))
        with c3:
            no_client_contact = st.checkbox("The team has not contacted the competition client", value=bool(current.get("no_client_contact")))
            no_paid_advisor = st.checkbox("No paid advisor, consultant, or prohibited course has been used", value=bool(current.get("no_paid_advisor")))
            student_owned_work = st.checkbox("Students created the strategy and made the decisions", value=bool(current.get("student_owned_work")))
            ai_cited = st.checkbox("Generated content is cited and is not presented as original student work", value=bool(current.get("ai_cited")))
            sources_cited = st.checkbox("All sources, images, and media are cited", value=bool(current.get("sources_cited")))
            school_permission = st.checkbox("School authorization on official letterhead is ready", value=bool(current.get("school_permission")))
        save_rules = st.form_submit_button("Save and Recalculate Compliance", type="primary", use_container_width=True)

    if save_rules:
        payload = {
            "team_size": int(team_size), "leader_age": int(leader_age), "advisor_team_count": int(advisor_team_count),
            "same_school": int(same_school), "eligible_students": int(eligible_students),
            "leader_designated": int(leader_designated), "advisor_is_teacher": int(advisor_is_teacher),
            "one_wins_account": int(one_wins_account), "members_single_team": int(members_single_team),
            "no_client_contact": int(no_client_contact), "no_paid_advisor": int(no_paid_advisor),
            "student_owned_work": int(student_owned_work), "ai_cited": int(ai_cited),
            "sources_cited": int(sources_cited), "school_permission": int(school_permission),
        }
        fields = list(payload)
        assignments = ", ".join(f"{field} = excluded.{field}" for field in fields)
        with get_connection() as conn:
            conn.execute(
                f"INSERT INTO competition_compliance (id, {', '.join(fields)}, updated_at, updated_by) "
                f"VALUES (1, {', '.join(['?'] * len(fields))}, ?, ?) "
                f"ON CONFLICT(id) DO UPDATE SET {assignments}, updated_at = excluded.updated_at, updated_by = excluded.updated_by",
                (*[payload[field] for field in fields], _now_iso(), str(profile["username"])),
            )
            conn.commit()
            if hasattr(conn, "sync"):
                conn.sync()
        st.success(f"Compliance declaration updated by {profile['username']}.")
        current = payload

    checks = evaluate_compliance(current, _fetch_competition_positions())
    passed = sum(item["status"] == "pass" for item in checks)
    failed = sum(item["status"] == "fail" for item in checks)
    pending = sum(item["status"] == "pending" for item in checks)
    k1, k2, k3 = st.columns(3)
    k1.metric("Passed", passed)
    k2.metric("Failed", failed)
    k3.metric("Awaiting Wharton", pending)
    status_map = {"pass": "Pass", "fail": "Fail", "pending": "Pending"}
    st.dataframe(pd.DataFrame([
        {"Status": status_map[item["status"]], "Rule": item["rule"], "Exact Check Result": item["detail"]}
        for item in checks
    ]), use_container_width=True, hide_index=True)
    if failed:
        st.error(f"The team or portfolio currently fails {failed} checked rule(s). The exact reasons are shown above.")
    elif pending:
        st.warning("All published, verifiable rules are satisfied; additional official materials are still pending.")
    else:
        st.success("All checked rules are satisfied.")


def _competition_live_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    try:
        end_date = date.today() + timedelta(days=1)
        prices = _fetch_close_prices_cached(tuple(sorted(set(tickers))), end_date - timedelta(days=10), end_date)
        result: dict[str, float] = {}
        for ticker in tickers:
            if ticker in prices.columns:
                clean = pd.Series(prices[ticker]).dropna()
                if not clean.empty:
                    result[ticker] = float(clean.iloc[-1])
        return result
    except Exception:
        return {}


def _render_wins_reconciliation(
    positions: list[dict[str, Any]],
    live_prices: dict[str, float],
) -> None:
    """Compare a user-supplied WInS snapshot without mutating tracked positions."""
    from src.portfolio_tracker.wins_reconciliation import (
        normalize_wins_rows,
        reconcile_wins_positions,
    )

    with st.expander("WInS Reconciliation", expanded=False):
        st.caption(
            "Upload a WInS positions snapshot to verify that the analytical tracker matches the "
            "competition account. The file is analysed in memory and never changes either system."
        )
        uploaded = st.file_uploader(
            "WInS positions snapshot",
            type=["csv", "xlsx", "xls"],
            key="wins_reconciliation_upload",
            help=(
                "Common headers such as Ticker/Symbol, Quantity/Shares, Cost Basis, Current Price, "
                "Market Value and Security Type are detected automatically."
            ),
        )
        if uploaded is None:
            st.info("Upload a CSV or Excel export to run the reconciliation check.")
            return

        try:
            if str(uploaded.name).lower().endswith((".xlsx", ".xls")):
                uploaded_rows = pd.read_excel(uploaded).to_dict(orient="records")
            else:
                uploaded_rows = pd.read_csv(uploaded, sep=None, engine="python").to_dict(orient="records")
        except Exception as exc:
            st.error(f"The uploaded snapshot could not be read ({type(exc).__name__}).")
            return

        normalized = normalize_wins_rows(uploaded_rows)
        if uploaded_rows and not normalized:
            st.error(
                "No ticker column was recognised. Include a column named Ticker, Symbol, "
                "Ticker Symbol or Security Symbol."
            )
            return

        tracked_snapshot: list[dict[str, Any]] = []
        for position in positions:
            row = dict(position)
            ticker = str(row.get("ticker") or "").upper()
            if str(row.get("status") or "open").lower() == "open":
                market_price = live_prices.get(ticker)
                if market_price is None:
                    market_price = row.get("last_price")
                if market_price in (None, "", 0, 0.0):
                    market_price = row.get("entry_price")
                row["current_price"] = market_price
            tracked_snapshot.append(row)

        result = reconcile_wins_positions(uploaded_rows, tracked_snapshot)
        summary = result["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Two-way Coverage", f"{summary['two_way_coverage_pct']:.1f}%")
        m2.metric("Exact Matches", int(summary["exact_matches"]))
        m3.metric("Value Differences", int(summary["mismatched_positions"]))
        m4.metric(
            "Missing / Extra",
            f"{int(summary['missing_positions'])} / {int(summary['extra_positions'])}",
        )

        status = str(result.get("status") or "no_data")
        if status == "reconciled":
            st.success("The WInS snapshot and all tracked open positions reconcile within tolerance.")
        elif status == "partial":
            st.warning(
                "All comparable values agree, but at least one required value is missing on one side."
            )
        elif status == "differences":
            st.error(
                "The snapshot does not fully reconcile. Review the position-level differences below."
            )
        else:
            st.info("Neither source contains an open position to compare.")

        comparison_rows: list[dict[str, Any]] = []
        for item in result["matched"]:
            wins = item["wins"]
            tracked = item["tracked"]
            comparison_rows.append({
                "Ticker": item["ticker"],
                "Result": str(item["status"]).replace("_", " ").title(),
                "WInS Quantity": wins.get("quantity"),
                "Tracker Quantity": tracked.get("quantity"),
                "Quantity Difference": item.get("quantity_difference"),
                "WInS Cost": wins.get("total_cost"),
                "Tracker Cost": tracked.get("total_cost"),
                "Cost Difference": item.get("cost_difference"),
                "WInS Value": wins.get("current_value"),
                "Tracker Value": tracked.get("current_value"),
                "Value Difference": item.get("value_difference"),
            })
        for bucket, label in (("missing", "Missing in WInS"), ("extra", "Extra in WInS")):
            for item in result[bucket]:
                is_extra = bucket == "extra"
                comparison_rows.append({
                    "Ticker": item["ticker"],
                    "Result": label,
                    "WInS Quantity": item.get("quantity") if is_extra else None,
                    "Tracker Quantity": None if is_extra else item.get("quantity"),
                    "Quantity Difference": item.get("quantity_difference"),
                    "WInS Cost": item.get("total_cost") if is_extra else None,
                    "Tracker Cost": None if is_extra else item.get("total_cost"),
                    "Cost Difference": item.get("cost_difference"),
                    "WInS Value": item.get("current_value") if is_extra else None,
                    "Tracker Value": None if is_extra else item.get("current_value"),
                    "Value Difference": item.get("value_difference"),
                })
        if comparison_rows:
            st.dataframe(
                pd.DataFrame(comparison_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "WInS Cost": st.column_config.NumberColumn(format="$%.2f"),
                    "Tracker Cost": st.column_config.NumberColumn(format="$%.2f"),
                    "Cost Difference": st.column_config.NumberColumn(format="$%.2f"),
                    "WInS Value": st.column_config.NumberColumn(format="$%.2f"),
                    "Tracker Value": st.column_config.NumberColumn(format="$%.2f"),
                    "Value Difference": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
        st.caption(
            "Differences use WInS minus tracker. Exact matches use a quantity tolerance of 1e-8 "
            "and a USD tolerance of $0.01."
        )


def _render_live_competition_analytics(
    positions: list[dict[str, Any]],
    live_prices: dict[str, float],
) -> None:
    from src.portfolio_tracker.live_analytics import build_live_competition_analytics
    from src.portfolio_tracker.research_health import assess_research_health
    from src.portfolio_tracker.strategy_store import list_holding_theses

    open_tickers = list(dict.fromkeys(
        str(row.get("ticker") or "").upper()
        for row in positions
        if row.get("ticker") and str(row.get("status") or "open").lower() == "open"
    ))
    st.markdown("#### Live Competition Portfolio Analytics")
    st.caption(
        "Actual P/L comes from the tracker ledger. Volatility, beta, correlation, and historical risk are "
        "current-weight proxies based on adjusted market history; they are not reconstructed WInS returns."
    )
    with st.form("live_competition_analytics_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            benchmark = st.text_input("Live-portfolio benchmark", value="SPY").strip().upper()
        with c2:
            lookback_years = st.slider("Historical proxy lookback (years)", 1, 5, 2)
        with c3:
            risk_free_pct = st.number_input("Risk-free assumption (%)", 0.0, 20.0, 3.0, 0.25)
        run_live_analytics = st.form_submit_button(
            "Run Live Portfolio Analytics", type="primary", use_container_width=True,
        )
    if run_live_analytics:
        end_date = date.today() + timedelta(days=1)
        start_date = end_date - timedelta(days=365 * int(lookback_years))
        history_symbols = list(dict.fromkeys([*open_tickers, *([benchmark] if benchmark else [])]))
        with st.spinner("Loading histories and calculating live-portfolio analytics…"):
            prices = _fetch_close_prices_cached(tuple(history_symbols), start_date, end_date)
            asset_prices = prices.reindex(columns=open_tickers) if open_tickers else pd.DataFrame(index=prices.index)
            asset_returns = asset_prices.sort_index().pct_change(fill_method=None).dropna(how="all")
            benchmark_returns = (
                pd.Series(prices[benchmark]).sort_index().pct_change(fill_method=None).dropna()
                if benchmark and benchmark in prices.columns else pd.Series(dtype=float)
            )
            with get_connection() as conn:
                thesis_records = list_holding_theses(conn)
                try:
                    from src.portfolio_tracker.governance_store import (
                        list_catalyst_events,
                        list_research_sources,
                    )
                    catalysts = list_catalyst_events(conn)
                    sources = list_research_sources(conn)
                except ImportError:
                    catalysts, sources = [], []
            thesis_by_ticker = {
                str(item.get("ticker") or "").upper(): item
                for item in thesis_records if item.get("ticker")
            }
            analytics_result = build_live_competition_analytics(
                positions,
                live_prices,
                asset_returns,
                benchmark_returns=benchmark_returns,
                benchmark_ticker=benchmark,
                risk_free_rate=float(risk_free_pct) / 100.0,
                thesis_by_ticker=thesis_by_ticker,
            )
            price_observations = {
                ticker: {"observed_at": date.today().isoformat(), "source": "Yahoo Finance"}
                for ticker in open_tickers if ticker in live_prices
            }
            analytics_result["research_health"] = assess_research_health(
                open_tickers,
                theses=thesis_records,
                sources=sources,
                catalysts=catalysts,
                price_observations=price_observations,
                as_of=date.today(),
            )
            analytics_result["ui_context"] = {
                "generated_at": _now_iso(),
                "benchmark": benchmark,
                "lookback_years": int(lookback_years),
                "risk_free_rate": float(risk_free_pct) / 100.0,
                "open_tickers": open_tickers,
            }
            st.session_state[LIVE_PORTFOLIO_ANALYTICS_KEY] = analytics_result

    analytics_result = st.session_state.get(LIVE_PORTFOLIO_ANALYTICS_KEY)
    if not isinstance(analytics_result, dict):
        st.info("Run the live analysis to connect current tracker positions to risk, benchmark, and attribution analytics.")
        return

    saved_tickers = analytics_result.get("ui_context", {}).get("open_tickers", [])
    if saved_tickers != open_tickers:
        st.warning("Portfolio holdings changed after this analysis. Run it again before relying on the results.")
    context = analytics_result.get("ui_context", {})
    st.caption(
        f"Generated {context.get('generated_at', '—')} · Benchmark {context.get('benchmark') or 'none'} · "
        f"Lookback {context.get('lookback_years', '—')} years."
    )

    coverage = analytics_result.get("coverage", {})
    risk_metrics = analytics_result.get("risk_metrics", {})
    benchmark_metrics = analytics_result.get("benchmark_metrics", {})
    top = st.columns(6)
    top[0].metric("Current equity", f"${float(analytics_result.get('current_equity') or 0):,.0f}")
    top[1].metric("Current cash", f"${float(analytics_result.get('current_cash') or 0):,.0f}")
    top[2].metric("History value coverage", f"{float(coverage.get('history_value_coverage_pct') or 0):.1%}")
    top[3].metric("Annualized volatility", _fmt_pct(risk_metrics.get("volatility")))
    top[4].metric("Historical VaR 95%", _fmt_pct(risk_metrics.get("historical_var_95")))
    top[5].metric("Beta vs benchmark", _fmt_float(benchmark_metrics.get("beta_to_benchmark")))

    risk_tab, attribution_tab, stress_tab, health_tab = st.tabs([
        "Risk & Benchmark Proxy", "Actual P/L Attribution", "Exposure Stress", "Evidence & Review Queue",
    ])
    with risk_tab:
        if not analytics_result.get("risk_proxy_available"):
            st.warning("Historical risk proxy is unavailable because too little compatible price history was found.")
        else:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Annualized return proxy", _fmt_pct(risk_metrics.get("annualized_return")))
            r2.metric("Maximum drawdown proxy", _fmt_pct(risk_metrics.get("max_drawdown")))
            r3.metric("CVaR 95% proxy", _fmt_pct(risk_metrics.get("historical_cvar_95")))
            r4.metric("Tracking error", _fmt_pct(benchmark_metrics.get("tracking_error")))
            portfolio_returns = analytics_result.get("portfolio_returns")
            if isinstance(portfolio_returns, pd.Series) and not portfolio_returns.empty:
                growth = (1.0 + portfolio_returns.fillna(0.0)).cumprod()
                st.line_chart(growth.rename("Current-weight historical growth proxy"))
        exposures = analytics_result.get("open_exposures")
        if isinstance(exposures, pd.DataFrame) and not exposures.empty:
            st.markdown("##### Current exposures including cash context")
            st.dataframe(exposures, use_container_width=True, hide_index=True)
        correlation = analytics_result.get("correlation")
        if isinstance(correlation, pd.DataFrame) and not correlation.empty:
            st.markdown("##### Risky-asset correlation")
            st.dataframe(correlation.round(3), use_container_width=True)
        risk_contribution = analytics_result.get("risk_contribution")
        if isinstance(risk_contribution, pd.DataFrame) and not risk_contribution.empty:
            st.markdown("##### Volatility-risk contribution")
            st.dataframe(risk_contribution, use_container_width=True, hide_index=True)
        for warning in analytics_result.get("warnings", []):
            st.warning(str(warning.get("message") or warning) if isinstance(warning, dict) else str(warning))

    with attribution_tab:
        st.caption("Return contributions are additive percentage points versus the initial USD 500,000 capital.")
        ticker_attr = analytics_result.get("ledger_attribution_by_ticker")
        if isinstance(ticker_attr, pd.DataFrame) and not ticker_attr.empty:
            st.dataframe(ticker_attr, use_container_width=True, hide_index=True)
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("##### Attribution by sector")
            sector_attr = analytics_result.get("ledger_attribution_by_sector")
            if isinstance(sector_attr, pd.DataFrame) and not sector_attr.empty:
                st.dataframe(sector_attr, use_container_width=True, hide_index=True)
            else:
                st.caption("Add sectors in Thesis Monitor.")
        with a2:
            st.markdown("##### Attribution by client goal")
            goal_attr = analytics_result.get("ledger_attribution_by_goal")
            if isinstance(goal_attr, pd.DataFrame) and not goal_attr.empty:
                st.dataframe(goal_attr, use_container_width=True, hide_index=True)
            else:
                st.caption("Assign client goals in Thesis Monitor.")

    with stress_tab:
        exposures = analytics_result.get("open_exposures")
        if not isinstance(exposures, pd.DataFrame) or exposures.empty:
            st.info("Open positions are required for exposure stress testing.")
        else:
            stress_frame = exposures[["Ticker", "Sector", "CurrentValue"]].copy()
            stress_frame["Shock %"] = 0.0
            edited_stress = st.data_editor(
                stress_frame,
                hide_index=True,
                use_container_width=True,
                key="live_portfolio_stress_editor",
                disabled=["Ticker", "Sector", "CurrentValue"],
                column_config={"Shock %": st.column_config.NumberColumn("Shock %", min_value=-100.0, max_value=500.0, format="%.1f")},
            )
            stress_result = edited_stress.copy()
            stress_result["ShockPnL"] = stress_result["CurrentValue"] * stress_result["Shock %"] / 100.0
            stressed_pnl = float(stress_result["ShockPnL"].sum())
            stressed_equity = float(analytics_result.get("current_equity") or 0.0) + stressed_pnl
            s1, s2, s3 = st.columns(3)
            s1.metric("Deterministic stress P/L", f"${stressed_pnl:+,.0f}")
            s2.metric("Stressed equity", f"${stressed_equity:,.0f}")
            s3.metric(
                "Portfolio shock",
                f"{stressed_pnl / float(analytics_result.get('current_equity') or 1.0):+.1%}",
            )
            st.dataframe(stress_result, use_container_width=True, hide_index=True)
            st.caption("Shocks are user-authored deterministic assumptions; no probability or forecast is implied.")

    with health_tab:
        health = analytics_result.get("research_health", {})
        summary = health.get("summary", {}) if isinstance(health, dict) else {}
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Fresh price coverage", f"{float(summary.get('fresh_price_coverage_pct') or 0):.0f}%")
        h2.metric("Thesis coverage", f"{float(summary.get('thesis_coverage_pct') or 0):.0f}%")
        h3.metric("Evidence coverage", f"{float(summary.get('evidence_coverage_pct') or 0):.0f}%")
        h4.metric("Review queue", int(summary.get("review_queue_count") or 0))
        if health.get("tickers"):
            st.dataframe(pd.DataFrame(health["tickers"]), use_container_width=True, hide_index=True)
        if health.get("review_queue"):
            st.markdown("##### Actionable data and review gaps")
            st.dataframe(pd.DataFrame(health["review_queue"]), use_container_width=True, hide_index=True)
        st.caption(str(health.get("methodology") or ""))
        st.caption(str(health.get("macro_policy") or ""))


def _render_competition_portfolio(profile: dict[str, str | int]) -> None:
    from src.portfolio_tracker.wharton_competition import INITIAL_CAPITAL_USD, calculate_portfolio_performance

    st.markdown("### Portfolio Tracker — Wharton 2026–2027")
    st.caption("Returns are measured from USD 500,000. Every position records its creator, dates, entry price, and performance.")
    with st.expander("Add a New Position", expanded=False):
        with st.form("competition_add_position", clear_on_submit=True):
            p1, p2, p3 = st.columns(3)
            with p1:
                ticker = st.text_input("Ticker", placeholder="e.g. MSFT")
                security_type = st.selectbox("Type", ["Stock", "ETF", "Bond", "Other"])
            with p2:
                quantity = st.number_input("Quantity", min_value=0.0, value=0.0, step=1.0)
                entry_price = st.number_input("Entry Price per Unit (USD)", min_value=0.0, value=0.0, step=0.01)
            with p3:
                entry_date = st.date_input("Opening Date", value=date.today())
                manual_price = st.number_input("Current Price (optional)", min_value=0.0, value=0.0, step=0.01)
            notes = st.text_area("Notes / Investment Thesis", height=80)
            add_position = st.form_submit_button("Add Position", type="primary", use_container_width=True)
        if add_position:
            clean_ticker = ticker.strip().upper()
            if not clean_ticker or quantity <= 0 or entry_price <= 0:
                st.error("Ticker, quantity, and entry price are required and must be positive.")
            else:
                with get_connection() as conn:
                    conn.execute(
                        "INSERT INTO competition_positions (ticker, security_type, quantity, entry_price, entry_date, opened_by, opened_at, last_price, notes, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')",
                        (clean_ticker, security_type, float(quantity), float(entry_price), entry_date.isoformat(), str(profile["username"]), _now_iso(), float(manual_price) if manual_price > 0 else None, notes.strip()),
                    )
                    conn.commit()
                    if hasattr(conn, "sync"):
                        conn.sync()
                st.success(f"Position {clean_ticker} was opened by {profile['username']}.")
                st.rerun()

    positions = _fetch_competition_positions()
    open_tickers = [str(row["ticker"]).upper() for row in positions if row["status"] == "open"]
    live_prices = _competition_live_prices(open_tickers)
    performance = calculate_portfolio_performance(positions, live_prices)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Portfolio Value", f"${performance['equity']:,.2f}")
    m2.metric("Total Return Since Inception", f"{performance['total_return_pct']:+.2f}%", f"${performance['total_pnl']:+,.2f}")
    m3.metric("Unrealized P/L", f"${performance['unrealized_pnl']:+,.2f}")
    m4.metric("Realized P/L", f"${performance['realized_pnl']:+,.2f}")
    st.caption(
        f"Initial capital: ${INITIAL_CAPITAL_USD:,.0f} · Uninvested cash before P/L: "
        f"${performance['cash_before_pnl']:,.2f} · Live prices: {len(live_prices)}/{len(set(open_tickers))} tickers."
    )
    _render_wins_reconciliation(positions, live_prices)
    if not performance["positions"]:
        st.info("No positions have been entered yet.")
        return

    from src.portfolio_tracker.strategy_store import list_holding_theses

    with get_connection() as conn:
        thesis_records = list_holding_theses(conn)
    thesis_by_ticker = {
        str(item.get("ticker") or "").upper(): item
        for item in thesis_records if item.get("ticker")
    }
    st.dataframe(pd.DataFrame([{
        "Status": "Open" if row["status"] == "open" else "Closed", "Ticker": row["ticker"],
        "Type": row["security_type"], "Quantity": row["quantity"], "Entry Price": f"${row['entry_price']:,.2f}",
        "Current / Exit Price": f"${row['current_price']:,.2f}", "Position Return": f"{row['return_pct']:+.2f}%",
        "P/L": f"${row['pnl']:+,.2f}", "Opened By": row["opened_by"], "Opening Date": row["entry_date"],
        "Price Source": row["price_source"],
        "Client Goal": _strategy_payload(thesis_by_ticker.get(str(row["ticker"]).upper(), {})).get("primary_goal") or "Unassigned",
        "Sector": _strategy_payload(thesis_by_ticker.get(str(row["ticker"]).upper(), {})).get("sector") or "Unassigned",
        "Thesis Status": str(thesis_by_ticker.get(str(row["ticker"]).upper(), {}).get("status") or "Missing").replace("_", " ").title(),
        "Conviction": thesis_by_ticker.get(str(row["ticker"]).upper(), {}).get("conviction") or "—",
        "Next Review": thesis_by_ticker.get(str(row["ticker"]).upper(), {}).get("next_review_at") or "Not scheduled",
    } for row in performance["positions"]]), use_container_width=True, hide_index=True)

    _render_live_competition_analytics(positions, live_prices)

    st.markdown("#### Manage Open Positions")
    for row in performance["positions"]:
        if row["status"] != "open":
            continue
        with st.expander(f"{row['ticker']} · {row['return_pct']:+.2f}% · opened by {row['opened_by']}"):
            st.write(row.get("notes") or "No notes.")
            update_col, close_col = st.columns(2)
            with update_col:
                with st.form(f"competition_update_price_{row['id']}"):
                    new_price = st.number_input("Manual Current Price", min_value=0.0, value=float(row["current_price"]), step=0.01, key=f"competition_price_{row['id']}")
                    if st.form_submit_button("Save Manual Price", use_container_width=True):
                        with get_connection() as conn:
                            conn.execute("UPDATE competition_positions SET last_price = ? WHERE id = ?", (float(new_price), int(row["id"])))
                            conn.commit()
                            if hasattr(conn, "sync"):
                                conn.sync()
                        st.rerun()
            with close_col:
                with st.form(f"competition_close_{row['id']}"):
                    exit_price = st.number_input("Exit Price", min_value=0.01, value=max(float(row["current_price"]), 0.01), step=0.01, key=f"competition_exit_price_{row['id']}")
                    exit_date = st.date_input("Closing Date", value=date.today(), key=f"competition_exit_date_{row['id']}")
                    if st.form_submit_button("Close Position", type="primary", use_container_width=True):
                        with get_connection() as conn:
                            conn.execute("UPDATE competition_positions SET status = 'closed', exit_price = ?, exit_date = ?, closed_by = ? WHERE id = ?", (float(exit_price), exit_date.isoformat(), str(profile["username"]), int(row["id"])))
                            conn.commit()
                            if hasattr(conn, "sync"):
                                conn.sync()
                        st.rerun()
            if st.button("Delete Incorrectly Entered Position", key=f"competition_delete_{row['id']}"):
                with get_connection() as conn:
                    conn.execute("DELETE FROM competition_positions WHERE id = ?", (int(row["id"]),))
                    conn.commit()
                    if hasattr(conn, "sync"):
                        conn.sync()
                st.rerun()


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_company_analysis_cached(ticker: str) -> dict[str, Any]:
    from src.analytics.company_analysis import fetch_company_data

    return fetch_company_data(ticker)


def _load_company_analysis_module(*required_names: str, minimum_macro_schema: int = 0) -> Any:
    """Reload the analytics module when Streamlit retained a pre-feature version."""
    module = importlib.import_module("src.analytics.company_analysis")
    if (
        any(not hasattr(module, name) for name in required_names)
        or int(getattr(module, "MACRO_DATA_SCHEMA_VERSION", 0)) < minimum_macro_schema
    ):
        module = importlib.reload(module)
    missing = [name for name in required_names if not hasattr(module, name)]
    if missing:
        raise ImportError(f"Company-analysis module is missing: {', '.join(missing)}")
    return module


@st.cache_data(ttl=21600, show_spinner=False)
def _fetch_macro_snapshot_cached(economy_code: str) -> dict[str, Any]:
    module = _load_company_analysis_module("fetch_macro_snapshot", minimum_macro_schema=5)
    from src.analytics.macro_snapshot_store import load_macro_snapshot, upsert_macro_snapshot

    code = str(economy_code or "").upper().strip()
    reference_year = int(module.MACRO_REFERENCE_YEAR)
    schema_version = int(module.MACRO_DATA_SCHEMA_VERSION)
    backend = _macro_database_backend()
    storage_warning = ""
    try:
        with get_connection() as conn:
            cached = load_macro_snapshot(conn, code, reference_year, schema_version)
        if cached is not None:
            cached.setdefault("cache_info", {})["backend"] = backend
            return cached
    except Exception as exc:
        storage_warning = f"Database cache read unavailable ({type(exc).__name__})."

    snapshot = module.fetch_macro_snapshot(code)
    cache_info = {
        "origin": "live",
        "backend": backend,
        "schema_version": schema_version,
        "persisted": False,
    }
    if snapshot.get("available"):
        try:
            with get_connection() as conn:
                cache_info["persisted"] = upsert_macro_snapshot(conn, snapshot, schema_version)
        except Exception as exc:
            storage_warning = f"Database cache write unavailable ({type(exc).__name__})."
    if storage_warning:
        cache_info["warning"] = storage_warning
    snapshot["cache_info"] = cache_info
    return snapshot


def _macro_database_backend() -> str:
    """Return the configured persistence mode without exposing credentials."""
    database_url = os.environ.get("TURSO_DATABASE_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN")
    try:
        database_url = st.secrets.get("TURSO_DATABASE_URL") or database_url
        auth_token = st.secrets.get("TURSO_AUTH_TOKEN") or auth_token
    except Exception:
        pass
    return "turso" if database_url and auth_token else "sqlite"


def _invalidate_macro_snapshot(economy_code: str) -> bool:
    module = _load_company_analysis_module(minimum_macro_schema=5)
    from src.analytics.macro_snapshot_store import delete_macro_snapshot

    try:
        with get_connection() as conn:
            delete_macro_snapshot(
                conn,
                economy_code,
                int(module.MACRO_REFERENCE_YEAR),
                int(module.MACRO_DATA_SCHEMA_VERSION),
            )
        return True
    except Exception:
        return False


@st.cache_data(ttl=604800, show_spinner=False)
def _fetch_management_biography_cached(name: str, company_name: str) -> dict[str, Any]:
    from src.analytics.company_analysis import fetch_management_biography

    return fetch_management_biography(name, company_name)


def _company_news_rows(news: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, item in enumerate(news, start=1):
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        canonical = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
        click_through = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}
        provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
        title = str(item.get("title") or content.get("title") or "Untitled")
        url = str(item.get("link") or canonical.get("url") or click_through.get("url", ""))
        publisher = str(item.get("publisher") or provider.get("displayName") or "Yahoo Finance")
        rows.append({"id": f"N{index}", "title": title, "url": url, "publisher": publisher})
    return rows


def _format_company_metric(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—"
        if any(token in key.lower() for token in ("margin", "growth", "yield", "returnon", "heldpercent")):
            return f"{value:.2%}"
        if abs(value) >= 1e9:
            return f"{value / 1e9:,.2f}B"
        if abs(value) >= 1e6:
            return f"{value / 1e6:,.2f}M"
        return f"{value:,.4g}"
    if isinstance(value, int) and abs(value) >= 1_000_000:
        return f"{value / 1e6:,.2f}M"
    return str(value)


def _render_statement_block(label: str, frame: pd.DataFrame) -> None:
    from src.analytics.company_analysis import format_statement

    st.markdown(f"##### {label}")
    display = format_statement(frame)
    if display.empty:
        st.info("Data are not available for this ticker.")
    else:
        st.caption("Values are in millions of the reporting currency; the newest period is shown first.")
        st.dataframe(display, use_container_width=True)


def _render_geographic_revenue_breakdown(snapshot: dict[str, Any], ticker: str, company_name: str) -> None:
    from src.analytics.company_analysis import analyze_geographic_revenue

    st.markdown("#### Revenue by Region")
    st.caption(
        "Automatic data come from the most recent SEC annual filing available through the configured sources "
        "and are used only when a non-overlapping Inline XBRL geographic table can be verified."
    )
    geographic = snapshot.get("geographic_revenue", {})
    analysis = geographic.get("analysis", {}) if isinstance(geographic, dict) else {}

    if not geographic.get("available"):
        st.warning(geographic.get("error", "A reliable geographic revenue breakdown was not found."))
        st.info(
            "Yahoo Finance does not normally publish revenue by region. You can enter values from the company's "
            "annual report below; QuantSim will analyze them without inventing missing regions."
        )
        manual_key = f"company_region_manual_{ticker}"
        starter = pd.DataFrame(
            [
                {"Region": "", "Revenue": None},
                {"Region": "", "Revenue": None},
            ]
        )
        manual = st.data_editor(
            starter,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=manual_key,
            column_config={
                "Region": st.column_config.TextColumn("Region", help="Use the exact label from the annual report."),
                "Revenue": st.column_config.NumberColumn("Revenue", min_value=0.0, format="%.2f"),
            },
        )
        manual_rows = [
            {"region": row.get("Region"), "revenue": row.get("Revenue")}
            for row in manual.to_dict("records")
        ]
        manual_analysis = analyze_geographic_revenue(manual_rows)
        if manual_analysis.get("available"):
            analysis = manual_analysis
            geographic = {
                "available": True,
                "rows": [{"region": row["region"], "revenue": row["revenue"]} for row in analysis["rows"]],
                "currency": "entered units",
                "source_name": "Manual annual-report input",
                "source_url": "",
                "report_date": "",
                "analysis": analysis,
            }
        else:
            st.caption("Enter at least two regions with positive revenue to generate the chart and rating.")
            return

    if not analysis.get("available"):
        analysis = analyze_geographic_revenue(geographic.get("rows", []))
    if not analysis.get("available"):
        st.warning(analysis.get("error", "Regional analysis is unavailable."))
        return

    score_cols = st.columns(5)
    score_cols[0].metric("Diversification rating", f"{analysis['score']}/{analysis['max_score']}")
    score_cols[1].metric("Assessment", analysis["label"])
    score_cols[2].metric("Largest region", str(analysis["top_region"]))
    score_cols[3].metric("Largest share", f"{analysis['top_region_share']:.1%}")
    score_cols[4].metric("Effective regions", f"{analysis['effective_regions']:.1f}")
    st.caption(analysis["warning"])

    rows = analysis["rows"]
    chart_col, detail_col = st.columns([0.56, 0.44], gap="large")
    with chart_col:
        import plotly.graph_objects as go

        figure = go.Figure(
            go.Pie(
                labels=[row["region"] for row in rows],
                values=[row["revenue"] for row in rows],
                hole=0.38,
                sort=False,
                textinfo="label+percent",
                hovertemplate="%{label}<br>Revenue: %{value:,.2f}<br>Share: %{percent}<extra></extra>",
            )
        )
        figure.update_layout(
            title=f"{company_name}: Revenue by Region",
            height=460,
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h", yanchor="top", y=-0.08),
        )
        st.plotly_chart(figure, use_container_width=True)
    with detail_col:
        st.markdown("##### Regional interpretation")
        st.write(analysis["interpretation"])
        for strength in analysis.get("strengths", []):
            st.success(strength)
        for risk in analysis.get("risks", []):
            st.warning(risk)

    currency = str(geographic.get("currency") or "reporting currency")
    table = pd.DataFrame([
        {
            "Region": row["region"],
            f"Revenue ({currency})": row["revenue"],
            "Revenue share": row["share"],
            "Strategic importance": row["strategic_importance"],
        }
        for row in rows
    ])
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Revenue share": st.column_config.ProgressColumn(
                "Revenue share", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )
    source_url = str(geographic.get("source_url") or "")
    source_name = str(geographic.get("source_name") or "Annual filing")
    report_date = str(geographic.get("report_date") or geographic.get("fiscal_end") or "")
    if source_url:
        st.markdown(f"Source: [{source_name}]({source_url}) · reporting period ended {report_date or 'not specified'}")
    else:
        st.caption(f"Source: {source_name}. Verify manually entered values against the cited annual report.")
    if geographic.get("coverage_ratio") is not None:
        st.caption(
            f"Named-region disclosure coverage before any residual bucket: {float(geographic['coverage_ratio']):.1%}. "
            f"XBRL concept: {geographic.get('concept', '—')}."
        )


def _render_macro_region_drilldown(snapshot: dict[str, Any], ticker: str) -> None:
    module = _load_company_analysis_module(
        "MACRO_ECONOMIES",
        "analyze_macro_snapshot",
        "infer_macro_economy",
        minimum_macro_schema=5,
    )
    macro_economies = module.MACRO_ECONOMIES

    st.markdown("#### Regional Macro Drill-down · 2024")
    st.caption(
        "All score inputs use the same 2024 reference year. Connect a disclosed revenue region to an editable "
        "economy proxy, then inspect inflation, government debt, interest rates, growth, unemployment, and the current account."
    )

    geographic = snapshot.get("geographic_revenue", {})
    geographic_analysis = geographic.get("analysis", {}) if isinstance(geographic, dict) else {}
    region_rows = geographic_analysis.get("rows", []) if isinstance(geographic_analysis, dict) else []
    region_names = [
        str(row.get("region"))
        for row in region_rows
        if isinstance(row, dict)
        and row.get("region")
        and str(row.get("region")).lower() != "not separately disclosed"
    ]
    if not region_names:
        region_names = ["Company-wide / choose proxy manually"]

    control_cols = st.columns(2)
    with control_cols[0]:
        selected_region = st.selectbox(
            "Revenue region",
            region_names,
            key=f"company_macro_region_{ticker}",
            help="This is the filing bucket whose macro backdrop you want to inspect.",
        )

    inferred_code = module.infer_macro_economy(selected_region)
    economy_names = list(macro_economies)
    inferred_name = next(
        (name for name, code in macro_economies.items() if code == inferred_code),
        "World",
    )
    regional_codes = set(getattr(module, "WORLD_BANK_REGIONAL_CODES", set()))
    exact_country_names = {
        name.lower(): name
        for name, code in macro_economies.items()
        if code not in regional_codes
    }
    exact_country_match = exact_country_names.get(selected_region.strip().lower())
    with control_cols[1]:
        if exact_country_match:
            selected_economy = st.selectbox(
                "Macro economy used",
                [exact_country_match],
                disabled=True,
                key=f"company_macro_economy_v5_{ticker}_{selected_region}",
                help="An exact country disclosure is automatically tied to that country's macro data.",
            )
        else:
            selected_economy = st.selectbox(
                "Macro economy proxy",
                economy_names,
                index=economy_names.index(inferred_name),
                key=f"company_macro_economy_v5_{ticker}_{selected_region}",
                help="Change the proxy whenever the company's disclosure bucket does not match a World Bank economy.",
            )
    economy_code = macro_economies[selected_economy]
    st.info(
        f"Proxy used: **{selected_region} → {selected_economy} ({economy_code})**. "
        "Revenue filing buckets and macro datasets rarely align exactly, so this mapping is an analytical proxy, "
        "not a claim about the company's exact country mix."
    )

    state_key = f"company_macro_snapshot_v5_{ticker}_{economy_code}"
    macro_snapshot = st.session_state.get(state_key)
    if not isinstance(macro_snapshot, dict):
        with st.spinner(f"Loading World Bank indicators for {selected_economy}…"):
            macro_snapshot = _fetch_macro_snapshot_cached(economy_code)
            st.session_state[state_key] = macro_snapshot

    if not macro_snapshot.get("available"):
        st.warning(macro_snapshot.get("error", "Macro indicators are not available for this economy proxy."))
        if st.button(
            "Retry macro data",
            key=f"company_macro_retry_{ticker}_{economy_code}",
            use_container_width=True,
        ):
            _invalidate_macro_snapshot(economy_code)
            _fetch_macro_snapshot_cached.clear()
            st.session_state.pop(state_key, None)
            st.rerun()
        return

    reference_year = int(macro_snapshot.get("reference_year") or 2024)
    cache_info = macro_snapshot.get("cache_info", {})
    if isinstance(cache_info, dict) and cache_info:
        backend_label = "shared Turso database" if cache_info.get("backend") == "turso" else "local SQLite cache"
        if cache_info.get("origin") == "database":
            st.caption(f"Macro snapshot loaded from the {backend_label}; expires {cache_info.get('expires_at', 'automatically')}.")
        elif cache_info.get("persisted"):
            st.caption(f"Fresh macro snapshot saved to the {backend_label} for reuse by other app sessions.")
        if cache_info.get("warning"):
            st.caption(str(cache_info["warning"]))
    rate_codes = ("FR.INR.RINR", "FR.INR.LEND")
    snapshot_indicators = macro_snapshot.get("indicators", {})
    missing_rate_codes = [
        code
        for code in rate_codes
        if not isinstance(snapshot_indicators.get(code), dict)
        or snapshot_indicators[code].get("latest_value") is None
    ]
    if missing_rate_codes:
        rate_proxy_names = [
            name for name, code in macro_economies.items() if code not in regional_codes
        ]
        selected_rate_proxy = st.selectbox(
            "Interest-rate country proxy",
            rate_proxy_names,
            index=rate_proxy_names.index("United States") if "United States" in rate_proxy_names else 0,
            key=f"company_macro_rate_proxy_v5_{ticker}_{economy_code}",
            help=(
                "Broad regions do not have a comparable lending or real-interest-rate aggregate. "
                "Choose a representative country; proxy rates are labeled separately and used transparently."
            ),
        )
        rate_proxy_code = macro_economies[selected_rate_proxy]
        with st.spinner(f"Loading interest-rate proxy for {selected_rate_proxy}…"):
            rate_snapshot = _fetch_macro_snapshot_cached(rate_proxy_code)
        proxy_indicators = rate_snapshot.get("indicators", {}) if isinstance(rate_snapshot, dict) else {}
        working_snapshot = deepcopy(macro_snapshot)
        working_indicators = working_snapshot.get("indicators", {})
        applied_rate_labels: list[str] = []
        for rate_code in missing_rate_codes:
            proxy_item = proxy_indicators.get(rate_code, {}) if isinstance(proxy_indicators, dict) else {}
            if not isinstance(proxy_item, dict) or proxy_item.get("latest_value") is None:
                continue
            replacement = deepcopy(proxy_item)
            base_label = str(replacement.get("label") or rate_code)
            replacement["label"] = f"{base_label} ({selected_rate_proxy} proxy)"
            replacement["definition"] = f"{base_label}; {selected_rate_proxy} country proxy"
            replacement["source_name"] = (
                f"{replacement.get('source_name') or 'World Bank World Development Indicators'} "
                f"· {selected_rate_proxy} rate proxy"
            )
            replacement["is_proxy"] = True
            working_indicators[rate_code] = replacement
            applied_rate_labels.append(base_label)
        if applied_rate_labels:
            macro_snapshot = working_snapshot
            st.info(
                f"Regional rate aggregates were unavailable. {', '.join(applied_rate_labels)} now use the "
                f"editable **{selected_rate_proxy}** country proxy; the macro economy itself remains "
                f"**{selected_economy}**."
            )

    analysis = module.analyze_macro_snapshot(macro_snapshot)
    if not analysis.get("available"):
        st.warning("There are not enough recent observations to calculate the macro score.")
        return

    score_cols = st.columns(4)
    score_cols[0].metric(f"Macro resilience ({reference_year})", f"{analysis['score']:.0f}/100")
    score_cols[1].metric("Assessment", analysis["label"])
    score_cols[2].metric("Scoring coverage", f"{analysis['data_coverage']:.0%}")
    score_cols[3].metric("Macro economy used", str(macro_snapshot.get("economy_name") or selected_economy))
    st.caption(analysis["warning"])

    components = analysis.get("components", [])
    metric_columns = st.columns(3)
    for index, component in enumerate(components):
        value = component.get("value")
        year = component.get("year")
        display_value = f"{float(value):.1f}%" if value is not None else "Not published"
        metric_columns[index % 3].metric(
            str(component.get("label")),
            display_value,
            delta=f"Observation: {year}" if year else f"No {reference_year} observation",
            delta_color="off",
        )

    interpretation_cols = st.columns(2)
    with interpretation_cols[0]:
        st.markdown("##### Supportive signals")
        strengths = analysis.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.success(strength)
        else:
            st.caption("No indicator currently meets the score's supportive threshold.")
    with interpretation_cols[1]:
        st.markdown("##### Watch items")
        risks = analysis.get("risks", [])
        if risks:
            for risk in risks:
                st.warning(risk)
        else:
            st.caption("No indicator currently breaches the score's watch threshold.")

    latest_table = pd.DataFrame([
        {
            "Indicator": component.get("label"),
            "Latest value": component.get("value"),
            "Observation year": component.get("year"),
            "Score weight": float(component.get("weight", 0.0)) / 100.0,
            "Component score": component.get("component_score"),
            "Source": component.get("source_name") or "World Bank World Development Indicators",
        }
        for component in components
    ])
    st.markdown("##### Latest observations and score inputs")
    st.dataframe(
        latest_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Latest value": st.column_config.NumberColumn("Latest value (%)", format="%.2f"),
            "Score weight": st.column_config.ProgressColumn(
                "Score weight", min_value=0.0, max_value=1.0, format="percent"
            ),
            "Component score": st.column_config.NumberColumn("Component score", format="%.0f"),
        },
    )

    st.markdown("##### Ten-year macro trends")
    chart_groups = [
        ("Inflation and interest rates", ["FP.CPI.TOTL.ZG", "FR.INR.RINR", "FR.INR.LEND"]),
        ("Growth and labor market", ["NY.GDP.MKTP.KD.ZG", "SL.UEM.TOTL.ZS"]),
        ("Debt and external balance", ["GC.DOD.TOTL.GD.ZS", "BN.CAB.XOKA.GD.ZS"]),
    ]
    indicators = macro_snapshot.get("indicators", {})
    import plotly.graph_objects as go

    for title, indicator_codes in chart_groups:
        figure = go.Figure()
        for indicator_code in indicator_codes:
            indicator = indicators.get(indicator_code, {}) if isinstance(indicators, dict) else {}
            series = indicator.get("series", []) if isinstance(indicator, dict) else []
            series = [point for point in series if int(point.get("year", 0)) <= reference_year]
            if not series:
                continue
            figure.add_trace(go.Scatter(
                x=[point["year"] for point in series],
                y=[point["value"] for point in series],
                name=str(indicator.get("label") or indicator_code),
                mode="lines+markers",
                hovertemplate="%{x}: %{y:.2f}%<extra>%{fullData.name}</extra>",
            ))
        if figure.data:
            figure.update_layout(
                title=title,
                height=340,
                margin=dict(l=10, r=10, t=55, b=20),
                yaxis_title="Percent",
                legend=dict(orientation="h", yanchor="top", y=-0.18),
            )
            st.plotly_chart(figure, use_container_width=True)

    source_url = str(macro_snapshot.get("source_url") or "")
    fetched_at = str(macro_snapshot.get("fetched_at") or "")
    if source_url:
        st.markdown(f"Source: [World Bank World Development Indicators]({source_url}) · fetched {fetched_at}")
    debt_component = next(
        (component for component in components if component.get("indicator_code") == "GC.DOD.TOTL.GD.ZS"),
        {},
    )
    if debt_component.get("is_fallback") and debt_component.get("source_url"):
        st.markdown(
            f"Debt fallback: [{debt_component.get('source_name')}]({debt_component.get('source_url')}) · "
            "general government gross debt, historical year only."
        )
    debt_definition = (
        "Debt is IMF WEO general-government gross debt because the World Bank series was unavailable."
        if debt_component.get("is_fallback")
        else "Debt is the World Bank central-government-debt series, not a complete general-government measure."
    )
    st.caption(
        f"{debt_definition} Rates shown are annual real and lending rates, not a live central-bank policy rate. "
        "Observation years may differ."
    )


def _render_geographic_revenue(snapshot: dict[str, Any], ticker: str, company_name: str) -> None:
    selected_view = st.radio(
        "Regional analysis view",
        ["Revenue Exposure", "Macro Drill-down"],
        horizontal=True,
        key=f"company_region_view_{ticker}",
    )
    if selected_view == "Revenue Exposure":
        _render_geographic_revenue_breakdown(snapshot, ticker, company_name)
    else:
        _render_macro_region_drilldown(snapshot, ticker)


def _render_industry_peer_analysis(
    profile: dict[str, str | int],
    selected_ticker: str,
    valid_results: dict[str, dict[str, Any]],
) -> None:
    from src.analytics.industry_analysis import (
        PORTER_FORCES,
        build_industry_analysis,
    )
    from src.portfolio_tracker.strategy_store import load_company_research, save_company_research

    with get_connection() as conn:
        research_record = load_company_research(conn, selected_ticker)
    current = _strategy_payload(research_record)
    peer_options = [ticker for ticker in valid_results if ticker != selected_ticker]
    saved_peers = [
        str(ticker).upper() for ticker in current.get("peer_tickers", [])
        if str(ticker).upper() in peer_options
    ]
    default_peers = saved_peers if "peer_tickers" in current else peer_options

    st.markdown("#### Industry Structure & Peer Comparison")
    st.caption(
        "Quantitative comparisons use only the loaded companies. Porter Five Forces and SWOT use only analyst-entered "
        "ratings and evidence; missing qualitative judgments remain missing."
    )
    selected_peers = st.multiselect(
        "Peer set from loaded companies",
        peer_options,
        default=default_peers,
        key=f"industry_peer_set_{selected_ticker}",
        help="Load additional tickers in Company Analysis to expand the peer set.",
    )

    porter_current = current.get("porter", {}) if isinstance(current.get("porter"), dict) else {}
    porter_frame = pd.DataFrame([{
        "Force key": key,
        "Competitive force": label,
        "Pressure (1–5)": _finite_form_number(
            porter_current.get(key, {}).get("rating") or 0.0
            if isinstance(porter_current.get(key), dict) else 0.0,
            0.0,
        ),
        "Evidence / rationale": "\n".join(porter_current.get(key, {}).get("evidence", []))
        if isinstance(porter_current.get(key), dict) else "",
    } for key, label, _ in PORTER_FORCES])
    swot_current = current.get("swot", {}) if isinstance(current.get("swot"), dict) else {}
    with st.form(f"industry_research_form_{selected_ticker}"):
        industry_thesis = st.text_area(
            "Industry thesis",
            value=str(current.get("industry_thesis") or ""),
            height=90,
            help="State the structural reason this industry should or should not create attractive long-term economics.",
        )
        st.markdown("##### Porter Five Forces — analyst assessment")
        edited_porter = st.data_editor(
            porter_frame,
            hide_index=True,
            use_container_width=True,
            disabled=["Force key", "Competitive force"],
            key=f"porter_editor_{selected_ticker}",
            column_config={
                "Force key": None,
                "Pressure (1–5)": st.column_config.NumberColumn(
                    "Pressure (1–5)", min_value=0.0, max_value=5.0, step=1.0,
                    help="0 means not assessed; 1 is low competitive pressure and 5 is high pressure.",
                ),
            },
        )
        st.markdown("##### SWOT — analyst evidence")
        s1, s2 = st.columns(2)
        with s1:
            strengths = st.text_area("Strengths (one per line)", value="\n".join(swot_current.get("strengths", [])), height=100)
            opportunities = st.text_area("Opportunities (one per line)", value="\n".join(swot_current.get("opportunities", [])), height=100)
        with s2:
            weaknesses = st.text_area("Weaknesses (one per line)", value="\n".join(swot_current.get("weaknesses", [])), height=100)
            threats = st.text_area("Threats (one per line)", value="\n".join(swot_current.get("threats", [])), height=100)
        evidence_notes = st.text_area(
            "Industry evidence and source notes",
            value=str(current.get("evidence_notes") or ""),
            height=80,
        )
        save_research = st.form_submit_button("Save Industry Research", type="primary", use_container_width=True)
    if save_research:
        porter_payload: dict[str, Any] = {}
        for row in edited_porter.to_dict("records"):
            rating = _finite_form_number(row.get("Pressure (1–5)"), 0.0)
            if rating <= 0:
                continue
            evidence = [
                item.strip() for item in str(row.get("Evidence / rationale") or "").splitlines()
                if item.strip()
            ]
            porter_payload[str(row.get("Force key"))] = {"rating": rating, "evidence": evidence}
        swot_payload = {
            "strengths": [item.strip() for item in strengths.splitlines() if item.strip()],
            "weaknesses": [item.strip() for item in weaknesses.splitlines() if item.strip()],
            "opportunities": [item.strip() for item in opportunities.splitlines() if item.strip()],
            "threats": [item.strip() for item in threats.splitlines() if item.strip()],
        }
        with get_connection() as conn:
            save_company_research(
                conn,
                selected_ticker,
                {
                    "peer_tickers": selected_peers,
                    "industry_thesis": industry_thesis.strip(),
                    "porter": porter_payload,
                    "swot": swot_payload,
                    "evidence_notes": evidence_notes.strip(),
                },
                updated_by=str(profile["username"]),
            )
        st.success("Industry and peer research saved to the shared database.")
        st.rerun()

    company_info = valid_results[selected_ticker].get("info", {})
    peer_metrics = {
        ticker: valid_results[ticker].get("info", {})
        for ticker in selected_peers if ticker in valid_results
    }
    analysis = build_industry_analysis(
        company_info,
        peer_metrics,
        porter_assessments=porter_current,
        swot_assessments=swot_current,
        company_name=selected_ticker,
        min_peer_count=2,
    )
    peers = analysis.get("peer_comparison", {})
    if peers.get("available"):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Peer-relative score", f"{float(peers.get('score') or 0):.0f}/100")
        p2.metric("Assessment", str(peers.get("rating") or "Not rated"))
        p3.metric("Data coverage", f"{float(peers.get('coverage_pct') or 0):.0f}%")
        p4.metric("Confidence", str(peers.get("confidence") or "Low"))
        category_rows = [
            {
                "Category": item.get("label"),
                "Score": item.get("score"),
                "Raw score": item.get("raw_score"),
                "Coverage %": item.get("coverage_pct"),
                "Metrics analyzed": item.get("metrics_analyzed"),
            }
            for item in peers.get("categories", {}).values() if isinstance(item, dict)
        ]
        st.dataframe(pd.DataFrame(category_rows), use_container_width=True, hide_index=True)
        metric_rows = [
            {
                "Category": str(row.get("category") or "").replace("_", " ").title(),
                "Metric": row.get("label"),
                "Company": row.get("company_value"),
                "Peer median": row.get("peer_median"),
                "Peer Q1": row.get("peer_q1"),
                "Peer Q3": row.get("peer_q3"),
                "Relative difference %": row.get("relative_difference_pct"),
                "Direction-adjusted percentile": row.get("desirability_percentile"),
                "Signal": str(row.get("relative_flag") or "").replace("_", " ").title(),
                "Peer observations": row.get("peer_count"),
            }
            for row in peers.get("metrics", []) if isinstance(row, dict)
        ]
        st.markdown("##### Metric-by-metric peer evidence")
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True, height=470)
    else:
        st.warning(
            "A robust peer score needs at least two loaded peer companies with comparable metrics. "
            "The app does not manufacture a peer set."
        )

    scatter_rows = []
    for ticker in [selected_ticker, *selected_peers]:
        info = valid_results.get(ticker, {}).get("info", {})
        forward_pe = info.get("forwardPE")
        growth = info.get("revenueGrowth")
        if forward_pe is None or growth is None:
            continue
        try:
            scatter_rows.append({
                "Ticker": ticker,
                "Forward P/E": float(forward_pe),
                "Revenue growth": float(growth),
                "Operating margin": float(info.get("operatingMargins") or 0.0),
                "Selected company": ticker == selected_ticker,
            })
        except (TypeError, ValueError):
            continue
    if scatter_rows:
        import plotly.express as px

        scatter_df = pd.DataFrame(scatter_rows)
        figure = px.scatter(
            scatter_df,
            x="Forward P/E",
            y="Revenue growth",
            text="Ticker",
            color="Selected company",
            size=scatter_df["Operating margin"].abs().clip(lower=0.01),
            title="Peer valuation versus growth",
        )
        figure.update_traces(textposition="top center")
        figure.update_layout(height=410, yaxis_tickformat=".1%")
        st.plotly_chart(figure, use_container_width=True)

    porter = analysis.get("porter_five_forces", {})
    swot = analysis.get("swot", {})
    st.markdown("##### Industry structure diagnostics")
    q1, q2, q3 = st.columns(3)
    q1.metric("Porter coverage", f"{float(porter.get('coverage_pct') or 0):.0f}%")
    q2.metric(
        "Industry attractiveness",
        f"{float(porter.get('industry_attractiveness_score')):.0f}/100"
        if porter.get("industry_attractiveness_score") is not None else "Not assessed",
    )
    q3.metric("SWOT coverage", f"{float(swot.get('coverage_pct') or 0):.0f}%")
    if porter.get("forces"):
        st.dataframe(pd.DataFrame([{
            "Force": row.get("label"),
            "Pressure": row.get("rating_label"),
            "Rating": row.get("rating"),
            "Evidence": "; ".join(row.get("evidence", [])),
        } for row in porter["forces"]]), use_container_width=True, hide_index=True)
    if swot.get("available"):
        swot_cols = st.columns(4)
        for column, key in zip(swot_cols, ("strengths", "weaknesses", "opportunities", "threats")):
            quadrant = swot.get("quadrants", {}).get(key, {})
            with column:
                st.markdown(f"**{quadrant.get('label', key.title())}**")
                items = quadrant.get("items", [])
                if items:
                    for item in items:
                        st.write(f"• {item}")
                else:
                    st.caption("Not assessed")
    if current.get("industry_thesis"):
        st.markdown("##### Saved industry thesis")
        st.write(current["industry_thesis"])
    if research_record:
        st.caption(
            f"Manual research last updated {research_record.get('updated_at', '—')} "
            f"by {research_record.get('updated_by') or 'unknown analyst'}."
        )


def _render_research_evidence(
    profile: dict[str, str | int],
    ticker: str,
    snapshot: dict[str, Any],
) -> None:
    from src.portfolio_tracker.governance_store import (
        RESEARCH_SOURCE_TYPES,
        delete_research_source,
        list_research_sources,
        upsert_research_source,
    )

    with get_connection() as conn:
        sources = list_research_sources(conn, ticker=ticker, include_global=True)
    source_labels = {
        "Annual report": "annual_report",
        "Quarterly report": "quarterly_report",
        "Regulatory filing": "regulatory_filing",
        "Company release": "company_release",
        "Earnings call": "earnings_call",
        "Official data": "official_data",
        "News": "news",
        "Analyst research": "analyst_research",
        "Academic research": "academic",
        "Website": "website",
        "Other": "other",
    }
    source_labels = {label: value for label, value in source_labels.items() if value in RESEARCH_SOURCE_TYPES}

    st.markdown("#### Research Evidence Registry")
    st.caption(
        "Store provenance separately from conclusions: who published the evidence, what period it covers, when the "
        "team accessed and verified it, and which claim it supports."
    )
    auto1, auto2, auto3 = st.columns(3)
    auto1.metric("Market-data provider", "Yahoo Finance")
    auto2.metric("Company snapshot fetched", str(snapshot.get("fetched_at") or "Unknown"))
    geographic = snapshot.get("geographic_revenue", {}) if isinstance(snapshot.get("geographic_revenue"), dict) else {}
    auto3.metric("Geographic source", str(geographic.get("source_name") or "Not disclosed"))

    with st.form(f"research_source_form_{ticker}"):
        s1, s2, s3 = st.columns(3)
        with s1:
            source_title = st.text_input("Source title")
            publisher = st.text_input("Publisher / issuer")
            source_type_label = st.selectbox("Source type", list(source_labels))
        with s2:
            source_url = st.text_input("Source URL")
            primary_source = st.checkbox("Primary source")
            verified = st.checkbox("Verified by current analyst")
        with s3:
            published_known = st.checkbox("Publication date known")
            published_at = st.date_input("Published", value=date.today(), disabled=not published_known)
            period_known = st.checkbox("Covered period end known")
            period_end = st.date_input("Period end", value=date.today(), disabled=not period_known)
        claim_supported = st.text_area("Claim or analytical input supported by this source", height=75)
        source_notes = st.text_area("Verification notes / limitations", height=75)
        save_source = st.form_submit_button("Add Evidence Source", type="primary", use_container_width=True)
    if save_source:
        try:
            with get_connection() as conn:
                upsert_research_source(
                    conn,
                    source_title,
                    ticker=ticker,
                    publisher=publisher,
                    url=source_url,
                    source_type=source_labels[source_type_label],
                    primary_source=bool(primary_source),
                    published_at=published_at.isoformat() if published_known else None,
                    period_end=period_end.isoformat() if period_known else None,
                    accessed_at=date.today().isoformat(),
                    verified_by=str(profile["username"]) if verified else "",
                    verified_at=_now_iso() if verified else None,
                    notes=source_notes.strip(),
                    payload={"claim_supported": claim_supported.strip()},
                    updated_by=str(profile["username"]),
                )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success("Evidence source saved to the shared registry.")
            st.rerun()

    if not sources:
        st.warning("No manual evidence sources are registered for this company.")
        return

    threshold_by_type = {
        "news": 7,
        "quarterly_report": 150,
        "annual_report": 450,
    }
    evidence_rows = []
    today = date.today()
    for item in sources:
        observed_date = None
        for candidate in (item.get("period_end"), item.get("published_at"), item.get("accessed_at")):
            try:
                observed_date = date.fromisoformat(str(candidate or "")[:10])
                break
            except ValueError:
                continue
        age_days = (today - observed_date).days if observed_date else None
        threshold = threshold_by_type.get(str(item.get("source_type") or ""), 180)
        freshness = (
            "Missing date" if age_days is None
            else "Future date" if age_days < 0
            else "Fresh" if age_days <= threshold
            else "Review due"
        )
        payload = _strategy_payload(item)
        evidence_rows.append({
            "ID": item.get("id"),
            "Scope": item.get("ticker") or "Global",
            "Title": item.get("title"),
            "Publisher": item.get("publisher") or "—",
            "Type": str(item.get("source_type") or "other").replace("_", " ").title(),
            "Primary": bool(item.get("primary_source")),
            "Published": item.get("published_at") or "Unknown",
            "Period end": item.get("period_end") or "Unknown",
            "Accessed": item.get("accessed_at") or "Unknown",
            "Verified by": item.get("verified_by") or "Unverified",
            "Age (days)": age_days,
            "Review threshold": threshold,
            "Freshness": freshness,
            "Claim supported": payload.get("claim_supported") or "",
            "URL": item.get("url") or "",
        })
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Registered sources", len(sources))
    e2.metric("Primary sources", sum(bool(item.get("primary_source")) for item in sources))
    e3.metric("Verified sources", sum(bool(item.get("verified_at") and item.get("verified_by")) for item in sources))
    e4.metric("Review due", sum(row["Freshness"] in {"Review due", "Missing date", "Future date"} for row in evidence_rows))
    st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Freshness flags are factual review reminders: news 7 days, quarterly evidence 150 days, annual evidence "
        "450 days, and other manual evidence 180 days. They do not change an investment score."
    )

    source_by_label = {
        f"#{item['id']} · {item.get('title') or 'Untitled'}": item for item in sources
        if item.get("ticker") == ticker
    }
    if source_by_label:
        selected_label = st.selectbox("Remove company-specific source", list(source_by_label), key=f"source_delete_{ticker}")
        selected = source_by_label[selected_label]
        if st.button("Delete Selected Evidence Source", key=f"source_delete_button_{ticker}_{selected['id']}"):
            with get_connection() as conn:
                delete_research_source(conn, int(selected["id"]), updated_by=str(profile["username"]))
            st.rerun()


def _render_company_analysis(profile: dict[str, str | int]) -> None:
    from src.analytics.company_analysis import (
        analyze_moat,
        analyze_track_record,
        build_dcf_scenarios,
        default_dcf_assumptions,
    )

    st.markdown("### Company Analysis")
    st.caption(
        "Company profile, all available metrics, financial statements, management, operating track record, "
        "moat screening, risks, news, and a configurable DCF for every company."
    )
    existing_positions = _fetch_competition_positions()
    suggested = ", ".join(dict.fromkeys(str(row["ticker"]).upper() for row in existing_positions))
    raw_tickers = st.text_area(
        "Tickers (comma, space, or newline separated; maximum 8)",
        value=suggested,
        placeholder="MSFT, ASML, NVDA",
        key="wharton_company_analysis_tickers",
        height=75,
    )
    run_analysis = st.button("Analyze Companies", type="primary", use_container_width=True)
    if run_analysis:
        import re

        tickers = list(dict.fromkeys(item.upper() for item in re.split(r"[\s,;]+", raw_tickers) if item.strip()))
        if not tickers:
            st.error("Enter at least one ticker.")
        elif len(tickers) > 8:
            st.error("You can analyze up to 8 companies at once.")
        else:
            results: dict[str, Any] = {}
            progress = st.progress(0.0, text="Loading company data…")
            for index, ticker in enumerate(tickers, start=1):
                try:
                    results[ticker] = _fetch_company_analysis_cached(ticker)
                except Exception as exc:
                    results[ticker] = {"ticker": ticker, "error": str(exc)}
                progress.progress(index / len(tickers), text=f"Loaded {index}/{len(tickers)}: {ticker}")
            progress.empty()
            st.session_state[COMPANY_ANALYSIS_KEY] = results

    results = st.session_state.get(COMPANY_ANALYSIS_KEY, {})
    if not isinstance(results, dict) or not results:
        st.info("Enter tickers and run the analysis. Tickers from Portfolio Tracker are prefilled automatically.")
        return

    valid_results = {ticker: data for ticker, data in results.items() if isinstance(data, dict) and not data.get("error")}
    for ticker, data in results.items():
        if isinstance(data, dict) and data.get("error"):
            st.error(f"{ticker}: {data['error']}")
    if not valid_results:
        return

    st.markdown("#### Company Comparison")
    comparison_rows: list[dict[str, Any]] = []
    for ticker, snapshot in valid_results.items():
        info = snapshot.get("info", {})
        moat = analyze_moat(info)
        base_dcf = build_dcf_scenarios(info).get("Base", {})
        geographic_analysis = snapshot.get("geographic_revenue", {}).get("analysis", {})
        comparison_rows.append({
            "Ticker": ticker,
            "Company": info.get("shortName") or info.get("longName") or ticker,
            "Sector": info.get("sector") or "—",
            "Price": _format_company_metric("price", info.get("currentPrice") or info.get("regularMarketPrice")),
            "Market cap": _format_company_metric("marketCap", info.get("marketCap")),
            "Revenue YoY": _format_company_metric("revenueGrowth", info.get("revenueGrowth")),
            "Operating margin": _format_company_metric("operatingMargins", info.get("operatingMargins")),
            "FCF": _format_company_metric("freeCashflow", info.get("freeCashflow")),
            "Moat signal": f"{moat['score']}/{moat['max_score']} · {moat['label']}",
            "Regional diversification": (
                f"{geographic_analysis.get('score')}/5 · {geographic_analysis.get('label')}"
                if geographic_analysis.get("available")
                else "Not disclosed"
            ),
            "Base DCF / share": f"${base_dcf['fair_value_per_share']:,.2f}" if base_dcf.get("available") else "N/A",
        })
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    selected_ticker = st.selectbox("Company Detail", list(valid_results), key="wharton_company_detail_ticker")
    snapshot = valid_results[selected_ticker]
    info = snapshot.get("info", {})
    company_name = info.get("longName") or info.get("shortName") or selected_ticker
    st.markdown(f"## {company_name} ({selected_ticker})")
    st.caption(f"Data fetched: {snapshot.get('fetched_at', '—')} · Market-data source: Yahoo Finance")

    overview_tab, regions_tab, industry_tab, evidence_tab, financials_tab, management_tab, moat_tab, dcf_tab, metrics_tab = st.tabs([
        "Overview", "Revenue by Region", "Industry & Peers", "Evidence & Sources", "Financial Statements", "Management", "Moat, Track Record & Risks", "DCF", "All Metrics",
    ])

    with overview_tab:
        kpis = st.columns(6)
        overview_metrics = [
            ("Price", info.get("currentPrice") or info.get("regularMarketPrice")),
            ("Market cap", info.get("marketCap")),
            ("Forward P/E", info.get("forwardPE")),
            ("Revenue growth", info.get("revenueGrowth")),
            ("Operating margin", info.get("operatingMargins")),
            ("ROE", info.get("returnOnEquity")),
        ]
        for column, (label, value) in zip(kpis, overview_metrics):
            column.metric(label, _format_company_metric(label, value))
        st.markdown("#### What the Company Does")
        st.write(info.get("longBusinessSummary") or "A company description is not available from the data source.")
        profile_rows = {
            "Sector": info.get("sector"), "Industry": info.get("industry"),
            "Country": info.get("country"), "City": info.get("city"),
            "Employees": info.get("fullTimeEmployees"), "Website": info.get("website"),
        }
        st.dataframe(pd.DataFrame([{"Field": key, "Value": value or "—"} for key, value in profile_rows.items()]), hide_index=True, use_container_width=True)
        history = snapshot.get("history")
        if isinstance(history, pd.DataFrame) and not history.empty and "Close" in history:
            st.markdown("#### Five-Year Price History")
            st.line_chart(history[["Close"]].rename(columns={"Close": selected_ticker}), use_container_width=True)

    with regions_tab:
        _render_geographic_revenue(snapshot, selected_ticker, str(company_name))

    with industry_tab:
        _render_industry_peer_analysis(profile, selected_ticker, valid_results)

    with evidence_tab:
        _render_research_evidence(profile, selected_ticker, snapshot)

    with financials_tab:
        annual, quarterly = st.tabs(["Annual", "Quarterly"])
        with annual:
            _render_statement_block("Income Statement", snapshot.get("income_statement", pd.DataFrame()))
            _render_statement_block("Balance Sheet", snapshot.get("balance_sheet", pd.DataFrame()))
            _render_statement_block("Cash Flow Statement", snapshot.get("cash_flow", pd.DataFrame()))
        with quarterly:
            _render_statement_block("Quarterly Income Statement", snapshot.get("quarterly_income_statement", pd.DataFrame()))
            _render_statement_block("Quarterly Balance Sheet", snapshot.get("quarterly_balance_sheet", pd.DataFrame()))
            _render_statement_block("Quarterly Cash Flow Statement", snapshot.get("quarterly_cash_flow", pd.DataFrame()))

    with management_tab:
        st.markdown("#### Current Management")
        officers = snapshot.get("officers", [])
        if officers:
            officer_rows = []
            for officer in officers:
                officer_rows.append({
                    "Name": officer.get("name") or "—", "Role": officer.get("title") or "—",
                    "Age": officer.get("age") or "—", "Year born": officer.get("yearBorn") or "—",
                    "Reported fiscal year": officer.get("fiscalYear") or "—",
                    "Total compensation": _format_company_metric("totalPay", officer.get("totalPay")),
                })
            st.dataframe(pd.DataFrame(officer_rows), use_container_width=True, hide_index=True)
        else:
            st.info("The source did not provide a management roster.")
        st.warning(
            "Yahoo Finance usually does not provide complete executive education and career histories. "
            "The app never invents them. Source-backed public biographies can be loaded below."
        )
        biography_key = f"wharton_management_biographies_{selected_ticker}"
        if officers and st.button(
            "Load Education and Career Histories",
            key=f"load_{biography_key}",
            use_container_width=True,
        ):
            biographies: dict[str, Any] = {}
            biography_progress = st.progress(0.0, text="Searching public biographies…")
            for officer_index, officer in enumerate(officers, start=1):
                officer_name = str(officer.get("name") or "").strip()
                if officer_name:
                    try:
                        biographies[officer_name] = _fetch_management_biography_cached(officer_name, company_name)
                    except Exception as exc:
                        biographies[officer_name] = {"available": False, "error": str(exc)}
                biography_progress.progress(
                    officer_index / len(officers),
                    text=f"Checked {officer_index}/{len(officers)}: {officer_name or 'Unnamed executive'}",
                )
            biography_progress.empty()
            st.session_state[biography_key] = biographies

        biographies = st.session_state.get(biography_key, {})
        if isinstance(biographies, dict) and biographies:
            st.markdown("#### Education and Historical Career")
            st.caption(
                "Profiles are matched to public English Wikipedia biographies and include a direct source link. "
                "Unmatched executives remain explicitly unavailable. Verify material facts against official company biographies or filings."
            )
            for officer in officers:
                officer_name = str(officer.get("name") or "").strip()
                biography = biographies.get(officer_name, {})
                with st.expander(f"{officer_name or 'Unnamed executive'} — {officer.get('title') or 'Role unavailable'}"):
                    if biography.get("available"):
                        st.markdown("**Education**")
                        st.write(biography.get("education"))
                        st.markdown("**Historical career**")
                        st.write(biography.get("career"))
                        st.markdown(f"[Source: {biography.get('matched_title')}]({biography.get('source_url')})")
                        st.caption(biography.get("verification_note") or "")
                    else:
                        st.info(f"Source-backed profile unavailable: {biography.get('error', 'No confident match found.')}")
        elif officers:
            st.info("Select **Load Education and Career Histories** to enrich the management roster with source-backed public biographies.")
        news_rows = _company_news_rows(snapshot.get("news", []))
        st.markdown("#### Current Evidence Sources")
        if news_rows:
            for item in news_rows[:10]:
                if item["url"]:
                    st.markdown(f"- **[{item['id']}]** [{item['title']}]({item['url']}) — {item['publisher']}")
                else:
                    st.markdown(f"- **[{item['id']}]** {item['title']} — {item['publisher']}")
        else:
            st.info("No current news is available.")

        ai_key = f"wharton_company_ai_{selected_ticker}"
        if st.button("Generate Evidence-Constrained Management Synthesis", key=f"run_{ai_key}"):
            from src.ai.ai_review import resolve_groq_api_key
            from src.ai.company_analysis import generate_company_deep_dive

            api_key = resolve_groq_api_key(st.secrets)
            evidence = {
                "ticker": selected_ticker, "company": company_name,
                "business_summary": info.get("longBusinessSummary"), "officers": officers,
                "management_biographies": biographies if isinstance(biographies, dict) else {},
                "selected_metrics": {key: info.get(key) for key in [
                    "revenueGrowth", "earningsGrowth", "operatingMargins", "profitMargins",
                    "returnOnEquity", "freeCashflow", "totalDebt", "marketCap",
                ]},
                "news": news_rows,
                "deterministic_track_record": analyze_track_record(info, snapshot.get("history")),
            }
            with st.spinner("Preparing a synthesis from the supplied evidence…"):
                st.session_state[ai_key] = generate_company_deep_dive(evidence, api_key)
        ai_result = st.session_state.get(ai_key)
        if isinstance(ai_result, dict):
            if ai_result.get("available"):
                st.markdown("#### Management-History Synthesis")
                st.write(ai_result.get("management_history") or "Insufficient evidence.")
                st.markdown("**Investment View**")
                st.write(ai_result.get("investment_view") or "—")
                st.caption(ai_result.get("evidence_limitations") or "")
            else:
                st.warning(f"Supplementary synthesis is unavailable: {ai_result.get('error', 'unknown error')}")

    with moat_tab:
        moat = analyze_moat(info)
        track = analyze_track_record(info, snapshot.get("history"))
        score_col, label_col = st.columns(2)
        score_col.metric("Quantitative Moat Score", f"{moat['score']}/{moat['max_score']}")
        label_col.metric("Result", moat["label"])
        st.caption(moat["warning"])
        st.dataframe(pd.DataFrame([{
            "Status": "Pass" if signal["passed"] else "Fail", "Area": signal["name"], "Evidence": signal["evidence"],
        } for signal in moat["signals"]]), use_container_width=True, hide_index=True)
        success_col, failure_col = st.columns(2)
        with success_col:
            st.markdown("#### Observable Successes")
            if track["successes"]:
                for item in track["successes"]:
                    st.success(item)
            else:
                st.info("Available metrics did not establish a clear positive signal.")
        with failure_col:
            st.markdown("#### Failures / Warning Signals")
            if track["failures"]:
                for item in track["failures"]:
                    st.error(item)
            else:
                st.info("Available metrics did not establish a clear negative signal.")
        if isinstance(ai_result := st.session_state.get(f"wharton_company_ai_{selected_ticker}"), dict) and ai_result.get("available"):
            st.markdown("#### Qualitative Moat Synthesis")
            st.write(ai_result.get("moat_analysis") or "—")
            ai_success, ai_failure = st.columns(2)
            with ai_success:
                for item in ai_result.get("successes", []):
                    st.success(item)
            with ai_failure:
                for item in ai_result.get("failures", []):
                    st.error(item)

    with dcf_tab:
        defaults = default_dcf_assumptions(info)
        st.markdown("#### Custom DCF Assumptions")
        with st.form(f"dcf_form_{selected_ticker}"):
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                fcf_b = st.number_input("Normalized FCF (billions)", value=float(defaults["free_cash_flow"]) / 1e9, step=0.1, key=f"dcf_fcf_{selected_ticker}")
                growth_pct = st.number_input("FCF Growth (%)", value=float(defaults["growth_rate"]) * 100, step=0.5, key=f"dcf_growth_{selected_ticker}")
            with d2:
                discount_pct = st.number_input("Discount Rate / WACC (%)", value=float(defaults["discount_rate"]) * 100, step=0.5, key=f"dcf_wacc_{selected_ticker}")
                terminal_pct = st.number_input("Terminal Growth (%)", value=float(defaults["terminal_growth_rate"]) * 100, step=0.25, key=f"dcf_terminal_{selected_ticker}")
            with d3:
                years = st.number_input("Explicit Forecast Years", 1, 20, int(defaults["years"]), key=f"dcf_years_{selected_ticker}")
                shares_m = st.number_input("Shares Outstanding (millions)", min_value=0.0, value=float(defaults["shares_outstanding"]) / 1e6, step=1.0, key=f"dcf_shares_{selected_ticker}")
            with d4:
                cash_b = st.number_input("Cash (billions)", value=float(defaults["cash"]) / 1e9, step=0.1, key=f"dcf_cash_{selected_ticker}")
                debt_b = st.number_input("Debt (billions)", value=float(defaults["debt"]) / 1e9, step=0.1, key=f"dcf_debt_{selected_ticker}")
            st.form_submit_button("Recalculate DCF", type="primary", use_container_width=True)
        assumptions = {
            "free_cash_flow": float(fcf_b) * 1e9, "growth_rate": float(growth_pct) / 100,
            "discount_rate": float(discount_pct) / 100, "terminal_growth_rate": float(terminal_pct) / 100,
            "years": int(years), "cash": float(cash_b) * 1e9, "debt": float(debt_b) * 1e9,
            "shares_outstanding": float(shares_m) * 1e6, "current_price": float(defaults["current_price"]),
        }
        scenarios = build_dcf_scenarios(info, assumptions)
        scenario_columns = st.columns(3)
        for column, (name, result) in zip(scenario_columns, scenarios.items()):
            with column:
                st.markdown(f"#### {name}")
                if result.get("available"):
                    st.metric("Fair Value / Share", f"${result['fair_value_per_share']:,.2f}", f"{result['upside_pct']:+.1%}" if result.get("upside_pct") is not None else None)
                    st.caption(f"Terminal value: {result['terminal_value_share']:.1%} EV")
                else:
                    st.error(result.get("error", "DCF cannot be calculated."))
        base_result = scenarios.get("Base", {})
        if base_result.get("available"):
            st.markdown("#### Base-Case Projection")
            st.dataframe(pd.DataFrame([{
                "Year": row["year"], "FCF": f"${row['free_cash_flow'] / 1e9:,.2f}B", "Present Value": f"${row['present_value'] / 1e9:,.2f}B",
            } for row in base_result["projected"]]), use_container_width=True, hide_index=True)
            if base_result["terminal_value_share"] > 0.75:
                st.warning("More than 75% of enterprise value comes from terminal value; the result is highly sensitive to WACC and terminal growth.")
        st.caption("DCF is a scenario model, not a price target or investment recommendation.")

    with metrics_tab:
        st.markdown(f"#### All Available Scalar Metrics ({len(snapshot.get('metrics', {}))})")
        metric_rows = [
            {"Metric / Key": key, "Value": _format_company_metric(key, value), "Raw": str(value)}
            for key, value in sorted(snapshot.get("metrics", {}).items())
        ]
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True, height=650)
        st.download_button(
            "Download Metrics as JSON",
            data=json.dumps(snapshot.get("metrics", {}), ensure_ascii=False, indent=2, default=str),
            file_name=f"{selected_ticker}_company_metrics.json",
            mime="application/json",
            use_container_width=True,
        )


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


def _render_custom_quant_context(result: dict[str, Any]) -> None:
    if result:
        generated = str(result.get("generated_at") or "unknown time")
        tickers = ", ".join(str(item) for item in result.get("tickers", [])) or "unknown universe"
        st.info(
            f"Context: latest manually configured Quant Sandbox run ({tickers}; generated {generated}). "
            "This is not automatically the live Portfolio Tracker."
        )


def render_wharton_cockpit() -> None:
    _inject_cockpit_styles()
    init_db()

    profile = _get_current_profile()
    if profile is None:
        _render_login()
        return

    _render_header(profile)

    # Fetch result from state if available.
    result = st.session_state.get(QUANT_RESULT_KEY, {})

    def _with_quant_context(renderer):
        def _render() -> None:
            _render_custom_quant_context(result)
            renderer(result)

        return _render

    tab_renderers = [
        ("Overview & Tasks", lambda: _render_overview_action_center(profile)),
        ("Strategy & Decisions", lambda: _render_strategy_workspace(profile, result)),
        ("Quant Engine", lambda: _render_quant_engine(profile)),
        ("Stock Screener", _render_stock_screener),
        ("Risk Cockpit", _with_quant_context(_render_risk_cockpit)),
        ("Factor Exposure", _with_quant_context(_render_factor_exposure)),
        ("Regime Detection", _with_quant_context(_render_regime_detection)),
        ("Scenario Playground", _with_quant_context(_render_scenario_playground)),
        ("Efficient Frontier", _with_quant_context(_render_efficient_frontier)),
        ("Monte Carlo", _with_quant_context(_render_monte_carlo)),
        ("Advanced Monte Carlo", _with_quant_context(_render_advanced_monte_carlo)),
        ("Advanced Analytics", _with_quant_context(_render_advanced_analytics)),
        ("Mind Map", _render_mindmap),
        ("Sub-Projects", lambda: _render_subprojects(profile)),
        ("War Room", lambda: _render_chat(profile)),
        ("File Vault", lambda: _render_file_center(profile)),
        ("Assignment & Rules", lambda: _render_competition_rules(profile)),
        ("Portfolio Tracker", lambda: _render_competition_portfolio(profile)),
        ("Company Analysis", lambda: _render_company_analysis(profile)),
    ]
    visible_tab_renderers = [
        (label, renderer)
        for label, renderer in tab_renderers
        if label not in HIDDEN_COCKPIT_TABS
    ]

    tabs = st.tabs([label for label, _ in visible_tab_renderers])
    for tab, (_, renderer) in zip(tabs, visible_tab_renderers, strict=False):
        with tab:
            renderer()


def main() -> None:
    render_wharton_cockpit()


if __name__ == "__main__":
    main()
