from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
import importlib
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

import bcrypt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_PATH = "data/wharton_production.db"
UPLOAD_DIR = "data/wharton_uploads"
DEFAULT_PASSWORD = "Wharton2026!"
USER_PROFILE_KEY = "user_profile"
TASK_EDITOR_VERSION_KEY = "wharton_task_editor_version"
QUANT_RESULT_KEY = "wharton_quant_result"
QUANT_ERROR_KEY = "wharton_quant_error"

TASK_PRIORITIES = ["Critical", "High", "Medium", "Low"]
GRAPH_NODE_TYPES = ["Policy", "Company", "Model", "Market", "Risk", "Research", "Other"]
QUANT_MODULES = [
    "Benchmark Analytics",
    "Cost-Aware Rebalance",
    "Performance Attribution",
    "Simulation",
]
QUANT_OPERATOR_USERS = {"Jakub", "Matfyz_Genius"}
DEFAULT_QUANT_TICKERS = ["ASML", "NVDA", "MSFT", "LLY", "JPM"]

DEFAULT_USERS = [
    {
        "username": "Jakub",
        "role": "Captain/Quant",
        "primary_module": "Quant Engine",
    },
    {
        "username": "Matěj",
        "role": "Oxford/CIO",
        "primary_module": "Dashboard & Strategy",
    },
    {
        "username": "Martin",
        "role": "Logistics/Risk",
        "primary_module": "Risk Operations",
    },
    {
        "username": "Lukáš",
        "role": "Geopolitics",
        "primary_module": "Macro Intelligence",
    },
    {
        "username": "Janek",
        "role": "Intelligence",
        "primary_module": "War Room",
    },
    {
        "username": "Matfyz_Genius",
        "role": "Quant/Math",
        "primary_module": "Quant Engine",
    },
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


def get_connection() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    connection = sqlite3.connect("data/wharton_production.db", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _inject_cockpit_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 100% !important;
            padding-top: 1.2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .wharton-hero {
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 24px;
            padding: 1.25rem 1.45rem;
            margin-bottom: 1rem;
            background:
                radial-gradient(circle at 4% 18%, rgba(20, 184, 166, 0.20), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(30, 41, 59, 0.94));
            color: #f8fafc;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.16);
        }
        .wharton-hero h1 {
            margin: 0;
            font-size: 2.05rem;
            letter-spacing: -0.04em;
        }
        .wharton-hero p {
            margin: 0.45rem 0 0;
            color: rgba(248, 250, 252, 0.78);
        }
        .wharton-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }
        .wharton-badge {
            border: 1px solid rgba(226, 232, 240, 0.22);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            background: rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
            font-size: 0.86rem;
        }
        .wharton-panel {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            background: rgba(248, 250, 252, 0.74);
            margin-bottom: 1rem;
        }
        .wharton-section-kicker {
            color: #0f766e;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 800;
            font-size: 0.76rem;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(20, 184, 166, 0.32);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            background:
                radial-gradient(circle at top left, rgba(45, 212, 191, 0.22), transparent 42%),
                linear-gradient(135deg, #0f172a, #164e63);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #f8fafc !important;
        }
        div[data-testid="stMetric"] svg {
            fill: #f8fafc !important;
        }
        .wharton-graph-shell {
            border: 1px solid rgba(20, 184, 166, 0.24);
            border-radius: 20px;
            padding: 0.75rem 0.75rem 0.25rem;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 89, 0.90));
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
            margin: 0.75rem 0 1.25rem;
        }
        .wharton-graph-shell strong {
            color: #ecfeff;
        }
        .wharton-graph-shell span {
            color: #a7f3d0;
        }
        div[data-testid="stTabs"] button {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_db() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                role TEXT,
                primary_module TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                username TEXT,
                message TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                priority TEXT,
                task_text TEXT,
                assignee TEXT,
                is_done INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                filename TEXT,
                uploaded_by TEXT,
                file_path TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mindmap_nodes (
                id TEXT PRIMARY KEY,
                label TEXT,
                type TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mindmap_edges (
                id TEXT PRIMARY KEY,
                source TEXT,
                target TEXT
            )
            """
        )

        existing_users = {
            str(row["username"]): str(row["password_hash"] or "")
            for row in connection.execute("SELECT username, password_hash FROM users").fetchall()
        }
        for user in DEFAULT_USERS:
            if existing_users.get(user["username"]):
                continue

            password_hash = bcrypt.hashpw(
                DEFAULT_PASSWORD.encode("utf-8"),
                bcrypt.gensalt(),
            ).decode("utf-8")
            if user["username"] in existing_users:
                connection.execute(
                    """
                    UPDATE users
                    SET password_hash = ?, role = ?, primary_module = ?
                    WHERE username = ?
                    """,
                    (
                        password_hash,
                        user["role"],
                        user["primary_module"],
                        user["username"],
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO users (username, password_hash, role, primary_module)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        user["username"],
                        password_hash,
                        user["role"],
                        user["primary_module"],
                    ),
                )

        node_count = int(connection.execute("SELECT COUNT(*) FROM mindmap_nodes").fetchone()[0])
        if node_count == 0:
            connection.executemany(
                """
                INSERT INTO mindmap_nodes (id, label, type)
                VALUES (?, ?, ?)
                """,
                DEFAULT_MINDMAP_NODES,
            )

        edge_count = int(connection.execute("SELECT COUNT(*) FROM mindmap_edges").fetchone()[0])
        if edge_count == 0:
            connection.executemany(
                """
                INSERT INTO mindmap_edges (id, source, target)
                VALUES (?, ?, ?)
                """,
                DEFAULT_MINDMAP_EDGES,
            )


def _fetch_users() -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, username, role, primary_module
            FROM users
            ORDER BY username COLLATE NOCASE
            """
        ).fetchall()


def authenticate_user(username: str, password: str) -> dict[str, str | int] | None:
    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT id, username, password_hash, role, primary_module
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if user is None:
        return None

    stored_hash = str(user["password_hash"] or "")
    if not stored_hash:
        return None

    password_ok = bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash.encode("utf-8"),
    )
    if not password_ok:
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
    st.title("Wharton Cockpit")
    st.caption("Production workspace for strategy, quant research, files, and team coordination.")

    users = _fetch_users()
    usernames = [str(user["username"]) for user in users]
    if not usernames:
        st.error("No seeded users were found. Refresh the app to run database initialization again.")
        st.stop()

    with st.form("wharton_login_form", clear_on_submit=False):
        username = st.selectbox("Username", options=usernames, index=0)
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter Cockpit", type="primary")

    st.caption("Seeded accounts use the initial password `Wharton2026!`. Rotate it before real deployment.")

    if submitted:
        profile = authenticate_user(username, password)
        if profile is None:
            st.error("Authentication failed. Check the username and password.")
            return

        st.session_state[USER_PROFILE_KEY] = profile
        st.rerun()


def _fetch_task_rows() -> pd.DataFrame:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, priority, task_text, assignee, is_done
            FROM tasks
            ORDER BY id ASC
            """
        ).fetchall()

    return pd.DataFrame(
        [
            {
                "id": int(row["id"]),
                "priority": str(row["priority"] or "Medium"),
                "task_text": str(row["task_text"] or ""),
                "assignee": str(row["assignee"] or ""),
                "is_done": bool(row["is_done"]),
            }
            for row in rows
        ],
        columns=["id", "priority", "task_text", "assignee", "is_done"],
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "done", "checked"}
    return bool(value)


def _clean_priority(value: object) -> str:
    priority = str(value or "Medium").strip()
    if priority not in TASK_PRIORITIES:
        return "Medium"
    return priority


def _clean_task_payload(row_data: dict[str, object], default_assignee: str) -> dict[str, str | int]:
    assignee = str(row_data.get("assignee") or default_assignee).strip()
    if not assignee:
        assignee = default_assignee

    return {
        "priority": _clean_priority(row_data.get("priority")),
        "task_text": str(row_data.get("task_text") or "").strip(),
        "assignee": assignee,
        "is_done": int(_truthy(row_data.get("is_done"))),
    }


def _coerce_row_index(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_editor_has_changes(editor_state: object) -> bool:
    if not isinstance(editor_state, dict):
        return False
    return bool(
        editor_state.get("edited_rows")
        or editor_state.get("added_rows")
        or editor_state.get("deleted_rows")
    )


def _apply_task_editor_changes(
    editor_state: dict[str, object],
    original_tasks: pd.DataFrame,
    default_assignee: str,
) -> None:
    edited_rows = editor_state.get("edited_rows") or {}
    added_rows = editor_state.get("added_rows") or []
    deleted_rows = editor_state.get("deleted_rows") or []

    if not isinstance(edited_rows, dict):
        edited_rows = {}
    if not isinstance(added_rows, list):
        added_rows = []
    if not isinstance(deleted_rows, list):
        deleted_rows = []

    deleted_indices = {
        row_index
        for row_index in (_coerce_row_index(value) for value in deleted_rows)
        if row_index is not None
    }

    with get_connection() as connection:
        for row_index in sorted(deleted_indices, reverse=True):
            if 0 <= row_index < len(original_tasks):
                task_id = int(original_tasks.iloc[row_index]["id"])
                connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

        for raw_index, changes in edited_rows.items():
            row_index = _coerce_row_index(raw_index)
            if row_index is None or row_index in deleted_indices:
                continue
            if not 0 <= row_index < len(original_tasks):
                continue
            if not isinstance(changes, dict):
                continue

            current_row = original_tasks.iloc[row_index].to_dict()
            current_row.update(changes)
            payload = _clean_task_payload(current_row, default_assignee)
            task_id = int(original_tasks.iloc[row_index]["id"])
            connection.execute(
                """
                UPDATE tasks
                SET priority = ?, task_text = ?, assignee = ?, is_done = ?
                WHERE id = ?
                """,
                (
                    payload["priority"],
                    payload["task_text"],
                    payload["assignee"],
                    payload["is_done"],
                    task_id,
                ),
            )

        for added_row in added_rows:
            if not isinstance(added_row, dict):
                continue
            payload = _clean_task_payload(added_row, default_assignee)
            if not payload["task_text"]:
                continue
            connection.execute(
                """
                INSERT INTO tasks (priority, task_text, assignee, is_done)
                VALUES (?, ?, ?, ?)
                """,
                (
                    payload["priority"],
                    payload["task_text"],
                    payload["assignee"],
                    payload["is_done"],
                ),
            )


def _render_task_manager(profile: dict[str, str | int]) -> None:
    st.markdown("### Mission Task Manager")
    st.caption("Edits, additions, and deletions commit directly to SQLite on rerun.")

    original_tasks = _fetch_task_rows()
    editor_version = int(st.session_state.get(TASK_EDITOR_VERSION_KEY, 0))
    editor_key = f"wharton_tasks_editor_{editor_version}"

    st.data_editor(
        original_tasks,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=editor_key,
        column_order=["id", "priority", "task_text", "assignee", "is_done"],
        height=520,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "priority": st.column_config.SelectboxColumn(
                "Priority",
                options=TASK_PRIORITIES,
                required=True,
                width="small",
            ),
            "task_text": st.column_config.TextColumn(
                "Task",
                required=True,
                width="large",
            ),
            "assignee": st.column_config.TextColumn(
                "Assignee",
                required=True,
                width="medium",
            ),
            "is_done": st.column_config.CheckboxColumn(
                "Done",
                help="Mark completed tasks without leaving the cockpit.",
                width="small",
            ),
        },
    )

    editor_state = st.session_state.get(editor_key, {})
    if _task_editor_has_changes(editor_state):
        _apply_task_editor_changes(
            editor_state=editor_state,
            original_tasks=original_tasks,
            default_assignee=str(profile["username"]),
        )
        st.session_state[TASK_EDITOR_VERSION_KEY] = editor_version + 1
        st.toast("Task board saved to SQLite.")
        st.rerun()


def _fetch_graph_rows() -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    with get_connection() as connection:
        nodes = connection.execute(
            """
            SELECT id, label, type
            FROM mindmap_nodes
            ORDER BY label COLLATE NOCASE
            """
        ).fetchall()
        edges = connection.execute(
            """
            SELECT id, source, target
            FROM mindmap_edges
            ORDER BY id COLLATE NOCASE
            """
        ).fetchall()
    return nodes, edges


def _node_display_name(row: sqlite3.Row) -> str:
    return f"{row['label']} ({row['type']})"


def _slugify_node_id(label: str) -> str:
    slug = "".join(
        character.lower() if character.isascii() and character.isalnum() else "_"
        for character in label.strip()
    )
    slug = "_".join(part for part in slug.split("_") if part)
    if not slug:
        slug = "node"
    if not slug.startswith("node_"):
        slug = f"node_{slug}"
    return slug[:96]


def _create_unique_node_id(label: str) -> str:
    base_id = _slugify_node_id(label)
    candidate = base_id
    suffix = 2
    with get_connection() as connection:
        while connection.execute(
            "SELECT 1 FROM mindmap_nodes WHERE id = ?",
            (candidate,),
        ).fetchone():
            candidate = f"{base_id}_{suffix}"
            suffix += 1
    return candidate


def _edge_id(source_id: str, target_id: str) -> str:
    safe_source = "".join(character if character.isalnum() else "_" for character in source_id)
    safe_target = "".join(character if character.isalnum() else "_" for character in target_id)
    return f"edge_{safe_source}_{safe_target}"[:180]


def _insert_node(label: str, node_type: str) -> None:
    node_id = _create_unique_node_id(label)
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO mindmap_nodes (id, label, type)
            VALUES (?, ?, ?)
            """,
            (node_id, label.strip(), node_type),
        )


def _insert_edge(source_id: str, target_id: str) -> bool:
    edge_id = _edge_id(source_id, target_id)
    with get_connection() as connection:
        exists = connection.execute(
            "SELECT 1 FROM mindmap_edges WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if exists:
            return False
        connection.execute(
            """
            INSERT INTO mindmap_edges (id, source, target)
            VALUES (?, ?, ?)
            """,
            (edge_id, source_id, target_id),
        )
    return True


def _render_mindmap() -> None:
    st.markdown("### Strategy Mind Map")
    st.caption("The graph uses SQLite-backed nodes and edges, with full-width rendering for actual thinking room.")

    node_rows, edge_rows = _fetch_graph_rows()
    node_ids = {str(row["id"]) for row in node_rows}
    valid_edge_count = sum(
        1
        for row in edge_rows
        if str(row["source"]) in node_ids and str(row["target"]) in node_ids
    )
    st.markdown(
        f"""
        <div class="wharton-graph-shell">
            <strong>Live Strategy Graph</strong>
            <span> | {len(node_rows)} nodes | {valid_edge_count} visible edges</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Identify connected nodes (nodes that have at least one edge)
    connected_node_ids = set()
    for row in edge_rows:
        connected_node_ids.add(str(row["source"]))
        connected_node_ids.add(str(row["target"]))

    graph_nodes = [
        Node(
            id=str(row["id"]),
            label=str(row["label"]),
            title=f"{row['label']} | {row['type']}",
            size=32,
            color=NODE_COLORS.get(str(row["type"]), NODE_COLORS["Other"]),
            font={"color": "#d1d8eb", "size": 18, "face": "Tahoma"},
            # Isolated nodes get higher mass (stickier, resist movement but can be dragged)
            # Connected nodes use default mass for normal force-directed behavior
            mass=4.0 if str(row["id"]) not in connected_node_ids else 1.0,
        )
        for row in node_rows
    ]
    graph_edges = [
        Edge(
            source=str(row["source"]),
            target=str(row["target"]),
            color="#64748b",
        )
        for row in edge_rows
        if str(row["source"]) in node_ids and str(row["target"]) in node_ids
    ]
    graph_config = Config(
        width=1800,
        height=760,
        directed=True,
        physics={
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.9,  # Higher damping = more resistance = smoother, stickier movement
                "avoidOverlap": 0.1,
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
        st.warning("No mind map nodes exist yet. Add the first node below to initialize the map.")

    st.divider()
    st.markdown("### Manage Knowledge Graph")

    with st.expander("Add a Node", expanded=True):
        with st.form("wharton_add_node_form", clear_on_submit=True):
            node_label = st.text_input(
                "Node Label",
                help="Example: Taiwan Export Controls, ASML Earnings, Monte Carlo Risk Model",
            )
            node_type = st.selectbox("Node Type", options=GRAPH_NODE_TYPES)
            add_node = st.form_submit_button("Add Node", type="primary")

        if add_node:
            clean_label = node_label.strip()
            if not clean_label:
                st.warning("Enter a node label before adding it.")
            else:
                _insert_node(clean_label, node_type)
                st.success(f"Added node: {clean_label}")
                st.rerun()

    with st.expander("Connect Existing Nodes", expanded=True):
        if len(node_rows) < 2:
            st.info("Add at least two nodes before creating an edge.")
        else:
            node_options = [str(row["id"]) for row in node_rows]
            row_by_id = {str(row["id"]): row for row in node_rows}
            with st.form("wharton_connect_nodes_form", clear_on_submit=False):
                source_id = st.selectbox(
                    "Source Node",
                    options=node_options,
                    format_func=lambda node_id: _node_display_name(row_by_id[node_id]),
                    key="wharton_edge_source_select",
                )
                target_id = st.selectbox(
                    "Target Node",
                    options=node_options,
                    index=1 if len(node_options) > 1 else 0,
                    format_func=lambda node_id: _node_display_name(row_by_id[node_id]),
                    key="wharton_edge_target_select",
                )
                connect_clicked = st.form_submit_button("Connect", type="primary")

            if connect_clicked:
                if source_id == target_id:
                    st.warning("Choose two different nodes to create a meaningful connection.")
                elif _insert_edge(source_id, target_id):
                    st.success(
                        f"Connected {_node_display_name(row_by_id[source_id])} "
                        f"to {_node_display_name(row_by_id[target_id])}."
                    )
                    st.rerun()
                else:
                    st.info("That connection already exists.")

    with st.expander("Current Graph Tables", expanded=False):
        st.markdown("#### Nodes")
        st.dataframe(
            pd.DataFrame(
                [{"ID": row["id"], "Label": row["label"], "Type": row["type"]} for row in node_rows]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Edges")
        st.dataframe(
            pd.DataFrame(
                [{"ID": row["id"], "Source": row["source"], "Target": row["target"]} for row in edge_rows]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _fetch_chat_history() -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, timestamp, username, message
            FROM chat
            ORDER BY id ASC
            """
        ).fetchall()


def _save_chat_message(username: str, message: str) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat (timestamp, username, message)
            VALUES (?, ?, ?)
            """,
            (_now_iso(), username, message),
        )


def _render_chat(profile: dict[str, str | int]) -> None:
    st.markdown("### War Room Chat")
    st.caption("Every message is persisted in the `chat` table.")

    history = _fetch_chat_history()
    if not history:
        st.info("No messages yet. Open the channel with the first update.")

    current_username = str(profile["username"])
    for row in history:
        chat_role = "user" if str(row["username"]) == current_username else "assistant"
        with st.chat_message(chat_role):
            st.markdown(f"**{row['username']}** · {row['timestamp']}")
            st.write(str(row["message"]))

    prompt = st.chat_input("Send a War Room update")
    if prompt and prompt.strip():
        _save_chat_message(current_username, prompt.strip())
        st.rerun()


def _safe_filename(filename: str) -> str:
    base_name = os.path.basename(filename).replace("\\", "_").strip()
    cleaned = "".join(
        character
        if character.isalnum() or character in {".", "_", "-"}
        else "_"
        for character in base_name
    )
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "upload.bin"
    return cleaned[:140]


def _save_uploaded_file(uploaded_file: object, uploaded_by: str) -> None:
    safe_name = _safe_filename(str(uploaded_file.name))
    unique_prefix = datetime.now().strftime("%Y%m%d%H%M%S%f")
    stored_filename = f"{unique_prefix}_{safe_name}"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    with open(file_path, "wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO files (timestamp, filename, uploaded_by, file_path)
            VALUES (?, ?, ?, ?)
            """,
            (_now_iso(), safe_name, uploaded_by, file_path),
        )


def _fetch_file_rows() -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, timestamp, filename, uploaded_by, file_path
            FROM files
            ORDER BY id DESC
            """
        ).fetchall()


def _render_file_center(profile: dict[str, str | int]) -> None:
    st.markdown("### Persistent File Vault")
    st.caption(f"Files are stored on disk under `{UPLOAD_DIR}/` and indexed in SQLite.")

    with st.form("wharton_file_upload_form", clear_on_submit=True):
        uploads = st.file_uploader(
            "Upload research, decks, model notes, datasets, or screenshots",
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Save Uploaded Files", type="primary")

    if submitted:
        if not uploads:
            st.warning("Choose at least one file before saving.")
        else:
            for uploaded_file in uploads:
                _save_uploaded_file(uploaded_file, str(profile["username"]))
            st.success(f"Saved {len(uploads)} file(s).")
            st.rerun()

    file_rows = _fetch_file_rows()
    if not file_rows:
        st.info("No files have been uploaded yet.")
        return

    files_table = pd.DataFrame(
        [
            {
                "ID": int(row["id"]),
                "Uploaded": str(row["timestamp"]),
                "Filename": str(row["filename"]),
                "Uploaded By": str(row["uploaded_by"]),
                "Path": str(row["file_path"]),
            }
            for row in file_rows
        ]
    )
    st.dataframe(files_table, use_container_width=True, hide_index=True, height=320)

    st.markdown("#### Downloads")
    for row in file_rows:
        file_path = str(row["file_path"])
        filename = str(row["filename"])
        if not os.path.exists(file_path):
            st.warning(f"`{filename}` is indexed but missing from disk.")
            continue

        with open(file_path, "rb") as file_handle:
            st.download_button(
                label=f"Download {filename}",
                data=file_handle.read(),
                file_name=filename,
                mime="application/octet-stream",
                key=f"download_file_{row['id']}",
            )


def _load_quant_modules() -> dict[str, Any]:
    return {
        "analytics": importlib.import_module("src.analytics"),
        "optimization": importlib.import_module("src.optimization"),
        "simulation": importlib.import_module("src.simulation"),
        "yahoo_fetcher": importlib.import_module("src.data.fetchers.yahoo_fetcher"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_close_prices_cached(
    symbols: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    modules = _load_quant_modules()
    fetcher = modules["yahoo_fetcher"].YahooFetcher()
    return fetcher.fetch_close_prices(list(symbols), start_date, end_date)


def _parse_tickers(raw_text: str) -> list[str]:
    tickers = []
    seen = set()
    for chunk in raw_text.replace(",", "\n").splitlines():
        ticker = chunk.strip().upper()
        if not ticker or ticker in seen:
            continue
        tickers.append(ticker)
        seen.add(ticker)
    if not tickers:
        raise ValueError("Enter at least one portfolio ticker.")
    return tickers


def _parse_weights(raw_text: str, tickers: list[str]) -> np.ndarray:
    if not raw_text.strip():
        return np.array([1.0 / len(tickers)] * len(tickers), dtype=float)

    values = []
    for chunk in raw_text.replace(",", "\n").splitlines():
        stripped = chunk.strip().replace("%", "")
        if stripped:
            values.append(float(stripped))

    if len(values) != len(tickers):
        raise ValueError("Weights must be empty or provide one value per ticker.")

    weights = np.asarray(values, dtype=float)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative for the current long-only engine.")
    if weights.sum() > 1.5:
        weights = weights / 100.0
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("Weights cannot sum to zero.")
    return weights / weights.sum()


def _align_weights_to_returns(
    requested_tickers: list[str],
    requested_weights: np.ndarray,
    available_columns: list[str],
) -> np.ndarray:
    weight_by_ticker = pd.Series(requested_weights, index=requested_tickers, dtype=float)
    aligned = weight_by_ticker.reindex(available_columns).fillna(0.0).to_numpy(dtype=float)
    if np.isclose(aligned.sum(), 0.0):
        return np.array([1.0 / len(available_columns)] * len(available_columns), dtype=float)
    return aligned / aligned.sum()


def _format_pct(value: object, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(numeric) or np.isinf(numeric):
        return "n/a"
    return f"{numeric:.{decimals}%}"


def _format_float(value: object, decimals: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(numeric) or np.isinf(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}"


def _compute_quant_run(
    tickers: list[str],
    weights: np.ndarray,
    benchmark_ticker: str,
    start_date: date,
    end_date: date,
    risk_free_rate: float,
    current_value: float,
    max_weight: float,
    turnover_limit: float,
    transaction_cost_bps: float,
    risk_aversion: float,
    simulation_days: int,
    n_simulations: int,
    random_seed: int,
) -> dict[str, Any]:
    modules = _load_quant_modules()
    analytics = modules["analytics"]
    optimization = modules["optimization"]
    simulation = modules["simulation"]

    prices = _fetch_close_prices_cached(tuple(tickers), start_date, end_date)
    if prices.empty:
        raise ValueError("No price data returned for the selected portfolio universe.")

    prices = prices.sort_index().ffill().dropna(how="all")
    available_columns = [str(column) for column in prices.columns if prices[column].notna().sum() > 2]
    prices = prices[available_columns].dropna(how="any")
    if prices.empty or len(available_columns) == 0:
        raise ValueError("Price data did not contain enough aligned observations.")

    returns = prices.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Return series is empty after alignment.")

    aligned_weights = _align_weights_to_returns(tickers, weights, list(returns.columns))
    portfolio_returns = analytics.calculate_portfolio_daily_returns(returns, aligned_weights)
    core_metrics = analytics.calculate_portfolio_core_metrics(portfolio_returns, risk_free_rate)
    concentration = analytics.calculate_concentration_metrics(aligned_weights)
    corr_matrix = analytics.calculate_correlation_matrix(returns)
    avg_corr = analytics.calculate_average_correlation(corr_matrix)

    benchmark_symbol = benchmark_ticker.strip().upper()
    benchmark_returns = pd.Series(dtype=float)
    if benchmark_symbol:
        if benchmark_symbol in returns.columns:
            benchmark_returns = returns[benchmark_symbol]
        else:
            benchmark_prices = _fetch_close_prices_cached((benchmark_symbol,), start_date, end_date)
            if benchmark_symbol in benchmark_prices.columns:
                benchmark_returns = benchmark_prices[benchmark_symbol].sort_index().pct_change().dropna()

    benchmark_metrics = analytics.calculate_active_risk_metrics(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        benchmark_ticker=benchmark_symbol,
        risk_free_rate=risk_free_rate,
    )
    return_contribution = analytics.calculate_return_contribution(returns, aligned_weights)
    risk_contribution = analytics.calculate_risk_contribution(returns, aligned_weights)
    min_variance = optimization.optimize_minimum_variance(returns, max_weight=max_weight)
    max_sharpe = optimization.optimize_maximum_sharpe(
        returns,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )
    cost_aware = optimization.optimize_cost_aware_rebalance(
        returns=returns,
        current_weights=aligned_weights,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        turnover_limit=turnover_limit,
        transaction_cost_bps=transaction_cost_bps,
        risk_aversion=risk_aversion,
    )
    portfolio_timeseries = analytics.build_portfolio_timeseries(
        portfolio_returns,
        initial_value=current_value,
    )
    price_paths, simulation_stats = simulation.run_monte_carlo_simulation(
        current_value=current_value,
        expected_return=float(core_metrics.get("annualized_return", 0.0)),
        volatility=float(core_metrics.get("volatility", 0.0)),
        time_horizon=simulation_days,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    metrics = {
        **core_metrics,
        **concentration,
        "avg_correlation": avg_corr,
        "observations": int(returns.shape[0]),
    }

    return {
        "generated_at": _now_iso(),
        "tickers": list(returns.columns),
        "requested_tickers": tickers,
        "weights": aligned_weights,
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
        "inputs": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "risk_free_rate": risk_free_rate,
            "current_value": current_value,
            "max_weight": max_weight,
            "turnover_limit": turnover_limit,
            "transaction_cost_bps": transaction_cost_bps,
            "risk_aversion": risk_aversion,
            "simulation_days": simulation_days,
            "n_simulations": n_simulations,
            "random_seed": random_seed,
        },
    }


def _render_quant_configuration() -> None:
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=365 * 2)

    with st.expander("Quant Run Configuration", expanded=QUANT_RESULT_KEY not in st.session_state):
        with st.form("wharton_quant_config_form", clear_on_submit=False):
            input_col, risk_col = st.columns([1, 1], gap="large")
            with input_col:
                tickers_text = st.text_area(
                    "Portfolio Tickers",
                    value="\n".join(DEFAULT_QUANT_TICKERS),
                    height=145,
                    help="One ticker per line or comma-separated.",
                )
                weights_text = st.text_area(
                    "Weights",
                    value="",
                    height=115,
                    help="Leave empty for equal weights. Use decimals or percentages.",
                )
                benchmark_ticker = st.text_input("Benchmark Ticker", value="SPY").strip().upper()
                current_value = st.number_input(
                    "Current Portfolio Value",
                    min_value=1_000.0,
                    value=100_000.0,
                    step=5_000.0,
                )

            with risk_col:
                start_date = st.date_input("Start Date", value=default_start)
                end_date = st.date_input("End Date", value=default_end)
                risk_free_rate = st.slider(
                    "Risk-Free Rate",
                    min_value=0.0,
                    max_value=0.15,
                    value=0.03,
                    step=0.005,
                    format="%.3f",
                )
                max_weight = st.slider(
                    "Max Asset Weight",
                    min_value=0.10,
                    max_value=1.0,
                    value=0.35,
                    step=0.01,
                )
                turnover_limit = st.slider(
                    "Turnover Limit",
                    min_value=0.05,
                    max_value=2.0,
                    value=0.30,
                    step=0.05,
                )
                transaction_cost_bps = st.slider(
                    "Transaction Cost (bps)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                )
                risk_aversion = st.slider(
                    "Risk Aversion",
                    min_value=0.5,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                )
                simulation_days = st.slider(
                    "Simulation Horizon (days)",
                    min_value=30,
                    max_value=1260,
                    value=252,
                    step=30,
                )
                n_simulations = st.slider(
                    "Simulation Count",
                    min_value=200,
                    max_value=5000,
                    value=1200,
                    step=100,
                )
                random_seed = st.number_input(
                    "Simulation Seed",
                    min_value=0,
                    value=42,
                    step=1,
                )

            run_clicked = st.form_submit_button("Run Full Quant Engine", type="primary")

    if run_clicked:
        st.session_state.pop(QUANT_ERROR_KEY, None)
        try:
            tickers = _parse_tickers(tickers_text)
            weights = _parse_weights(weights_text, tickers)
            if start_date >= end_date:
                raise ValueError("Start Date must be before End Date.")

            with st.spinner("Fetching market data and running quant stack..."):
                st.session_state[QUANT_RESULT_KEY] = _compute_quant_run(
                    tickers=tickers,
                    weights=weights,
                    benchmark_ticker=benchmark_ticker,
                    start_date=start_date,
                    end_date=end_date,
                    risk_free_rate=float(risk_free_rate),
                    current_value=float(current_value),
                    max_weight=float(max_weight),
                    turnover_limit=float(turnover_limit),
                    transaction_cost_bps=float(transaction_cost_bps),
                    risk_aversion=float(risk_aversion),
                    simulation_days=int(simulation_days),
                    n_simulations=int(n_simulations),
                    random_seed=int(random_seed),
                )
            st.success("Quant engine run completed.")
            st.rerun()
        except Exception as exc:
            st.session_state[QUANT_ERROR_KEY] = str(exc)


def _weights_frame(symbols: list[str], current_weights: np.ndarray, optimized_weights: np.ndarray) -> pd.DataFrame:
    optimized = np.asarray(optimized_weights, dtype=float)
    if optimized.size != len(symbols):
        optimized = np.zeros(len(symbols), dtype=float)
    current = np.asarray(current_weights, dtype=float)
    return pd.DataFrame(
        {
            "Ticker": symbols,
            "Current Weight": current,
            "Optimized Weight": optimized,
            "Delta": optimized - current,
        }
    )


def _render_weight_table(frame: pd.DataFrame) -> None:
    view = frame.copy()
    for column in ["Current Weight", "Optimized Weight", "Delta"]:
        view[column] = view[column].map(lambda value: _format_pct(value))
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_benchmark_analytics(result: dict[str, Any], advanced: bool) -> None:
    metrics = result["metrics"]
    benchmark_metrics = result["benchmark_metrics"]

    st.markdown("### Benchmark Analytics")
    metric_row = st.columns(4)
    metric_row[0].metric("Total Return", _format_pct(metrics.get("total_return")))
    metric_row[1].metric("Annualized Return", _format_pct(metrics.get("annualized_return")))
    metric_row[2].metric("Volatility", _format_pct(metrics.get("volatility")))
    metric_row[3].metric("Sharpe Ratio", _format_float(metrics.get("sharpe_ratio")))

    st.markdown("#### Portfolio Value Path")
    value_series = result["portfolio_timeseries"][["value"]].rename(columns={"value": "Portfolio Value"})
    st.line_chart(value_series, use_container_width=True, height=340)

    st.markdown("#### Active Risk")
    if benchmark_metrics.get("benchmark_available"):
        active_row = st.columns(4)
        active_row[0].metric("Benchmark", str(benchmark_metrics.get("benchmark_ticker", "")))
        active_row[1].metric("Active Return Ann.", _format_pct(benchmark_metrics.get("active_return_annualized")))
        active_row[2].metric("Tracking Error", _format_pct(benchmark_metrics.get("tracking_error")))
        active_row[3].metric("Information Ratio", _format_float(benchmark_metrics.get("information_ratio")))

        secondary_row = st.columns(4)
        secondary_row[0].metric("Beta", _format_float(benchmark_metrics.get("beta_to_benchmark")))
        secondary_row[1].metric("Alpha Ann.", _format_pct(benchmark_metrics.get("alpha_to_benchmark")))
        secondary_row[2].metric("Up Capture", _format_float(benchmark_metrics.get("up_capture")))
        secondary_row[3].metric("Down Capture", _format_float(benchmark_metrics.get("down_capture")))
    else:
        st.warning(
            "Benchmark metrics are unavailable for this run: "
            f"{benchmark_metrics.get('reason', 'unknown reason')}."
        )

    if advanced:
        with st.expander("Advanced Benchmark Diagnostics", expanded=True):
            st.write(f"Observations: **{metrics.get('observations', 0)}**")
            st.write(f"Average correlation: **{_format_float(metrics.get('avg_correlation'))}**")
            st.write(f"Effective holdings: **{_format_float(metrics.get('effective_holdings'))}**")
            st.write(f"Max weight: **{_format_pct(metrics.get('max_weight'))}**")
            st.markdown("##### Correlation Matrix")
            st.dataframe(result["correlation"].round(3), use_container_width=True)
            st.markdown("##### Return Tail")
            st.dataframe(result["returns"].tail(15), use_container_width=True)


def _render_cost_aware_rebalance(result: dict[str, Any], advanced: bool) -> None:
    cost_aware = result["cost_aware"]
    min_variance = result["min_variance"]
    max_sharpe = result["max_sharpe"]
    symbols = list(result["tickers"])
    weights = np.asarray(result["weights"], dtype=float)

    st.markdown("### Cost-Aware Rebalance")
    if not cost_aware.get("success"):
        st.warning(f"Cost-aware optimization returned a warning: {cost_aware.get('message', 'unknown')}")

    metric_row = st.columns(4)
    metric_row[0].metric("Expected Return", _format_pct(cost_aware.get("expected_return")))
    metric_row[1].metric("Volatility", _format_pct(cost_aware.get("volatility")))
    metric_row[2].metric("Sharpe", _format_float(cost_aware.get("sharpe_ratio")))
    metric_row[3].metric("Turnover", _format_pct(cost_aware.get("turnover")))

    weights_df = _weights_frame(symbols, weights, np.asarray(cost_aware.get("weights", []), dtype=float))
    st.markdown("#### Rebalance Weights")
    _render_weight_table(weights_df)
    chart_frame = weights_df.set_index("Ticker")[["Current Weight", "Optimized Weight"]]
    st.bar_chart(chart_frame, use_container_width=True, height=360)

    st.markdown("#### Optimizer Comparison")
    comparison = pd.DataFrame(
        [
            {
                "Model": "Current Portfolio",
                "Expected Return": result["metrics"].get("annualized_return", 0.0),
                "Volatility": result["metrics"].get("volatility", 0.0),
                "Sharpe": result["metrics"].get("sharpe_ratio", 0.0),
            },
            {
                "Model": "Minimum Variance",
                "Expected Return": min_variance.get("expected_return", 0.0),
                "Volatility": min_variance.get("volatility", 0.0),
                "Sharpe": min_variance.get("sharpe_ratio", 0.0),
            },
            {
                "Model": "Maximum Sharpe",
                "Expected Return": max_sharpe.get("expected_return", 0.0),
                "Volatility": max_sharpe.get("volatility", 0.0),
                "Sharpe": max_sharpe.get("sharpe_ratio", 0.0),
            },
            {
                "Model": "Cost-Aware Rebalance",
                "Expected Return": cost_aware.get("expected_return", 0.0),
                "Volatility": cost_aware.get("volatility", 0.0),
                "Sharpe": cost_aware.get("sharpe_ratio", 0.0),
            },
        ]
    )
    comparison_view = comparison.copy()
    for column in ["Expected Return", "Volatility"]:
        comparison_view[column] = comparison_view[column].map(lambda value: _format_pct(value))
    comparison_view["Sharpe"] = comparison_view["Sharpe"].map(lambda value: _format_float(value))
    st.dataframe(comparison_view, use_container_width=True, hide_index=True)

    if advanced:
        with st.expander("Advanced Optimizer Diagnostics", expanded=True):
            st.json(
                {
                    "success": bool(cost_aware.get("success")),
                    "message": str(cost_aware.get("message", "")),
                    "utility_score": float(cost_aware.get("utility_score", 0.0)),
                    "transaction_cost_drag": float(cost_aware.get("transaction_cost_drag", 0.0)),
                    "turnover_limit": float(cost_aware.get("turnover_limit", 0.0)),
                    "max_weight": float(cost_aware.get("max_weight", 0.0)),
                    "risk_aversion": float(cost_aware.get("risk_aversion", 0.0)),
                }
            )
            min_var_weights = _weights_frame(
                symbols,
                weights,
                np.asarray(min_variance.get("weights", []), dtype=float),
            )
            max_sharpe_weights = _weights_frame(
                symbols,
                weights,
                np.asarray(max_sharpe.get("weights", []), dtype=float),
            )
            st.markdown("##### Minimum Variance Weights")
            _render_weight_table(min_var_weights)
            st.markdown("##### Maximum Sharpe Weights")
            _render_weight_table(max_sharpe_weights)


def _render_performance_attribution(result: dict[str, Any], advanced: bool) -> None:
    st.markdown("### Performance Attribution")
    return_contribution = result["return_contribution"].copy()
    risk_contribution = result["risk_contribution"].copy()

    st.markdown("#### Return Contribution")
    if return_contribution.empty:
        st.info("Return contribution is unavailable for this run.")
    else:
        return_chart = return_contribution.set_index("Ticker")[["AnnualizedContributionApprox"]]
        st.bar_chart(return_chart, use_container_width=True, height=320)
        return_view = return_contribution.copy()
        for column in [
            "Weight",
            "TotalContributionApprox",
            "AnnualizedContributionApprox",
            "ContributionShare",
            "MeanDailyContribution",
        ]:
            return_view[column] = return_view[column].map(lambda value: _format_pct(value))
        st.dataframe(return_view, use_container_width=True, hide_index=True)

    st.markdown("#### Risk Contribution")
    if risk_contribution.empty:
        st.info("Risk contribution is unavailable for this run.")
    else:
        risk_chart = risk_contribution.set_index("Ticker")[["RiskBudgetPct"]]
        st.bar_chart(risk_chart, use_container_width=True, height=320)
        risk_view = risk_contribution.copy()
        for column in ["Weight", "MarginalVolatility", "RiskContribution", "RiskBudgetPct"]:
            risk_view[column] = risk_view[column].map(lambda value: _format_pct(value))
        st.dataframe(risk_view, use_container_width=True, hide_index=True)

    if advanced:
        with st.expander("Advanced Attribution Diagnostics", expanded=True):
            weighted_returns = result["returns"].multiply(result["weights"], axis=1)
            st.markdown("##### Weighted Daily Return Snapshot")
            st.dataframe(weighted_returns.tail(20), use_container_width=True)
            st.markdown("##### Contribution Summary")
            st.dataframe(weighted_returns.describe().T, use_container_width=True)


def _simulation_percentile_frame(price_paths: np.ndarray) -> pd.DataFrame:
    percentiles = [5, 25, 50, 75, 95]
    data = {
        f"p{percentile}": np.percentile(price_paths, percentile, axis=1)
        for percentile in percentiles
    }
    return pd.DataFrame(data)


def _render_simulation(result: dict[str, Any], advanced: bool) -> None:
    st.markdown("### Simulation")
    stats = result["simulation_stats"]
    price_paths = result["price_paths"]

    metric_row = st.columns(4)
    metric_row[0].metric("Mean Final Value", f"${stats['mean']:,.0f}")
    metric_row[1].metric("Median Final Value", f"${stats['median']:,.0f}")
    metric_row[2].metric("5th Percentile", f"${stats['percentile_5']:,.0f}")
    metric_row[3].metric("95th Percentile", f"${stats['percentile_95']:,.0f}")

    percentile_frame = _simulation_percentile_frame(price_paths)
    st.markdown("#### Monte Carlo Percentile Paths")
    st.line_chart(percentile_frame, use_container_width=True, height=420)

    final_values = pd.Series(price_paths[-1], name="Final Value")
    hist_counts, hist_bins = np.histogram(final_values, bins=40)
    hist_frame = pd.DataFrame(
        {
            "Final Value Bucket": [
                f"{hist_bins[index]:,.0f}-{hist_bins[index + 1]:,.0f}"
                for index in range(len(hist_counts))
            ],
            "Count": hist_counts,
        }
    )
    st.markdown("#### Final Value Distribution")
    st.bar_chart(hist_frame.set_index("Final Value Bucket"), use_container_width=True, height=320)

    if advanced:
        with st.expander("Advanced Simulation Diagnostics", expanded=True):
            st.write(f"Simulation matrix shape: **{price_paths.shape[0]} x {price_paths.shape[1]}**")
            st.dataframe(final_values.describe().to_frame(), use_container_width=True)
            st.dataframe(percentile_frame.tail(20), use_container_width=True)


def _render_quant_engine(profile: dict[str, str | int]) -> None:
    username = str(profile["username"])
    is_quant_operator = username in QUANT_OPERATOR_USERS
    diagnostic_key = f"wharton_quant_diagnostics_{username}"
    if diagnostic_key not in st.session_state:
        st.session_state[diagnostic_key] = is_quant_operator

    st.markdown("### Full Quant Engine")
    st.caption(
        "This tab dynamically imports `src.analytics`, `src.optimization`, and `src.simulation` "
        "and renders live outputs from the repository engine."
    )

    _render_quant_configuration()
    if QUANT_ERROR_KEY in st.session_state:
        st.error(st.session_state[QUANT_ERROR_KEY])

    result = st.session_state.get(QUANT_RESULT_KEY)
    if not isinstance(result, dict):
        st.info("Configure a universe and run the engine to populate benchmark, rebalance, attribution, and simulation views.")
        return

    st.success(f"Latest quant run generated at {result.get('generated_at', 'unknown time')}.")
    advanced = st.checkbox(
        "Show advanced diagnostics",
        key=diagnostic_key,
        help="Jakub and Matfyz_Genius see diagnostics enabled by default.",
    )

    nav_col, content_col = st.columns([0.18, 0.82], gap="large")
    with nav_col:
        st.markdown("#### Engine Module")
        selected_module = st.radio(
            "Module",
            options=QUANT_MODULES,
            label_visibility="collapsed",
            key="wharton_quant_module_selector",
        )
        st.markdown("#### Universe")
        st.write(", ".join(result["tickers"]))
        st.markdown("#### Inputs")
        st.write(f"Benchmark: **{result.get('benchmark_ticker') or 'None'}**")
        st.write(f"Range: **{result['inputs']['start_date']}** to **{result['inputs']['end_date']}**")

    with content_col:
        if selected_module == "Benchmark Analytics":
            _render_benchmark_analytics(result, advanced)
        elif selected_module == "Cost-Aware Rebalance":
            _render_cost_aware_rebalance(result, advanced)
        elif selected_module == "Performance Attribution":
            _render_performance_attribution(result, advanced)
        elif selected_module == "Simulation":
            _render_simulation(result, advanced)


def _render_overview_action_center(profile: dict[str, str | int]) -> None:
    st.markdown("### Overview & Action Center")

    with get_connection() as connection:
        open_tasks = int(
            connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE COALESCE(is_done, 0) = 0"
            ).fetchone()[0]
        )
        completed_tasks = int(
            connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE COALESCE(is_done, 0) = 1"
            ).fetchone()[0]
        )
        chat_messages = int(connection.execute("SELECT COUNT(*) FROM chat").fetchone()[0])
        files_count = int(connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        node_count = int(connection.execute("SELECT COUNT(*) FROM mindmap_nodes").fetchone()[0])
        edge_count = int(connection.execute("SELECT COUNT(*) FROM mindmap_edges").fetchone()[0])

    metric_row = st.columns(6)
    metric_row[0].metric("Open Tasks", open_tasks)
    metric_row[1].metric("Completed", completed_tasks)
    metric_row[2].metric("Chat Messages", chat_messages)
    metric_row[3].metric("Vault Files", files_count)
    metric_row[4].metric("Map Nodes", node_count)
    metric_row[5].metric("Map Edges", edge_count)

    st.markdown(
        f"""
        <div class="wharton-panel">
          <div class="wharton-section-kicker">Active Desk</div>
          <strong>{escape(str(profile['username']))}</strong> is operating as <strong>{escape(str(profile['role']))}</strong>.
          Primary module: <strong>{escape(str(profile['primary_module']))}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_task_manager(profile)


def _render_war_room_and_mindmap() -> None:
    st.markdown("### War Room & Mind Map")
    st.caption("A dedicated wide-screen knowledge layer for strategy links, risk themes, and research dependencies.")
    _render_mindmap()


def _render_comms_and_vault(profile: dict[str, str | int]) -> None:
    st.markdown("### Comms & Vault")
    comm_tabs = st.tabs(["Chat", "File Vault"])
    with comm_tabs[0]:
        _render_chat(profile)
    with comm_tabs[1]:
        _render_file_center(profile)


def _render_header(profile: dict[str, str | int]) -> None:
    username = escape(str(profile["username"]))
    role = escape(str(profile["role"]))
    primary_module = escape(str(profile["primary_module"]))
    st.markdown(
        f"""
        <div class="wharton-hero">
          <h1>Wharton Cockpit</h1>
          <p>Production command center for execution, graph intelligence, quant analytics, and team memory.</p>
          <div class="wharton-badge-row">
            <span class="wharton-badge">Active user: {username}</span>
            <span class="wharton-badge">Role: {role}</span>
            <span class="wharton-badge">Primary module: {primary_module}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Logout Wharton Cockpit", use_container_width=True):
        _logout()


def render_wharton_cockpit() -> None:
    _inject_cockpit_styles()
    init_db()

    profile = _get_current_profile()
    if profile is None:
        _render_login()
        return

    _render_header(profile)
    tabs = st.tabs(
        [
            "Overview & Action Center",
            "War Room & Mind Map",
            "Full Quant Engine",
            "Comms & Vault",
        ]
    )

    with tabs[0]:
        _render_overview_action_center(profile)
    with tabs[1]:
        _render_war_room_and_mindmap()
    with tabs[2]:
        _render_quant_engine(profile)
    with tabs[3]:
        _render_comms_and_vault(profile)


def main() -> None:
    render_wharton_cockpit()


if __name__ == "__main__":
    main()
