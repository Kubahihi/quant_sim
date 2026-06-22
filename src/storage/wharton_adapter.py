"""
Wharton File Vault storage adapter.

Integrates the new StorageBackend layer with the existing Wharton File Vault
database schema and UI.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib
import mimetypes
import os
import sqlite3
import uuid

from .backend import StorageBackend, LocalStorageBackend
from .file_manager import FileManager
from .exceptions import FileValidationError, StorageFileNotFoundError


# Default configuration
DEFAULT_STORAGE_PATH = "data/storage/wharton"
DEFAULT_DB_PATH = "data/wharton_production.db"
ALLOWED_EXTENSIONS = {
    ".pdf", ".xlsx", ".xls", ".csv", ".docx", ".doc",
    ".txt", ".md", ".png", ".jpg", ".jpeg", ".gif",
    ".pptx", ".ppt", ".json", ".py", ".ipynb", ".zip",
}
MAX_FILE_SIZE_MB = 50


def get_storage_backend(storage_path: Optional[str] = None) -> StorageBackend:
    """
    Get storage backend. Uses R2 in production (when [storage] secrets are set),
    falls back to local storage for development.
    """
    from .backend import storage_config
    
    try:
        loaded = storage_config.load_from_secrets()
        if loaded:
            backend = storage_config.create_backend()
            return backend
    except Exception as e:
        try:
            import streamlit as st
            st.warning(f" R2 storage init failed ({e}), using local storage. Files will not persist.")
        except Exception:
            pass

    # Local fallback
    path = storage_path or DEFAULT_STORAGE_PATH
    return LocalStorageBackend(path)


def get_file_manager(
    storage_path: Optional[str] = None,
) -> FileManager:
    """
    Get a FileManager instance configured for Wharton File Vault.
    
    Args:
        storage_path: Optional custom storage path
        
    Returns:
        FileManager instance
    """
    backend = get_storage_backend(storage_path)
    
    return FileManager(backend=backend)


def _now_iso() -> str:
    """Get current time as ISO string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    base = os.path.basename(filename).replace("\\", "_").strip()
    cleaned = "".join(c if c.isalnum() or c in {".", "_", "-"} else "_" for c in base)
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "upload.bin"
    return cleaned[:140]


def init_storage_db(db_path: Optional[str] = None) -> None:
    """
    Initialize or migrate the database schema for the new storage layer.
    
    Adds new columns to the existing files table if they don't exist.
    
    Args:
        db_path: Optional custom database path
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get existing columns
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()
    }
    
    # Add new columns for storage backend support
    new_columns = [
        ("storage_backend", "TEXT DEFAULT 'local'"),
        ("storage_key", "TEXT DEFAULT ''"),
        ("sha256", "TEXT DEFAULT ''"),
        ("mime_type", "TEXT DEFAULT ''"),
    ]
    
    for col_name, col_def in new_columns:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE files ADD COLUMN {col_name} {col_def}")
    
    conn.commit()
    conn.close()


def _validate_file(filename: str, file_bytes: bytes) -> tuple[bool, Optional[str]]:
    """Validate file extension and size."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type `{ext}` is not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    size_bytes = len(file_bytes)
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_size_bytes:
        return False, f"File `{filename}` exceeds {MAX_FILE_SIZE_MB} MB limit ({size_bytes / 1024 / 1024:.1f} MB)."
    return True, None


def _detect_content_type(filename: str, file_bytes: bytes) -> str:
    """Detect MIME type from filename extension."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def _generate_storage_key(filename: str) -> str:
    """Generate a unique storage key for the file."""
    safe_name = _safe_filename(filename)
    unique_id = uuid.uuid4().hex[:12]
    return f"{unique_id}_{safe_name}"


def save_uploaded_file(
    uploaded_file: object,
    uploaded_by: str,
    db_path: Optional[str] = None,
    storage_path: Optional[str] = None,
    project_name: str = "",
    description: str = "",
    tags: str = "",
) -> dict:
    """
    Save an uploaded file using the storage backend.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        uploaded_by: Username of the uploader
        db_path: Optional custom database path
        storage_path: Optional custom storage path
        project_name: Optional project/category
        description: Optional description
        tags: Optional comma-separated tags
        
    Returns:
        dict with file metadata including 'id' and 'error' if any
        
    Raises:
        FileValidationError: if validation fails
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Get file content
    file_bytes = uploaded_file.getbuffer()
    original_filename = str(uploaded_file.name)
    
    # Validate file
    is_valid, error = _validate_file(original_filename, file_bytes)
    if not is_valid:
        raise FileValidationError(error)
    
    # Detect content type
    content_type = _detect_content_type(original_filename, file_bytes)
    
    # Generate storage key
    storage_key = _generate_storage_key(original_filename)
    
    # Get backend and save to storage
    backend = get_storage_backend(storage_path)
    upload_metadata = backend.upload(file_bytes, original_filename, content_type)
    
    # Use the storage key generated by the backend
    storage_key = upload_metadata.storage_key
    
    # Compute SHA256
    sha256 = upload_metadata.sha256
    
    # Use the filename stored by the backend
    stored_filename = original_filename
    
    # Save metadata to database
    conn = sqlite3.connect(db_path)
    try:
        # Get existing columns
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()
        }
        
        # Build INSERT query based on available columns
        columns = ["timestamp", "filename", "original_filename", "uploaded_by", "file_path"]
        values = [_now_iso(), stored_filename, original_filename, uploaded_by, storage_key]
        
        # Add new columns if they exist
        if "file_size_bytes" in existing_cols:
            columns.append("file_size_bytes")
            values.append(len(file_bytes))
        if "mime_type" in existing_cols:
            columns.append("mime_type")
            values.append(content_type)
        if "project_name" in existing_cols:
            columns.append("project_name")
            values.append(project_name.strip())
        if "description" in existing_cols:
            columns.append("description")
            values.append(description.strip())
        if "tags" in existing_cols:
            columns.append("tags")
            values.append(tags.strip())
        if "storage_backend" in existing_cols:
            columns.append("storage_backend")
            values.append(backend.backend_name)
        if "storage_key" in existing_cols:
            columns.append("storage_key")
            values.append(storage_key)
        if "sha256" in existing_cols:
            columns.append("sha256")
            values.append(sha256)
        
        placeholders = ", ".join(["?"] * len(columns))
        columns_str = ", ".join(columns)
        
        cursor = conn.execute(
            f"INSERT INTO files ({columns_str}) VALUES ({placeholders})",
            tuple(values),
        )
        file_id = cursor.lastrowid
        conn.commit()
        
        return {
            "id": file_id,
            "storage_key": storage_key,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "size": len(file_bytes),
            "content_type": content_type,
            "sha256": sha256,
            "error": None,
        }
    finally:
        conn.close()


def download_file(
    file_id: int,
    db_path: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> tuple[bytes, str, str]:
    """
    Download a file from storage.
    
    Args:
        file_id: Database ID of the file
        db_path: Optional custom database path
        storage_path: Optional custom storage path
        
    Returns:
        Tuple of (content, original_filename, content_type)
        
    Raises:
        StorageFileNotFoundError: if file doesn't exist in storage
        FileNotFoundError: if file record not found in database
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Get file metadata from database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        
        if row is None:
            raise FileNotFoundError(f"File with id {file_id} not found")
        
        # Handle columns that may not exist in older schemas
        column_names = set(row.keys())
        if "storage_key" in column_names and row["storage_key"]:
            storage_key = row["storage_key"]
        elif "file_path" in column_names:
            storage_key = row["file_path"]
        else:
            raise FileNotFoundError(f"File with id {file_id} has no storage key")
        original_filename = row["original_filename"]
        content_type = row["mime_type"] if "mime_type" in column_names and row["mime_type"] else "application/octet-stream"
    finally:
        conn.close()
    
    # Get file from storage backend
    backend = get_storage_backend(storage_path)
    
    try:
        content = backend.download(storage_key)
    except FileNotFound:
        raise StorageFileNotFoundError(
            f"File '{original_filename}' (ID: {file_id}) is indexed but missing from storage"
        )
    
    return content, original_filename, content_type


def file_exists(
    file_id: int,
    db_path: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> bool:
    """
    Check if a file exists in storage.
    
    Args:
        file_id: Database ID of the file
        db_path: Optional custom database path
        storage_path: Optional custom storage path
        
    Returns:
        True if file exists in storage
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Get storage key from database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        
        if row is None:
            return False
        
        column_names = set(row.keys())
        if "storage_key" in column_names and row["storage_key"]:
            storage_key = row["storage_key"]
        elif "file_path" in column_names:
            storage_key = row["file_path"]
        else:
            return False
    finally:
        conn.close()
    
    # Check if file exists in storage
    backend = get_storage_backend(storage_path)
    return backend.exists(storage_key)


def verify_file_integrity(
    file_id: int,
    db_path: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> bool:
    """
    Verify file integrity by checking SHA256 hash.
    
    Args:
        file_id: Database ID of the file
        db_path: Optional custom database path
        storage_path: Optional custom storage path
        
    Returns:
        True if integrity is verified
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Get stored hash from database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        
        if row is None:
            return False
        
        column_names = set(row.keys())
        stored_hash = row["sha256"] if "sha256" in column_names else ""
        if "storage_key" in column_names and row["storage_key"]:
            storage_key = row["storage_key"]
        elif "file_path" in column_names:
            storage_key = row["file_path"]
        else:
            return False
    finally:
        conn.close()
    
    if not stored_hash:
        # No hash stored (old file), can't verify
        return True
    
    # Download and compute hash
    backend = get_storage_backend(storage_path)
    try:
        content = backend.download(storage_key)
        actual_hash = hashlib.sha256(content).hexdigest()
        return actual_hash == stored_hash
    except Exception:
        return False


def delete_file(
    file_id: int,
    db_path: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> bool:
    """
    Delete a file from storage and database.
    
    Args:
        file_id: Database ID of the file
        db_path: Optional custom database path
        storage_path: Optional custom storage path
        
    Returns:
        True if file was deleted
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Get storage key from database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        
        if row is None:
            return False
        
        column_names = set(row.keys())
        if "storage_key" in column_names and row["storage_key"]:
            storage_key = row["storage_key"]
        elif "file_path" in column_names:
            storage_key = row["file_path"]
        else:
            return False
    finally:
        conn.close()
    
    # Delete from storage backend
    backend = get_storage_backend(storage_path)
    backend.delete(storage_key)
    
    # Delete from database
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()
    
    return True


def get_file_status(
    file_id: int,
    db_path: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> dict:
    """
    Get comprehensive status of a file.
    
    Args:
        file_id: Database ID of the file
        db_path: Optional custom database path
        storage_path: Optional custom storage path
        
    Returns:
        dict with status information
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Get file metadata from database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        
        if row is None:
            return {
                "id": file_id,
                "exists_in_db": False,
                "exists_in_storage": False,
                "status": "not_found",
            }
        
        column_names = set(row.keys())
        if "storage_key" in column_names and row["storage_key"]:
            storage_key = row["storage_key"]
        elif "file_path" in column_names:
            storage_key = row["file_path"]
        else:
            storage_key = ""
        original_filename = row["original_filename"]
        file_size = row["file_size_bytes"] if "file_size_bytes" in column_names else 0
        stored_hash = row["sha256"] if "sha256" in column_names else ""
    finally:
        conn.close()
    
    # Check storage
    backend = get_storage_backend(storage_path)
    exists_in_storage = backend.exists(storage_key)
    
    # Check integrity if file exists
    integrity_ok = False
    if exists_in_storage and stored_hash:
        try:
            content = backend.download(storage_key)
            actual_hash = hashlib.sha256(content).hexdigest()
            integrity_ok = actual_hash == stored_hash
        except Exception:
            integrity_ok = False
    
    # Determine status
    if not exists_in_storage:
        status = "missing"
    elif stored_hash and not integrity_ok:
        status = "corrupted"
    else:
        status = "available"
    
    return {
        "id": file_id,
        "exists_in_db": True,
        "exists_in_storage": exists_in_storage,
        "status": status,
        "original_filename": original_filename,
        "file_size": file_size,
        "integrity_ok": integrity_ok or (not stored_hash),
        "storage_backend": row["storage_backend"] if "storage_backend" in column_names else "local",
    }


def list_files_with_status(
    db_path: Optional[str] = None,
    storage_path: Optional[str] = None,
    search_query: str = "",
) -> list[dict]:
    """
    List all files with their availability status.
    
    Args:
        db_path: Optional custom database path
        storage_path: Optional custom storage path
        search_query: Optional search query to filter files
        
    Returns:
        List of file info dicts with status
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Get all files from database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM files ORDER BY id DESC",
        ).fetchall()
    finally:
        conn.close()
    
    # Get storage backend
    backend = get_storage_backend(storage_path)
    
    # Get column names from the first row to handle schema variations
    column_names = set(rows[0].keys()) if rows else set()
    
    result = []
    for row in rows:
        file_id = row["id"]
        original_filename = row["original_filename"]
        uploaded_by = row["uploaded_by"]
        project_name = row["project_name"] if "project_name" in column_names else ""
        tags = row["tags"] if "tags" in column_names else ""
        description = row["description"] if "description" in column_names else ""
        file_size = row["file_size_bytes"] if "file_size_bytes" in column_names else 0
        timestamp = row["timestamp"]
        # Handle storage_key and file_path columns that may not exist in older schemas
        if "storage_key" in column_names and row["storage_key"]:
            storage_key = row["storage_key"]
        elif "file_path" in column_names:
            storage_key = row["file_path"]
        else:
            # Fallback: try to get file_path from the row (legacy behavior)
            storage_key = row["file_path"] if "file_path" in column_names else ""
        
        # Apply search filter
        if search_query:
            q = search_query.lower()
            haystack = f"{original_filename} {uploaded_by} {project_name} {tags}".lower()
            if q not in haystack:
                continue
        
        # Check storage status
        exists_in_storage = backend.exists(storage_key)
        
        # Determine status
        if not exists_in_storage:
            status = "missing"
            status_label = " Missing from storage"
        else:
            status = "available"
            status_label = " Available"
        
        # Format size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.0f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        result.append({
            "id": file_id,
            "timestamp": timestamp,
            "filename": original_filename,  # Alias for backward compatibility
            "original_filename": original_filename,
            "uploaded_by": uploaded_by,
            "project_name": project_name,
            "tags": tags,
            "description": description,
            "size": file_size,
            "file_size_bytes": file_size,  # Alias for backward compatibility
            "size_str": size_str,
            "status": status,
            "status_label": status_label,
            "exists_in_storage": exists_in_storage,
            "file_path": storage_key,  # Alias for backward compatibility
        })
    
    return result