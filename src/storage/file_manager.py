"""
File Manager for handling file uploads and downloads with storage backends.

Includes:
- FileManager: High-level file operations with limit enforcement
- WhartonFileVault: User-friendly interface for file management
"""

import io
import json
import streamlit as st
from typing import Optional, Dict, Any, List
from pathlib import Path

from .backend import (
    StorageBackend, 
    StorageMetadata, 
    StorageLimits,
    get_storage_backend, 
    initialize_storage,
    check_storage_health,
    get_storage_usage,
    storage_config,
)
from .exceptions import (
    StorageError, 
    FileNotFound,
    FileSizeLimitExceeded,
    TotalStorageLimitExceeded,
    FileCountLimitExceeded,
    UserFileCountLimitExceeded,
    DuplicateFileError,
    ProductionConfigError,
)


class FileManager:
    """
    High-level file manager that uses storage backends for file operations.
    
    Enforces hard limits:
    - MAX_FILE_SIZE_MB: Maximum size per file
    - MAX_TOTAL_STORAGE_MB: Maximum total storage
    - MAX_FILES: Maximum number of files
    - MAX_FILES_PER_USER: Maximum files per user
    """
    
    def __init__(self, backend: Optional[StorageBackend] = None):
        """
        Initialize FileManager with a storage backend.
        
        Args:
            backend: Optional storage backend. If None, uses the global configured backend.
        """
        self._backend = backend
        self._initialized = False
        self._init_error = None
    
    @property
    def backend(self) -> StorageBackend:
        """Get the storage backend, initializing if needed."""
        if not self._initialized:
            try:
                if self._backend is None:
                    self._backend = get_storage_backend()
                self._initialized = True
            except Exception as e:
                self._init_error = str(e)
                raise StorageError(f"Failed to initialize storage backend: {e}")
        return self._backend
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the file manager and run health checks.
        
        Returns:
            Dictionary with initialization status.
        """
        result = initialize_storage()
        if not result["success"]:
            self._init_error = result.get("error", "Unknown error")
            raise StorageError(f"Storage initialization failed: {self._init_error}")
        
        self._initialized = True
        return result
    
    def upload_file(
        self, 
        file_data: bytes, 
        filename: str, 
        content_type: str = "application/octet-stream",
        uploaded_by: Optional[str] = None,
        project_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        check_limits: bool = True,
        **kwargs
    ) -> StorageMetadata:
        """
        Upload a file to storage with limit enforcement.
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            content_type: MIME type of the file
            uploaded_by: User identifier who uploaded the file
            project_name: Optional project name for categorization
            description: Optional file description
            tags: Optional list of tags
            check_limits: Whether to enforce hard limits (default True)
            **kwargs: Additional metadata
        
        Returns:
            StorageMetadata for the uploaded file
        
        Raises:
            FileSizeLimitExceeded: If file exceeds size limit
            TotalStorageLimitExceeded: If upload would exceed total storage
            FileCountLimitExceeded: If file count would exceed limit
            UserFileCountLimitExceeded: If user file count would exceed limit
            DuplicateFileError: If file with same SHA256 already exists
        """
        metadata = {
            "uploaded_by": uploaded_by,
            "project_name": project_name,
            "description": description,
            "tags": tags or [],
            **kwargs
        }
        
        # Validate upload against limits (if enabled)
        if check_limits:
            # This will raise appropriate exceptions if limits are exceeded
            self.backend.validate_upload(file_data, uploaded_by=uploaded_by, check_duplicate=True)
        
        return self.backend.upload(file_data, filename, content_type, metadata)
    
    def download_file(self, storage_key: str) -> bytes:
        """
        Download a file from storage.
        
        Args:
            storage_key: Unique identifier of the file
        
        Returns:
            Raw file bytes
        
        Raises:
            FileNotFound: If file doesn't exist
        """
        try:
            return self.backend.download(storage_key)
        except FileNotFoundError:
            raise FileNotFound(f"File not found: {storage_key}")
    
    def download_file_buffer(self, storage_key: str) -> io.BytesIO:
        """
        Download a file and return as BytesIO buffer (safe for Streamlit).
        
        Args:
            storage_key: Unique identifier of the file
        
        Returns:
            BytesIO object containing file data
        """
        if hasattr(self.backend, 'download_to_buffer'):
            return self.backend.download_to_buffer(storage_key)
        return io.BytesIO(self.download_file(storage_key))
    
    def get_metadata(self, storage_key: str) -> Optional[StorageMetadata]:
        """
        Get metadata for a file.
        
        Args:
            storage_key: Unique identifier of the file
        
        Returns:
            StorageMetadata or None if not found
        """
        # List files and find the one with matching key
        files = self.backend.list_files()
        for file_meta in files:
            if file_meta.storage_key == storage_key:
                return file_meta
        return None
    
    def delete_file(self, storage_key: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            storage_key: Unique identifier of the file
        
        Returns:
            True if deleted successfully
        """
        return self.backend.delete(storage_key)
    
    def list_files(self, prefix: Optional[str] = None) -> List[StorageMetadata]:
        """
        List files in storage.
        
        Args:
            prefix: Optional prefix to filter files
        
        Returns:
            List of StorageMetadata objects
        """
        return self.backend.list_files(prefix)
    
    def file_exists(self, storage_key: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            storage_key: Unique identifier of the file
        
        Returns:
            True if file exists
        """
        return self.backend.exists(storage_key)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the storage backend.
        
        Returns:
            Health check results
        """
        return check_storage_health()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current storage usage statistics.
        
        Returns:
            Dictionary with usage statistics including:
            - total_files
            - total_storage_mb
            - max_files
            - max_storage_mb
            - storage_usage_percent
            - files_usage_percent
            - approaching_storage_limit
            - approaching_file_limit
        """
        return get_storage_usage()
    
    def find_duplicate(self, file_data: bytes) -> Optional[StorageMetadata]:
        """
        Check if a file with the same content already exists.
        
        Args:
            file_data: Raw file bytes to check
        
        Returns:
            StorageMetadata of existing file if duplicate found, None otherwise
        """
        import hashlib
        sha256 = hashlib.sha256(file_data).hexdigest()
        return self.backend.find_by_sha256(sha256)
    
    @property
    def is_initialized(self) -> bool:
        """Check if the file manager is initialized."""
        return self._initialized
    
    @property
    def backend_name(self) -> str:
        """Get the name of the current backend."""
        if not self._initialized:
            return "not_initialized"
        return self.backend.backend_name


class WhartonFileVault:
    """
    File vault for Wharton application with user-friendly interface.
    
    Provides Streamlit-optimized file operations with limit display
    and user-friendly error handling.
    """
    
    def __init__(self, file_manager: Optional[FileManager] = None):
        """
        Initialize Wharton File Vault.
        
        Args:
            file_manager: Optional FileManager instance. If None, creates a new one.
        """
        self.file_manager = file_manager or FileManager()
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the file vault.
        
        Returns:
            Initialization status
        """
        return self.file_manager.initialize()
    
    def upload_from_streamlit(
        self, 
        uploaded_file, 
        uploaded_by: Optional[str] = None,
        project_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> StorageMetadata:
        """
        Upload a file from Streamlit's file_uploader widget.
        
        Args:
            uploaded_file: File object from st.file_uploader
            uploaded_by: User identifier
            project_name: Optional project name
            description: Optional file description
            tags: Optional list of tags
        
        Returns:
            StorageMetadata for the uploaded file
        
        Raises:
            Various limit exceptions if limits are exceeded
        """
        file_data = uploaded_file.getvalue()
        filename = uploaded_file.name
        content_type = uploaded_file.type or "application/octet-stream"
        
        return self.file_manager.upload_file(
            file_data=file_data,
            filename=filename,
            content_type=content_type,
            uploaded_by=uploaded_by,
            project_name=project_name,
            description=description,
            tags=tags,
        )
    
    def download_to_streamlit(self, storage_key: str) -> io.BytesIO:
        """
        Download a file and return as BytesIO for Streamlit.
        
        Args:
            storage_key: Unique identifier of the file
        
        Returns:
            BytesIO object for download button
        """
        return self.file_manager.download_file_buffer(storage_key)
    
    def get_user_files(self, user_id: str) -> List[StorageMetadata]:
        """
        Get all files uploaded by a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            List of StorageMetadata objects
        """
        all_files = self.file_manager.list_files()
        return [f for f in all_files if f.uploaded_by == user_id]
    
    def display_file_list(self, files: List[StorageMetadata]):
        """
        Display a list of files in Streamlit.
        
        Args:
            files: List of StorageMetadata to display
        """
        if not files:
            st.info("No files found.")
            return
        
        for file_meta in files:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{file_meta.original_filename}**")
                    st.caption(f"Size: {self._format_size(file_meta.file_size_bytes)} | Type: {file_meta.content_type}")
                    if file_meta.description:
                        st.caption(file_meta.description)
                    if file_meta.tags:
                        st.caption(f"Tags: {', '.join(file_meta.tags)}")
                
                with col2:
                    st.caption(f"Uploaded: {file_meta.created_at}")
                    if file_meta.uploaded_by:
                        st.caption(f"By: {file_meta.uploaded_by}")
                    if file_meta.project_name:
                        st.caption(f"Project: {file_meta.project_name}")
                
                with col3:
                    # Download button
                    try:
                        file_bytes = self.file_manager.download_file(file_meta.storage_key)
                        st.download_button(
                            label=" Download",
                            data=file_bytes,
                            file_name=file_meta.original_filename,
                            mime=file_meta.content_type,
                            key=f"download_{file_meta.storage_key}"
                        )
                    except Exception as e:
                        st.error(f"Download failed: {e}")
                
                st.divider()
    
    def display_usage_stats(self):
        """
        Display storage usage statistics in a user-friendly format.
        """
        stats = self.file_manager.get_usage_stats()
        
        # Storage usage
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Storage Used",
                value=f"{stats['total_storage_mb']:.1f} MB / {stats['max_storage_mb']} MB",
                delta=f"{stats['storage_usage_percent']}%",
                delta_color="normal" if stats['storage_usage_percent'] < 50 else "inverse"
            )
            
            # Storage progress bar
            storage_progress = stats['total_storage_mb'] / stats['max_storage_mb']
            st.progress(storage_progress)
            
            if stats['approaching_storage_limit']:
                st.warning(f" Approaching storage limit! {stats['storage_usage_percent']}% used.")
        
        with col2:
            st.metric(
                label="Files",
                value=f"{stats['total_files']} / {stats['max_files']}",
                delta=f"{stats['files_usage_percent']}%",
                delta_color="normal" if stats['files_usage_percent'] < 50 else "inverse"
            )
            
            # Files progress bar
            files_progress = stats['total_files'] / stats['max_files']
            st.progress(files_progress)
            
            if stats['approaching_file_limit']:
                st.warning(f" Approaching file limit! {stats['files_usage_percent']}% used.")
    
    def display_limit_info(self):
        """
        Display information about current limits.
        """
        with st.expander(" Storage Limits (Free Tier)", expanded=False):
            st.write("""
            #### Current Limits:
            - **Max file size:** 20 MB per file
            - **Max total storage:** 500 MB
            - **Max files:** 100 total
            - **Max files per user:** 50
            
            These limits ensure the application stays within the free tier of Cloudflare R2.
            """)
            
            # Show current usage
            stats = self.file_manager.get_usage_stats()
            st.write("**Current Usage:**")
            st.write(f"- Files: {stats['total_files']} / {stats['max_files']}")
            st.write(f"- Storage: {stats['total_storage_mb']:.1f} MB / {stats['max_storage_mb']} MB")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def handle_upload_error(self, error: Exception) -> str:
        """
        Handle upload errors and return user-friendly message.
        
        Args:
            error: The exception that occurred
        
        Returns:
            User-friendly error message
        """
        if isinstance(error, FileSizeLimitExceeded):
            return f" File too large! Maximum file size is {StorageLimits.MAX_FILE_SIZE_MB} MB. Your file is {error.current}."
        
        elif isinstance(error, TotalStorageLimitExceeded):
            return f" Storage limit reached! Upload would exceed the {StorageLimits.MAX_TOTAL_STORAGE_MB} MB limit. Current usage: {error.current}."
        
        elif isinstance(error, FileCountLimitExceeded):
            return f" File limit reached! Maximum {StorageLimits.MAX_FILES} files allowed. You have {error.current} files."
        
        elif isinstance(error, UserFileCountLimitExceeded):
            return f" Per-user file limit reached! Maximum {StorageLimits.MAX_FILES_PER_USER} files per user."
        
        elif isinstance(error, DuplicateFileError):
            return f" Duplicate file detected! A file with the same content already exists."
        
        elif isinstance(error, ProductionConfigError):
            return f" Configuration error! Storage is not properly configured for production. Please contact the administrator."
        
        else:
            return f" Upload failed: {str(error)}"


# Global file manager instance
_file_manager = None
_wharton_vault = None


def get_file_manager() -> FileManager:
    """Get the global FileManager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager


def get_wharton_vault() -> WhartonFileVault:
    """Get the global WhartonFileVault instance."""
    global _wharton_vault
    if _wharton_vault is None:
        _wharton_vault = WhartonFileVault()
    return _wharton_vault


def initialize_file_manager() -> Dict[str, Any]:
    """
    Initialize the global file manager.
    
    Returns:
        Initialization status
    """
    fm = get_file_manager()
    return fm.initialize()