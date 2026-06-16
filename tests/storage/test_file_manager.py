"""
Tests for file manager module.
"""

import io
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.storage.file_manager import (
    FileManager,
    WhartonFileVault,
    get_file_manager,
    get_wharton_vault,
    initialize_file_manager,
)
from src.storage.backend import StorageMetadata, LocalStorageBackend
from src.storage.exceptions import StorageError, FileNotFound


class TestFileManager:
    """Test FileManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create a FileManager instance with local backend."""
        backend = LocalStorageBackend(base_path=temp_dir)
        return FileManager(backend=backend)
    
    def test_initialization_with_backend(self, file_manager):
        """Test FileManager initialization with provided backend."""
        assert file_manager._initialized is False
        assert file_manager._backend is not None
        
        # Backend should be initialized on first access
        backend = file_manager.backend
        assert file_manager._initialized is True
        assert isinstance(backend, LocalStorageBackend)
    
    def test_initialize_method(self, file_manager):
        """Test explicit initialization."""
        result = file_manager.initialize()
        
        assert result["success"] is True
        assert result["backend"] == "local"
        assert file_manager._initialized is True
    
    def test_upload_file(self, file_manager):
        """Test uploading a file."""
        file_data = b"Hello, World!"
        filename = "test.txt"
        content_type = "text/plain"
        
        metadata = file_manager.upload_file(
            file_data=file_data,
            filename=filename,
            content_type=content_type,
            uploaded_by="test_user"
        )
        
        assert isinstance(metadata, StorageMetadata)
        assert metadata.storage_backend == "local"
        assert metadata.original_filename == filename
        assert metadata.content_type == content_type
        assert metadata.file_size_bytes == len(file_data)
        assert metadata.uploaded_by == "test_user"
    
    def test_download_file(self, file_manager):
        """Test downloading a file."""
        file_data = b"Test content"
        filename = "test.txt"
        
        # Upload first
        metadata = file_manager.upload_file(file_data, filename, "text/plain")
        
        # Download
        downloaded = file_manager.download_file(metadata.storage_key)
        assert downloaded == file_data
    
    def test_download_file_not_found(self, file_manager):
        """Test downloading a non-existent file."""
        with pytest.raises(FileNotFound):
            file_manager.download_file("nonexistent_key")
    
    def test_get_metadata(self, file_manager):
        """Test getting file metadata."""
        file_data = b"Test data"
        metadata = file_manager.upload_file(file_data, "test.txt", "text/plain")
        
        retrieved = file_manager.get_metadata(metadata.storage_key)
        
        assert retrieved is not None
        assert retrieved.storage_key == metadata.storage_key
        assert retrieved.original_filename == metadata.original_filename
    
    def test_get_metadata_not_found(self, file_manager):
        """Test getting metadata for non-existent file."""
        result = file_manager.get_metadata("nonexistent_key")
        assert result is None
    
    def test_delete_file(self, file_manager):
        """Test deleting a file."""
        file_data = b"Test data"
        metadata = file_manager.upload_file(file_data, "test.txt", "text/plain")
        
        # Verify file exists
        assert file_manager.file_exists(metadata.storage_key)
        
        # Delete
        result = file_manager.delete_file(metadata.storage_key)
        assert result is True
        
        # Verify file is deleted
        assert not file_manager.file_exists(metadata.storage_key)
    
    def test_list_files(self, file_manager):
        """Test listing files."""
        # Upload multiple files
        file_manager.upload_file(b"data1", "file1.txt", "text/plain")
        file_manager.upload_file(b"data2", "file2.txt", "text/plain")
        file_manager.upload_file(b"data3", "file3.txt", "text/plain")
        
        files = file_manager.list_files()
        assert len(files) == 3
    
    def test_list_files_with_prefix(self, file_manager):
        """Test listing files with prefix."""
        file_manager.upload_file(b"data1", "prefix_file1.txt", "text/plain")
        file_manager.upload_file(b"data2", "prefix_file2.txt", "text/plain")
        file_manager.upload_file(b"data3", "other_file.txt", "text/plain")
        
        files = file_manager.list_files(prefix="prefix")
        assert len(files) == 2
        
        for f in files:
            assert "prefix" in f.original_filename
    
    def test_file_exists(self, file_manager):
        """Test checking if file exists."""
        metadata = file_manager.upload_file(b"data", "test.txt", "text/plain")
        
        assert file_manager.file_exists(metadata.storage_key)
        assert not file_manager.file_exists("nonexistent_key")
    
    def test_health_check(self, file_manager):
        """Test health check."""
        health = file_manager.health_check()
        
        assert health["status"] == "healthy"
        assert health["backend"] == "local"
    
    def test_backend_name_property(self, file_manager):
        """Test backend name property."""
        assert file_manager.backend_name == "not_initialized"
        
        # Access backend to initialize
        _ = file_manager.backend
        
        assert file_manager.backend_name == "local"
    
    def test_upload_with_extra_metadata(self, file_manager):
        """Test uploading with additional metadata."""
        file_data = b"Test data"
        metadata = file_manager.upload_file(
            file_data=file_data,
            filename="test.txt",
            content_type="text/plain",
            uploaded_by="user1",
            custom_field="custom_value",
            another_field=123
        )
        
        # Download and verify metadata includes custom fields
        retrieved = file_manager.get_metadata(metadata.storage_key)
        assert retrieved is not None
        assert retrieved.extra.get("custom_field") == "custom_value"
        assert retrieved.extra.get("another_field") == 123


class TestWhartonFileVault:
    """Test WhartonFileVault class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def vault(self, temp_dir):
        """Create a WhartonFileVault instance."""
        backend = LocalStorageBackend(base_path=temp_dir)
        file_manager = FileManager(backend=backend)
        return WhartonFileVault(file_manager=file_manager)
    
    def test_initialization(self, vault):
        """Test WhartonFileVault initialization."""
        assert vault.file_manager is not None
        assert isinstance(vault.file_manager, FileManager)
    
    def test_initialize(self, vault):
        """Test initializing the vault."""
        result = vault.initialize()
        
        assert result["success"] is True
        assert result["backend"] == "local"
    
    @patch('streamlit.file_uploader')
    def test_upload_from_streamlit(self, mock_uploader, vault):
        """Test uploading from Streamlit file uploader."""
        # Mock uploaded file
        mock_file = Mock()
        mock_file.getvalue.return_value = b"File content"
        mock_file.name = "uploaded.txt"
        mock_file.type = "text/plain"
        
        metadata = vault.upload_from_streamlit(mock_file, uploaded_by="user1")
        
        assert isinstance(metadata, StorageMetadata)
        assert metadata.original_filename == "uploaded.txt"
        assert metadata.content_type == "text/plain"
        assert metadata.uploaded_by == "user1"
    
    def test_download_to_streamlit(self, vault):
        """Test downloading to Streamlit BytesIO."""
        file_data = b"Test content"
        metadata = vault.file_manager.upload_file(file_data, "test.txt", "text/plain")
        
        bytes_io = vault.download_to_streamlit(metadata.storage_key)
        
        assert isinstance(bytes_io, io.BytesIO)
        assert bytes_io.getvalue() == file_data
    
    def test_get_user_files(self, vault):
        """Test getting files uploaded by a specific user."""
        # Upload files from different users
        vault.file_manager.upload_file(b"data1", "file1.txt", "text/plain", uploaded_by="user1")
        vault.file_manager.upload_file(b"data2", "file2.txt", "text/plain", uploaded_by="user2")
        vault.file_manager.upload_file(b"data3", "file3.txt", "text/plain", uploaded_by="user1")
        
        user1_files = vault.get_user_files("user1")
        assert len(user1_files) == 2
        
        user2_files = vault.get_user_files("user2")
        assert len(user2_files) == 1
    
    def test_format_size(self, vault):
        """Test file size formatting."""
        assert vault._format_size(500) == "500.0 B"
        assert vault._format_size(1024) == "1.0 KB"
        assert vault._format_size(1048576) == "1.0 MB"
        assert vault._format_size(1073741824) == "1.0 GB"


class TestGlobalInstances:
    """Test global file manager instances."""
    
    def test_get_file_manager_creates_instance(self):
        """Test that get_file_manager creates an instance if none exists."""
        from src.storage import file_manager as fm_module
        
        # Reset global instance
        original = fm_module._file_manager
        fm_module._file_manager = None
        
        try:
            manager = get_file_manager()
            assert manager is not None
            assert isinstance(manager, FileManager)
            
            # Second call should return same instance
            manager2 = get_file_manager()
            assert manager2 is manager
        finally:
            fm_module._file_manager = original
    
    def test_get_wharton_vault_creates_instance(self):
        """Test that get_wharton_vault creates an instance if none exists."""
        from src.storage import file_manager as fm_module
        
        # Reset global instance
        original = fm_module._wharton_vault
        fm_module._wharton_vault = None
        
        try:
            vault = get_wharton_vault()
            assert vault is not None
            assert isinstance(vault, WhartonFileVault)
            
            # Second call should return same instance
            vault2 = get_wharton_vault()
            assert vault2 is vault
        finally:
            fm_module._wharton_vault = original


class TestInitializeFileManager:
    """Test initialize_file_manager function."""
    
    @patch('src.storage.file_manager.get_file_manager')
    def test_initialize_file_manager_success(self, mock_get_manager):
        """Test successful file manager initialization."""
        mock_manager = Mock()
        mock_manager.initialize.return_value = {
            "success": True,
            "backend": "local",
            "health": {"status": "healthy"}
        }
        mock_get_manager.return_value = mock_manager
        
        result = initialize_file_manager()
        
        assert result["success"] is True
        assert result["backend"] == "local"
        mock_manager.initialize.assert_called_once()
    
    @patch('src.storage.file_manager.get_file_manager')
    def test_initialize_file_manager_failure(self, mock_get_manager):
        """Test file manager initialization failure."""
        mock_manager = Mock()
        mock_manager.initialize.return_value = {
            "success": False,
            "error": "Configuration error"
        }
        mock_get_manager.return_value = mock_manager
        
        with pytest.raises(StorageError):
            initialize_file_manager()