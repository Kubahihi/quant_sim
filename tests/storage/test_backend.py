"""
Tests for storage backend module.

Includes tests for:
- StorageMetadata
- LocalStorageBackend
- R2StorageBackend (mocked)
- StorageConfig
- StorageLimits
- SHA256 deduplication
- Hard limits enforcement
"""

import json
import os
import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.storage.backend import (
    StorageMetadata,
    StorageLimits,
    LocalStorageBackend,
    StorageConfig,
)
from src.storage.exceptions import (
    StorageError,
    FileSizeLimitExceeded,
    TotalStorageLimitExceeded,
    FileCountLimitExceeded,
    UserFileCountLimitExceeded,
    DuplicateFileError,
    ProductionConfigError,
)


class TestStorageLimits:
    """Test StorageLimits class."""
    
    def test_max_file_size_bytes(self):
        """Test max file size conversion to bytes."""
        assert StorageLimits.max_file_size_bytes() == 20 * 1024 * 1024
    
    def test_max_total_storage_bytes(self):
        """Test max total storage conversion to bytes."""
        assert StorageLimits.max_total_storage_bytes() == 500 * 1024 * 1024
    
    def test_limit_values(self):
        """Test that limit values are set correctly."""
        assert StorageLimits.MAX_FILE_SIZE_MB == 20
        assert StorageLimits.MAX_TOTAL_STORAGE_MB == 500
        assert StorageLimits.MAX_FILES == 100
        assert StorageLimits.MAX_FILES_PER_USER == 50
    
    def test_warning_threshold(self):
        """Test warning threshold value."""
        assert StorageLimits.get_usage_warning_threshold() == 0.8


class TestStorageMetadata:
    """Test StorageMetadata class."""
    
    def test_create_metadata(self):
        """Test creating metadata with all fields."""
        meta = StorageMetadata(
            storage_backend="r2",
            storage_key="test_key",
            original_filename="test.txt",
            content_type="text/plain",
            file_size_bytes=1024,
            sha256="abc123",
            created_at="2024-01-01T00:00:00",
            uploaded_by="user1",
            project_name="test_project",
            description="Test file",
            tags=["test", "sample"]
        )
        
        assert meta.storage_backend == "r2"
        assert meta.storage_key == "test_key"
        assert meta.original_filename == "test.txt"
        assert meta.content_type == "text/plain"
        assert meta.file_size_bytes == 1024
        assert meta.sha256 == "abc123"
        assert meta.created_at == "2024-01-01T00:00:00"
        assert meta.uploaded_by == "user1"
        assert meta.project_name == "test_project"
        assert meta.description == "Test file"
        assert meta.tags == ["test", "sample"]
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        meta = StorageMetadata(
            storage_backend="local",
            storage_key="key1",
            original_filename="file.pdf",
            content_type="application/pdf",
            file_size_bytes=2048,
            sha256="def456",
            created_at="2024-01-02T12:00:00",
            uploaded_by="user2",
            project_name="proj1",
            tags=["tag1", "tag2"]
        )
        
        d = meta.to_dict()
        
        assert d["storage_backend"] == "local"
        assert d["storage_key"] == "key1"
        assert d["original_filename"] == "file.pdf"
        assert d["content_type"] == "application/pdf"
        assert d["file_size_bytes"] == 2048
        assert d["sha256"] == "def456"
        assert d["created_at"] == "2024-01-02T12:00:00"
        assert d["uploaded_by"] == "user2"
        assert d["project_name"] == "proj1"
        assert d["tags"] == ["tag1", "tag2"]
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "storage_backend": "r2",
            "storage_key": "key2",
            "original_filename": "doc.docx",
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "file_size_bytes": 4096,
            "sha256": "ghi789",
            "created_at": "2024-01-03T18:30:00",
            "uploaded_by": "user3",
            "project_name": "proj2",
            "description": "A test document",
            "tags": ["doc", "test"],
            "custom_field": "custom_value"
        }
        
        meta = StorageMetadata.from_dict(data)
        
        assert meta.storage_backend == "r2"
        assert meta.storage_key == "key2"
        assert meta.original_filename == "doc.docx"
        assert meta.uploaded_by == "user3"
        assert meta.project_name == "proj2"
        assert meta.description == "A test document"
        assert meta.tags == ["doc", "test"]
        assert meta.extra["custom_field"] == "custom_value"
    
    def test_metadata_size_mb_property(self):
        """Test size_mb property."""
        meta = StorageMetadata(
            storage_backend="local",
            storage_key="test",
            original_filename="test.txt",
            content_type="text/plain",
            file_size_bytes=1048576,  # 1 MB
            sha256="hash",
            created_at="2024-01-01"
        )
        
        assert meta.size_mb == 1.0
    
    def test_metadata_repr(self):
        """Test metadata string representation."""
        meta = StorageMetadata(
            storage_backend="local",
            storage_key="test123",
            original_filename="report.pdf",
            content_type="application/pdf",
            file_size_bytes=1000,
            sha256="hash",
            created_at="2024-01-01"
        )
        
        repr_str = repr(meta)
        assert "test123" in repr_str
        assert "report.pdf" in repr_str


class TestLocalStorageBackend:
    """Test LocalStorageBackend class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def backend(self, temp_dir):
        """Create a LocalStorageBackend instance."""
        return LocalStorageBackend(base_path=temp_dir)
    
    def test_backend_name(self, backend):
        """Test backend name property."""
        assert backend.backend_name == "local"
    
    def test_upload_and_download(self, backend):
        """Test uploading and downloading files."""
        file_data = b"Hello, World!"
        filename = "test.txt"
        content_type = "text/plain"
        
        # Upload
        metadata = backend.upload(file_data, filename, content_type)
        
        assert metadata.storage_backend == "local"
        assert metadata.original_filename == filename
        assert metadata.content_type == content_type
        assert metadata.file_size_bytes == len(file_data)
        assert metadata.sha256 == hashlib.sha256(file_data).hexdigest()
        
        # Download
        downloaded = backend.download(metadata.storage_key)
        assert downloaded == file_data
    
    def test_exists(self, backend):
        """Test checking if file exists."""
        file_data = b"Test data"
        metadata = backend.upload(file_data, "test.txt", "text/plain")
        
        assert backend.exists(metadata.storage_key)
        assert not backend.exists("nonexistent_key")
    
    def test_delete(self, backend):
        """Test deleting files."""
        file_data = b"Test data"
        metadata = backend.upload(file_data, "test.txt", "text/plain")
        
        # Verify file exists
        assert backend.exists(metadata.storage_key)
        
        # Delete
        result = backend.delete(metadata.storage_key)
        assert result is True
        
        # Verify file is deleted
        assert not backend.exists(metadata.storage_key)
    
    def test_list_files(self, backend):
        """Test listing files."""
        # Upload multiple files
        backend.upload(b"data1", "file1.txt", "text/plain")
        backend.upload(b"data2", "file2.txt", "text/plain")
        backend.upload(b"data3", "file3.txt", "text/plain")
        
        files = backend.list_files()
        assert len(files) == 3
        
        # Check that all files have required metadata
        for f in files:
            assert f.storage_key is not None
            assert f.original_filename is not None
            assert f.content_type is not None
            assert f.file_size_bytes > 0
            assert f.sha256 is not None
            assert f.created_at is not None
    
    def test_list_files_with_prefix(self, backend):
        """Test listing files with prefix filter."""
        backend.upload(b"data1", "prefix_file1.txt", "text/plain")
        backend.upload(b"data2", "prefix_file2.txt", "text/plain")
        backend.upload(b"data3", "other_file.txt", "text/plain")
        
        files = backend.list_files(prefix="prefix")
        assert len(files) == 2
        
        for f in files:
            assert "prefix" in f.original_filename
    
    def test_health_check(self, backend):
        """Test health check."""
        health = backend.health_check()
        
        assert health["status"] == "healthy"
        assert health["backend"] == "local"
        assert "path" in health
        assert health["writable"] is True
        assert health["readable"] is True
    
    def test_metadata_persistence(self, backend):
        """Test that metadata is persisted to disk."""
        file_data = b"Test data"
        metadata = backend.upload(file_data, "test.txt", "text/plain")
        
        # Check that metadata file exists
        meta_path = Path(backend.base_path) / f"{metadata.storage_key}.meta.json"
        assert meta_path.exists()
        
        # Read and verify metadata
        saved_meta = json.loads(meta_path.read_text())
        assert saved_meta["storage_key"] == metadata.storage_key
        assert saved_meta["original_filename"] == "test.txt"
    
    def test_get_total_storage_used(self, backend):
        """Test getting total storage used."""
        backend.upload(b"12345", "file1.txt", "text/plain")
        backend.upload(b"1234567890", "file2.txt", "text/plain")
        
        total = backend.get_total_storage_used()
        assert total == 5 + 10  # 15 bytes
    
    def test_get_file_count(self, backend):
        """Test getting file count."""
        backend.upload(b"data1", "file1.txt", "text/plain")
        backend.upload(b"data2", "file2.txt", "text/plain")
        
        assert backend.get_file_count() == 2
    
    def test_get_user_file_count(self, backend):
        """Test getting user file count."""
        backend.upload(b"data1", "file1.txt", "text/plain", {"uploaded_by": "user1"})
        backend.upload(b"data2", "file2.txt", "text/plain", {"uploaded_by": "user1"})
        backend.upload(b"data3", "file3.txt", "text/plain", {"uploaded_by": "user2"})
        
        assert backend.get_user_file_count("user1") == 2
        assert backend.get_user_file_count("user2") == 1
        assert backend.get_user_file_count("user3") == 0
    
    def test_find_by_sha256(self, backend):
        """Test finding file by SHA256."""
        file_data = b"Test data for SHA256"
        sha256 = hashlib.sha256(file_data).hexdigest()
        
        metadata = backend.upload(file_data, "test.txt", "text/plain")
        
        found = backend.find_by_sha256(sha256)
        assert found is not None
        assert found.storage_key == metadata.storage_key
        
        # Non-existent hash
        found = backend.find_by_sha256("nonexistent")
        assert found is None
    
    def test_get_usage_stats(self, backend):
        """Test getting usage statistics."""
        backend.upload(b"12345", "file1.txt", "text/plain")
        backend.upload(b"1234567890", "file2.txt", "text/plain")
        
        stats = backend.get_usage_stats()
        
        assert stats["total_files"] == 2
        assert stats["total_storage_bytes"] == 15
        assert stats["max_files"] == StorageLimits.MAX_FILES
        assert stats["max_storage_mb"] == StorageLimits.MAX_TOTAL_STORAGE_MB
        assert isinstance(stats["storage_usage_percent"], float)
        assert isinstance(stats["files_usage_percent"], float)
    
    def test_validate_upload_file_size_limit(self, backend):
        """Test that file size limit is enforced."""
        # Create data larger than max (20 MB)
        large_data = b"x" * (StorageLimits.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        
        with pytest.raises(FileSizeLimitExceeded):
            backend.validate_upload(large_data)
    
    def test_validate_upload_duplicate(self, backend):
        """Test that duplicate files are detected."""
        file_data = b"Duplicate content"
        backend.upload(file_data, "original.txt", "text/plain")
        
        # Try to upload same content
        with pytest.raises(DuplicateFileError):
            backend.validate_upload(file_data, check_duplicate=True)
    
    def test_validate_upload_file_count_limit(self, backend):
        """Test that file count limit is enforced."""
        # Upload max files
        for i in range(StorageLimits.MAX_FILES):
            backend.upload(f"data{i}".encode(), f"file{i}.txt", "text/plain")
        
        # Try to upload one more
        with pytest.raises(FileCountLimitExceeded):
            backend.validate_upload(b"new file")
    
    def test_validate_upload_user_file_count_limit(self, backend):
        """Test that per-user file count limit is enforced."""
        user_id = "test_user"
        
        # Upload max files for user
        for i in range(StorageLimits.MAX_FILES_PER_USER):
            backend.upload(
                f"data{i}".encode(), 
                f"file{i}.txt", 
                "text/plain",
                {"uploaded_by": user_id}
            )
        
        # Try to upload one more for same user
        with pytest.raises(UserFileCountLimitExceeded):
            backend.validate_upload(b"new file", uploaded_by=user_id)
    
    def test_validate_upload_success(self, backend):
        """Test successful validation."""
        file_data = b"Valid file content"
        
        is_valid, error_type, error_msg = backend.validate_upload(file_data)
        
        assert is_valid is True
        assert error_type is None
        assert error_msg is None


class TestStorageConfig:
    """Test StorageConfig class."""
    
    def test_load_from_secrets_missing(self):
        """Test loading config when secrets are missing."""
        with patch('src.storage.backend.st.secrets', {}):
            config = StorageConfig()
            result = config.load_from_secrets()
            assert result is False
            assert config._config is None
    
    def test_load_from_secrets_success(self):
        """Test loading config from secrets."""
        mock_secrets = {
            'storage': {
                'STORAGE_BACKEND': 'r2',
                'R2_BUCKET': 'test-bucket',
                'R2_ENDPOINT_URL': 'https://test.r2.cloudflarestorage.com',
                'R2_ACCESS_KEY_ID': 'test-key',
                'R2_SECRET_ACCESS_KEY': 'test-secret',
                'R2_REGION': 'auto'
            }
        }
        
        with patch('src.storage.backend.st.secrets', mock_secrets):
            config = StorageConfig()
            result = config.load_from_secrets()
            
            assert result is True
            assert config._config is not None
            assert config._config['backend'] == 'r2'
            assert config._config['r2_bucket'] == 'test-bucket'
            assert config._config['r2_endpoint_url'] == 'https://test.r2.cloudflarestorage.com'
            assert config._config['r2_access_key_id'] == 'test-key'
            assert config._config['r2_secret_access_key'] == 'test-secret'
            assert config._config['r2_region'] == 'auto'
    
    def test_validate_r2_config_complete(self):
        """Test validating complete R2 configuration."""
        config = StorageConfig()
        config._config = {
            'backend': 'r2',
            'r2_bucket': 'test-bucket',
            'r2_endpoint_url': 'https://test.r2.cloudflarestorage.com',
            'r2_access_key_id': 'test-key',
            'r2_secret_access_key': 'test-secret',
            'r2_region': 'auto'
        }
        
        missing = config.validate_r2_config()
        assert len(missing) == 0
    
    def test_validate_r2_config_incomplete(self):
        """Test validating incomplete R2 configuration."""
        config = StorageConfig()
        config._config = {
            'backend': 'r2',
            'r2_bucket': 'test-bucket',
            # Missing other required fields
        }
        
        missing = config.validate_r2_config()
        assert len(missing) == 3  # Missing endpoint, access key, secret key
        assert 'r2_endpoint_url' in missing
        assert 'r2_access_key_id' in missing
        assert 'r2_secret_access_key' in missing
    
    def test_is_production_mode(self):
        """Test production mode detection."""
        config = StorageConfig()
        
        # Without STREAMLIT_SERVER_PORT
        with patch.dict(os.environ, {}, clear=True):
            assert config.is_production_mode() is False
        
        # With STREAMLIT_SERVER_PORT
        with patch.dict(os.environ, {'STREAMLIT_SERVER_PORT': '8501'}):
            assert config.is_production_mode() is True
    
    def test_create_backend_local(self):
        """Test creating local backend."""
        config = StorageConfig()
        config._config = {'backend': 'local'}
        
        backend = config.create_backend()
        assert isinstance(backend, LocalStorageBackend)
        assert backend.backend_name == 'local'
    
    def test_create_backend_r2_missing_secrets_in_production(self):
        """Test that R2 backend raises error in production with missing secrets."""
        config = StorageConfig()
        config._config = {
            'backend': 'r2',
            'r2_bucket': 'test-bucket'
            # Missing other required fields
        }
        
        with patch.dict(os.environ, {'STREAMLIT_SERVER_PORT': '8501'}):
            with pytest.raises(ProductionConfigError) as excinfo:
                config.create_backend()
            
            assert 'missing secrets' in str(excinfo.value).lower()
    
    def test_create_backend_r2_missing_secrets_in_development(self):
        """Test that R2 backend falls back to local in development with missing secrets."""
        config = StorageConfig()
        config._config = {
            'backend': 'r2',
            'r2_bucket': 'test-bucket'
            # Missing other required fields
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('src.storage.backend.st.warning'):
                backend = config.create_backend()
                assert isinstance(backend, LocalStorageBackend)
    
    @patch('src.storage.backend.R2StorageBackend')
    def test_create_backend_r2_complete_config(self, mock_r2):
        """Test creating R2 backend with complete configuration."""
        mock_r2.return_value = Mock(backend_name='r2')
        
        config = StorageConfig()
        config._config = {
            'backend': 'r2',
            'r2_bucket': 'test-bucket',
            'r2_endpoint_url': 'https://test.r2.cloudflarestorage.com',
            'r2_access_key_id': 'test-key',
            'r2_secret_access_key': 'test-secret',
            'r2_region': 'auto'
        }
        
        backend = config.create_backend()
        
        mock_r2.assert_called_once_with(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='auto'
        )
    
    def test_backend_property_caches(self):
        """Test that backend property caches the backend instance."""
        config = StorageConfig()
        config._config = {'backend': 'local'}
        
        backend1 = config.backend
        backend2 = config.backend
        
        assert backend1 is backend2  # Same instance


class TestInitializeStorage:
    """Test storage initialization functions."""
    
    @patch('src.storage.backend.StorageConfig')
    def test_initialize_storage_success(self, mock_config_class):
        """Test successful storage initialization."""
        mock_config = Mock()
        mock_config.load_from_secrets.return_value = True
        mock_config.backend = Mock()
        mock_config.backend.backend_name = 'local'
        mock_config.backend.health_check.return_value = {'status': 'healthy'}
        mock_config.backend.get_usage_stats.return_value = {
            'total_files': 0,
            'total_storage_mb': 0,
            'storage_usage_percent': 0,
            'files_usage_percent': 0,
        }
        mock_config.is_production_mode.return_value = False
        
        mock_config_class.return_value = mock_config
        
        from src.storage.backend import storage_config, initialize_storage
        
        # Replace the global config
        original_config = storage_config
        import src.storage.backend as backend_module
        backend_module.storage_config = mock_config
        
        try:
            result = initialize_storage()
            
            assert result['success'] is True
            assert result['backend'] == 'local'
            assert result['config_loaded'] is True
            assert result['health']['status'] == 'healthy'
            assert result['production_mode'] is False
        finally:
            backend_module.storage_config = original_config


class TestR2StorageBackend:
    """Test R2StorageBackend class."""
    
    @patch('boto3.client')
    def test_r2_backend_initialization(self, mock_boto3_client):
        """Test R2 backend initialization."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='auto'
        )
        
        assert backend.backend_name == 'r2'
        assert backend.bucket == 'test-bucket'
        assert backend.endpoint_url == 'https://test.r2.cloudflarestorage.com'
        assert backend.region == 'auto'
        
        mock_boto3_client.assert_called_once_with(
            's3',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            aws_access_key_id='test-key',
            aws_secret_access_key='test-secret',
            region_name='auto'
        )
    
    @patch('boto3.client')
    def test_r2_upload(self, mock_boto3_client):
        """Test R2 upload."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        file_data = b"Test file content"
        metadata = backend.upload(file_data, "test.txt", "text/plain")
        
        # Verify S3 put_object was called
        assert mock_client.put_object.call_count == 2  # File + metadata
        
        # Verify metadata
        assert metadata.storage_backend == 'r2'
        assert metadata.original_filename == 'test.txt'
        assert metadata.content_type == 'text/plain'
        assert metadata.file_size_bytes == len(file_data)
        assert metadata.sha256 == hashlib.sha256(file_data).hexdigest()
    
    @patch('boto3.client')
    def test_r2_download(self, mock_boto3_client):
        """Test R2 download."""
        mock_client = Mock()
        mock_client.get_object.return_value = {'Body': MagicMock(read=lambda: b"file content")}
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        content = backend.download('test-key')
        assert content == b"file content"
        
        mock_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-key'
        )
    
    @patch('boto3.client')
    def test_r2_download_to_buffer(self, mock_boto3_client):
        """Test R2 download to buffer."""
        mock_client = Mock()
        mock_client.get_object.return_value = {'Body': MagicMock(read=lambda: b"file content")}
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        buffer = backend.download_to_buffer('test-key')
        assert buffer.read() == b"file content"
    
    @patch('boto3.client')
    def test_r2_download_not_found(self, mock_boto3_client):
        """Test R2 download when file not found."""
        from src.storage.exceptions import FileNotFound
        
        mock_client = Mock()
        mock_client.get_object.side_effect = Exception('404 Not Found')
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        with pytest.raises(FileNotFound):
            backend.download('nonexistent-key')
    
    @patch('boto3.client')
    def test_r2_exists(self, mock_boto3_client):
        """Test checking if file exists in R2."""
        mock_client = Mock()
        mock_client.head_object.return_value = {}
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        assert backend.exists('test-key') is True
        
        mock_client.head_object.side_effect = Exception('404')
        assert backend.exists('nonexistent-key') is False
    
    @patch('boto3.client')
    def test_r2_delete(self, mock_boto3_client):
        """Test deleting file from R2."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        result = backend.delete('test-key')
        assert result is True
        
        # Verify delete_object was called twice (file + metadata)
        assert mock_client.delete_object.call_count == 2
    
    @patch('boto3.client')
    def test_r2_health_check_success(self, mock_boto3_client):
        """Test successful R2 health check."""
        mock_client = Mock()
        # Configure mock to return proper responses for health check sequence
        mock_client.head_bucket.return_value = {}
        mock_client.put_object.return_value = {}
        mock_client.get_object.return_value = {'Body': MagicMock(read=lambda: b"health check")}
        mock_client.delete_object.return_value = {}
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        health = backend.health_check()
        
        assert health['status'] == 'healthy'
        assert health['backend'] == 'r2'
        assert health['bucket'] == 'test-bucket'
        assert health['endpoint'] == 'https://test.r2.cloudflarestorage.com'
        assert health['region'] == 'auto'
    
    @patch('boto3.client')
    def test_r2_health_check_failure(self, mock_boto3_client):
        """Test failed R2 health check."""
        mock_client = Mock()
        mock_client.head_bucket.side_effect = Exception('Access Denied')
        mock_boto3_client.return_value = mock_client
        
        from src.storage.backend import R2StorageBackend
        
        backend = R2StorageBackend(
            bucket='test-bucket',
            endpoint_url='https://test.r2.cloudflarestorage.com',
            access_key_id='test-key',
            secret_access_key='test-secret'
        )
        
        health = backend.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['backend'] == 'r2'
        assert 'error' in health


class TestTotalStorageLimitExceeded:
    """Test TotalStorageLimitExceeded exception."""
    
    @pytest.fixture
    def backend_with_files(self):
        """Create a backend with files close to the limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(base_path=tmpdir)
            
            # Calculate size to get close to limit (but under)
            # Use a smaller limit for testing
            original_limit = StorageLimits.MAX_TOTAL_STORAGE_MB
            
            # We'll test by mocking the limit
            yield backend
    
    def test_total_storage_limit_validation(self):
        """Test that total storage limit is enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalStorageBackend(base_path=tmpdir)
            
            # Mock the limit to a small value for testing
            original_limit = StorageLimits.MAX_TOTAL_STORAGE_MB
            StorageLimits.MAX_TOTAL_STORAGE_MB = 1  # 1 MB limit
            
            try:
                # Upload a file that's under the limit
                small_file = b"x" * 100  # 100 bytes
                backend.upload(small_file, "small.txt", "text/plain")
                
                # Now try to upload a file that would exceed the limit
                large_file = b"x" * (1024 * 1024 + 100)  # Just over 1 MB total
                
                with pytest.raises(TotalStorageLimitExceeded):
                    backend.validate_upload(large_file)
            finally:
                StorageLimits.MAX_TOTAL_STORAGE_MB = original_limit