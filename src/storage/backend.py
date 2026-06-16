"""
Storage backend implementations for production deployment.

Supports:
- LocalStorageBackend: For local development only
- R2StorageBackend: For production on Streamlit Cloud with Cloudflare R2

Hard Limits (Free Tier Protection):
- MAX_FILE_SIZE_MB = 20
- MAX_TOTAL_STORAGE_MB = 500
- MAX_FILES = 100
- MAX_FILES_PER_USER = 50 (optional)
"""

import hashlib
import io
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from .exceptions import (
    StorageError,
    FileNotFound,
    ConfigurationError,
    ProductionConfigError,
    FileSizeLimitExceeded,
    TotalStorageLimitExceeded,
    FileCountLimitExceeded,
    UserFileCountLimitExceeded,
    DuplicateFileError,
)


# ============================================
# HARD LIMITS - Free Tier Protection
# ============================================
class StorageLimits:
    """Hard limits to prevent exceeding free tier."""
    
    MAX_FILE_SIZE_MB: int = 20
    MAX_TOTAL_STORAGE_MB: int = 500
    MAX_FILES: int = 100
    MAX_FILES_PER_USER: int = 50
    
    @classmethod
    def max_file_size_bytes(cls) -> int:
        return cls.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @classmethod
    def max_total_storage_bytes(cls) -> int:
        return cls.MAX_TOTAL_STORAGE_MB * 1024 * 1024
    
    @classmethod
    def get_usage_warning_threshold(cls) -> float:
        """Return threshold (0-1) at which to show warnings."""
        return 0.8  # 80%


class StorageMetadata:
    """Metadata for stored files."""
    
    def __init__(
        self,
        storage_backend: str,
        storage_key: str,
        original_filename: str,
        content_type: str,
        file_size_bytes: int,
        sha256: str,
        created_at: str,
        uploaded_by: Optional[str] = None,
        project_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        self.storage_backend = storage_backend
        self.storage_key = storage_key
        self.original_filename = original_filename
        self.content_type = content_type
        self.file_size_bytes = file_size_bytes
        self.sha256 = sha256
        self.created_at = created_at
        self.uploaded_by = uploaded_by
        self.project_name = project_name
        self.description = description
        self.tags = tags or []
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "storage_backend": self.storage_backend,
            "storage_key": self.storage_key,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "uploaded_by": self.uploaded_by,
            "project_name": self.project_name,
            "description": self.description,
            "tags": self.tags,
            **self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageMetadata":
        """Create metadata from dictionary."""
        extra = {k: v for k, v in data.items() if k not in [
            "storage_backend", "storage_key", "original_filename",
            "content_type", "file_size_bytes", "sha256", "created_at",
            "uploaded_by", "project_name", "description", "tags"
        ]}
        return cls(
            storage_backend=data.get("storage_backend", ""),
            storage_key=data.get("storage_key", ""),
            original_filename=data.get("original_filename", ""),
            content_type=data.get("content_type", ""),
            file_size_bytes=data.get("file_size_bytes", 0),
            sha256=data.get("sha256", ""),
            created_at=data.get("created_at", ""),
            uploaded_by=data.get("uploaded_by"),
            project_name=data.get("project_name"),
            description=data.get("description"),
            tags=data.get("tags", []),
            **extra
        )
    
    def __repr__(self) -> str:
        return f"StorageMetadata(key={self.storage_key}, filename={self.original_filename})"
    
    @property
    def size_mb(self) -> float:
        """Return file size in MB."""
        return self.file_size_bytes / (1024 * 1024)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def upload(self, file_data: bytes, filename: str, content_type: str, metadata: Optional[Dict] = None) -> StorageMetadata:
        """Upload a file and return metadata."""
        pass
    
    @abstractmethod
    def download(self, storage_key: str) -> bytes:
        """Download a file by its storage key."""
        pass
    
    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Check if a file exists."""
        pass
    
    @abstractmethod
    def delete(self, storage_key: str) -> bool:
        """Delete a file by its storage key."""
        pass
    
    @abstractmethod
    def list_files(self, prefix: Optional[str] = None) -> List[StorageMetadata]:
        """List files, optionally filtered by prefix."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check and return status."""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of the backend."""
        pass
    
    def get_total_storage_used(self) -> int:
        """Get total storage used in bytes."""
        files = self.list_files()
        return sum(f.file_size_bytes for f in files)
    
    def get_file_count(self) -> int:
        """Get total number of files."""
        return len(self.list_files())
    
    def get_user_file_count(self, user_id: str) -> int:
        """Get number of files uploaded by a user."""
        files = self.list_files()
        return sum(1 for f in files if f.uploaded_by == user_id)
    
    def find_by_sha256(self, sha256: str) -> Optional[StorageMetadata]:
        """Find a file by its SHA256 hash."""
        files = self.list_files()
        for f in files:
            if f.sha256 == sha256:
                return f
        return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current storage usage statistics."""
        files = self.list_files()
        total_bytes = sum(f.file_size_bytes for f in files)
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            "total_files": len(files),
            "total_storage_bytes": total_bytes,
            "total_storage_mb": round(total_mb, 2),
            "max_files": StorageLimits.MAX_FILES,
            "max_storage_mb": StorageLimits.MAX_TOTAL_STORAGE_MB,
            "storage_usage_percent": round((total_mb / StorageLimits.MAX_TOTAL_STORAGE_MB) * 100, 1),
            "files_usage_percent": round((len(files) / StorageLimits.MAX_FILES) * 100, 1),
            "warning_threshold_percent": StorageLimits.get_usage_warning_threshold() * 100,
            "approaching_storage_limit": total_mb / StorageLimits.MAX_TOTAL_STORAGE_MB >= StorageLimits.get_usage_warning_threshold(),
            "approaching_file_limit": len(files) / StorageLimits.MAX_FILES >= StorageLimits.get_usage_warning_threshold(),
        }
    
    def validate_upload(
        self, 
        file_data: bytes, 
        uploaded_by: Optional[str] = None,
        check_duplicate: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate an upload against all limits.
        
        Returns:
            Tuple of (is_valid, error_type, error_message)
            If valid, returns (True, None, None)
        """
        file_size_bytes = len(file_data)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Check 1: File size limit
        if file_size_bytes > StorageLimits.max_file_size_bytes():
            raise FileSizeLimitExceeded(file_size_mb, StorageLimits.MAX_FILE_SIZE_MB)
        
        # Check 2: Compute SHA256 and check for duplicates
        sha256 = hashlib.sha256(file_data).hexdigest()
        if check_duplicate:
            existing = self.find_by_sha256(sha256)
            if existing:
                raise DuplicateFileError(sha256, existing.storage_key)
        
        # Check 3: Total storage limit
        current_storage_bytes = self.get_total_storage_used()
        current_storage_mb = current_storage_bytes / (1024 * 1024)
        new_total_mb = current_storage_mb + file_size_mb
        if new_total_mb > StorageLimits.MAX_TOTAL_STORAGE_MB:
            raise TotalStorageLimitExceeded(current_storage_mb, file_size_mb, StorageLimits.MAX_TOTAL_STORAGE_MB)
        
        # Check 4: File count limit
        current_file_count = self.get_file_count()
        if current_file_count + 1 > StorageLimits.MAX_FILES:
            raise FileCountLimitExceeded(current_file_count, StorageLimits.MAX_FILES)
        
        # Check 5: Per-user file count limit
        if uploaded_by:
            user_file_count = self.get_user_file_count(uploaded_by)
            if user_file_count + 1 > StorageLimits.MAX_FILES_PER_USER:
                raise UserFileCountLimitExceeded(uploaded_by, user_file_count, StorageLimits.MAX_FILES_PER_USER)
        
        return (True, None, None)


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend for development only."""
    
    def __init__(self, base_path: str = "./data/wharton_uploads"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def backend_name(self) -> str:
        return "local"
    
    def _get_file_path(self, storage_key: str) -> Path:
        """Get the local file path for a storage key."""
        return self.base_path / storage_key
    
    def _compute_sha256(self, data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def upload(
        self, 
        file_data: bytes, 
        filename: str, 
        content_type: str, 
        metadata: Optional[Dict] = None
    ) -> StorageMetadata:
        """Upload a file to local storage."""
        storage_key = f"{uuid.uuid4().hex}_{filename}"
        file_path = self._get_file_path(storage_key)
        
        # Write file
        file_path.write_bytes(file_data)
        
        # Create metadata
        sha256 = self._compute_sha256(file_data)
        created_at = datetime.utcnow().isoformat()
        
        # Extract known fields from metadata to avoid duplicates
        meta_copy = dict(metadata) if metadata else {}
        meta_copy.pop("uploaded_by", None)
        meta_copy.pop("project_name", None)
        meta_copy.pop("description", None)
        meta_copy.pop("tags", None)
        
        storage_metadata = StorageMetadata(
            storage_backend="local",
            storage_key=storage_key,
            original_filename=filename,
            content_type=content_type,
            file_size_bytes=len(file_data),
            sha256=sha256,
            created_at=created_at,
            uploaded_by=metadata.get("uploaded_by") if metadata else None,
            project_name=metadata.get("project_name") if metadata else None,
            description=metadata.get("description") if metadata else None,
            tags=metadata.get("tags", []) if metadata else [],
            **meta_copy
        )
        
        # Save metadata file
        meta_path = self.base_path / f"{storage_key}.meta.json"
        meta_path.write_text(json.dumps(storage_metadata.to_dict(), indent=2))
        
        return storage_metadata
    
    def download(self, storage_key: str) -> bytes:
        """Download a file from local storage."""
        file_path = self._get_file_path(storage_key)
        if not file_path.exists():
            raise FileNotFound(f"File not found: {storage_key}")
        return file_path.read_bytes()
    
    def exists(self, storage_key: str) -> bool:
        """Check if a file exists in local storage."""
        return self._get_file_path(storage_key).exists()
    
    def delete(self, storage_key: str) -> bool:
        """Delete a file from local storage."""
        file_path = self._get_file_path(storage_key)
        meta_path = self.base_path / f"{storage_key}.meta.json"
        
        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
        
        return deleted
    
    def list_files(self, prefix: Optional[str] = None) -> List[StorageMetadata]:
        """List files in local storage."""
        metadata_files = list(self.base_path.glob("*.meta.json"))
        if prefix:
            metadata_files = [f for f in metadata_files if prefix in f.name]
        
        files = []
        for meta_file in metadata_files:
            try:
                data = json.loads(meta_file.read_text())
                files.append(StorageMetadata.from_dict(data))
            except Exception as e:
                st.error(f"Error reading metadata {meta_file.name}: {e}")
        
        return files
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check for local storage."""
        try:
            # Test write
            test_key = f"health_check_{int(time.time())}.txt"
            test_data = b"health check"
            test_path = self._get_file_path(test_key)
            test_path.write_bytes(test_data)
            
            # Test read
            read_data = test_path.read_bytes()
            assert read_data == test_data, "Data mismatch in health check"
            
            # Cleanup
            test_path.unlink()
            
            return {
                "status": "healthy",
                "backend": "local",
                "path": str(self.base_path),
                "writable": os.access(self.base_path, os.W_OK),
                "readable": os.access(self.base_path, os.R_OK)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "local",
                "error": str(e)
            }


class R2StorageBackend(StorageBackend):
    """Cloudflare R2 storage backend for production."""
    
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "auto"
    ):
        """
        Initialize R2 storage backend.
        
        Args:
            bucket: R2 bucket name
            endpoint_url: R2 endpoint URL (e.g., https://accountid.r2.cloudflarestorage.com)
            access_key_id: R2 access key ID
            secret_access_key: R2 secret access key
            region: Region (typically "auto" for R2)
        """
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region = region
        
        # Import boto3 lazily to avoid dependency if not used
        try:
            import boto3
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region
            )
        except ImportError:
            raise ImportError("boto3 is required for R2 storage backend. Install with: pip install boto3")
    
    @property
    def backend_name(self) -> str:
        return "r2"
    
    def _compute_sha256(self, data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def upload(
        self, 
        file_data: bytes, 
        filename: str, 
        content_type: str, 
        metadata: Optional[Dict] = None
    ) -> StorageMetadata:
        """Upload a file to R2 storage."""
        storage_key = f"{uuid.uuid4().hex}_{filename}"
        
        # Upload file data
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=storage_key,
            Body=file_data,
            ContentType=content_type
        )
        
        # Create and upload metadata
        sha256 = self._compute_sha256(file_data)
        created_at = datetime.utcnow().isoformat()
        
        # Extract known fields from metadata to avoid duplicates
        meta_copy = dict(metadata) if metadata else {}
        meta_copy.pop("uploaded_by", None)
        meta_copy.pop("project_name", None)
        meta_copy.pop("description", None)
        meta_copy.pop("tags", None)
        
        storage_metadata = StorageMetadata(
            storage_backend="r2",
            storage_key=storage_key,
            original_filename=filename,
            content_type=content_type,
            file_size_bytes=len(file_data),
            sha256=sha256,
            created_at=created_at,
            uploaded_by=metadata.get("uploaded_by") if metadata else None,
            project_name=metadata.get("project_name") if metadata else None,
            description=metadata.get("description") if metadata else None,
            tags=metadata.get("tags", []) if metadata else [],
            **meta_copy
        )
        
        # Upload metadata as separate object
        meta_key = f"{storage_key}.meta.json"
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=meta_key,
            Body=json.dumps(storage_metadata.to_dict(), indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        
        return storage_metadata
    
    def download(self, storage_key: str) -> bytes:
        """Download a file from R2 storage using streaming."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=storage_key
            )
            return response['Body'].read()
        except Exception as e:
            if 'NoSuchKey' in str(e) or '404' in str(e):
                raise FileNotFound(f"File not found: {storage_key}")
            raise
    
    def download_to_buffer(self, storage_key: str) -> io.BytesIO:
        """Download a file and return as BytesIO buffer (safe for Streamlit)."""
        data = self.download(storage_key)
        return io.BytesIO(data)
    
    def exists(self, storage_key: str) -> bool:
        """Check if a file exists in R2 storage."""
        try:
            self.s3_client.head_object(
                Bucket=self.bucket,
                Key=storage_key
            )
            return True
        except Exception:
            return False
    
    def delete(self, storage_key: str) -> bool:
        """Delete a file from R2 storage."""
        try:
            # Delete main file
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=storage_key
            )
            
            # Delete metadata file
            meta_key = f"{storage_key}.meta.json"
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=meta_key
            )
            
            return True
        except Exception:
            return False
    
    def list_files(self, prefix: Optional[str] = None) -> List[StorageMetadata]:
        """List files in R2 storage."""
        files = []
        
        try:
            # List objects with prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix or ""
            )
            
            if 'Contents' not in response:
                return files
            
            # Get metadata for each file (excluding .meta.json files)
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.meta.json'):
                    continue
                
                # Try to get metadata
                meta_key = f"{key}.meta.json"
                try:
                    meta_response = self.s3_client.get_object(
                        Bucket=self.bucket,
                        Key=meta_key
                    )
                    meta_data = json.loads(meta_response['Body'].read())
                    files.append(StorageMetadata.from_dict(meta_data))
                except Exception:
                    # If no metadata, create basic metadata from object info
                    files.append(StorageMetadata(
                        storage_backend="r2",
                        storage_key=key,
                        original_filename=key,
                        content_type="application/octet-stream",
                        file_size_bytes=obj['Size'],
                        sha256="",
                        created_at=obj['LastModified'].isoformat() if 'LastModified' in obj else ""
                    ))
        
        except Exception as e:
            st.error(f"Error listing files: {e}")
        
        return files
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check for R2 storage."""
        try:
            # Test bucket accessibility
            self.s3_client.head_bucket(Bucket=self.bucket)
            
            # Test write
            test_key = f"health_check_{int(time.time())}.txt"
            test_data = b"health check"
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=test_key,
                Body=test_data
            )
            
            # Test read
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=test_key
            )
            read_data = response['Body'].read()
            assert read_data == test_data, "Data mismatch in health check"
            
            # Cleanup
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=test_key
            )
            
            return {
                "status": "healthy",
                "backend": "r2",
                "bucket": self.bucket,
                "endpoint": self.endpoint_url,
                "region": self.region
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "r2",
                "bucket": self.bucket,
                "error": str(e)
            }


class StorageConfig:
    """Configuration for storage backend selection."""
    
    def __init__(self):
        self._config = None
        self._backend = None
    
    def load_from_secrets(self):
        """Load configuration from Streamlit secrets."""
        if 'storage' not in st.secrets:
            return False
        
        storage_secrets = st.secrets['storage']
        
        self._config = {
            'backend': storage_secrets.get('STORAGE_BACKEND', 'local'),
            'r2_bucket': storage_secrets.get('R2_BUCKET'),
            'r2_endpoint_url': storage_secrets.get('R2_ENDPOINT_URL'),
            'r2_access_key_id': storage_secrets.get('R2_ACCESS_KEY_ID'),
            'r2_secret_access_key': storage_secrets.get('R2_SECRET_ACCESS_KEY'),
            'r2_region': storage_secrets.get('R2_REGION', 'auto')
        }
        
        return True
    
    def validate_r2_config(self) -> List[str]:
        """Validate that all required R2 configuration is present."""
        if not self._config:
            return ["No storage configuration loaded"]
        
        required_keys = [
            'r2_bucket',
            'r2_endpoint_url',
            'r2_access_key_id',
            'r2_secret_access_key'
        ]
        
        missing = []
        for key in required_keys:
            if not self._config.get(key):
                missing.append(key)
        
        return missing
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode (Streamlit Cloud)."""
        # Check if running on Streamlit Cloud
        return 'STREAMLIT_SERVER_PORT' in os.environ
    
    def create_backend(self) -> StorageBackend:
        """Create the appropriate storage backend based on configuration."""
        if self._config is None:
            self.load_from_secrets()
        
        backend_type = self._config.get('backend', 'local')
        
        if backend_type == 'r2':
            # Validate R2 configuration
            missing = self.validate_r2_config()
            if missing:
                if self.is_production_mode():
                    # In production, raise error - no fallback
                    raise ProductionConfigError(missing)
                else:
                    # In development, warn and fall back to local
                    st.warning(
                        f"R2 backend selected but missing configuration: {', '.join(missing)}. "
                        f"Falling back to local storage for development."
                    )
                    return LocalStorageBackend()
            
            # Create R2 backend
            return R2StorageBackend(
                bucket=self._config['r2_bucket'],
                endpoint_url=self._config['r2_endpoint_url'],
                access_key_id=self._config['r2_access_key_id'],
                secret_access_key=self._config['r2_secret_access_key'],
                region=self._config.get('r2_region', 'auto')
            )
        
        elif backend_type == 'local':
            if self.is_production_mode():
                # In production with local backend - show error
                st.error(
                    "Local storage backend selected in production mode. "
                    "This is not recommended as files will not persist across redeployments. "
                    "Please configure R2 storage backend in secrets."
                )
            return LocalStorageBackend()
        
        else:
            raise ValueError(f"Unknown storage backend: {backend_type}")
    
    @property
    def backend(self) -> StorageBackend:
        """Get or create the storage backend."""
        if self._backend is None:
            self._backend = self.create_backend()
        return self._backend
    
    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get the loaded configuration."""
        return self._config


# Global storage configuration instance
storage_config = StorageConfig()


def get_storage_backend() -> StorageBackend:
    """Get the configured storage backend."""
    return storage_config.backend


def initialize_storage() -> Dict[str, Any]:
    """
    Initialize storage backend and perform health check.
    
    Returns:
        Dictionary with initialization status and health check results.
    """
    try:
        # Load configuration
        config_loaded = storage_config.load_from_secrets()
        
        # Create backend
        backend = storage_config.backend
        
        # Perform health check
        health = backend.health_check()
        
        # Get usage stats
        usage_stats = backend.get_usage_stats()
        
        return {
            "success": True,
            "backend": backend.backend_name,
            "config_loaded": config_loaded,
            "health": health,
            "production_mode": storage_config.is_production_mode(),
            "usage_stats": usage_stats
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "backend": "none",
            "production_mode": 'STREAMLIT_SERVER_PORT' in os.environ
        }


def check_storage_health() -> Dict[str, Any]:
    """
    Check storage health without reinitializing.
    
    Returns:
        Health check results.
    """
    if storage_config._backend is None:
        return {
            "status": "not_initialized",
            "message": "Storage backend not initialized"
        }
    
    return storage_config._backend.health_check()


def get_storage_usage() -> Dict[str, Any]:
    """
    Get current storage usage statistics.
    
    Returns:
        Dictionary with usage statistics.
    """
    if storage_config._backend is None:
        return {
            "total_files": 0,
            "total_storage_bytes": 0,
            "total_storage_mb": 0,
            "max_files": StorageLimits.MAX_FILES,
            "max_storage_mb": StorageLimits.MAX_TOTAL_STORAGE_MB,
            "storage_usage_percent": 0,
            "files_usage_percent": 0,
            "warning_threshold_percent": 80,
            "approaching_storage_limit": False,
            "approaching_file_limit": False,
        }
    
    return storage_config._backend.get_usage_stats()