"""
Custom exceptions for storage module.
"""


class StorageError(Exception):
    """Base exception for storage errors."""
    pass


class FileNotFound(StorageError):
    """Raised when a file is not found."""
    pass


class StorageBackendError(StorageError):
    """Raised when there's an error with the storage backend."""
    pass


class ConfigurationError(StorageError):
    """Raised when there's a configuration error."""
    pass


class HealthCheckError(StorageError):
    """Raised when a health check fails."""
    pass


class StorageLimitExceeded(StorageError):
    """Raised when a storage limit would be exceeded."""
    
    def __init__(self, limit_type: str, current: any, limit: any, message: str = None):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        super().__init__(message or f"Storage limit exceeded: {limit_type} ({current}/{limit})")


class FileSizeLimitExceeded(StorageLimitExceeded):
    """Raised when a file exceeds the maximum file size limit."""
    
    def __init__(self, file_size_mb: float, max_size_mb: float):
        super().__init__(
            limit_type="file_size",
            current=f"{file_size_mb:.2f} MB",
            limit=f"{max_size_mb:.2f} MB",
            message=f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed ({max_size_mb:.2f} MB)"
        )


class TotalStorageLimitExceeded(StorageLimitExceeded):
    """Raised when total storage would exceed the limit."""
    
    def __init__(self, current_mb: float, new_file_mb: float, max_mb: float):
        super().__init__(
            limit_type="total_storage",
            current=f"{current_mb:.2f} MB (+{new_file_mb:.2f} MB)",
            limit=f"{max_mb:.2f} MB",
            message=f"Upload would exceed total storage limit ({current_mb:.2f} + {new_file_mb:.2f} > {max_mb:.2f} MB)"
        )


class FileCountLimitExceeded(StorageLimitExceeded):
    """Raised when file count would exceed the limit."""
    
    def __init__(self, current_count: int, max_count: int):
        super().__init__(
            limit_type="file_count",
            current=str(current_count),
            limit=str(max_count),
            message=f"File count ({current_count}) would exceed maximum allowed ({max_count})"
        )


class UserFileCountLimitExceeded(StorageLimitExceeded):
    """Raised when a user's file count would exceed the per-user limit."""
    
    def __init__(self, user_id: str, current_count: int, max_count: int):
        super().__init__(
            limit_type="user_file_count",
            current=str(current_count),
            limit=str(max_count),
            message=f"User '{user_id}' file count ({current_count}) would exceed limit ({max_count})"
        )


class DuplicateFileError(StorageError):
    """Raised when a duplicate file (by SHA256) is detected."""
    
    def __init__(self, sha256: str, existing_storage_key: str = None):
        self.sha256 = sha256
        self.existing_storage_key = existing_storage_key
        super().__init__(
            f"Duplicate file detected (SHA256: {sha256[:16]}...). "
            f"Existing key: {existing_storage_key}" if existing_storage_key
            else f"Duplicate file detected (SHA256: {sha256[:16]}...)"
        )


class ProductionConfigError(StorageError):
    """Raised when production configuration is missing or invalid."""
    
    def __init__(self, missing_secrets: list):
        self.missing_secrets = missing_secrets
        super().__init__(
            f"Production storage configuration missing. Missing secrets: {', '.join(missing_secrets)}. "
            f"Please add these to Streamlit Cloud secrets."
        )


class StorageFileNotFoundError(StorageError):
    """Raised when a file is not found in storage."""
    
    def __init__(self, message: str = "File not found in storage"):
        super().__init__(message)


class FileValidationError(StorageError):
    """Raised when file validation fails."""
    
    def __init__(self, reason: str):
        super().__init__(f"File validation failed: {reason}")
