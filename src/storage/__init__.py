"""
Storage module for handling file uploads and downloads.

Supports multiple backends:
- LocalStorageBackend: For local development
- R2StorageBackend: For production on Streamlit Cloud

Features:
- Hard limits to prevent exceeding free tier
- SHA256 deduplication
- Comprehensive health checks
- Migration support from local to R2
"""

from .backend import (
    StorageBackend,
    StorageMetadata,
    StorageLimits,
    LocalStorageBackend,
    R2StorageBackend,
    StorageConfig,
    storage_config,
    get_storage_backend,
    initialize_storage,
    check_storage_health,
    get_storage_usage,
)
from .file_manager import (
    FileManager,
    WhartonFileVault,
    get_file_manager,
    get_wharton_vault,
    initialize_file_manager,
)
from .exceptions import (
    StorageError,
    FileNotFound,
    StorageBackendError,
    ConfigurationError,
    HealthCheckError,
    StorageLimitExceeded,
    FileSizeLimitExceeded,
    TotalStorageLimitExceeded,
    FileCountLimitExceeded,
    UserFileCountLimitExceeded,
    DuplicateFileError,
    ProductionConfigError,
    StorageFileNotFoundError,
    FileValidationError,
)
from .health import (
    run_storage_startup_check,
    run_enhanced_startup_check,
    display_storage_status,
    show_production_error_message,
    validate_storage_for_production,
    display_migration_ui,
    check_migration_needed,
)
from .wharton_adapter import (
    init_storage_db,
    save_uploaded_file,
    download_file,
    file_exists,
    verify_file_integrity,
    delete_file,
    get_file_status,
    list_files_with_status,
)

__all__ = [
    # Backend classes
    "StorageBackend",
    "LocalStorageBackend",
    "R2StorageBackend",
    "StorageMetadata",
    "StorageLimits",
    "StorageConfig",
    "storage_config",
    
    # Functions
    "get_storage_backend",
    "initialize_storage",
    "check_storage_health",
    "get_storage_usage",
    "FileManager",
    "WhartonFileVault",
    "get_file_manager",
    "get_wharton_vault",
    "initialize_file_manager",
    
    # Health check functions
    "run_storage_startup_check",
    "run_enhanced_startup_check",
    "display_storage_status",
    "show_production_error_message",
    "validate_storage_for_production",
    "display_migration_ui",
    "check_migration_needed",
    
    # Exceptions
    "StorageError",
    "FileNotFound",
    "StorageBackendError",
    "ConfigurationError",
    "HealthCheckError",
    "StorageLimitExceeded",
    "FileSizeLimitExceeded",
    "TotalStorageLimitExceeded",
    "FileCountLimitExceeded",
    "UserFileCountLimitExceeded",
    "DuplicateFileError",
    "ProductionConfigError",
    "StorageFileNotFoundError",
    "FileValidationError",
    
    # Wharton adapter functions
    "init_storage_db",
    "save_uploaded_file",
    "download_file",
    "file_exists",
    "verify_file_integrity",
    "delete_file",
    "get_file_status",
    "list_files_with_status",
]
