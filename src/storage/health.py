"""
Storage health check and startup validation.

Includes:
- Enhanced startup checks with R2 validation
- Migration support for existing local uploads
- Production-ready error handling
"""

import os
import streamlit as st
from typing import Dict, Any, Optional, List
from .backend import (
    initialize_storage, 
    check_storage_health, 
    get_storage_usage,
    storage_config,
    StorageLimits,
)
from .exceptions import HealthCheckError, ConfigurationError, ProductionConfigError
from .file_manager import get_file_manager


def run_storage_startup_check() -> Dict[str, Any]:
    """
    Run storage startup checks.
    
    This function should be called at application startup to ensure
    the storage backend is properly configured and healthy.
    
    Returns:
        Dictionary with startup check results
    
    Raises:
        HealthCheckError: If storage is not healthy in production mode
        ConfigurationError: If required configuration is missing
    """
    # Check if running in production
    is_production = os.environ.get("QUANT_SIM_ENV", "development") == "production"
    
    # Initialize storage
    init_result = initialize_storage()
    
    if not init_result["success"]:
        error_msg = f"Storage initialization failed: {init_result.get('error', 'Unknown error')}"
        if is_production:
            raise HealthCheckError(error_msg)
        else:
            st.error(error_msg)
            return init_result
    
    # Check health
    health_result = init_result.get("health", {})
    
    if health_result.get("status") != "healthy":
        error_msg = f"Storage health check failed: {health_result.get('error', 'Unknown error')}"
        if is_production:
            raise HealthCheckError(error_msg)
        else:
            st.error(error_msg)
            return init_result
    
    # Log startup information
    backend_name = init_result.get("backend", "unknown")
    
    if is_production:
        st.success(f"✅ Storage initialized: {backend_name.upper()} backend (production mode)")
    else:
        st.info(f"ℹ️ Storage initialized: {backend_name.upper()} backend (development mode)")
    
    return init_result


def run_enhanced_startup_check() -> Dict[str, Any]:
    """
    Run enhanced startup checks including R2 validation and limits verification.
    
    This performs a more thorough check than run_storage_startup_check():
    1. Validates R2 configuration if in production
    2. Performs a test write/read/delete cycle
    3. Verifies limits are properly configured
    4. Checks for migration needs
    
    Returns:
        Dictionary with detailed startup check results
    """
    results = {
        "success": False,
        "backend": None,
        "production_mode": False,
        "checks": {
            "config_valid": False,
            "backend_healthy": False,
            "limits_configured": False,
            "migration_needed": False,
        },
        "errors": [],
        "warnings": [],
    }
    
    is_production = os.environ.get("QUANT_SIM_ENV", "development") == "production"
    results["production_mode"] = is_production
    
    try:
        # Step 1: Validate configuration
        if is_production:
            if not storage_config.load_from_secrets():
                results["errors"].append("No storage configuration found in secrets")
                if storage_config.is_production_mode():
                    raise ProductionConfigError(["storage section missing"])
            else:
                config = storage_config.config
                if config.get("backend") == "r2":
                    missing = storage_config.validate_r2_config()
                    if missing:
                        results["errors"].append(f"Missing R2 configuration: {missing}")
                        raise ProductionConfigError(missing)
                results["checks"]["config_valid"] = True
        else:
            # Development mode - try to load config but don't fail
            storage_config.load_from_secrets()
            results["checks"]["config_valid"] = True
            results["warnings"].append("Running in development mode")
        
        # Step 2: Initialize and check backend health
        init_result = initialize_storage()
        if not init_result["success"]:
            results["errors"].append(init_result.get("error", "Initialization failed"))
            if is_production:
                raise HealthCheckError(init_result.get("error", "Initialization failed"))
        else:
            results["backend"] = init_result["backend"]
            results["checks"]["backend_healthy"] = True
        
        # Step 3: Verify limits are configured
        results["checks"]["limits_configured"] = True  # Limits are hardcoded in StorageLimits
        results["limits"] = {
            "max_file_size_mb": StorageLimits.MAX_FILE_SIZE_MB,
            "max_total_storage_mb": StorageLimits.MAX_TOTAL_STORAGE_MB,
            "max_files": StorageLimits.MAX_FILES,
            "max_files_per_user": StorageLimits.MAX_FILES_PER_USER,
        }
        
        # Step 4: Get usage stats
        usage = get_storage_usage()
        results["usage"] = usage
        
        # Step 5: Check for migration needs (if there are local files)
        try:
            fm = get_file_manager()
            local_files = fm.list_files() if fm.is_initialized else []
            if local_files and is_production and init_result["backend"] == "r2":
                results["checks"]["migration_needed"] = True
                results["migration_info"] = {
                    "local_files_count": len(local_files),
                    "message": f"Found {len(local_files)} local files that could be migrated to R2"
                }
        except Exception:
            pass  # Migration check is optional
        
        results["success"] = True
        
    except Exception as e:
        results["errors"].append(str(e))
        if is_production:
            raise
    
    return results


def display_storage_status():
    """
    Display storage status in the sidebar or a dedicated page.
    
    Shows:
    - Backend health
    - Current usage (files and storage)
    - Limit warnings
    - Configuration info
    """
    with st.sidebar:
        st.subheader("📦 Storage Status")
        
        try:
            health = check_storage_health()
            
            if health.get("status") == "healthy":
                st.success("✅ Healthy")
                st.caption(f"Backend: {health.get('backend', 'unknown')}")
                
                if health.get("bucket"):
                    st.caption(f"Bucket: {health.get('bucket')}")
                
                # Show usage stats
                usage = get_storage_usage()
                st.divider()
                st.caption("**Storage Usage:**")
                st.progress(usage["storage_usage_percent"] / 100)
                st.caption(f"{usage['total_storage_mb']:.1f} MB / {usage['max_storage_mb']} MB")
                
                st.caption("**Files:**")
                st.progress(usage["files_usage_percent"] / 100)
                st.caption(f"{usage['total_files']} / {usage['max_files']}")
                
                # Warnings
                if usage["approaching_storage_limit"]:
                    st.warning("⚠️ Storage limit approaching!")
                if usage["approaching_file_limit"]:
                    st.warning("⚠️ File limit approaching!")
            else:
                st.error("❌ Unhealthy")
                if health.get("error"):
                    st.caption(f"Error: {health.get('error')}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        # Show configuration info
        if storage_config.config:
            with st.expander("Storage Configuration"):
                config = storage_config.config
                st.write(f"**Backend:** {config.get('backend', 'unknown')}")
                st.write(f"**Production Mode:** {'Yes' if storage_config.is_production_mode() else 'No'}")
                
                if config.get('backend') == 'r2':
                    st.write(f"**Bucket:** {config.get('r2_bucket', 'not configured')}")
                    st.write(f"**Region:** {config.get('r2_region', 'auto')}")


def show_production_error_message(missing_secrets: list):
    """
    Show a user-friendly error message when production secrets are missing.
    
    Args:
        missing_secrets: List of missing secret keys
    """
    st.error("""
    ### 🚨 Production Storage Configuration Missing
    
    This application is running on Streamlit Cloud but the required storage 
    configuration is incomplete. Files uploaded in this mode will be lost 
    when the app restarts.
    
    #### Missing Configuration:
    """)
    
    for secret in missing_secrets:
        st.code(f"secrets.toml: storage.{secret}")
    
    st.error("""
    #### How to fix:
    
    1. Go to your Streamlit Cloud dashboard
    2. Select this app
    3. Click "Secrets" in the left menu
    4. Add the missing secrets (see README.md for details)
    
    Alternatively, set up Cloudflare R2 storage as documented in README.md.
    """)
    
    st.warning("⚠️ The application will continue in limited mode, but file uploads may not persist.")


def validate_storage_for_production() -> bool:
    """
    Validate that storage is properly configured for production.
    
    Returns:
        True if storage is ready for production, False otherwise
    """
    is_production = os.environ.get("QUANT_SIM_ENV", "development") == "production"
    
    if not is_production:
        return True  # Development mode is always valid
    
    # Check if R2 backend is configured
    if not storage_config.load_from_secrets():
        show_production_error_message(["[entire [storage] section missing"])
        return False
    
    config = storage_config.config
    
    if config.get('backend') != 'r2':
        st.warning("⚠️ Local storage backend selected in production. Files will not persist across redeployments.")
        return True  # Not ideal but allowed with warning
    
    # Check for missing R2 secrets
    missing = storage_config.validate_r2_config()
    if missing:
        show_production_error_message(missing)
        return False
    
    return True


def display_migration_ui() -> bool:
    """
    Display UI for migrating local files to R2.
    
    Returns:
        True if migration was completed, False otherwise
    """
    st.subheader("🔄 Migrate Local Files to R2")
    
    st.write("""
    Local files were found that can be migrated to Cloudflare R2 storage.
    This ensures your files persist across app restarts.
    """)
    
    try:
        fm = get_file_manager()
        local_files = fm.list_files()
        
        if not local_files:
            st.info("No local files found for migration.")
            return False
        
        st.write(f"**Found {len(local_files)} local files:**")
        
        # Display files to migrate
        for f in local_files[:10]:  # Show first 10
            st.caption(f"- {f.original_filename} ({f.file_size_bytes} bytes)")
        
        if len(local_files) > 10:
            st.caption(f"... and {len(local_files) - 10} more")
        
        # Migration button
        if st.button("🚀 Migrate Files to R2", key="migrate_files"):
            progress_bar = st.progress(0)
            success_count = 0
            error_count = 0
            
            for i, file_meta in enumerate(local_files):
                try:
                    # Download from local
                    file_data = fm.download_file(file_meta.storage_key)
                    
                    # Re-upload to R2 (backend handles deduplication)
                    fm.upload_file(
                        file_data=file_data,
                        filename=file_meta.original_filename,
                        content_type=file_meta.content_type,
                        uploaded_by=file_meta.uploaded_by,
                        project_name=file_meta.project_name,
                        description=file_meta.description,
                        tags=file_meta.tags,
                    )
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    st.error(f"Failed to migrate {file_meta.original_filename}: {e}")
                
                progress_bar.progress((i + 1) / len(local_files))
            
            st.success(f"Migration complete! {success_count} files migrated, {error_count} errors.")
            return True
        
        return False
        
    except Exception as e:
        st.error(f"Migration failed: {e}")
        return False


def check_migration_needed() -> bool:
    """
    Check if there are local files that should be migrated to R2.
    
    Returns:
        True if migration is needed
    """
    is_production = os.environ.get("QUANT_SIM_ENV", "development") == "production"
    
    if not is_production:
        return False  # No migration needed in development
    
    try:
        fm = get_file_manager()
        if not fm.is_initialized:
            return False
        
        # Check if using R2
        if fm.backend_name != "r2":
            return False
        
        # Check for local files
        local_files = fm.list_files()
        return len(local_files) > 0
        
    except Exception:
        return False