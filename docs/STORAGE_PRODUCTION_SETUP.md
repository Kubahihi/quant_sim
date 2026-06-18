# Production Storage Setup Guide

This document provides a comprehensive guide for setting up production-ready storage for Streamlit Cloud deployment with Cloudflare R2.

## Overview

The quant_sim application uses a robust storage system with:

1. **LocalStorageBackend** - For local development only
2. **R2StorageBackend** - For production on Streamlit Cloud (Cloudflare R2)

### Key Features

- **Hard Limits**: Enforced limits to stay within Cloudflare R2 free tier
- **SHA256 Deduplication**: Prevents storing duplicate files
- **Comprehensive Metadata**: Full file tracking with user, project, and tags
- **Health Checks**: Startup validation and ongoing monitoring
- **Migration Support**: Easy migration from local to R2 storage
- **No Silent Fallbacks**: Production mode fails explicitly if R2 is not configured

## Hard Limits (Free Tier Protection)

The following limits are enforced to ensure the application never exceeds the Cloudflare R2 free tier:

| Limit | Value | Description |
|-------|-------|-------------|
| Max file size | 20 MB | Maximum size per individual file |
| Max total storage | 500 MB | Maximum total storage used |
| Max files | 100 | Maximum number of files |
| Max files per user | 50 | Maximum files per individual user |
| Warning threshold | 80% | Show warnings when approaching limits |

### Cloudflare R2 Free Tier Includes

- 10 GB storage per month
- 10 million Class B reads per month
- 1 million Class A writes per month

Our conservative limits (500 MB, 100 files) ensure we never exceed free tier even with heavy usage.

## Configuration

### Secrets Structure

Add the following to your Streamlit Cloud secrets (Settings > Secrets):

```toml
[storage]
# Storage backend: "r2" for production (REQUIRED on Streamlit Cloud)
STORAGE_BACKEND = "r2"

# R2 Bucket name (must be globally unique)
R2_BUCKET = "your-bucket-name"

# R2 Endpoint URL (format: https://<ACCOUNT_ID>.r2.cloudflarestorage.com)
R2_ENDPOINT_URL = "https://your-account-id.r2.cloudflarestorage.com"

# R2 API Access Key ID
R2_ACCESS_KEY_ID = "your-access-key-id"

# R2 API Secret Access Key
R2_SECRET_ACCESS_KEY = "your-secret-access-key"

# R2 Region (typically "auto" for R2)
R2_REGION = "auto"
```

### Setting Up Cloudflare R2

1. **Go to Cloudflare Dashboard** → R2
2. **Create a bucket** with a unique name (e.g., `quant-sim-files-2024`)
3. **Go to "Manage R2 API Tokens"** → Create new token
4. **Select permissions**:
   - Object Read
   - Object Write
   - Object Delete
   - Admin Read and Write (for bucket operations)
5. **Copy the credentials** to your Streamlit Cloud secrets

### Backend Selection Logic

1. If `STORAGE_BACKEND = "r2"` and all R2 secrets present → Use R2
2. If `STORAGE_BACKEND = "local"` → Use local storage (development only)
3. If on Streamlit Cloud without proper config → **Show error and block file operations**
4. **No silent fallback** in production mode

## Metadata Structure

All uploaded files include comprehensive metadata:

```python
{
    "storage_backend": "r2",                    # Backend used
    "storage_key": "uuid_filename",             # Unique identifier
    "original_filename": "file.pdf",            # Original name
    "content_type": "application/pdf",          # MIME type
    "file_size_bytes": 1024,                    # File size
    "sha256": "hash...",                        # File hash (for deduplication)
    "created_at": "2024-01-01T00:00:00",       # Upload timestamp
    "uploaded_by": "user_id",                   # User who uploaded
    "project_name": "project_name",             # Optional project
    "description": "File description",          # Optional description
    "tags": ["tag1", "tag2"]                    # Optional tags
}
```

## SHA256 Deduplication

The system automatically detects duplicate files by their SHA256 hash:

- If you try to upload a file with the same content as an existing file, you'll get a `DuplicateFileError`
- This saves storage space and prevents accidental duplicates
- The existing file's metadata is returned so you can reference it

## Health Check Process

On application startup:

1. **Load configuration** from secrets
2. **Validate R2 configuration** if in production mode
3. **Create storage backend** (R2 or local)
4. **Perform health check**:
   - Test bucket accessibility
   - Test write operation
   - Test read operation
   - Verify data integrity
   - Clean up test data
5. **Display status** to user
6. **If production and unhealthy** → Raise error and block file operations

## Usage Statistics

The system tracks and displays:

- Total files used / maximum files
- Total storage used / maximum storage
- Usage percentages with progress bars
- Warnings when approaching limits (80% threshold)

### Display in UI

```python
from src.storage import get_wharton_vault

vault = get_wharton_vault()
vault.display_usage_stats()  # Shows metrics and progress bars
vault.display_limit_info()   # Shows limit details in expander
```

## Migration from Local to R2

If you have existing files in local storage when deploying to R2:

### Automatic Migration UI

```python
from src.storage import check_migration_needed, display_migration_ui

if check_migration_needed():
    display_migration_ui()  # Shows migration interface
```

### Manual Migration

1. Deploy with R2 configuration
2. Download existing files from local storage
3. Re-upload through the application
4. Files will now be stored in R2 and persist

The migration is:
- **Safe**: Original files are not deleted until migration is verified
- **Idempotent**: Can be run multiple times safely
- **Deduplication-aware**: Won't create duplicates if files already exist in R2

## Testing

Run the complete test suite:

```bash
pytest tests/storage/ -v
```

### Test Coverage

- ✅ StorageLimits configuration
- ✅ StorageMetadata with all fields
- ✅ LocalStorageBackend operations
- ✅ R2StorageBackend (mocked)
- ✅ File size limit enforcement
- ✅ Total storage limit enforcement
- ✅ File count limit enforcement
- ✅ Per-user file count limit enforcement
- ✅ SHA256 deduplication
- ✅ Secrets configuration parsing
- ✅ Backend selection logic
- ✅ Production mode validation
- ✅ Health check functionality
- ✅ Upload/download roundtrip
- ✅ Failure handling when secrets missing

## Deployment Checklist

### Before Deploying to Streamlit Cloud

- [ ] Create Cloudflare R2 bucket
- [ ] Generate R2 API token with required permissions
- [ ] Note endpoint URL, bucket name, access keys
- [ ] Add secrets to Streamlit Cloud dashboard (Settings > Secrets)
- [ ] Set `STORAGE_BACKEND = "r2"`
- [ ] Test health check in development mode first

### After Deployment

- [ ] Verify storage status in sidebar
- [ ] Check that R2 backend is active
- [ ] Test file upload (small file first)
- [ ] Test file download
- [ ] Restart app and verify files persist
- [ ] Check usage statistics display correctly
- [ ] Verify limits are enforced

## Troubleshooting

### Common Issues

1. **"Storage initialization failed: Production storage configuration missing"**
   - Add all required R2 secrets to Streamlit Cloud
   - Ensure `STORAGE_BACKEND = "r2"`
   - Check for typos in secret names

2. **"Health check failed: Access Denied"**
   - Verify R2 API token has correct permissions
   - Check that bucket exists
   - Ensure endpoint URL is correct

3. **"File size exceeds maximum allowed"**
   - Maximum file size is 20 MB
   - Compress or split large files

4. **"Storage limit reached"**
   - Total storage limit is 500 MB
   - Delete unused files to free space
   - Consider upgrading R2 plan if needed

5. **"File limit reached"**
   - Maximum 100 files total
   - Maximum 50 files per user
   - Delete unused files

### Error Messages

The application provides clear error messages:

- **Development Mode**: Warnings and fallback to local storage
- **Production Mode**: Clear errors with instructions to fix configuration
- **Limit Exceeded**: Specific message about which limit was hit

## Security Considerations

1. **Never commit secrets** - Use `.gitignore` for `secrets.toml`
2. **Rotate keys regularly** - Update secrets in Streamlit Cloud
3. **Monitor usage** - Check Cloudflare dashboard for R2 usage
4. **Use minimum permissions** - R2 token should only have necessary access
5. **No local fallback in production** - Files won't silently go to local storage

## Cost Considerations

Cloudflare R2 pricing:
- Storage: $0.015 per GB per month
- Class A operations (write, list): $4.50 per million
- Class B operations (read): $0.36 per million

With our limits (500 MB storage, 100 files), expected monthly cost is **$0.00** (well within free tier).

## API Reference

### FileManager

```python
from src.storage import FileManager, StorageLimits

fm = FileManager()

# Upload with limit enforcement
metadata = fm.upload_file(
    file_data=bytes,
    filename="file.txt",
    content_type="text/plain",
    uploaded_by="user_id",
    project_name="project",
    description="Description",
    tags=["tag1", "tag2"]
)

# Download
data = fm.download_file(storage_key)

# Get usage stats
stats = fm.get_usage_stats()
# Returns: {
#   "total_files": int,
#   "total_storage_mb": float,
#   "max_files": int,
#   "max_storage_mb": int,
#   "storage_usage_percent": float,
#   "files_usage_percent": float,
#   "approaching_storage_limit": bool,
#   "approaching_file_limit": bool
# }
```

### WhartonFileVault

```python
from src.storage import get_wharton_vault

vault = get_wharton_vault()

# Upload from Streamlit uploader
metadata = vault.upload_from_streamlit(
    uploaded_file,
    uploaded_by="user_id"
)

# Display usage
vault.display_usage_stats()
vault.display_limit_info()

# Handle upload errors
try:
    vault.upload_from_streamlit(file)
except Exception as e:
    st.error(vault.handle_upload_error(e))
```

## Additional Resources

- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Boto3 S3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html) (R2 is S3-compatible)