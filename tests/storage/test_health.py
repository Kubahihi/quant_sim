"""
Tests for storage health module.
"""

import pytest
from unittest.mock import patch, Mock

from src.storage.health import (
    run_storage_startup_check,
    display_storage_status,
    show_production_error_message,
    validate_storage_for_production,
)
from src.storage.exceptions import HealthCheckError, ConfigurationError


class TestRunStorageStartupCheck:
    """Test run_storage_startup_check function."""
    
    @patch('src.storage.health.initialize_storage')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'development'}, clear=True)
    def test_startup_check_success_development(self, mock_init):
        """Test successful startup check in development mode."""
        mock_init.return_value = {
            "success": True,
            "backend": "local",
            "health": {"status": "healthy"},
            "config_loaded": True
        }
        
        with patch('src.storage.health.st.success') as mock_success, \
             patch('src.storage.health.st.error') as mock_error:
            
            result = run_storage_startup_check()
            
            assert result["success"] is True
            assert result["backend"] == "local"
            mock_success.assert_not_called()  # No success message in dev
            mock_error.assert_not_called()
    
    @patch('src.storage.health.initialize_storage')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'production'}, clear=True)
    def test_startup_check_success_production(self, mock_init):
        """Test successful startup check in production mode."""
        mock_init.return_value = {
            "success": True,
            "backend": "r2",
            "health": {"status": "healthy"},
            "config_loaded": True
        }
        
        with patch('src.storage.health.st.success') as mock_success:
            
            result = run_storage_startup_check()
            
            assert result["success"] is True
            assert result["backend"] == "r2"
            mock_success.assert_called_once()
    
    @patch('src.storage.health.initialize_storage')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'development'}, clear=True)
    def test_startup_check_failure_development(self, mock_init):
        """Test failed startup check in development mode."""
        mock_init.return_value = {
            "success": False,
            "error": "Configuration error"
        }
        
        with patch('src.storage.health.st.error') as mock_error:
            
            result = run_storage_startup_check()
            
            assert result["success"] is False
            mock_error.assert_called_once()
    
    @patch('src.storage.health.initialize_storage')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'production'}, clear=True)
    def test_startup_check_failure_production_raises(self, mock_init):
        """Test failed startup check in production raises exception."""
        mock_init.return_value = {
            "success": False,
            "error": "Configuration error"
        }
        
        with pytest.raises(HealthCheckError) as excinfo:
            run_storage_startup_check()
        
        assert "Storage initialization failed" in str(excinfo.value)
    
    @patch('src.storage.health.initialize_storage')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'production'}, clear=True)
    def test_startup_check_unhealthy_production_raises(self, mock_init):
        """Test unhealthy storage in production raises exception."""
        mock_init.return_value = {
            "success": True,
            "backend": "r2",
            "health": {"status": "unhealthy", "error": "Bucket not accessible"},
            "config_loaded": True
        }
        
        with pytest.raises(HealthCheckError) as excinfo:
            run_storage_startup_check()
        
        assert "Storage health check failed" in str(excinfo.value)


class TestDisplayStorageStatus:
    """Test display_storage_status function."""
    
    @patch('src.storage.health.check_storage_health')
    @patch('src.storage.health.storage_config')
    def test_display_storage_status_healthy(self, mock_config, mock_health):
        """Test displaying healthy storage status."""
        mock_health.return_value = {
            "status": "healthy",
            "backend": "local"
        }
        
        mock_config.config = {"backend": "local"}
        
        with patch('src.storage.health.st.sidebar', create=True) as mock_sidebar, \
             patch('src.storage.health.st.subheader') as mock_subheader, \
             patch('src.storage.health.st.success') as mock_success, \
             patch('src.storage.health.st.caption') as mock_caption, \
             patch('src.storage.health.st.expander') as mock_expander:
            
            # Mock expander context manager
            mock_expander.return_value.__enter__ = Mock(return_value=None)
            mock_expander.return_value.__exit__ = Mock(return_value=False)
            
            display_storage_status()
            
            mock_success.assert_called_once_with(" Healthy")
            mock_caption.assert_any_call("Backend: local")
    
    @patch('src.storage.health.check_storage_health')
    @patch('src.storage.health.storage_config')
    def test_display_storage_status_unhealthy(self, mock_config, mock_health):
        """Test displaying unhealthy storage status."""
        mock_health.return_value = {
            "status": "unhealthy",
            "error": "Connection failed"
        }
        
        mock_config.config = {"backend": "r2"}
        
        with patch('src.storage.health.st.sidebar', create=True), \
             patch('src.storage.health.st.subheader'), \
             patch('src.storage.health.st.error') as mock_error, \
             patch('src.storage.health.st.caption') as mock_caption:
            
            display_storage_status()
            
            mock_error.assert_called_with(" Unhealthy")
            mock_caption.assert_any_call("Error: Connection failed")
    
    @patch('src.storage.health.check_storage_health')
    @patch('src.storage.health.storage_config')
    def test_display_storage_status_r2_shows_bucket(self, mock_config, mock_health):
        """Test that R2 backend shows bucket information."""
        mock_health.return_value = {
            "status": "healthy",
            "backend": "r2",
            "bucket": "test-bucket"
        }
        
        mock_config.config = {"backend": "r2", "r2_bucket": "test-bucket"}
        
        with patch('src.storage.health.st.sidebar', create=True), \
             patch('src.storage.health.st.subheader'), \
             patch('src.storage.health.st.success'), \
             patch('src.storage.health.st.caption') as mock_caption, \
             patch('src.storage.health.st.expander') as mock_expander:
            
            mock_expander.return_value.__enter__ = Mock(return_value=None)
            mock_expander.return_value.__exit__ = Mock(return_value=False)
            
            display_storage_status()
            
            # Check that bucket info is displayed
            caption_calls = [call[0] for call in mock_caption.call_args_list]
            assert any("test-bucket" in str(call) for call in caption_calls)


class TestShowProductionErrorMessage:
    """Test show_production_error_message function."""
    
    def test_show_production_error_message(self):
        """Test showing production error message."""
        missing_secrets = ["R2_BUCKET", "R2_ENDPOINT_URL"]
        
        with patch('src.storage.health.st.error') as mock_error, \
             patch('src.storage.health.st.code') as mock_code, \
             patch('src.storage.health.st.warning') as mock_warning:
            
            show_production_error_message(missing_secrets)
            
            # Should show error header
            assert mock_error.call_count >= 1
            
            # Should show missing secrets
            assert mock_code.call_count == len(missing_secrets)
            
            # Should show warning about limited mode
            mock_warning.assert_called_once()


class TestValidateStorageForProduction:
    """Test validate_storage_for_production function."""
    
    @patch('src.storage.health.storage_config')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'development'}, clear=True)
    def test_validate_development_always_passes(self, mock_config):
        """Test that validation passes in development mode."""
        mock_config.load_from_secrets.return_value = False
        
        result = validate_storage_for_production()
        
        assert result is True
    
    @patch('src.storage.health.storage_config')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'production'}, clear=True)
    def test_validate_production_missing_entire_section(self, mock_config):
        """Test validation fails when storage section is missing."""
        mock_config.load_from_secrets.return_value = False
        
        with patch('src.storage.health.show_production_error_message') as mock_show:
            result = validate_storage_for_production()
            
            assert result is False
            mock_show.assert_called_once()
    
    @patch('src.storage.health.storage_config')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'production'}, clear=True)
    def test_validate_production_local_backend_warns(self, mock_config):
        """Test that local backend in production shows warning but passes."""
        mock_config.load_from_secrets.return_value = True
        mock_config.config = {"backend": "local"}
        
        with patch('src.storage.health.st.warning') as mock_warning:
            result = validate_storage_for_production()
            
            assert result is True
            mock_warning.assert_called_once()
    
    @patch('src.storage.health.storage_config')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'production'}, clear=True)
    def test_validate_production_r2_missing_secrets(self, mock_config):
        """Test validation fails when R2 secrets are missing."""
        mock_config.load_from_secrets.return_value = True
        mock_config.config = {
            "backend": "r2",
            "r2_bucket": "test-bucket"
            # Missing other required secrets
        }
        mock_config.validate_r2_config.return_value = ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
        
        with patch('src.storage.health.show_production_error_message') as mock_show:
            result = validate_storage_for_production()
            
            assert result is False
            mock_show.assert_called_once_with(["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"])
    
    @patch('src.storage.health.storage_config')
    @patch.dict('os.environ', {'QUANT_SIM_ENV': 'production'}, clear=True)
    def test_validate_production_r2_complete(self, mock_config):
        """Test validation passes when R2 is fully configured."""
        mock_config.load_from_secrets.return_value = True
        mock_config.config = {
            "backend": "r2",
            "r2_bucket": "test-bucket",
            "r2_endpoint_url": "https://test.r2.cloudflarestorage.com",
            "r2_access_key_id": "test-key",
            "r2_secret_access_key": "test-secret",
            "r2_region": "auto"
        }
        mock_config.validate_r2_config.return_value = []
        
        result = validate_storage_for_production()
        
        assert result is True