from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.storage import wharton_adapter
from src.storage.backend import LocalStorageBackend
from src.storage.exceptions import ProductionConfigError


def test_wharton_storage_refuses_local_fallback_in_production(monkeypatch, tmp_path):
    config = Mock()
    config.load_from_secrets.return_value = False
    config.is_production_mode.return_value = True
    monkeypatch.setattr("src.storage.backend.storage_config", config)

    with pytest.raises(ProductionConfigError, match="R2 storage secrets"):
        wharton_adapter.get_storage_backend(str(tmp_path))


def test_wharton_storage_keeps_local_fallback_in_development(monkeypatch, tmp_path):
    config = Mock()
    config.load_from_secrets.return_value = False
    config.is_production_mode.return_value = False
    monkeypatch.setattr("src.storage.backend.storage_config", config)

    backend = wharton_adapter.get_storage_backend(str(tmp_path))

    assert isinstance(backend, LocalStorageBackend)
