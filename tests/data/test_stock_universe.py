from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import stock_universe
from src.data.universe_enrichment import STANDARD_COLUMNS, TEXT_COLUMNS


def _blank_snapshot_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=STANDARD_COLUMNS)
    frame.loc[0, "Ticker"] = "AAA"
    frame.loc[0, "Source"] = "seed"
    frame.loc[0, "SourceCount"] = 1
    return frame


def test_normalize_snapshot_columns_keeps_text_fields_writable():
    frame = _blank_snapshot_frame()
    normalized = stock_universe._normalize_snapshot_columns(frame)

    normalized.loc[0, "Sector"] = "Healthcare"
    normalized.loc[0, "Industry"] = "Biotech"
    normalized.loc[0, "Company"] = "Acme Corp"

    for column in TEXT_COLUMNS:
        assert normalized[column].dtype == object
    assert normalized.loc[0, "Sector"] == "Healthcare"
    assert normalized.loc[0, "Industry"] == "Biotech"


def test_load_snapshot_from_csv_recasts_text_columns(tmp_path, monkeypatch):
    csv_path = tmp_path / "universe_snapshot.csv"
    parquet_path = tmp_path / "universe_snapshot.parquet"

    frame = _blank_snapshot_frame()
    frame.loc[0, "MarketCap"] = 123456789.0
    frame.loc[0, "Sector"] = np.nan
    frame.loc[0, "Industry"] = np.nan
    frame.to_csv(csv_path, index=False)

    monkeypatch.setattr(stock_universe, "SNAPSHOT_CSV_PATH", csv_path)
    monkeypatch.setattr(stock_universe, "SNAPSHOT_PARQUET_PATH", parquet_path)

    loaded = stock_universe._load_universe_snapshot_from_disk()
    loaded.loc[0, "Sector"] = "Healthcare"
    loaded.loc[0, "Industry"] = "Software"

    assert loaded.loc[0, "Sector"] == "Healthcare"
    assert loaded.loc[0, "Industry"] == "Software"
    assert loaded["Sector"].dtype == object
    assert loaded["Industry"].dtype == object
    assert float(loaded.loc[0, "MarketCap"]) == 123456789.0
