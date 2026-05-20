from __future__ import annotations

from src.data import universe_sources


def test_is_likely_us_common_symbol_keeps_plain_exchange_tickers():
    assert universe_sources._is_likely_us_common_symbol("AAPL") is True
    assert universe_sources._is_likely_us_common_symbol("GOOGL") is True
    assert universe_sources._is_likely_us_common_symbol("BRK-B") is True


def test_is_likely_us_common_symbol_filters_otc_and_warrant_suffix_noise():
    assert universe_sources._is_likely_us_common_symbol("ABCDY") is False
    assert universe_sources._is_likely_us_common_symbol("ABCDQ") is False
    assert universe_sources._is_likely_us_common_symbol("ABCDW") is False
    assert universe_sources._is_likely_us_common_symbol("ABCDF") is False
    assert universe_sources._is_likely_us_common_symbol("ABCDEF") is False
