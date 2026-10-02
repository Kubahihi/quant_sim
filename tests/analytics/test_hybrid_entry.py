from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analytics.hybrid_entry import HybridEntryConfig, hybrid_entry_signal, study_hybrid_entry


def histories(n=1_250):
    index = pd.bdate_range('2020-01-02', periods=n)
    generator = np.random.default_rng(42)
    benchmark = pd.Series(100 * np.exp(np.cumsum(generator.normal(.0002, .009, n))), index=index)
    security = pd.Series(100 * np.exp(np.cumsum(generator.normal(.00035, .012, n))), index=index)
    return security, benchmark


def test_hybrid_study_uses_common_non_overlapping_terminal_dates():
    security, benchmark = histories()
    study = study_hybrid_entry(security, benchmark, as_of=security.index[-1])
    assert len(study.cases) >= 12
    assert study.cases.terminal_date.nunique() == len(study.cases)
    assert study.cases.wait_sessions.between(1, 20).all()
    assert study.summary.strategy.tolist() == ['Lump sum', 'Fixed DCA', 'Hybrid DCA']
    assert {'hybrid_vs_lump', 'hybrid_vs_fixed'} <= set(study.cases)


def test_hybrid_signal_does_not_see_prices_after_as_of():
    security, benchmark = histories()
    cutoff = security.index[1_100]
    first = hybrid_entry_signal(security, benchmark, as_of=cutoff)
    security.loc[security.index > cutoff] *= 10
    benchmark.loc[benchmark.index > cutoff] *= .1
    second = hybrid_entry_signal(security, benchmark, as_of=cutoff)
    assert first.to_dict() == second.to_dict()


def test_hybrid_config_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        replace(HybridEntryConfig(), first_fraction=1.).validate()


def test_hybrid_study_marks_a_short_history_as_not_report_ready():
    security, benchmark = histories(800)
    study = study_hybrid_entry(security, benchmark, as_of=security.index[-1])
    assert len(study.cases) < study.config['minimum_cases']
    assert any('at least' in warning for warning in study.warnings)
