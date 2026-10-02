from dataclasses import replace

import numpy as np
import pandas as pd

from src.analytics.entry_algorithm import EntryAlgorithmConfig, entry_algorithm_signal, validate_entry_algorithm


def histories(n=1_250):
    index = pd.bdate_range('2020-01-02', periods=n)
    generator = np.random.default_rng(17)
    benchmark = pd.Series(100 * np.exp(np.cumsum(generator.normal(.0002, .009, n))), index=index)
    security = pd.Series(100 * np.exp(np.cumsum(generator.normal(.00035, .012, n))), index=index)
    return security, benchmark


def test_entry_algorithm_is_past_only_and_has_a_forced_deadline():
    security, benchmark = histories()
    cutoff = security.index[1_100]
    original = entry_algorithm_signal(security, benchmark, as_of=cutoff)
    security.loc[security.index > cutoff] *= 10
    benchmark.loc[benchmark.index > cutoff] *= .1
    changed = entry_algorithm_signal(security, benchmark, as_of=cutoff)
    assert original.to_dict() == changed.to_dict()
    assert original.completion_deadline == str((cutoff + pd.offsets.BDay(20)).date())
    assert original.action in {'Buy remaining tranche now', 'Wait for next model review', 'Calendar control'}


def test_walk_forward_has_common_terminal_and_no_overlapping_origins():
    security, benchmark = histories()
    result = validate_entry_algorithm(security, benchmark, as_of=security.index[-1])
    assert result.summary['cases'] >= 12
    assert result.observations.terminal_date.nunique() == len(result.observations)
    assert result.observations.wait_sessions.between(0, 20).all()
    assert {'immediate_wealth', 'signal_wealth', 'net_advantage'} <= set(result.observations)


def test_invalid_algorithm_configuration_is_rejected():
    config = replace(EntryAlgorithmConfig(), maximum_wait_sessions=19)
    try:
        config.validate()
    except ValueError:
        pass
    else:
        raise AssertionError('Expected invalid review cadence to fail validation.')
