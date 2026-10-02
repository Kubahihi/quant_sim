"""Past-only, signal-gated entry research for a remaining position tranche.

The module deliberately uses one fixed, regularised specification.  It does not
search parameters or submit orders.  A signal can defer a tranche only until a
saved completion review; its walk-forward study compares that choice to buying at
the first eligible session using the same terminal date and cost assumption.
"""
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from src.analytics.deployment import day


FEATURE_NAMES = (
    'security_return_5d', 'security_return_20d', 'security_return_63d',
    'distance_from_sma_20d', 'distance_from_sma_63d', 'drawdown_20d',
    'realized_volatility_20d', 'benchmark_return_20d', 'relative_return_20d',
)


@dataclass(frozen=True)
class EntryAlgorithmConfig:
    horizon_sessions: int = 21
    review_interval_sessions: int = 5
    maximum_wait_sessions: int = 20
    minimum_training_observations: int = 252
    training_window_sessions: int = 756
    ridge_penalty: float = 10.
    predicted_excess_threshold: float = 0.
    annual_cash_rate: float = .03
    cost_bps: float = 10.
    minimum_out_of_sample_cases: int = 12

    def validate(self):
        if any(type(value) is not int for value in (
                self.horizon_sessions, self.review_interval_sessions,
                self.maximum_wait_sessions, self.minimum_training_observations,
                self.training_window_sessions, self.minimum_out_of_sample_cases)):
            raise ValueError('Session counts must be integers.')
        if (not 5 <= self.horizon_sessions <= 126 or not 1 <= self.review_interval_sessions <= 21 or
                not self.review_interval_sessions <= self.maximum_wait_sessions <= 63 or
                not 126 <= self.minimum_training_observations <= self.training_window_sessions or
                not 4 <= self.minimum_out_of_sample_cases <= 100):
            raise ValueError('Entry algorithm session settings are outside the supported range.')
        if self.maximum_wait_sessions % self.review_interval_sessions:
            raise ValueError('Maximum wait must land exactly on a model review.')
        for value, name in ((self.ridge_penalty, 'ridge penalty'),
                            (self.predicted_excess_threshold, 'predicted excess threshold'),
                            (self.annual_cash_rate, 'annual cash rate'), (self.cost_bps, 'cost bps')):
            if not isfinite(value):
                raise ValueError(f'{name} must be finite.')
        if self.ridge_penalty < 0 or not -1 <= self.annual_cash_rate <= 1 or not 0 <= self.cost_bps <= 1000:
            raise ValueError('Unsupported regularisation, cash-rate or cost assumption.')


@dataclass
class EntryAlgorithmSignal:
    as_of: str
    status: str
    action: str
    next_review: str | None
    completion_deadline: str | None
    predicted_excess_return: float | None
    training_observations: int
    validation_cases: int
    validation_mean_advantage: float | None
    validation_hit_rate: float | None
    model_coefficients: dict[str, float]
    warnings: list[str]
    config: dict

    def to_dict(self):
        return asdict(self)


@dataclass
class EntryAlgorithmValidation:
    observations: pd.DataFrame
    summary: dict
    warnings: list[str]


def _clean_pair(prices, benchmark_prices, cutoff):
    def clean(series, name):
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f'{name} requires dated daily prices.')
        value = pd.to_numeric(series.copy(), errors='coerce')
        value.index = value.index.tz_localize(None).normalize()
        value = value.loc[:cutoff].where(value > 0).dropna().sort_index()
        if value.index.has_duplicates:
            raise ValueError(f'{name} requires unique session dates.')
        return value
    security, benchmark = clean(prices, 'Security'), clean(benchmark_prices, 'Benchmark')
    common = security.index.intersection(benchmark.index)
    if len(common) < 400:
        raise ValueError('At least 400 common adjusted-price observations are required.')
    return security.loc[common], benchmark.loc[common]


def _feature_matrix(prices, benchmark):
    """Vectorised features. Rows retain their date alignment for past-only slices."""
    returns = prices.pct_change()
    security_5 = prices.pct_change(5)
    security_20 = prices.pct_change(20)
    security_63 = prices.pct_change(63)
    benchmark_20 = benchmark.pct_change(20)
    matrix = pd.DataFrame({
        'security_return_5d': security_5,
        'security_return_20d': security_20,
        'security_return_63d': security_63,
        'distance_from_sma_20d': prices / prices.rolling(20).mean() - 1,
        'distance_from_sma_63d': prices / prices.rolling(63).mean() - 1,
        'drawdown_20d': prices / prices.rolling(20).max() - 1,
        'realized_volatility_20d': returns.rolling(20).std(ddof=1) * np.sqrt(252),
        'benchmark_return_20d': benchmark_20,
        'relative_return_20d': security_20 - benchmark_20,
    })
    return matrix.reindex(columns=FEATURE_NAMES).to_numpy(dtype=float)


def _training_data(features, prices, benchmark, review_index, config):
    latest_start = review_index - config.horizon_sessions
    first = max(63, latest_start - config.training_window_sessions + 1)
    rows = features[first:latest_start + 1]
    target = (prices.to_numpy()[first + config.horizon_sessions:latest_start + config.horizon_sessions + 1] /
              prices.to_numpy()[first:latest_start + 1] -
              benchmark.to_numpy()[first + config.horizon_sessions:latest_start + config.horizon_sessions + 1] /
              benchmark.to_numpy()[first:latest_start + 1])
    usable = np.isfinite(rows).all(axis=1) & np.isfinite(target)
    return rows[usable], target[usable]


def _fit_predict(train_x, train_y, current_x, config):
    if len(train_x) < config.minimum_training_observations:
        return None, {}
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    normalized = (train_x - mean) / scale
    y_mean = float(train_y.mean())
    lhs = normalized.T @ normalized + config.ridge_penalty * np.eye(normalized.shape[1])
    rhs = normalized.T @ (train_y - y_mean)
    coefficients = (np.linalg.solve(lhs, rhs) if config.ridge_penalty else
                    np.linalg.lstsq(normalized, train_y - y_mean, rcond=None)[0])
    prediction = float(y_mean + ((current_x - mean) / scale) @ coefficients)
    raw_coefficients = coefficients / scale
    intercept = y_mean - float(mean @ raw_coefficients)
    reported = {'intercept': intercept, **{name: float(value) for name, value in zip(FEATURE_NAMES, raw_coefficients)}}
    return prediction, reported


def _signal_at(features, prices, benchmark, review_index, config):
    current = features[review_index]
    if not np.isfinite(current).all():
        return None, {}, 0
    train_x, train_y = _training_data(features, prices, benchmark, review_index, config)
    predicted, coefficients = _fit_predict(train_x, train_y, current, config)
    return predicted, coefficients, len(train_y)


def validate_entry_algorithm(prices, benchmark_prices, *, as_of, config=None):
    """Non-overlapping walk-forward policy test with a common end date per origin."""
    config = config or EntryAlgorithmConfig()
    config.validate()
    cutoff = day(as_of)
    security, benchmark = _clean_pair(prices, benchmark_prices, cutoff)
    features = _feature_matrix(security, benchmark)
    terminal_span = config.maximum_wait_sessions + config.horizon_sessions + 1
    first_origin = 63 + config.horizon_sessions + config.minimum_training_observations
    origins = range(first_origin, len(security) - terminal_span, terminal_span)
    rows = []
    for origin in origins:
        purchase_index = None
        last_prediction = None
        for offset in range(0, config.maximum_wait_sessions + 1, config.review_interval_sessions):
            review_index = origin + offset
            prediction, _, training_count = _signal_at(features, security, benchmark, review_index, config)
            last_prediction = prediction
            if offset == config.maximum_wait_sessions or (prediction is not None and
                                                           prediction >= config.predicted_excess_threshold):
                purchase_index = review_index + 1
                break
        if purchase_index is None:
            continue
        immediate_index = origin + 1
        terminal_index = origin + terminal_span
        if terminal_index >= len(security):
            continue
        fee = config.cost_bps / 10_000
        wait_sessions = purchase_index - immediate_index
        immediate_wealth = (1 - fee) * float(security.iloc[terminal_index] / security.iloc[immediate_index])
        signal_wealth = ((1 + config.annual_cash_rate) ** (wait_sessions / 252) * (1 - fee) *
                         float(security.iloc[terminal_index] / security.iloc[purchase_index]))
        rows.append(dict(origin_date=str(security.index[origin].date()),
            execution_date=str(security.index[purchase_index].date()), terminal_date=str(security.index[terminal_index].date()),
            wait_sessions=wait_sessions, predicted_excess_return=last_prediction,
            training_observations=training_count, immediate_wealth=immediate_wealth,
            signal_wealth=signal_wealth, net_advantage=signal_wealth / immediate_wealth - 1))
    observations = pd.DataFrame(rows)
    if observations.empty:
        summary = dict(cases=0, mean_net_advantage=None, median_net_advantage=None, hit_rate=None,
                       research_ready=False)
    else:
        advantage = observations['net_advantage']
        cases = len(observations)
        mean = float(advantage.mean())
        summary = dict(cases=cases, mean_net_advantage=mean, median_net_advantage=float(advantage.median()),
            hit_rate=float((advantage > 0).mean()),
            research_ready=bool(cases >= config.minimum_out_of_sample_cases and mean > 0))
    warnings = [
        'Walk-forward origins are non-overlapping, but they are still a single-security historical study.',
        'The model is fixed: no threshold, feature, horizon or penalty search is performed by this function.',
        'A positive result is not proof of future alpha. Revised prices, regime change and selection bias remain material.',
        'The comparison includes one stated cost assumption and cash yield; it does not model tax, market impact or liquidity.',
    ]
    return EntryAlgorithmValidation(observations, summary, warnings)


def entry_algorithm_signal(prices, benchmark_prices, *, as_of, config=None):
    """Current decision: buy now if the fixed model qualifies, otherwise review again or force completion."""
    config = config or EntryAlgorithmConfig()
    config.validate()
    today = day(as_of)
    security, benchmark = _clean_pair(prices, benchmark_prices, today)
    features = _feature_matrix(security, benchmark)
    review_index = len(security) - 1
    prediction, coefficients, training_count = _signal_at(features, security, benchmark, review_index, config)
    validation = validate_entry_algorithm(security, benchmark, as_of=today, config=config)
    next_review = today + pd.offsets.BDay(config.review_interval_sessions)
    deadline = today + pd.offsets.BDay(config.maximum_wait_sessions)
    warnings = list(validation.warnings)
    if prediction is None:
        return EntryAlgorithmSignal(str(today.date()), 'insufficient_data', 'Calendar control',
            str(next_review.date()), str(deadline.date()), None, training_count,
            validation.summary['cases'], validation.summary['mean_net_advantage'], validation.summary['hit_rate'],
            coefficients, warnings + ['Insufficient completed historical outcomes to fit the fixed model.'], asdict(config))
    if not validation.summary['research_ready']:
        return EntryAlgorithmSignal(str(today.date()), 'validation_incomplete', 'Calendar control',
            str(next_review.date()), str(deadline.date()), prediction, training_count,
            validation.summary['cases'], validation.summary['mean_net_advantage'], validation.summary['hit_rate'],
            coefficients, warnings + ['The model is not research-ready under its fixed sample and advantage gate.'], asdict(config))
    action = 'Buy remaining tranche now' if prediction >= config.predicted_excess_threshold else 'Wait for next model review'
    return EntryAlgorithmSignal(str(today.date()), 'research_ready', action,
        None if action.startswith('Buy') else str(next_review.date()), str(deadline.date()), prediction,
        training_count, validation.summary['cases'], validation.summary['mean_net_advantage'],
        validation.summary['hit_rate'], coefficients, warnings, asdict(config))
