from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from .registry import ModelRegistry
from .results import ModelResult

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover
    ARIMA = None  # type: ignore

try:
    from arch import arch_model
except Exception:  # pragma: no cover
    arch_model = None  # type: ignore


def _safe_model(name: str, family: str, fn: Any, series: pd.Series, context: Dict[str, Any]) -> ModelResult:
    try:
        result = fn(series, context)
        if isinstance(result, ModelResult):
            return result
        return ModelResult(name=name, family=family, available=False, error="Invalid model output type")
    except Exception as exc:
        return ModelResult(name=name, family=family, available=False, error=str(exc))


def _series(series: pd.Series) -> pd.Series:
    return pd.Series(series).dropna().astype(float)


def _linear_regression_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 10:
        raise ValueError("linear regression requires at least 10 observations")
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    next_ret = float(intercept + slope * len(clean))
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return ModelResult(
        name="linear_regression",
        family="classical",
        available=True,
        metrics={
            "expected_daily_return": next_ret,
            "expected_annual_return": next_ret * 252.0,
            "trend_slope_daily": float(slope),
            "confidence": r2,
        },
        payload={"generated_at": datetime.utcnow().isoformat()},
        confidence=r2,
    )


def _ridge_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 25:
        raise ValueError("ridge regression requires at least 25 observations")
    x = np.column_stack([np.ones(len(clean) - 1), clean.shift(1).dropna().to_numpy(dtype=float)])
    y = clean.iloc[1:].to_numpy(dtype=float)
    alpha = 2.0
    xtx = x.T @ x
    ridge = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), x.T @ y)
    next_ret = float(ridge[0] + ridge[1] * clean.iloc[-1])
    return ModelResult(
        name="ridge",
        family="classical",
        available=True,
        metrics={
            "expected_daily_return": next_ret,
            "expected_annual_return": next_ret * 252.0,
            "beta_lag1": float(ridge[1]),
            "confidence": float(max(0.0, min(1.0, 1.0 - abs(next_ret) * 20.0))),
        },
        confidence=float(max(0.0, min(1.0, 1.0 - abs(next_ret) * 20.0))),
    )


def _lasso_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 25:
        raise ValueError("lasso regression requires at least 25 observations")
    x = clean.shift(1).dropna().to_numpy(dtype=float)
    y = clean.iloc[1:].to_numpy(dtype=float)
    corr = float(np.dot(x, y) / max(1.0, np.dot(x, x)))
    lam = 0.001
    beta = np.sign(corr) * max(0.0, abs(corr) - lam)
    next_ret = float(beta * clean.iloc[-1])
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    corr_strength = 0.0
    if x_std > 1e-12 and y_std > 1e-12:
        corr_strength = float(abs(np.corrcoef(x, y)[0, 1]))
    shrinkage_retention = float(min(1.0, abs(beta) / max(abs(corr), 1e-8))) if abs(corr) > 1e-8 else 0.0
    forecast_strength = float(
        min(1.0, abs(next_ret) / max(float(clean.std()), 1e-8) * 20.0)
    )
    confidence = float(
        max(
            0.0,
            min(
                1.0,
                corr_strength * shrinkage_retention * (0.25 + 0.75 * forecast_strength),
            ),
        )
    )
    return ModelResult(
        name="lasso",
        family="classical",
        available=True,
        metrics={
            "expected_daily_return": next_ret,
            "expected_annual_return": next_ret * 252.0,
            "sparse_beta": float(beta),
            "confidence": confidence,
        },
        confidence=confidence,
    )


def _arima_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 30:
        raise ValueError("arima requires at least 30 observations")
    if ARIMA is None:
        raise ImportError("statsmodels not available")
    fitted = ARIMA(clean, order=(1, 0, 1)).fit()
    forecast = fitted.get_forecast(steps=1)
    next_ret = float(np.asarray(forecast.predicted_mean)[0])
    conf = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
    spread = float(conf[0, 1] - conf[0, 0]) if conf.shape[1] >= 2 else 0.0
    confidence = float(max(0.0, min(1.0, 1.0 / (1.0 + spread * 100.0))))
    return ModelResult(
        name="arima",
        family="classical",
        available=True,
        metrics={
            "next_period_return_forecast": next_ret,
            "forecast_spread": spread,
            "confidence": confidence,
        },
        confidence=confidence,
    )


def _garch_model(series: pd.Series, context: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 50:
        raise ValueError("garch requires at least 50 observations")

    if arch_model is not None:
        scaled = clean * 100.0
        fitted = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        forecast = fitted.forecast(horizon=1, reindex=False)
        variance = float(forecast.variance.values[-1, 0])
        cond_vol = max(0.0, np.sqrt(variance) / 100.0)
    else:
        # Graceful fallback to EWMA volatility if arch is unavailable.
        lambda_ = float(context.get("ewma_lambda", 0.94))
        var = 0.0
        for ret in clean:
            var = lambda_ * var + (1 - lambda_) * float(ret) ** 2
        cond_vol = float(np.sqrt(max(var, 0.0)))

    ann = float(cond_vol * np.sqrt(252.0))
    confidence = float(max(0.0, min(1.0, 1.0 - min(0.95, ann * 0.8))))
    return ModelResult(
        name="garch",
        family="classical",
        available=True,
        metrics={
            "conditional_volatility": float(cond_vol),
            "volatility_annualized": ann,
            "confidence": confidence,
        },
        confidence=confidence,
    )


def _ewma_vol_model(series: pd.Series, context: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 10:
        raise ValueError("ewma requires at least 10 observations")
    lambda_ = float(context.get("ewma_lambda", 0.94))
    var = float(clean.var())
    for ret in clean:
        var = lambda_ * var + (1 - lambda_) * float(ret) ** 2
    vol = float(np.sqrt(max(0.0, var)))
    return ModelResult(
        name="ewma",
        family="classical",
        available=True,
        metrics={
            "conditional_volatility": vol,
            "volatility_annualized": vol * np.sqrt(252.0),
            "confidence": float(max(0.0, min(1.0, 1.0 - vol * 10.0))),
        },
        confidence=float(max(0.0, min(1.0, 1.0 - vol * 10.0))),
    )


def _kalman_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 10:
        raise ValueError("kalman requires at least 10 observations")
    q = 1e-6
    r = 1e-4
    x = 0.0
    p = 1.0
    filtered_states = []
    for z in clean:
        p = p + q
        k = p / (p + r)
        x = x + k * (float(z) - x)
        p = (1 - k) * p
        filtered_states.append(float(x))
    clean_std = float(clean.std())
    filtered_array = np.asarray(filtered_states, dtype=float)
    residual_std = float(np.std(clean.to_numpy(dtype=float) - filtered_array))
    signal_strength = float(np.tanh(abs(x) / max(clean_std, 1e-8) * 3.0))
    tracking_quality = float(1.0 / (1.0 + residual_std / max(clean_std, 1e-8)))
    state_certainty = float(1.0 / (1.0 + p * 100_000.0))
    confidence = float(
        max(0.0, min(0.85, signal_strength * 0.45 + tracking_quality * 0.35 + state_certainty * 0.20))
    )
    return ModelResult(
        name="kalman_filter",
        family="classical",
        available=True,
        metrics={
            "state_estimate": float(x),
            "expected_annual_return": float(x * 252.0),
            "residual_volatility": residual_std,
            "confidence": confidence,
        },
        confidence=confidence,
    )


def _cointegration_model(_: pd.Series, context: Dict[str, Any]) -> ModelResult:
    returns_df = context.get("returns_df")
    if not isinstance(returns_df, pd.DataFrame) or returns_df.shape[1] < 2:
        raise ValueError("cointegration model requires at least 2 assets")
    cols = returns_df.columns[:2]
    spread = returns_df[cols[0]] - returns_df[cols[1]]
    z = float((spread.iloc[-1] - spread.mean()) / max(1e-9, spread.std()))
    confidence = float(max(0.0, min(1.0, 1.0 - min(1.0, abs(z) / 4.0))))
    return ModelResult(
        name="cointegration_pairs",
        family="classical",
        available=True,
        metrics={
            "spread_zscore": z,
            "confidence": confidence,
            "mean_reversion_strength": float(max(0.0, 1.0 - abs(z) / 3.0)),
        },
        payload={"pair": [str(cols[0]), str(cols[1])]},
        confidence=confidence,
    )


def _pca_model(_: pd.Series, context: Dict[str, Any]) -> ModelResult:
    returns_df = context.get("returns_df")
    if not isinstance(returns_df, pd.DataFrame) or returns_df.shape[1] < 2:
        raise ValueError("pca factor model requires at least 2 assets")
    matrix = returns_df.dropna(how="any").to_numpy(dtype=float)
    if matrix.shape[0] < 5:
        raise ValueError("pca factor model requires at least 5 observations")
    matrix = matrix - matrix.mean(axis=0)
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    total = float(np.sum(s ** 2))
    explained = float((s[0] ** 2) / total) if total > 0 else 0.0
    return ModelResult(
        name="pca_factor_exposure",
        family="classical",
        available=True,
        metrics={
            "pc1_explained_variance": explained,
            "confidence": float(max(0.0, min(1.0, explained))),
        },
        confidence=float(max(0.0, min(1.0, explained))),
    )


def _bayesian_drift_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 15:
        raise ValueError("bayesian drift requires at least 15 observations")
    prior_mean = 0.0
    prior_var = 1e-4
    obs_var = float(max(1e-8, clean.var()))
    n = len(clean)
    sample_mean = float(clean.mean())
    post_var = 1.0 / (1.0 / prior_var + n / obs_var)
    post_mean = post_var * (prior_mean / prior_var + n * sample_mean / obs_var)
    ci_half = 1.96 * np.sqrt(post_var)
    confidence = float(max(0.0, min(1.0, 1.0 - ci_half * 300.0)))
    return ModelResult(
        name="bayesian_drift",
        family="bayesian",
        available=True,
        metrics={
            "posterior_mean_daily_return": float(post_mean),
            "posterior_annual_return": float(post_mean * 252.0),
            "posterior_interval_width": float(ci_half * 2.0),
            "confidence": confidence,
        },
        confidence=confidence,
    )


def _bayesian_regression_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 20:
        raise ValueError("bayesian regression requires at least 20 observations")
    x = clean.shift(1).dropna().to_numpy(dtype=float)
    y = clean.iloc[1:].to_numpy(dtype=float)
    x_mat = np.column_stack([np.ones_like(x), x])
    prior_precision = np.eye(2) * 10.0
    noise_var = float(max(1e-8, np.var(y - y.mean())))
    post_precision = prior_precision + (x_mat.T @ x_mat) / noise_var
    post_cov = np.linalg.inv(post_precision)
    post_mean = post_cov @ ((x_mat.T @ y) / noise_var)
    next_ret = float(post_mean[0] + post_mean[1] * clean.iloc[-1])
    coef_var = float(post_cov[1, 1])
    confidence = float(max(0.0, min(1.0, 1.0 / (1.0 + coef_var * 5000.0))))
    return ModelResult(
        name="bayesian_regression",
        family="bayesian",
        available=True,
        metrics={
            "expected_daily_return": next_ret,
            "expected_annual_return": next_ret * 252.0,
            "posterior_beta": float(post_mean[1]),
            "posterior_beta_var": coef_var,
            "confidence": confidence,
        },
        confidence=confidence,
    )


def _bayesian_vol_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 20:
        raise ValueError("bayesian volatility requires at least 20 observations")
    alpha0 = 3.0
    beta0 = 1e-4
    residual = clean - clean.mean()
    n = len(residual)
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * float(np.sum(np.square(residual.to_numpy(dtype=float))))
    var_post = beta_n / max(alpha_n - 1.0, 1.0)
    vol = float(np.sqrt(max(0.0, var_post)))
    conf = float(max(0.0, min(1.0, 1.0 - vol * 12.0)))
    return ModelResult(
        name="bayesian_volatility",
        family="bayesian",
        available=True,
        metrics={
            "posterior_volatility_daily": vol,
            "posterior_volatility_annualized": vol * np.sqrt(252.0),
            "confidence": conf,
        },
        confidence=conf,
    )


def _regime_probability_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 30:
        raise ValueError("regime probability model requires at least 30 observations")
    slow = float(clean.rolling(20).std().iloc[-1])
    fast = float(clean.rolling(5).std().iloc[-1])
    ratio = fast / max(1e-8, slow)
    risk_off_prob = float(1.0 / (1.0 + np.exp(-(ratio - 1.0) * 4.0)))
    return ModelResult(
        name="regime_probability",
        family="bayesian",
        available=True,
        metrics={
            "risk_off_probability": risk_off_prob,
            "risk_on_probability": 1.0 - risk_off_prob,
            "confidence": float(max(risk_off_prob, 1.0 - risk_off_prob)),
        },
        confidence=float(max(risk_off_prob, 1.0 - risk_off_prob)),
    )


def _bma_model(_: pd.Series, context: Dict[str, Any]) -> ModelResult:
    interim = context.get("interim_models", {})
    candidates = [
        interim.get("bayesian_drift"),
        interim.get("linear_regression"),
        interim.get("arima"),
    ]
    usable = [m for m in candidates if isinstance(m, ModelResult) and m.available]
    if not usable:
        raise ValueError("bayesian model averaging requires prior model outputs")

    values = []
    weights = []
    for model in usable:
        if "posterior_annual_return" in model.metrics:
            ann = float(model.metrics["posterior_annual_return"])
        elif "expected_annual_return" in model.metrics:
            ann = float(model.metrics["expected_annual_return"])
        else:
            ann = float(model.metrics.get("next_period_return_forecast", 0.0) * 252.0)
        w = max(0.01, float(model.confidence or model.metrics.get("confidence", 0.25)))
        values.append(ann)
        weights.append(w)

    total_w = sum(weights)
    weighted = float(sum(v * w for v, w in zip(values, weights, strict=False)) / total_w)
    disagreement = float(np.std(values)) if len(values) > 1 else 0.0
    conf = float(max(0.0, min(1.0, 1.0 / (1.0 + disagreement * 3.0))))
    return ModelResult(
        name="bayesian_model_averaging",
        family="bayesian",
        available=True,
        metrics={
            "bma_expected_annual_return": weighted,
            "disagreement": disagreement,
            "confidence": conf,
        },
        confidence=conf,
    )


def _logistic_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 40:
        raise ValueError("logistic model requires at least 40 observations")
    lag = clean.shift(1).dropna().to_numpy(dtype=float)
    y = (clean.iloc[1:] > 0).astype(float).to_numpy(dtype=float)

    w0 = 0.0
    w1 = 0.0
    lr = 0.2
    for _ in range(120):
        z = w0 + w1 * lag
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        w0 -= lr * float(np.mean(err))
        w1 -= lr * float(np.mean(err * lag))

    z_next = w0 + w1 * float(clean.iloc[-1])
    prob_up = float(1.0 / (1.0 + np.exp(-np.clip(z_next, -30, 30))))
    conf = float(abs(prob_up - 0.5) * 2.0)
    return ModelResult(
        name="logistic_regression",
        family="ml",
        available=True,
        metrics={
            "probability_up": prob_up,
            "confidence": conf,
            "weight_lag1": float(w1),
        },
        confidence=conf,
    )


def _tree_model(series: pd.Series, _: Dict[str, Any]) -> ModelResult:
    clean = _series(series)
    if len(clean) < 40:
        raise ValueError("tree model requires at least 40 observations")
    lag = clean.shift(1).dropna().to_numpy(dtype=float)
    target = (clean.iloc[1:] > 0).astype(float).to_numpy(dtype=float)
    threshold = float(np.median(lag))
    left = target[lag <= threshold]
    right = target[lag > threshold]
    p_left = float(left.mean()) if len(left) else 0.5
    p_right = float(right.mean()) if len(right) else 0.5
    p_next = p_left if float(clean.iloc[-1]) <= threshold else p_right
    conf = float(abs(p_next - 0.5) * 2.0)
    return ModelResult(
        name="tree_based",
        family="ml",
        available=True,
        metrics={
            "probability_up": p_next,
            "split_threshold": threshold,
            "confidence": conf,
        },
        confidence=conf,
    )


def _ensemble_model(_: pd.Series, context: Dict[str, Any]) -> ModelResult:
    interim = context.get("interim_models", {})
    logit = interim.get("logistic_regression")
    tree = interim.get("tree_based")
    probs = []
    confs = []
    for model in [logit, tree]:
        if isinstance(model, ModelResult) and model.available:
            probs.append(float(model.metrics.get("probability_up", 0.5)))
            confs.append(float(model.confidence or model.metrics.get("confidence", 0.5)))
    if not probs:
        raise ValueError("ensemble requires logistic or tree model")
    prob = float(np.mean(probs))
    conf = float(np.mean(confs))
    return ModelResult(
        name="ensemble",
        family="ml",
        available=True,
        metrics={
            "probability_up": prob,
            "confidence": conf,
        },
        confidence=conf,
    )


def _calibrated_model(_: pd.Series, context: Dict[str, Any]) -> ModelResult:
    interim = context.get("interim_models", {})
    base = interim.get("ensemble")
    if not isinstance(base, ModelResult) or not base.available:
        raise ValueError("calibrated probabilities require ensemble model")
    p = float(base.metrics.get("probability_up", 0.5))
    calibrated = float(0.15 * 0.5 + 0.85 * p)
    conf = float(abs(calibrated - 0.5) * 2.0)
    return ModelResult(
        name="calibrated_probabilities",
        family="ml",
        available=True,
        metrics={
            "probability_up": calibrated,
            "confidence": conf,
        },
        confidence=conf,
    )


def _black_litterman_model(_: pd.Series, context: Dict[str, Any]) -> ModelResult:
    returns_df = context.get("returns_df")
    if not isinstance(returns_df, pd.DataFrame) or returns_df.shape[1] < 2:
        raise ValueError("black-litterman model requires returns for at least 2 assets")

    clean = returns_df.dropna(how="any")
    if clean.shape[0] < 20:
        raise ValueError("black-litterman model requires at least 20 observations")

    tickers = [str(col) for col in clean.columns]
    n_assets = len(tickers)

    raw_weights = context.get("market_weights", [])
    if isinstance(raw_weights, (list, tuple, np.ndarray)) and len(raw_weights) == n_assets:
        market_weights = np.asarray(raw_weights, dtype=float)
    else:
        market_weights = np.ones(n_assets, dtype=float) / n_assets

    weight_sum = float(market_weights.sum())
    if weight_sum <= 0:
        market_weights = np.ones(n_assets, dtype=float) / n_assets
    else:
        market_weights = market_weights / weight_sum

    cov_annual = clean.cov().to_numpy(dtype=float) * 252.0
    if not np.all(np.isfinite(cov_annual)):
        raise ValueError("black-litterman covariance matrix contains invalid values")

    risk_aversion = float(context.get("bl_risk_aversion", 2.5))
    tau = float(context.get("bl_tau", 0.05))
    tau = max(1e-6, tau)

    pi = risk_aversion * cov_annual @ market_weights

    raw_views = context.get("bl_views", {})
    views = raw_views if isinstance(raw_views, dict) else {}
    view_keys = [ticker for ticker in tickers if ticker in views]

    if not view_keys:
        posterior = pi
        confidence = 0.35
    else:
        p = np.zeros((len(view_keys), n_assets), dtype=float)
        q = np.zeros(len(view_keys), dtype=float)
        for idx, ticker in enumerate(view_keys):
            p[idx, tickers.index(ticker)] = 1.0
            q[idx] = float(views.get(ticker, 0.0))

        omega = np.diag(np.maximum(np.diag(p @ (tau * cov_annual) @ p.T), 1e-6))
        inv_tau_cov = np.linalg.pinv(tau * cov_annual)
        inv_omega = np.linalg.pinv(omega)
        posterior_cov = np.linalg.pinv(inv_tau_cov + p.T @ inv_omega @ p)
        posterior = posterior_cov @ (inv_tau_cov @ pi + p.T @ inv_omega @ q)
        confidence = float(
            max(
                0.0,
                min(
                    0.9,
                    (1.0 / (1.0 + float(np.std(posterior - pi)) * 25.0))
                    * min(1.0, 0.45 + len(view_keys) * 0.15),
                ),
            )
        )

    implied_alpha = posterior - pi
    posterior_portfolio_return = float(np.dot(market_weights, posterior))
    implied_alpha_portfolio = float(np.dot(market_weights, implied_alpha))
    tilt_strength = float(np.max(np.abs(implied_alpha)))

    return ModelResult(
        name="black_litterman",
        family="portfolio",
        available=True,
        metrics={
            "posterior_expected_annual_return": posterior_portfolio_return,
            "implied_alpha_portfolio": implied_alpha_portfolio,
            "tilt_strength": tilt_strength,
            "view_count": float(len(view_keys)),
            "passive_reference_only": float(1.0 if not view_keys else 0.0),
            "confidence": confidence,
        },
        payload={
            "tickers": tickers,
            "market_weights": market_weights.tolist(),
            "implied_equilibrium_return": {ticker: float(value) for ticker, value in zip(tickers, pi, strict=False)},
            "posterior_return": {ticker: float(value) for ticker, value in zip(tickers, posterior, strict=False)},
            "implied_alpha": {ticker: float(value) for ticker, value in zip(tickers, implied_alpha, strict=False)},
            "view_count": len(view_keys),
        },
        confidence=confidence,
    )


def build_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("linear_regression", "classical", lambda s, c: _safe_model("linear_regression", "classical", _linear_regression_model, s, c))
    registry.register("ridge", "classical", lambda s, c: _safe_model("ridge", "classical", _ridge_model, s, c))
    registry.register("lasso", "classical", lambda s, c: _safe_model("lasso", "classical", _lasso_model, s, c))
    registry.register("arima", "classical", lambda s, c: _safe_model("arima", "classical", _arima_model, s, c))
    registry.register("garch", "classical", lambda s, c: _safe_model("garch", "classical", _garch_model, s, c))
    registry.register("ewma", "classical", lambda s, c: _safe_model("ewma", "classical", _ewma_vol_model, s, c))
    registry.register("kalman_filter", "classical", lambda s, c: _safe_model("kalman_filter", "classical", _kalman_model, s, c))
    registry.register("cointegration_pairs", "classical", lambda s, c: _safe_model("cointegration_pairs", "classical", _cointegration_model, s, c))
    registry.register("pca_factor_exposure", "classical", lambda s, c: _safe_model("pca_factor_exposure", "classical", _pca_model, s, c))

    registry.register("bayesian_regression", "bayesian", lambda s, c: _safe_model("bayesian_regression", "bayesian", _bayesian_regression_model, s, c))
    registry.register("bayesian_drift", "bayesian", lambda s, c: _safe_model("bayesian_drift", "bayesian", _bayesian_drift_model, s, c))
    registry.register("bayesian_volatility", "bayesian", lambda s, c: _safe_model("bayesian_volatility", "bayesian", _bayesian_vol_model, s, c))
    registry.register("regime_probability", "bayesian", lambda s, c: _safe_model("regime_probability", "bayesian", _regime_probability_model, s, c))

    registry.register("logistic_regression", "ml", lambda s, c: _safe_model("logistic_regression", "ml", _logistic_model, s, c))
    registry.register("tree_based", "ml", lambda s, c: _safe_model("tree_based", "ml", _tree_model, s, c))
    registry.register("black_litterman", "portfolio", lambda s, c: _safe_model("black_litterman", "portfolio", _black_litterman_model, s, c))

    return registry


def run_model_bundle(series: pd.Series, context: Dict[str, Any] | None = None) -> Dict[str, ModelResult]:
    registry = build_model_registry()
    run_context = dict(context or {})
    interim: Dict[str, ModelResult] = {}

    for entry in registry.items():
        run_context["interim_models"] = interim
        interim[entry.name] = entry.runner(series, run_context)

    # Dependent bundle models run after base models.
    interim["bayesian_model_averaging"] = _safe_model(
        "bayesian_model_averaging",
        "bayesian",
        _bma_model,
        series,
        {**run_context, "interim_models": interim},
    )
    interim["ensemble"] = _safe_model(
        "ensemble",
        "ml",
        _ensemble_model,
        series,
        {**run_context, "interim_models": interim},
    )
    interim["calibrated_probabilities"] = _safe_model(
        "calibrated_probabilities",
        "ml",
        _calibrated_model,
        series,
        {**run_context, "interim_models": interim},
    )

    return interim
