from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .portfolio_metrics import calculate_portfolio_core_metrics


CASH_SYMBOLS = {"BIL", "SGOV", "SHV", "TBIL", "CLIP", "MINT", "CASH", "USD"}
BOND_SYMBOLS = {
    "BND",
    "AGG",
    "TLT",
    "IEF",
    "SHY",
    "TIP",
    "LQD",
    "HYG",
    "MUB",
    "VGIT",
    "VGSH",
    "BIV",
    "IEI",
}
GOLD_SYMBOLS = {"GLD", "IAU", "GLDM", "PHYS", "SGOL", "GOLD"}
COMMODITY_SYMBOLS = {"DBC", "GSG", "PDBC", "USO", "DBA", "XLE", "XOP"}
CRYPTO_SYMBOLS = {"BTC", "ETH", "SOL", "MSTR", "COIN", "IBIT", "FBTC", "ARKB", "BITB"}
TECH_TICKERS = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "TSM",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "QQQ",
    "SMH",
    "SOXX",
    "CRM",
    "NOW",
    "SHOP",
}
ROLE_COLUMNS = ["equity", "bond", "gold", "commodity", "crypto", "cash"]


SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Flash Crash": {
        "category": "Synthetic Stress",
        "era": "Microstructure shock",
        "description": "One-day liquidity vacuum followed by a shaky rebound.",
        "playbook": "Protect beta and preserve dry powder before averaging into the rebound.",
        "horizon_days": 30,
        "phases": [
            {
                "label": "Gap lower",
                "duration": 1,
                "vol_multiplier": 2.8,
                "role_overlays": {
                    "equity": -0.12,
                    "bond": 0.010,
                    "gold": 0.028,
                    "commodity": -0.035,
                    "crypto": -0.20,
                    "cash": 0.0,
                },
            },
            {
                "label": "Aftershock",
                "duration": 4,
                "vol_multiplier": 1.9,
                "role_overlays": {
                    "equity": -0.008,
                    "bond": 0.001,
                    "gold": 0.002,
                    "commodity": -0.002,
                    "crypto": -0.015,
                    "cash": 0.0,
                },
            },
            {
                "label": "Fragile bounce",
                "duration": 25,
                "vol_multiplier": 1.25,
                "role_overlays": {
                    "equity": 0.0014,
                    "bond": -0.0002,
                    "gold": -0.0004,
                    "commodity": 0.0002,
                    "crypto": 0.0010,
                    "cash": 0.0,
                },
            },
        ],
    },
    "Stagflation Grind": {
        "category": "Synthetic Stress",
        "era": "Inflation regime",
        "description": "Equities and duration assets leak lower while inflation hedges outperform.",
        "playbook": "Reduce long-duration exposure and look for real-asset hedges or pricing power names.",
        "horizon_days": 60,
        "phases": [
            {
                "label": "Sticky inflation",
                "duration": 60,
                "vol_multiplier": 1.5,
                "role_overlays": {
                    "equity": -0.0015,
                    "bond": -0.0012,
                    "gold": 0.0008,
                    "commodity": 0.0011,
                    "crypto": -0.0022,
                    "cash": 0.0001,
                },
            }
        ],
    },
    "Rate Shock": {
        "category": "Synthetic Stress",
        "era": "Rates repricing",
        "description": "Rates spike higher and duration-heavy exposures reprice aggressively.",
        "playbook": "Shorten duration, prefer cash-flow resilient assets, and reassess leverage assumptions.",
        "horizon_days": 45,
        "phases": [
            {
                "label": "Bond selloff",
                "duration": 10,
                "vol_multiplier": 1.7,
                "role_overlays": {
                    "equity": -0.0014,
                    "bond": -0.0042,
                    "gold": -0.0003,
                    "commodity": 0.0002,
                    "crypto": -0.0025,
                    "cash": 0.0001,
                },
            },
            {
                "label": "Secondary repricing",
                "duration": 35,
                "vol_multiplier": 1.25,
                "role_overlays": {
                    "equity": -0.0005,
                    "bond": -0.0011,
                    "gold": 0.0002,
                    "commodity": 0.0003,
                    "crypto": -0.0010,
                    "cash": 0.0001,
                },
            },
        ],
    },
    "Tech Decompression": {
        "category": "Synthetic Stress",
        "era": "Crowded growth unwind",
        "description": "Crowded growth leadership unwinds while defensives hold up relatively better.",
        "playbook": "Cut concentration in mega-cap winners and rebalance toward balanced factor exposure.",
        "horizon_days": 35,
        "phases": [
            {
                "label": "Leadership unwind",
                "duration": 5,
                "vol_multiplier": 2.0,
                "role_overlays": {
                    "equity": -0.005,
                    "bond": 0.0008,
                    "gold": 0.0010,
                    "commodity": -0.0005,
                    "crypto": -0.012,
                    "cash": 0.0,
                },
                "group_overlays": {
                    "tech": -0.009,
                },
            },
            {
                "label": "Multiple compression",
                "duration": 30,
                "vol_multiplier": 1.35,
                "role_overlays": {
                    "equity": -0.0010,
                    "bond": 0.0003,
                    "gold": 0.0002,
                    "commodity": 0.0,
                    "crypto": -0.0020,
                    "cash": 0.0,
                },
                "group_overlays": {
                    "tech": -0.0025,
                },
            },
        ],
    },
    "Liquidity Crunch": {
        "category": "Synthetic Stress",
        "era": "Cross-asset deleveraging",
        "description": "Cross-asset stress with correlations jumping toward one.",
        "playbook": "Focus on survival: gross down, cash up, and avoid assuming diversification will rescue the book.",
        "horizon_days": 25,
        "phases": [
            {
                "label": "Forced liquidation",
                "duration": 7,
                "vol_multiplier": 2.4,
                "role_overlays": {
                    "equity": -0.013,
                    "bond": -0.0015,
                    "gold": -0.0020,
                    "commodity": -0.0040,
                    "crypto": -0.025,
                    "cash": 0.0,
                },
            },
            {
                "label": "Funding stress",
                "duration": 18,
                "vol_multiplier": 1.6,
                "role_overlays": {
                    "equity": -0.0015,
                    "bond": 0.0002,
                    "gold": 0.0006,
                    "commodity": -0.0008,
                    "crypto": -0.0035,
                    "cash": 0.0001,
                },
            },
        ],
    },
    "2008 Global Financial Crisis": {
        "category": "Historical Crisis",
        "era": "2008-2009",
        "description": "Banking-system stress, violent deleveraging, and a deep recessionary second leg.",
        "playbook": "Prioritize balance-sheet quality, liquidity, and explicit downside buffers over mean-reversion instincts.",
        "horizon_days": 90,
        "phases": [
            {
                "label": "Credit seizure",
                "duration": 12,
                "vol_multiplier": 2.5,
                "role_overlays": {
                    "equity": -0.013,
                    "bond": 0.0012,
                    "gold": -0.0010,
                    "commodity": -0.0080,
                    "crypto": -0.030,
                    "cash": 0.0001,
                },
                "group_overlays": {"tech": -0.0025},
            },
            {
                "label": "Forced deleveraging",
                "duration": 28,
                "vol_multiplier": 1.9,
                "role_overlays": {
                    "equity": -0.0045,
                    "bond": 0.0003,
                    "gold": 0.0005,
                    "commodity": -0.0035,
                    "crypto": -0.010,
                    "cash": 0.0001,
                },
            },
            {
                "label": "Policy bounce, weak tape",
                "duration": 50,
                "vol_multiplier": 1.35,
                "role_overlays": {
                    "equity": 0.0004,
                    "bond": -0.0001,
                    "gold": 0.0004,
                    "commodity": -0.0003,
                    "crypto": 0.0002,
                    "cash": 0.0,
                },
            },
        ],
    },
    "Dot-Com Bust": {
        "category": "Historical Crisis",
        "era": "2000-2002",
        "description": "Long-duration growth multiples collapse while broad diversification helps only partially.",
        "playbook": "Reduce narrative-driven concentration and test what happens if leadership names rerate for longer than expected.",
        "horizon_days": 75,
        "phases": [
            {
                "label": "Valuation break",
                "duration": 18,
                "vol_multiplier": 1.9,
                "role_overlays": {
                    "equity": -0.0035,
                    "bond": 0.0007,
                    "gold": 0.0005,
                    "commodity": -0.0004,
                    "crypto": -0.010,
                    "cash": 0.0001,
                },
                "group_overlays": {"tech": -0.0100},
            },
            {
                "label": "Hope rally fade",
                "duration": 17,
                "vol_multiplier": 1.45,
                "role_overlays": {
                    "equity": 0.0004,
                    "bond": 0.0002,
                    "gold": -0.0001,
                    "commodity": 0.0,
                    "crypto": -0.0015,
                    "cash": 0.0,
                },
                "group_overlays": {"tech": -0.0020},
            },
            {
                "label": "Second tech washout",
                "duration": 40,
                "vol_multiplier": 1.6,
                "role_overlays": {
                    "equity": -0.0018,
                    "bond": 0.0002,
                    "gold": 0.0002,
                    "commodity": -0.0003,
                    "crypto": -0.0030,
                    "cash": 0.0,
                },
                "group_overlays": {"tech": -0.0045},
            },
        ],
    },
    "1970s Stagflation Spiral": {
        "category": "Historical Crisis",
        "era": "1973-1974 / late 1970s",
        "description": "Inflation shock, weak growth, and poor diversification between stocks and bonds.",
        "playbook": "Treat real assets and pricing-power businesses as insurance, not as optional style tilts.",
        "horizon_days": 90,
        "phases": [
            {
                "label": "Oil shock",
                "duration": 20,
                "vol_multiplier": 1.8,
                "role_overlays": {
                    "equity": -0.0028,
                    "bond": -0.0018,
                    "gold": 0.0012,
                    "commodity": 0.0025,
                    "crypto": -0.0040,
                    "cash": 0.0002,
                },
            },
            {
                "label": "Inflation persistence",
                "duration": 45,
                "vol_multiplier": 1.4,
                "role_overlays": {
                    "equity": -0.0015,
                    "bond": -0.0010,
                    "gold": 0.0009,
                    "commodity": 0.0014,
                    "crypto": -0.0015,
                    "cash": 0.0002,
                },
            },
            {
                "label": "Policy whiplash",
                "duration": 25,
                "vol_multiplier": 1.55,
                "role_overlays": {
                    "equity": -0.0004,
                    "bond": -0.0007,
                    "gold": 0.0005,
                    "commodity": 0.0006,
                    "crypto": -0.0010,
                    "cash": 0.0001,
                },
            },
        ],
    },
    "1929 Great Depression Analog": {
        "category": "Historical Crisis",
        "era": "1929-1932 analog",
        "description": "A cascading equity collapse with failed relief rallies and a grinding second leg lower.",
        "playbook": "Assume reflexive drawdowns can last longer than intuition suggests; survival comes from cash, patience, and rules.",
        "horizon_days": 100,
        "phases": [
            {
                "label": "Crash week",
                "duration": 5,
                "vol_multiplier": 3.1,
                "role_overlays": {
                    "equity": -0.065,
                    "bond": 0.0020,
                    "gold": 0.0060,
                    "commodity": -0.0150,
                    "crypto": -0.120,
                    "cash": 0.0,
                },
                "group_overlays": {"tech": -0.0120},
            },
            {
                "label": "Relief rally",
                "duration": 10,
                "vol_multiplier": 1.9,
                "role_overlays": {
                    "equity": 0.0040,
                    "bond": 0.0002,
                    "gold": -0.0003,
                    "commodity": 0.0001,
                    "crypto": 0.0060,
                    "cash": 0.0,
                },
            },
            {
                "label": "Depression slide",
                "duration": 85,
                "vol_multiplier": 1.75,
                "role_overlays": {
                    "equity": -0.0040,
                    "bond": 0.0004,
                    "gold": 0.0007,
                    "commodity": -0.0014,
                    "crypto": -0.0060,
                    "cash": 0.0001,
                },
            },
        ],
    },
    "2020 Covid Crash": {
        "category": "Historical Crisis",
        "era": "Q1 2020",
        "description": "Shock collapse, correlation spike, then an unusually fast liquidity-fueled rebound.",
        "playbook": "Use it to test whether your process can stay invested through violent mean reversion after the initial panic.",
        "horizon_days": 45,
        "phases": [
            {
                "label": "Lockdown panic",
                "duration": 8,
                "vol_multiplier": 2.7,
                "role_overlays": {
                    "equity": -0.018,
                    "bond": 0.0010,
                    "gold": -0.0010,
                    "commodity": -0.0100,
                    "crypto": -0.030,
                    "cash": 0.0,
                },
            },
            {
                "label": "Policy floor",
                "duration": 10,
                "vol_multiplier": 1.9,
                "role_overlays": {
                    "equity": 0.0060,
                    "bond": -0.0001,
                    "gold": 0.0006,
                    "commodity": 0.0012,
                    "crypto": 0.0120,
                    "cash": 0.0,
                },
                "group_overlays": {"tech": 0.0025},
            },
            {
                "label": "V-shape chase",
                "duration": 27,
                "vol_multiplier": 1.25,
                "role_overlays": {
                    "equity": 0.0018,
                    "bond": -0.0002,
                    "gold": 0.0001,
                    "commodity": 0.0008,
                    "crypto": 0.0025,
                    "cash": 0.0,
                },
                "group_overlays": {"tech": 0.0018},
            },
        ],
    },
}


def classify_asset_role(symbol: str) -> str:
    sym = str(symbol or "").upper().strip()
    if sym in CASH_SYMBOLS:
        return "cash"
    if sym in BOND_SYMBOLS:
        return "bond"
    if sym in GOLD_SYMBOLS:
        return "gold"
    if sym in COMMODITY_SYMBOLS:
        return "commodity"
    if sym in CRYPTO_SYMBOLS or any(token in sym for token in ("BTC", "ETH", "SOL")):
        return "crypto"
    return "equity"


def list_scenario_presets() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "category": str(config.get("category", "Scenario")),
            "era": str(config.get("era", "")),
            "description": str(config.get("description", "")),
            "playbook": str(config.get("playbook", "")),
            "horizon_days": int(config.get("horizon_days", 30)),
        }
        for name, config in SCENARIO_PRESETS.items()
    ]


def build_role_exposure_table(tickers: List[str], weights: np.ndarray) -> pd.DataFrame:
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.size == 0:
        weight_array = np.zeros(len(tickers), dtype=float)
    rows = []
    for ticker, weight in zip(tickers, weight_array, strict=False):
        rows.append(
            {
                "Ticker": str(ticker),
                "Role": classify_asset_role(str(ticker)),
                "Weight": float(weight),
                "Tech Bucket": bool(str(ticker).upper() in TECH_TICKERS),
            }
        )
    return pd.DataFrame(rows)


def _normalized_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    raw = np.asarray(weights, dtype=float)
    if raw.size != n_assets:
        raw = np.ones(n_assets, dtype=float)
    total = float(raw.sum())
    if total <= 0:
        return np.ones(n_assets, dtype=float) / float(n_assets)
    return raw / total


def _baseline_asset_returns(returns_df: pd.DataFrame, horizon_days: int) -> np.ndarray:
    clean = returns_df.dropna(how="any")
    if clean.empty:
        raise ValueError("Scenario engine requires return history.")

    lookback = min(len(clean), max(40, horizon_days * 2))
    base = clean.tail(lookback).to_numpy(dtype=float)
    if len(base) >= horizon_days:
        return base[-horizon_days:].copy()

    reps = int(np.ceil(horizon_days / max(1, len(base))))
    return np.tile(base, (reps, 1))[:horizon_days].copy()


def _group_indices(tickers: List[str]) -> Dict[str, List[int]]:
    tech = [idx for idx, ticker in enumerate(tickers) if str(ticker).upper() in TECH_TICKERS]
    return {"tech": tech}


def _action_cue(total_return: float, max_drawdown: float, worst_day: float) -> str:
    if max_drawdown <= -0.18 or worst_day <= -0.08:
        return "High alert"
    if max_drawdown <= -0.10 or total_return <= -0.08:
        return "Hedge / rebalance"
    if total_return <= -0.03:
        return "Monitor closely"
    return "Contained"


def _series_stats(portfolio_returns: pd.Series, initial_value: float) -> Dict[str, Any]:
    core = calculate_portfolio_core_metrics(portfolio_returns, risk_free_rate=0.0)
    path = initial_value * (1.0 + portfolio_returns).cumprod()
    path = pd.concat(
        [
            pd.Series([initial_value], index=[0], dtype=float),
            pd.Series(path.to_numpy(dtype=float), index=np.arange(1, len(path) + 1), dtype=float),
        ]
    )
    running_max = path.cummax()
    drawdown = path / running_max - 1.0

    recovery_day = None
    recovered = np.where(path.to_numpy(dtype=float)[1:] >= initial_value)[0]
    if recovered.size:
        recovery_day = int(recovered[0] + 1)

    return {
        **core,
        "final_value": float(path.iloc[-1]),
        "worst_day": float(portfolio_returns.min()),
        "days_underwater": int((path.iloc[1:] < initial_value).sum()),
        "recovery_day": recovery_day,
        "path": path.rename("value"),
        "drawdown_path": drawdown.rename("drawdown"),
    }


def run_scenario_preset(
    returns_df: pd.DataFrame,
    tickers: List[str],
    weights: np.ndarray,
    preset_name: str,
    severity: float = 1.0,
    initial_value: float = 100_000.0,
    horizon_override: int | None = None,
) -> Dict[str, Any]:
    if preset_name not in SCENARIO_PRESETS:
        raise KeyError(f"Unknown scenario preset: {preset_name}")

    preset = SCENARIO_PRESETS[preset_name]
    horizon_days = int(horizon_override or preset.get("horizon_days", 30))
    if horizon_days <= 0:
        raise ValueError("Scenario horizon must be positive.")

    tickers_list = [str(ticker) for ticker in tickers]
    if returns_df.empty or returns_df.shape[1] == 0:
        raise ValueError("Scenario engine requires asset return columns.")

    weight_array = _normalized_weights(weights, returns_df.shape[1])
    roles = [classify_asset_role(ticker) for ticker in tickers_list]
    group_map = _group_indices(tickers_list)
    phase_rows: List[Dict[str, Any]] = []
    shock_rows: List[Dict[str, Any]] = []
    daily_phase_labels: List[str] = []

    baseline_asset_returns = _baseline_asset_returns(returns_df, horizon_days)
    stressed_asset_returns = baseline_asset_returns.copy()
    baseline_mean = baseline_asset_returns.mean(axis=0, keepdims=True)

    cursor = 0
    for phase in preset.get("phases", []):
        duration = int(phase.get("duration", 0))
        if duration <= 0 or cursor >= horizon_days:
            continue
        end = min(horizon_days, cursor + duration)
        idx = slice(cursor, end)
        vol_multiplier = float(phase.get("vol_multiplier", 1.0))
        label = str(phase.get("label", f"Phase {len(phase_rows) + 1}"))
        stressed_asset_returns[idx] = baseline_mean + (stressed_asset_returns[idx] - baseline_mean) * vol_multiplier
        role_overlays = dict(phase.get("role_overlays", {}))
        group_overlays = dict(phase.get("group_overlays", {}))

        for role, delta in role_overlays.items():
            role_cols = [col for col, asset_role in enumerate(roles) if asset_role == role]
            if role_cols:
                stressed_asset_returns[idx][:, role_cols] += float(delta) * float(severity)

        for group_name, delta in group_overlays.items():
            cols = group_map.get(str(group_name), [])
            if cols:
                stressed_asset_returns[idx][:, cols] += float(delta) * float(severity)

        phase_rows.append(
            {
                "Phase": label,
                "Start Day": int(cursor + 1),
                "End Day": int(end),
                "Duration": int(end - cursor),
                "Vol Multiplier": vol_multiplier,
            }
        )
        shock_rows.append(
            {
                "Phase": label,
                **{role: float(role_overlays.get(role, 0.0) * float(severity)) for role in ROLE_COLUMNS},
                "tech": float(group_overlays.get("tech", 0.0) * float(severity)),
            }
        )
        daily_phase_labels.extend([label] * int(end - cursor))

        cursor = end

    stressed_asset_returns = np.clip(stressed_asset_returns, -0.85, 0.85)

    baseline_portfolio_returns = pd.Series(
        baseline_asset_returns @ weight_array,
        index=np.arange(1, horizon_days + 1),
        name="baseline",
        dtype=float,
    )
    stressed_portfolio_returns = pd.Series(
        stressed_asset_returns @ weight_array,
        index=np.arange(1, horizon_days + 1),
        name="stressed",
        dtype=float,
    )

    baseline_stats = _series_stats(baseline_portfolio_returns, initial_value=initial_value)
    stressed_stats = _series_stats(stressed_portfolio_returns, initial_value=initial_value)

    baseline_asset_terminal = (1.0 + pd.DataFrame(baseline_asset_returns, columns=tickers_list)).prod(axis=0) - 1.0
    stressed_asset_terminal = (1.0 + pd.DataFrame(stressed_asset_returns, columns=tickers_list)).prod(axis=0) - 1.0
    impact_proxy = (
        initial_value
        * weight_array
        * (stressed_asset_terminal.to_numpy(dtype=float) - baseline_asset_terminal.to_numpy(dtype=float))
    )
    impact_series = pd.Series(impact_proxy, index=tickers_list, name="Stress Gap")

    return {
        "name": preset_name,
        "category": str(preset.get("category", "Scenario")),
        "era": str(preset.get("era", "")),
        "description": str(preset.get("description", "")),
        "playbook": str(preset.get("playbook", "")),
        "horizon_days": horizon_days,
        "severity": float(severity),
        "phase_table": pd.DataFrame(phase_rows),
        "shock_map": pd.DataFrame(shock_rows).set_index("Phase") if shock_rows else pd.DataFrame(),
        "daily_phase_labels": daily_phase_labels,
        "baseline_returns": baseline_portfolio_returns,
        "stressed_returns": stressed_portfolio_returns,
        "baseline_path": baseline_stats["path"],
        "stressed_path": stressed_stats["path"],
        "baseline_drawdown": baseline_stats["drawdown_path"],
        "stressed_drawdown": stressed_stats["drawdown_path"],
        "asset_impact_proxy": impact_series.sort_values(),
        "baseline_stats": baseline_stats,
        "stressed_stats": stressed_stats,
        "role_exposures": build_role_exposure_table(tickers_list, weight_array),
        "action_cue": _action_cue(
            total_return=float(stressed_stats["total_return"]),
            max_drawdown=float(stressed_stats["max_drawdown"]),
            worst_day=float(stressed_stats["worst_day"]),
        ),
    }


def build_scenario_suite(
    returns_df: pd.DataFrame,
    tickers: List[str],
    weights: np.ndarray,
    severity: float = 1.0,
    initial_value: float = 100_000.0,
    horizon_override: int | None = None,
) -> Dict[str, Any]:
    scenarios: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for preset_name in SCENARIO_PRESETS:
        scenario = run_scenario_preset(
            returns_df=returns_df,
            tickers=tickers,
            weights=weights,
            preset_name=preset_name,
            severity=severity,
            initial_value=initial_value,
            horizon_override=horizon_override,
        )
        scenarios[preset_name] = scenario

        stressed_stats = scenario["stressed_stats"]
        baseline_stats = scenario["baseline_stats"]
        rows.append(
            {
                "Scenario": preset_name,
                "Category": str(scenario["category"]),
                "Era": str(scenario["era"]),
                "Horizon": int(scenario["horizon_days"]),
                "Final Value": float(stressed_stats["final_value"]),
                "Total Return": float(stressed_stats["total_return"]),
                "Max Drawdown": float(stressed_stats["max_drawdown"]),
                "Worst Day": float(stressed_stats["worst_day"]),
                "Stress Gap": float(stressed_stats["total_return"] - baseline_stats["total_return"]),
                "Days Underwater": int(stressed_stats["days_underwater"]),
                "Action Cue": scenario["action_cue"],
            }
        )

    return {
        "rows": pd.DataFrame(rows).sort_values(by="Total Return"),
        "scenarios": scenarios,
    }
