from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.visualization.charts_2d import plot_monte_carlo_fan
from src.visualization.charts_3d import plot_monte_carlo_percentile_surface


def _percentile_frame(paths: np.ndarray) -> pd.DataFrame:
    percentiles = tuple(range(5, 100, 5))
    values = np.percentile(paths, percentiles, axis=1).T
    frame = pd.DataFrame(values, columns=[f"p{value}" for value in percentiles])
    frame.insert(0, "day", np.arange(paths.shape[0]))
    return frame


def test_fan_chart_from_compact_percentiles_matches_full_paths() -> None:
    rng = np.random.default_rng(42)
    paths = 100_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (40, 200)), axis=0))
    frame = _percentile_frame(paths)

    full_figure = plot_monte_carlo_fan(paths)
    compact_figure = plot_monte_carlo_fan(percentile_frame=frame)
    try:
        full_lines = full_figure.axes[0].lines
        compact_lines = compact_figure.axes[0].lines
        assert len(full_lines) == len(compact_lines) == 5
        for full_line, compact_line in zip(full_lines, compact_lines, strict=True):
            np.testing.assert_allclose(full_line.get_ydata(), compact_line.get_ydata())
    finally:
        plt.close(full_figure)
        plt.close(compact_figure)


def test_surface_from_compact_percentiles_matches_full_paths() -> None:
    rng = np.random.default_rng(7)
    paths = 100_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (30, 150)), axis=0))
    frame = _percentile_frame(paths)

    full_figure = plot_monte_carlo_percentile_surface(paths)
    compact_figure = plot_monte_carlo_percentile_surface(percentile_frame=frame)

    np.testing.assert_allclose(full_figure.data[0].z, compact_figure.data[0].z)
    np.testing.assert_allclose(full_figure.data[1].z, compact_figure.data[1].z)
