from .runner import run_advanced_models

__all__ = [
    "LinearRegressionModel",
    "ARIMAModel",
    "GARCHModel",
    "ExponentialSmoothingModel",
    "run_advanced_models",
]


def __getattr__(name: str):
    """Load optional model implementations only when explicitly requested."""
    if name == "ARIMAModel":
        from .arima_model import ARIMAModel

        return ARIMAModel
    if name == "GARCHModel":
        from .garch_model import GARCHModel

        return GARCHModel
    if name == "LinearRegressionModel":
        from .linear_regression_model import LinearRegressionModel

        return LinearRegressionModel
    if name == "ExponentialSmoothingModel":
        from .exponential_smoothing_model import ExponentialSmoothingModel

        return ExponentialSmoothingModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
