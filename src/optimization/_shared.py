import numpy as np
import pandas as pd

def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    clean = pd.DataFrame(returns).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty:
        raise ValueError("returns are empty after cleaning.")
    if clean.shape[1] < 1:
        raise ValueError("returns must contain at least one asset.")
    if clean.shape[0] < 2:
        raise ValueError("returns must contain at least two observations.")
    return clean.astype(float)
