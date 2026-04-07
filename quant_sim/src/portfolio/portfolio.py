from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


@dataclass
class Asset:
    """Single asset representation"""
    symbol: str
    name: Optional[str] = None
    asset_type: str = "stock"
    sector: Optional[str] = None


class Portfolio:
    """Portfolio container with analytics"""
    
    def __init__(
        self,
        assets: list[Asset],
        weights: np.ndarray,
        prices: Dict[str, pd.DataFrame],
    ):
        self.assets = assets
        self.weights = np.array(weights)
        self.prices = prices
        
        self.validate_weights()
        
        self.symbols = [asset.symbol for asset in assets]
        self._returns_cache = None
    
    def validate_weights(self):
        """Validate portfolio weights"""
        if len(self.weights) != len(self.assets):
            raise ValueError("Number of weights must match number of assets")
        
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-6):
            logger.warning(f"Weights sum to {self.weights.sum()}, not 1.0")
    
    def get_returns(self) -> pd.DataFrame:
        """Calculate returns for all assets"""
        if self._returns_cache is not None:
            return self._returns_cache
        
        returns_dict = {}
        
        for asset in self.assets:
            if asset.symbol in self.prices:
                price_data = self.prices[asset.symbol]
                if "close" in price_data.columns:
                    returns_dict[asset.symbol] = price_data["close"].pct_change()
        
        self._returns_cache = pd.DataFrame(returns_dict).dropna()
        return self._returns_cache
    
    def calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns"""
        returns = self.get_returns()
        portfolio_returns = (returns * self.weights).sum(axis=1)
        return portfolio_returns
    
    def calculate_portfolio_volatility(self, annualize: bool = True) -> float:
        """Calculate portfolio volatility"""
        returns = self.get_returns()
        cov_matrix = returns.cov()
        
        portfolio_variance = self.weights.T @ cov_matrix @ self.weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        if annualize:
            portfolio_vol *= np.sqrt(252)
        
        return float(portfolio_vol)
    
    def calculate_metrics(self, risk_free_rate: float = 0.03) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        from src.analytics.returns import calculate_annualized_return
        from src.analytics.risk_metrics import (
            calculate_sharpe_ratio,
            calculate_max_drawdown,
            calculate_sortino_ratio,
        )
        
        portfolio_returns = self.calculate_portfolio_returns()
        
        metrics = {
            "total_return": float((1 + portfolio_returns).prod() - 1),
            "annualized_return": calculate_annualized_return(portfolio_returns),
            "volatility": self.calculate_portfolio_volatility(),
            "sharpe_ratio": calculate_sharpe_ratio(portfolio_returns, risk_free_rate),
            "sortino_ratio": calculate_sortino_ratio(portfolio_returns, risk_free_rate),
            "max_drawdown": calculate_max_drawdown(portfolio_returns),
        }
        
        return metrics
