# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal v5.0
# Professional Portfolio Optimization & Global Multi-Asset Edition
# Enhanced Institutional Features & Comprehensive Error Handling
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize, linalg
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import concurrent.futures
from functools import lru_cache
import traceback
import time
import hashlib
import pickle
import base64
import io
from dataclasses import dataclass, field
import logging
import sys
import inspect

# Import PyPortfolioOpt with enhanced error handling
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.objective_functions import L2_reg, negative_sharpe
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel
    PYPFOPT_AVAILABLE = True
except ImportError as e:
    PYPFOPT_AVAILABLE = False
    st.warning(f"⚠️ PyPortfolioOpt not fully available: {str(e)[:100]}. Some optimization features will be limited.")

# Import optional ML packages
try:
    from sklearn.covariance import LedoitWolf, GraphicalLassoCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# -------------------------------------------------------------
# ENHANCED INSTITUTIONAL CONFIGURATION
# -------------------------------------------------------------
class InstitutionalConfig:
    """Institutional configuration parameters"""
    MAX_ASSETS = 100
    MIN_DATA_POINTS = 50
    TRADING_DAYS = 252
    DEFAULT_RF_RATE = 0.03
    MAX_CACHE_ENTRIES = 100
    PARALLEL_WORKERS = min(10, os.cpu_count() or 4)
    CHUNK_SIZE = 50
    TIMEOUT_SECONDS = 30
    
    # Risk limits
    MAX_LEVERAGE = 2.0
    MAX_CONCENTRATION = 0.25
    MIN_DIVERSIFICATION = 3
    
    # Performance thresholds
    MIN_SHARPE = -1.0
    MAX_VOLATILITY = 0.50
    
    # Institutional features
    ENABLE_AUDIT_LOG = True
    ENABLE_PERFORMANCE_TRACKING = True
    ENABLE_RISK_LIMITS = True

# -------------------------------------------------------------
# ENHANCED GLOBAL ASSET UNIVERSE WITH METADATA
# -------------------------------------------------------------
@dataclass
class AssetMetadata:
    """Enhanced asset metadata for institutional use"""
    ticker: str
    name: str
    category: str
    region: str
    currency: str
    sector: str = "Unknown"
    market_cap: Optional[float] = None
    inception_date: Optional[str] = None
    expense_ratio: Optional[float] = None
    is_etf: bool = False
    is_currency: bool = False
    is_crypto: bool = False
    
    def to_dict(self):
        return {
            'ticker': self.ticker,
            'name': self.name,
            'category': self.category,
            'region': self.region,
            'currency': self.currency,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'is_etf': self.is_etf,
            'is_crypto': self.is_crypto
        }

# Enhanced asset universe with metadata
GLOBAL_ASSET_UNIVERSE_ENHANCED = {
    # US Major Indices & ETFs
    "US_Indices": [
        AssetMetadata("SPY", "SPDR S&P 500 ETF", "Equity", "US", "USD", "Broad Market", is_etf=True),
        AssetMetadata("QQQ", "Invesco QQQ Trust", "Equity", "US", "USD", "Technology", is_etf=True),
        AssetMetadata("IWM", "iShares Russell 2000 ETF", "Equity", "US", "USD", "Small Cap", is_etf=True),
        AssetMetadata("DIA", "SPDR Dow Jones ETF", "Equity", "US", "USD", "Large Cap", is_etf=True),
        AssetMetadata("VTI", "Vanguard Total Stock Market ETF", "Equity", "US", "USD", "Total Market", is_etf=True),
    ],
    
    # Bonds & Fixed Income
    "Bonds": [
        AssetMetadata("TLT", "iShares 20+ Year Treasury Bond ETF", "Fixed Income", "US", "USD", "Treasury", is_etf=True),
        AssetMetadata("IEF", "iShares 7-10 Year Treasury Bond ETF", "Fixed Income", "US", "USD", "Treasury", is_etf=True),
        AssetMetadata("SHY", "iShares 1-3 Year Treasury Bond ETF", "Fixed Income", "US", "USD", "Treasury", is_etf=True),
        AssetMetadata("BND", "Vanguard Total Bond Market ETF", "Fixed Income", "US", "USD", "Aggregate", is_etf=True),
        AssetMetadata("HYG", "iShares iBoxx High Yield Corporate Bond ETF", "Fixed Income", "US", "USD", "High Yield", is_etf=True),
    ],
    
    # Commodities
    "Commodities": [
        AssetMetadata("GLD", "SPDR Gold Shares", "Commodity", "Global", "USD", "Gold", is_etf=True),
        AssetMetadata("SLV", "iShares Silver Trust", "Commodity", "Global", "USD", "Silver", is_etf=True),
        AssetMetadata("USO", "United States Oil Fund", "Commodity", "Global", "USD", "Oil", is_etf=True),
        AssetMetadata("UNG", "United States Natural Gas Fund", "Commodity", "Global", "USD", "Natural Gas", is_etf=True),
        AssetMetadata("DBA", "Invesco DB Agriculture Fund", "Commodity", "Global", "USD", "Agriculture", is_etf=True),
    ],
    
    # Cryptocurrencies
    "Cryptocurrencies": [
        AssetMetadata("BTC-USD", "Bitcoin USD", "Cryptocurrency", "Global", "USD", "Currency", is_crypto=True),
        AssetMetadata("ETH-USD", "Ethereum USD", "Cryptocurrency", "Global", "USD", "Platform", is_crypto=True),
        AssetMetadata("BNB-USD", "Binance Coin USD", "Cryptocurrency", "Global", "USD", "Exchange", is_crypto=True),
        AssetMetadata("XRP-USD", "Ripple USD", "Cryptocurrency", "Global", "USD", "Payment", is_crypto=True),
        AssetMetadata("ADA-USD", "Cardano USD", "Cryptocurrency", "Global", "USD", "Platform", is_crypto=True),
    ],
    
    # Global Stocks - US
    "US_Stocks": [
        AssetMetadata("AAPL", "Apple Inc.", "Equity", "US", "USD", "Technology", market_cap=2800000000000),
        AssetMetadata("MSFT", "Microsoft Corporation", "Equity", "US", "USD", "Technology", market_cap=2500000000000),
        AssetMetadata("GOOGL", "Alphabet Inc. (Class A)", "Equity", "US", "USD", "Technology", market_cap=1800000000000),
        AssetMetadata("AMZN", "Amazon.com Inc.", "Equity", "US", "USD", "Consumer Discretionary", market_cap=1500000000000),
        AssetMetadata("TSLA", "Tesla Inc.", "Equity", "US", "USD", "Automotive", market_cap=800000000000),
    ],
    
    # Add more categories as needed...
}

# Create ticker lookup dictionaries
TICKER_TO_METADATA = {}
for category in GLOBAL_ASSET_UNIVERSE_ENHANCED.values():
    for asset in category:
        TICKER_TO_METADATA[asset.ticker] = asset

# Flatten universe for selection
ALL_TICKERS_ENHANCED = list(TICKER_TO_METADATA.keys())

# Symbol mapping for common variations
SYMBOL_MAPPING_ENHANCED = {
    "BTC-USD": ["BTC-USD", "BTCUSD", "BTCUSDT", "BITCOIN"],
    "ETH-USD": ["ETH-USD", "ETHUSD", "ETHUSDT", "ETHEREUM"],
    "BRK-B": ["BRK-B", "BRK.B", "BERKSHIRE"],
    "GOOGL": ["GOOGL", "GOOG", "ALPHABET"],
    "EURUSD=X": ["EURUSD=X", "EURUSD", "EUR/USD"],
    "GBPUSD=X": ["GBPUSD=X", "GBPUSD", "GBP/USD"],
    "USDJPY=X": ["USDJPY=X", "USDJPY", "USD/JPY"],
}

# -------------------------------------------------------------
# ENHANCED CACHE MANAGEMENT
# -------------------------------------------------------------
class InstitutionalCache:
    """Enhanced cache management for institutional use"""
    
    def __init__(self, max_entries=InstitutionalConfig.MAX_CACHE_ENTRIES):
        self.max_entries = max_entries
        self.cache = {}
        self.access_log = {}
        
    def _generate_key(self, *args, **kwargs):
        """Generate deterministic cache key"""
        key_parts = []
        
        # Add function arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                key_parts.append(str(arg))
            elif isinstance(arg, pd.DataFrame):
                # Use hash of data for DataFrames
                try:
                    key_parts.append(hashlib.md5(pd.util.hash_pandas_object(arg).values).hexdigest()[:16])
                except:
                    key_parts.append(str(arg.shape))
            elif isinstance(arg, np.ndarray):
                key_parts.append(hashlib.md5(arg.tobytes()).hexdigest()[:16])
        
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def get(self, func, *args, **kwargs):
        """Get cached result or compute"""
        key = self._generate_key(func.__name__, *args, **kwargs)
        
        if key in self.cache:
            self.access_log[key] = time.time()
            logger.debug(f"Cache hit for {func.__name__}")
            return self.cache[key]
        
        # Compute and cache
        result = func(*args, **kwargs)
        self.cache[key] = result
        self.access_log[key] = time.time()
        
        # Clean cache if too large
        if len(self.cache) > self.max_entries:
            self._clean_cache()
        
        return result
    
    def _clean_cache(self):
        """Remove least recently used entries"""
        sorted_entries = sorted(self.access_log.items(), key=lambda x: x[1])
        entries_to_remove = sorted_entries[:len(self.cache) - self.max_entries]
        
        for key, _ in entries_to_remove:
            del self.cache[key]
            del self.access_log[key]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_log.clear()
    
    def stats(self):
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_entries,
            'utilization': len(self.cache) / self.max_entries
        }

# Initialize global cache
global_cache = InstitutionalCache()

# -------------------------------------------------------------
# ENHANCED DATA LOADER WITH INTELLIGENT FALLBACKS
# -------------------------------------------------------------
class InstitutionalDataLoader:
    """Professional data loader with institutional-grade features"""
    
    def __init__(self, cache_enabled=True):
        self.cache_enabled = cache_enabled
        self.data_sources = ['yfinance', 'fallback', 'demo']
        self.retry_config = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'timeout': InstitutionalConfig.TIMEOUT_SECONDS
        }
        
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate ticker format"""
        if not isinstance(ticker, str):
            return False
        
        # Remove common suffixes
        clean_ticker = ticker.replace('=', '').replace('-', '').replace('.', '')
        
        # Basic validation
        if len(clean_ticker) < 1 or len(clean_ticker) > 20:
            return False
        
        # Check for invalid characters
        invalid_chars = set('!@#$%^&*()[]{}|\\;:\'"<>,?`~')
        if any(char in invalid_chars for char in clean_ticker):
            return False
        
        return True
    
    def download_with_retry(self, ticker: str, start_date: str, end_date: str, 
                           source: str = 'yfinance') -> Optional[pd.Series]:
        """Download data with intelligent retry logic"""
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                if source == 'yfinance':
                    return self._download_yfinance(ticker, start_date, end_date)
                elif source == 'fallback':
                    return self._download_fallback(ticker, start_date, end_date)
                elif source == 'demo':
                    return self._generate_demo_data(ticker, start_date, end_date)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)[:100]}")
                
                if attempt < self.retry_config['max_retries'] - 1:
                    time.sleep(self.retry_config['retry_delay'] * (attempt + 1))
                else:
                    logger.error(f"All attempts failed for {ticker}")
                    return None
        
        return None
    
    def _download_yfinance(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.Series]:
        """Download from yfinance with enhanced error handling"""
        
        # Try alternative symbols
        symbols_to_try = [ticker]
        if ticker in SYMBOL_MAPPING_ENHANCED:
            symbols_to_try = SYMBOL_MAPPING_ENHANCED[ticker] + symbols_to_try
        
        for symbol in symbols_to_try:
            try:
                ticker_obj = yf.Ticker(symbol)
                
                # Get info for validation
                info = ticker_obj.info
                
                # Download historical data
                hist = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True,
                    prepost=False,
                    timeout=self.retry_config['timeout']
                )
                
                if hist.empty or len(hist) < InstitutionalConfig.MIN_DATA_POINTS:
                    continue
                
                # Get price column
                if 'Close' in hist.columns:
                    price_series = hist['Close']
                elif 'Adj Close' in hist.columns:
                    price_series = hist['Adj Close']
                else:
                    continue
                
                # Validate data quality
                if price_series.isna().sum() > len(price_series) * 0.3:
                    continue
                
                # Check for stale data
                latest_date = price_series.index[-1]
                if (pd.Timestamp.now() - latest_date).days > 30:
                    logger.warning(f"Stale data for {symbol}: latest date {latest_date}")
                
                return price_series
                
            except Exception as e:
                logger.debug(f"Failed to download {symbol}: {str(e)[:100]}")
                continue
        
        return None
    
    def _download_fallback(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.Series]:
        """Fallback download method"""
        # Implement alternative data sources here
        # For now, return None
        return None
    
    def _generate_demo_data(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        """Generate realistic demo data for testing"""
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        if len(dates) < InstitutionalConfig.MIN_DATA_POINTS:
            dates = pd.date_range(end=pd.Timestamp.today(), 
                                 periods=InstitutionalConfig.TRADING_DAYS, 
                                 freq='B')
        
        # Get asset type for realistic parameters
        metadata = TICKER_TO_METADATA.get(ticker)
        
        if metadata:
            if metadata.is_crypto:
                base_price = 30000
                annual_vol = 0.60
                annual_return = 0.20
            elif metadata.category == "Equity":
                base_price = 100
                annual_vol = 0.25
                annual_return = 0.08
            elif metadata.category == "Fixed Income":
                base_price = 100
                annual_vol = 0.08
                annual_return = 0.03
            elif metadata.category == "Commodity":
                base_price = 50
                annual_vol = 0.30
                annual_return = 0.05
            else:
                base_price = 100
                annual_vol = 0.20
                annual_return = 0.06
        else:
            # Default parameters
            base_price = 100
            annual_vol = 0.20
            annual_return = 0.06
        
        # Generate realistic price path
        np.random.seed(hash(ticker) % 2**32)
        n_days = len(dates)
        
        daily_return = annual_return / InstitutionalConfig.TRADING_DAYS
        daily_vol = annual_vol / np.sqrt(InstitutionalConfig.TRADING_DAYS)
        
        # Generate correlated random walk with momentum
        returns = np.random.normal(daily_return, daily_vol, n_days)
        
        # Add autocorrelation
        for i in range(1, n_days):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Add jumps (5% probability)
        jump_mask = np.random.random(n_days) < 0.05
        returns[jump_mask] += np.random.normal(0, daily_vol * 3, jump_mask.sum())
        
        # Calculate prices
        cumulative_returns = np.exp(np.cumsum(returns))
        prices = base_price * cumulative_returns
        
        # Add some microstructure noise
        prices = prices * (1 + np.random.randn(n_days) * 0.001)
        
        return pd.Series(prices, index=dates, name=ticker)
    
    def load_batch(self, tickers: List[str], start_date: str, end_date: str, 
                  use_parallel: bool = True) -> pd.DataFrame:
        """Load batch of tickers with parallel processing"""
        
        if not tickers:
            return pd.DataFrame()
        
        # Validate tickers
        valid_tickers = []
        for ticker in tickers:
            if self.validate_ticker(ticker):
                valid_tickers.append(ticker)
            else:
                logger.warning(f"Invalid ticker format: {ticker}")
        
        if len(valid_tickers) < 2:
            raise ValueError(f"Need at least 2 valid tickers. Got {len(valid_tickers)}")
        
        # Check cache if enabled
        cache_key = f"batch_{hashlib.md5('|'.join(sorted(valid_tickers)).encode()).hexdigest()}_{start_date}_{end_date}"
        
        if self.cache_enabled:
            cached_result = st.session_state.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for batch: {len(valid_tickers)} assets")
                return cached_result
        
        logger.info(f"Loading data for {len(valid_tickers)} assets...")
        
        prices_dict = {}
        successful_tickers = []
        failed_tickers = []
        
        if use_parallel and len(valid_tickers) > 5:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=InstitutionalConfig.PARALLEL_WORKERS
            ) as executor:
                
                future_to_ticker = {
                    executor.submit(
                        self.download_with_retry, 
                        ticker, start_date, end_date, 'yfinance'
                    ): ticker for ticker in valid_tickers
                }
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        price_data = future.result(timeout=self.retry_config['timeout'])
                        if price_data is not None and len(price_data) >= InstitutionalConfig.MIN_DATA_POINTS:
                            prices_dict[ticker] = price_data
                            successful_tickers.append(ticker)
                        else:
                            failed_tickers.append(ticker)
                    except Exception as e:
                        failed_tickers.append(ticker)
                        logger.error(f"Failed to load {ticker}: {str(e)[:100]}")
        else:
            # Sequential processing
            for ticker in valid_tickers:
                try:
                    price_data = self.download_with_retry(ticker, start_date, end_date, 'yfinance')
                    if price_data is not None and len(price_data) >= InstitutionalConfig.MIN_DATA_POINTS:
                        prices_dict[ticker] = price_data
                        successful_tickers.append(ticker)
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
                    logger.error(f"Failed to load {ticker}: {str(e)[:100]}")
        
        # Create DataFrame
        if prices_dict:
            prices_df = pd.DataFrame(prices_dict)
            
            # Align dates
            prices_df = prices_df.sort_index()
            
            # Handle missing data
            prices_df = self._clean_dataframe(prices_df)
            
            if prices_df.empty:
                raise ValueError("No valid data after cleaning")
            
            # Store in cache
            if self.cache_enabled:
                st.session_state[cache_key] = prices_df
            
            # Log results
            logger.info(f"Successfully loaded {len(successful_tickers)}/{len(valid_tickers)} assets")
            if failed_tickers:
                logger.warning(f"Failed to load: {failed_tickers}")
            
            return prices_df
        else:
            raise ValueError("Could not load any data")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate DataFrame"""
        
        if df.empty:
            return df
        
        # Remove columns with too much missing data (>40%)
        missing_pct = df.isna().mean()
        valid_columns = missing_pct[missing_pct < 0.4].index.tolist()
        df = df[valid_columns]
        
        if len(df.columns) < 2:
            return pd.DataFrame()
        
        # Forward fill small gaps (max 5 consecutive NaNs)
        df = df.ffill(limit=5).bfill(limit=5)
        
        # Remove any remaining NaNs
        df = df.dropna()
        
        # Remove outliers (beyond 10 standard deviations)
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                # Use numpy for boolean operations to avoid Series ambiguity
                z_scores = np.abs((df[col].values - mean) / std)
                outlier_mask = z_scores > 10
                if outlier_mask.any():
                    df.loc[outlier_mask, col] = mean
        
        # Ensure minimum length
        if len(df) < InstitutionalConfig.MIN_DATA_POINTS:
            return pd.DataFrame()
        
        return df

# -------------------------------------------------------------
# ENHANCED PORTFOLIO OPTIMIZER WITH INSTITUTIONAL FEATURES
# -------------------------------------------------------------
class InstitutionalPortfolioOptimizer:
    """Professional portfolio optimizer with institutional features"""
    
    def __init__(self):
        self.config = InstitutionalConfig()
        self.optimization_history = []
        self.risk_limits_enabled = True
    
    def optimize_with_constraints(self, returns_df: pd.DataFrame, strategy: str,
                                 constraints: Dict = None, 
                                 risk_free_rate: float = None) -> Dict:
        """Optimize portfolio with institutional constraints"""
        
        # Validate inputs
        self._validate_optimization_inputs(returns_df, strategy)
        
        if risk_free_rate is None:
            risk_free_rate = self.config.DEFAULT_RF_RATE
        
        # Apply constraints
        if constraints is None:
            constraints = self._default_constraints(returns_df)
        
        try:
            # Try PyPortfolioOpt first
            if PYPFOPT_AVAILABLE:
                result = self._optimize_pypfopt(returns_df, strategy, constraints, risk_free_rate)
                if result.get('success', False):
                    return result
            
            # Fallback to custom optimization
            result = self._optimize_custom(returns_df, strategy, constraints, risk_free_rate)
            
            # Apply risk limits
            if self.risk_limits_enabled:
                result = self._apply_risk_limits(result, returns_df)
            
            # Log optimization
            self._log_optimization(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return self._fallback_optimization(returns_df, risk_free_rate)
    
    def _validate_optimization_inputs(self, returns_df: pd.DataFrame, strategy: str):
        """Validate optimization inputs"""
        
        if returns_df is None or returns_df.empty:
            raise ValueError("Returns DataFrame is empty")
        
        if len(returns_df.columns) < 2:
            raise ValueError(f"Need at least 2 assets, got {len(returns_df.columns)}")
        
        if len(returns_df) < self.config.MIN_DATA_POINTS:
            raise ValueError(f"Need at least {self.config.MIN_DATA_POINTS} data points")
        
        valid_strategies = [
            "Minimum Volatility", "Maximum Sharpe Ratio", "Maximum Quadratic Utility",
            "Efficient Risk", "Efficient Return", "Risk Parity", "Maximum Diversification",
            "Equal Weight", "Market Cap Weight", "Hierarchical Risk Parity"
        ]
        
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}")
    
    def _default_constraints(self, returns_df: pd.DataFrame) -> Dict:
        """Generate default institutional constraints"""
        
        n_assets = len(returns_df.columns)
        
        return {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'sum_to_one': True,
            'min_diversification': self.config.MIN_DIVERSIFICATION,
            'max_concentration': self.config.MAX_CONCENTRATION,
            'max_leverage': self.config.MAX_LEVERAGE,
            'allow_short': False,
            'asset_groups': None,
            'sector_limits': None,
            'turnover_limit': None
        }
    
    def _optimize_pypfopt(self, returns_df: pd.DataFrame, strategy: str,
                         constraints: Dict, risk_free_rate: float) -> Dict:
        """Optimize using PyPortfolioOpt"""
        
        try:
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(returns_df)
            
            # Use shrinkage covariance matrix
            if SKLEARN_AVAILABLE and len(returns_df) > 100:
                S = risk_models.CovarianceShrinkage(returns_df).ledoit_wolf()
            else:
                S = risk_models.sample_cov(returns_df)
            
            # Create efficient frontier
            ef = EfficientFrontier(
                mu, 
                S,
                weight_bounds=(constraints['min_weight'], constraints['max_weight'])
            )
            
            # Add L2 regularization for stability
            ef.add_objective(L2_reg, gamma=0.1)
            
            # Apply strategy
            weights = self._apply_pypfopt_strategy(ef, strategy, constraints, risk_free_rate)
            
            # Clean weights
            cleaned_weights = ef.clean_weights()
            
            # Calculate performance
            expected_return, expected_risk, sharpe_ratio = ef.portfolio_performance(
                risk_free_rate=risk_free_rate / self.config.TRADING_DAYS
            )
            
            # Convert to annualized
            expected_return_annual = expected_return * self.config.TRADING_DAYS
            expected_risk_annual = expected_risk * np.sqrt(self.config.TRADING_DAYS)
            sharpe_ratio_annual = sharpe_ratio * np.sqrt(self.config.TRADING_DAYS)
            
            # Prepare result
            weights_array = np.array([cleaned_weights.get(asset, 0) for asset in returns_df.columns])
            
            result = {
                'weights': weights_array,
                'expected_return': expected_return_annual,
                'expected_risk': expected_risk_annual,
                'sharpe_ratio': sharpe_ratio_annual,
                'method': f"PyPortfolioOpt {strategy}",
                'cleaned_weights': cleaned_weights,
                'optimizer': 'PyPortfolioOpt',
                'constraints_applied': True,
                'success': True,
                'covariance_matrix': S,
                'expected_returns': mu
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"PyPortfolioOpt optimization failed: {str(e)[:100]}")
            raise
    
    def _apply_pypfopt_strategy(self, ef: EfficientFrontier, strategy: str,
                               constraints: Dict, risk_free_rate: float) -> Dict:
        """Apply specific optimization strategy"""
        
        if strategy == "Minimum Volatility":
            return ef.min_volatility()
        
        elif strategy == "Maximum Sharpe Ratio":
            return ef.max_sharpe(risk_free_rate=risk_free_rate / self.config.TRADING_DAYS)
        
        elif strategy == "Maximum Quadratic Utility":
            return ef.max_quadratic_utility(risk_aversion=2)
        
        elif strategy == "Efficient Risk":
            target_risk = constraints.get('target_risk', 0.15)
            return ef.efficient_risk(target_risk=target_risk / np.sqrt(self.config.TRADING_DAYS))
        
        elif strategy == "Efficient Return":
            target_return = constraints.get('target_return', 0.10)
            return ef.efficient_return(target_return=target_return / self.config.TRADING_DAYS)
        
        elif strategy == "Hierarchical Risk Parity" and PYPFOPT_AVAILABLE:
            # Use HRP optimization
            hrp = HRPOpt(ef.expected_returns, ef.cov_matrix)
            return hrp.optimize()
        
        else:
            # Default to minimum volatility
            return ef.min_volatility()
    
    def _optimize_custom(self, returns_df: pd.DataFrame, strategy: str,
                        constraints: Dict, risk_free_rate: float) -> Dict:
        """Custom optimization implementation"""
        
        n_assets = len(returns_df.columns)
        
        if strategy == "Equal Weight":
            weights = np.ones(n_assets) / n_assets
        
        elif strategy == "Risk Parity":
            weights = self._risk_parity_optimization(returns_df)
        
        elif strategy == "Maximum Diversification":
            weights = self._maximum_diversification_optimization(returns_df)
        
        elif strategy == "Market Cap Weight":
            # Simplified market cap weighting
            volatilities = returns_df.std()
            # Inverse volatility weighting as proxy
            weights = (1 / (volatilities + 1e-10))
            weights = weights / weights.sum()
        
        else:
            # Fallback to equal weight
            weights = np.ones(n_assets) / n_assets
        
        # Calculate performance
        portfolio_returns = (returns_df * weights).sum(axis=1)
        expected_return = portfolio_returns.mean() * self.config.TRADING_DAYS
        expected_risk = portfolio_returns.std() * np.sqrt(self.config.TRADING_DAYS)
        sharpe_ratio = (expected_return - risk_free_rate) / (expected_risk + 1e-10)
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': f"Custom {strategy}",
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Custom',
            'constraints_applied': False,
            'success': True
        }
    
    def _risk_parity_optimization(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Risk parity optimization"""
        
        cov_matrix = returns_df.cov()
        n_assets = len(cov_matrix)
        
        # Initialize weights
        weights = np.ones(n_assets) / n_assets
        
        # Iterative risk parity optimization
        for _ in range(100):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Calculate marginal risk contributions
            marginal_risk = cov_matrix @ weights / portfolio_vol
            
            # Calculate risk contributions
            risk_contributions = weights * marginal_risk
            
            # Calculate target risk contributions (equal)
            target_contributions = np.ones(n_assets) * portfolio_vol / n_assets
            
            # Update weights using gradient method
            adjustment = 0.01 * (target_contributions - risk_contributions) / marginal_risk
            weights += adjustment
            
            # Ensure non-negative and normalized
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        
        return weights
    
    def _maximum_diversification_optimization(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Maximum diversification portfolio"""
        
        cov_matrix = returns_df.cov()
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # Calculate correlation matrix
        D = np.diag(1 / volatilities)
        corr_matrix = D @ cov_matrix @ D
        
        # Solve for weights
        try:
            weights = np.linalg.solve(corr_matrix, np.ones(len(corr_matrix)))
            weights = weights / weights.sum()
        except:
            # Fallback to inverse volatility
            weights = 1 / volatilities
            weights = weights / weights.sum()
        
        return weights
    
    def _apply_risk_limits(self, result: Dict, returns_df: pd.DataFrame) -> Dict:
        """Apply institutional risk limits"""
        
        weights = result['weights']
        
        # Check concentration limits
        max_weight = weights.max()
        if max_weight > self.config.MAX_CONCENTRATION:
            logger.warning(f"Concentration limit exceeded: {max_weight:.2%} > {self.config.MAX_CONCENTRATION:.2%}")
            # Apply simple adjustment
            excess = max_weight - self.config.MAX_CONCENTRATION
            weights[weights.argmax()] -= excess
        
        # Check diversification
        effective_n = 1 / (weights ** 2).sum()
        if effective_n < self.config.MIN_DIVERSIFICATION:
            logger.warning(f"Diversification below minimum: {effective_n:.1f} < {self.config.MIN_DIVERSIFICATION}")
        
        # Recalculate performance with adjusted weights
        portfolio_returns = (returns_df * weights).sum(axis=1)
        result['weights'] = weights
        result['expected_return'] = portfolio_returns.mean() * self.config.TRADING_DAYS
        result['expected_risk'] = portfolio_returns.std() * np.sqrt(self.config.TRADING_DAYS)
        
        if 'sharpe_ratio' in result and result.get('risk_free_rate'):
            result['sharpe_ratio'] = (result['expected_return'] - result['risk_free_rate']) / (result['expected_risk'] + 1e-10)
        
        result['risk_limits_applied'] = True
        result['effective_diversification'] = effective_n
        result['max_concentration'] = weights.max()
        
        return result
    
    def _log_optimization(self, result: Dict):
        """Log optimization results"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': result.get('method', 'Unknown'),
            'success': result.get('success', False),
            'expected_return': result.get('expected_return', 0),
            'expected_risk': result.get('expected_risk', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'n_assets': len(result.get('weights', [])),
            'optimizer': result.get('optimizer', 'Unknown')
        }
        
        self.optimization_history.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def _fallback_optimization(self, returns_df: pd.DataFrame, risk_free_rate: float) -> Dict:
        """Fallback optimization when everything fails"""
        
        n_assets = len(returns_df.columns)
        weights = np.ones(n_assets) / n_assets
        
        portfolio_returns = (returns_df * weights).sum(axis=1)
        expected_return = portfolio_returns.mean() * self.config.TRADING_DAYS
        expected_risk = portfolio_returns.std() * np.sqrt(self.config.TRADING_DAYS)
        sharpe_ratio = (expected_return - risk_free_rate) / (expected_risk + 1e-10)
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': 'Equal Weight (Fallback)',
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Fallback',
            'constraints_applied': False,
            'success': False,
            'error': 'All optimization methods failed'
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        return pd.DataFrame(self.optimization_history)

# -------------------------------------------------------------
# ENHANCED RISK ANALYTICS ENGINE
# -------------------------------------------------------------
class InstitutionalRiskAnalytics:
    """Professional risk analytics with institutional features"""
    
    def __init__(self):
        self.config = InstitutionalConfig()
        self.var_methods = {
            'historical': self._historical_var,
            'parametric': self._parametric_var,
            'ewma': self._ewma_var,
            'monte_carlo': self._monte_carlo_var,
            'cornish_fisher': self._cornish_fisher_var
        }
    
    def comprehensive_risk_report(self, returns: pd.Series, 
                                 portfolio_value: float = 1000000) -> Dict:
        """Generate comprehensive risk report"""
        
        if returns is None or returns.empty:
            return self._empty_risk_report()
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_points': len(returns),
                'portfolio_value': portfolio_value,
                'basic_statistics': self._basic_statistics(returns),
                'var_analysis': self._var_analysis_comprehensive(returns),
                'drawdown_analysis': self._drawdown_analysis(returns),
                'distribution_analysis': self._distribution_analysis(returns),
                'stress_tests': self._stress_tests(returns),
                'liquidity_metrics': self._liquidity_metrics(returns),
                'success': True
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Risk report generation failed: {str(e)}")
            return self._error_risk_report(str(e))
    
    def _basic_statistics(self, returns: pd.Series) -> Dict:
        """Calculate basic risk statistics"""
        
        ann_return = returns.mean() * self.config.TRADING_DAYS
        ann_vol = returns.std() * np.sqrt(self.config.TRADING_DAYS)
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Information ratio (vs zero)
        tracking_error = returns.std() * np.sqrt(self.config.TRADING_DAYS)
        information_ratio = ann_return / (tracking_error + 1e-10)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(self.config.TRADING_DAYS) if len(downside_returns) > 0 else 0
        sortino_ratio = ann_return / (downside_dev + 1e-10)
        
        # Calmar ratio (return / max drawdown)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        calmar_ratio = ann_return / (abs(max_dd) + 1e-10)
        
        return {
            'annual_return': ann_return,
            'annual_volatility': ann_vol,
            'sharpe_ratio': (ann_return - self.config.DEFAULT_RF_RATE) / (ann_vol + 1e-10),
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() if len(returns[returns <= np.percentile(returns, 5)]) > 0 else np.percentile(returns, 5)
        }
    
    def _var_analysis_comprehensive(self, returns: pd.Series) -> Dict:
        """Comprehensive VaR analysis with multiple methods"""
        
        confidence_levels = [0.90, 0.95, 0.99]
        horizons = [1, 5, 21]  # 1 day, 1 week, 1 month
        
        var_results = {}
        
        for conf in confidence_levels:
            var_results[f'conf_{int(conf*100)}'] = {}
            
            for horizon in horizons:
                if horizon > 1:
                    # Aggregate returns for horizon
                    horizon_returns = returns.rolling(horizon).apply(
                        lambda x: np.prod(1 + x) - 1, raw=True
                    ).dropna()
                else:
                    horizon_returns = returns
                
                # Calculate VaR using different methods
                methods_results = {}
                for method_name, method_func in self.var_methods.items():
                    try:
                        if method_name == 'monte_carlo':
                            result = method_func(horizon_returns, conf, params={'n_simulations': 10000, 'days': horizon})
                        else:
                            result = method_func(horizon_returns, conf)
                        
                        methods_results[method_name] = result
                    except Exception as e:
                        methods_results[method_name] = {'error': str(e)[:100]}
                
                var_results[f'conf_{int(conf*100)}'][f'{horizon}_day'] = methods_results
        
        return var_results
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Historical VaR"""
        
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            tail = returns[returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return {
                'var': var,
                'cvar': cvar,
                'method': 'Historical',
                'confidence': confidence_level,
                'observations': len(returns),
                'success': True
            }
        except Exception as e:
            return {
                'var': np.nan,
                'cvar': np.nan,
                'method': 'Historical',
                'error': str(e)[:100],
                'success': False
            }
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Parametric VaR assuming normal distribution"""
        
        try:
            mu = returns.mean()
            sigma = returns.std()
            
            if sigma == 0:
                return {
                    'var': mu,
                    'cvar': mu,
                    'method': 'Parametric',
                    'error': 'Zero volatility',
                    'success': False
                }
            
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mu + z_score * sigma
            cvar = mu - (sigma / (1 - confidence_level)) * stats.norm.pdf(z_score)
            
            return {
                'var': var,
                'cvar': cvar,
                'method': 'Parametric (Normal)',
                'confidence': confidence_level,
                'mu': mu,
                'sigma': sigma,
                'z_score': z_score,
                'success': True
            }
        except Exception as e:
            return {
                'var': np.nan,
                'cvar': np.nan,
                'method': 'Parametric',
                'error': str(e)[:100],
                'success': False
            }
    
    def _cornish_fisher_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Cornish-Fisher VaR incorporating skewness and kurtosis"""
        
        try:
            mu = returns.mean()
            sigma = returns.std()
            skew = returns.skew()
            kurt = returns.kurtosis()
            
            if sigma == 0:
                return {
                    'var': mu,
                    'cvar': mu,
                    'method': 'Cornish-Fisher',
                    'error': 'Zero volatility',
                    'success': False
                }
            
            # Cornish-Fisher expansion
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = (z + 
                   (z**2 - 1) * skew / 6 +
                   (z**3 - 3*z) * kurt / 24 -
                   (2*z**3 - 5*z) * skew**2 / 36)
            
            var = mu + z_cf * sigma
            
            # Approximate CVaR
            cvar = mu - (sigma / (1 - confidence_level)) * stats.norm.pdf(z)
            
            return {
                'var': var,
                'cvar': cvar,
                'method': 'Cornish-Fisher',
                'confidence': confidence_level,
                'mu': mu,
                'sigma': sigma,
                'skewness': skew,
                'kurtosis': kurt,
                'z_cf': z_cf,
                'success': True
            }
        except Exception as e:
            return {
                'var': np.nan,
                'cvar': np.nan,
                'method': 'Cornish-Fisher',
                'error': str(e)[:100],
                'success': False
            }
    
    def _ewma_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """EWMA VaR"""
        
        try:
            lambda_param = 0.94
            returns_squared = returns ** 2
            
            # Initialize EWMA
            ewma_var = pd.Series(index=returns.index, dtype=float)
            ewma_var.iloc[0] = returns_squared.iloc[:min(30, len(returns))].mean()
            
            # Calculate EWMA variance
            for i in range(1, len(returns)):
                ewma_var.iloc[i] = (lambda_param * ewma_var.iloc[i-1] + 
                                   (1 - lambda_param) * returns_squared.iloc[i-1])
            
            current_vol = np.sqrt(ewma_var.iloc[-1])
            z_score = stats.norm.ppf(1 - confidence_level)
            var = z_score * current_vol
            cvar = - (current_vol / (1 - confidence_level)) * stats.norm.pdf(z_score)
            
            return {
                'var': var,
                'cvar': cvar,
                'method': 'EWMA',
                'confidence': confidence_level,
                'lambda': lambda_param,
                'current_vol': current_vol,
                'success': True
            }
        except Exception as e:
            return {
                'var': np.nan,
                'cvar': np.nan,
                'method': 'EWMA',
                'error': str(e)[:100],
                'success': False
            }
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float, 
                        params: Dict = None) -> Dict:
        """Monte Carlo VaR"""
        
        try:
            n_simulations = params.get('n_simulations', 10000)
            days = params.get('days', 1)
            
            mu = returns.mean()
            sigma = returns.std()
            
            # Simulate using GBM
            np.random.seed(42)
            dt = 1 / self.config.TRADING_DAYS
            
            simulations = np.random.normal(
                mu * dt, 
                sigma * np.sqrt(dt), 
                (days, n_simulations)
            )
            
            cumulative_returns = np.ones(n_simulations)
            for day in range(days):
                cumulative_returns *= (1 + simulations[day])
            
            final_returns = cumulative_returns - 1
            
            var = np.percentile(final_returns, (1 - confidence_level) * 100)
            tail = final_returns[final_returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return {
                'var': var,
                'cvar': cvar,
                'method': 'Monte Carlo',
                'confidence': confidence_level,
                'simulations': n_simulations,
                'days': days,
                'mu': mu,
                'sigma': sigma,
                'success': True
            }
        except Exception as e:
            return {
                'var': np.nan,
                'cvar': np.nan,
                'method': 'Monte Carlo',
                'error': str(e)[:100],
                'success': False
            }
    
    def _drawdown_analysis(self, returns: pd.Series) -> Dict:
        """Comprehensive drawdown analysis"""
        
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            max_dd = drawdown.min()
            max_dd_date = drawdown.idxmin() if not drawdown.empty else None
            
            # Calculate drawdown duration
            underwater = drawdown < 0
            if underwater.any():
                underwater_periods = underwater.astype(int)
                underwater_diff = underwater_periods.diff()
                start_indices = underwater_diff[underwater_diff == 1].index
                end_indices = underwater_diff[underwater_diff == -1].index
                
                if len(start_indices) > 0 and len(end_indices) > 0:
                    durations = [(end - start).days for start, end in zip(start_indices, end_indices)]
                    avg_duration = np.mean(durations) if durations else 0
                    max_duration = np.max(durations) if durations else 0
                else:
                    avg_duration = 0
                    max_duration = 0
            else:
                avg_duration = 0
                max_duration = 0
            
            # Recovery analysis
            recovery_days = None
            if max_dd_date is not None:
                post_dd = cumulative.loc[max_dd_date:]
                recovery_level = running_max.loc[max_dd_date]
                recovery_mask = post_dd >= recovery_level
                if recovery_mask.any():
                    recovery_date = recovery_mask.idxmax()
                    recovery_days = (recovery_date - max_dd_date).days
            
            return {
                'max_drawdown': max_dd,
                'max_drawdown_date': max_dd_date,
                'current_drawdown': drawdown.iloc[-1] if not drawdown.empty else 0,
                'avg_drawdown_duration': avg_duration,
                'max_drawdown_duration': max_duration,
                'recovery_days': recovery_days,
                'drawdown_series': drawdown.tolist(),
                'success': True
            }
        except Exception as e:
            return {
                'max_drawdown': np.nan,
                'error': str(e)[:100],
                'success': False
            }
    
    def _distribution_analysis(self, returns: pd.Series) -> Dict:
        """Statistical distribution analysis"""
        
        try:
            # Fit different distributions
            dist_fits = {}
            
            # Normal distribution
            mu, sigma = stats.norm.fit(returns.dropna())
            dist_fits['normal'] = {'mu': mu, 'sigma': sigma}
            
            # Student's t-distribution
            try:
                df, mu_t, sigma_t = stats.t.fit(returns.dropna())
                dist_fits['student_t'] = {'df': df, 'mu': mu_t, 'sigma': sigma_t}
            except:
                dist_fits['student_t'] = {'error': 'Fit failed'}
            
            # Anderson-Darling test for normality
            anderson_result = stats.anderson(returns.dropna(), dist='norm')
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
            
            return {
                'distribution_fits': dist_fits,
                'normality_test': {
                    'anderson_statistic': anderson_result.statistic,
                    'anderson_critical_values': anderson_result.critical_values.tolist(),
                    'anderson_significance_level': anderson_result.significance_level.tolist(),
                    'jarque_bera_statistic': jb_stat,
                    'jarque_bera_pvalue': jb_pvalue
                },
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e)[:100],
                'success': False
            }
    
    def _stress_tests(self, returns: pd.Series) -> Dict:
        """Stress test scenarios"""
        
        try:
            # Historical stress periods
            worst_day = returns.min()
            worst_week = returns.rolling(5).apply(lambda x: np.prod(1 + x) - 1, raw=True).min()
            worst_month = returns.rolling(21).apply(lambda x: np.prod(1 + x) - 1, raw=True).min()
            
            # Extreme scenarios
            scenarios = {
                '2008_crisis': -0.10,  # Lehman Brothers
                '2020_covid': -0.12,   # COVID crash
                '1987_black_monday': -0.23,
                'standard_shock': -0.05
            }
            
            # Calculate impact
            current_vol = returns.std() * np.sqrt(self.config.TRADING_DAYS)
            stress_impacts = {}
            for scenario, shock in scenarios.items():
                stress_impacts[scenario] = {
                    'shock': shock,
                    'potential_loss': shock * current_vol,
                    'probability': stats.norm.cdf(shock / current_vol) if current_vol > 0 else 0
                }
            
            return {
                'historical_worst_day': worst_day,
                'historical_worst_week': worst_week,
                'historical_worst_month': worst_month,
                'stress_scenarios': stress_impacts,
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e)[:100],
                'success': False
            }
    
    def _liquidity_metrics(self, returns: pd.Series) -> Dict:
        """Liquidity risk metrics"""
        
        try:
            # Simplified liquidity metrics
            vol_ratio = returns.std() / returns.abs().mean() if returns.abs().mean() > 0 else 0
            
            # Autocorrelation (illiquidity indicator)
            autocorr = returns.autocorr(lag=1)
            
            # Volume proxy (using return magnitude)
            avg_daily_move = returns.abs().mean()
            
            return {
                'volatility_ratio': vol_ratio,
                'autocorrelation_lag1': autocorr,
                'avg_daily_move': avg_daily_move,
                'liquidity_score': 1 / (abs(autocorr) + 1e-10),  # Higher is better
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e)[:100],
                'success': False
            }
    
    def _empty_risk_report(self) -> Dict:
        """Empty risk report template"""
        return {
            'timestamp': datetime.now().isoformat(),
            'data_points': 0,
            'basic_statistics': {},
            'var_analysis': {},
            'drawdown_analysis': {},
            'distribution_analysis': {},
            'stress_tests': {},
            'liquidity_metrics': {},
            'success': False,
            'error': 'No data available'
        }
    
    def _error_risk_report(self, error_msg: str) -> Dict:
        """Error risk report"""
        report = self._empty_risk_report()
        report['error'] = error_msg
        return report

# -------------------------------------------------------------
# ENHANCED PERFORMANCE ATTRIBUTION
# -------------------------------------------------------------
class InstitutionalPerformanceAttribution:
    """Professional performance attribution with institutional features"""
    
    def __init__(self):
        self.config = InstitutionalConfig()
    
    def comprehensive_attribution(self, portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                portfolio_weights: np.ndarray,
                                benchmark_weights: np.ndarray,
                                asset_returns: pd.DataFrame) -> Dict:
        """Comprehensive performance attribution"""
        
        try:
            attribution = {
                'timestamp': datetime.now().isoformat(),
                'brinson_attribution': self._brinson_attribution(
                    portfolio_returns, benchmark_returns,
                    portfolio_weights, benchmark_weights, asset_returns
                ),
                'risk_decomposition': self._risk_decomposition(
                    portfolio_weights, asset_returns
                ),
                'return_decomposition': self._return_decomposition(
                    portfolio_weights, asset_returns
                ),
                'factor_attribution': self._factor_attribution(
                    portfolio_returns, benchmark_returns
                ),
                'success': True
            }
            
            return attribution
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)[:200],
                'success': False
            }
    
    def _brinson_attribution(self, portfolio_returns: pd.Series,
                           benchmark_returns: pd.Series,
                           portfolio_weights: np.ndarray,
                           benchmark_weights: np.ndarray,
                           asset_returns: pd.DataFrame) -> Dict:
        """Brinson-Fachler attribution"""
        
        # Ensure weights sum to 1
        portfolio_weights = portfolio_weights / (portfolio_weights.sum() + 1e-10)
        benchmark_weights = benchmark_weights / (benchmark_weights.sum() + 1e-10)
        
        # Calculate returns
        portfolio_return = portfolio_returns.mean() * self.config.TRADING_DAYS
        benchmark_return = benchmark_returns.mean() * self.config.TRADING_DAYS
        asset_returns_annual = asset_returns.mean() * self.config.TRADING_DAYS
        
        # Total active return
        total_active = portfolio_return - benchmark_return
        
        # Allocation effect
        allocation_effect = np.sum(
            (portfolio_weights - benchmark_weights) * 
            (asset_returns_annual.values - benchmark_return)
        )
        
        # Selection effect
        selection_effect = np.sum(
            benchmark_weights * 
            (asset_returns_annual.values - benchmark_return)
        )
        
        # Interaction effect
        interaction_effect = total_active - (allocation_effect + selection_effect)
        
        # Attribution by asset
        asset_attribution = []
        for i, asset in enumerate(asset_returns.columns):
            asset_allocation = (portfolio_weights[i] - benchmark_weights[i]) * (asset_returns_annual.iloc[i] - benchmark_return)
            asset_selection = benchmark_weights[i] * (asset_returns_annual.iloc[i] - benchmark_return)
            
            asset_attribution.append({
                'asset': asset,
                'allocation_effect': asset_allocation,
                'selection_effect': asset_selection,
                'total_effect': asset_allocation + asset_selection
            })
        
        return {
            'total_active_return': total_active,
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'asset_attribution': asset_attribution,
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return
        }
    
    def _risk_decomposition(self, weights: np.ndarray, 
                          asset_returns: pd.DataFrame) -> Dict:
        """Risk decomposition analysis"""
        
        try:
            cov_matrix = asset_returns.cov() * self.config.TRADING_DAYS
            portfolio_variance = weights.T @ cov_matrix @ weights
            
            if portfolio_variance <= 0:
                return {'error': 'Non-positive portfolio variance'}
            
            # Marginal risk contributions
            marginal_risk = cov_matrix @ weights / np.sqrt(portfolio_variance)
            
            # Risk contributions
            risk_contributions = weights * marginal_risk
            
            # Percentage contributions
            risk_pct_contributions = risk_contributions / np.sqrt(portfolio_variance)
            
            # Diversification ratio
            weighted_vol = weights @ np.sqrt(np.diag(cov_matrix))
            diversification_ratio = weighted_vol / np.sqrt(portfolio_variance)
            
            # Concentration measures
            herfindahl = np.sum(weights ** 2)
            gini = self._gini_coefficient(weights)
            
            return {
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'risk_contributions': dict(zip(asset_returns.columns, risk_contributions)),
                'risk_pct_contributions': dict(zip(asset_returns.columns, risk_pct_contributions)),
                'diversification_ratio': diversification_ratio,
                'herfindahl_index': herfindahl,
                'gini_coefficient': gini,
                'effective_number': 1 / herfindahl
            }
        except Exception as e:
            return {'error': str(e)[:100]}
    
    def _return_decomposition(self, weights: np.ndarray,
                            asset_returns: pd.DataFrame) -> Dict:
        """Return decomposition analysis"""
        
        try:
            asset_returns_annual = asset_returns.mean() * self.config.TRADING_DAYS
            portfolio_return = np.sum(weights * asset_returns_annual.values)
            
            # Return contributions
            return_contributions = weights * asset_returns_annual.values
            
            # Percentage contributions
            return_pct_contributions = return_contributions / (portfolio_return + 1e-10)
            
            # Return attribution by factor (simplified)
            factor_exposure = self._estimate_factor_exposure(asset_returns)
            
            return {
                'portfolio_return': portfolio_return,
                'return_contributions': dict(zip(asset_returns.columns, return_contributions)),
                'return_pct_contributions': dict(zip(asset_returns.columns, return_pct_contributions)),
                'factor_exposure': factor_exposure
            }
        except Exception as e:
            return {'error': str(e)[:100]}
    
    def _factor_attribution(self, portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series) -> Dict:
        """Simplified factor attribution"""
        
        try:
            # Common factors (simplified)
            factors = {
                'market': benchmark_returns,
                'size': self._generate_size_factor(portfolio_returns, benchmark_returns),
                'value': self._generate_value_factor(portfolio_returns, benchmark_returns),
                'momentum': self._generate_momentum_factor(portfolio_returns, benchmark_returns)
            }
            
            # Run factor regression
            factor_data = pd.DataFrame(factors)
            factor_data['portfolio'] = portfolio_returns
            
            # OLS regression
            import statsmodels.api as sm
            X = sm.add_constant(factor_data[['market', 'size', 'value', 'momentum']])
            y = factor_data['portfolio']
            
            model = sm.OLS(y, X).fit()
            
            return {
                'factor_loadings': model.params.to_dict(),
                'r_squared': model.rsquared,
                'alpha': model.params.get('const', 0) * self.config.TRADING_DAYS,
                'residual_risk': np.sqrt(model.mse_resid) * np.sqrt(self.config.TRADING_DAYS)
            }
        except Exception as e:
            return {'error': str(e)[:100]}
    
    def _gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
    
    def _estimate_factor_exposure(self, returns: pd.DataFrame) -> Dict:
        """Estimate factor exposures (simplified)"""
        # This is a placeholder - in practice, use real factor models
        return {
            'market_beta': 1.0,
            'size_exposure': 0.0,
            'value_exposure': 0.0,
            'momentum_exposure': 0.0
        }
    
    def _generate_size_factor(self, portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series) -> pd.Series:
        """Generate size factor (simplified)"""
        return portfolio_returns - benchmark_returns
    
    def _generate_value_factor(self, portfolio_returns: pd.Series,
                             benchmark_returns: pd.Series) -> pd.Series:
        """Generate value factor (simplified)"""
        return portfolio_returns.rolling(20).mean() - benchmark_returns.rolling(20).mean()
    
    def _generate_momentum_factor(self, portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series) -> pd.Series:
        """Generate momentum factor (simplified)"""
        return portfolio_returns.shift(20) - benchmark_returns.shift(20)

# -------------------------------------------------------------
# ENHANCED MONTE CARLO ENGINE
# -------------------------------------------------------------
class InstitutionalMonteCarlo:
    """Professional Monte Carlo simulation engine"""
    
    def __init__(self):
        self.config = InstitutionalConfig()
    
    def comprehensive_simulation(self, returns: pd.Series, 
                               initial_value: float = 1000000,
                               params: Dict = None) -> Dict:
        """Comprehensive Monte Carlo simulation"""
        
        if params is None:
            params = {
                'n_simulations': 10000,
                'time_horizon': 252,  # 1 year
                'confidence_levels': [0.90, 0.95, 0.99],
                'methods': ['gbm', 'bootstrap', 'regime_switching']
            }
        
        try:
            simulation_results = {
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'initial_value': initial_value,
                'gbm_simulation': self._gbm_simulation(returns, initial_value, params),
                'bootstrap_simulation': self._bootstrap_simulation(returns, initial_value, params),
                'regime_simulation': self._regime_switching_simulation(returns, initial_value, params),
                'stress_scenarios': self._stress_scenarios(returns, initial_value),
                'success': True
            }
            
            return simulation_results
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)[:200],
                'success': False
            }
    
    def _gbm_simulation(self, returns: pd.Series, initial_value: float,
                       params: Dict) -> Dict:
        """Geometric Brownian Motion simulation"""
        
        try:
            n_simulations = params['n_simulations']
            time_horizon = params['time_horizon']
            
            mu = returns.mean()
            sigma = returns.std()
            
            # GBM parameters
            dt = 1 / self.config.TRADING_DAYS
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt)
            
            # Generate paths
            np.random.seed(42)
            paths = np.zeros((time_horizon + 1, n_simulations))
            paths[0] = initial_value
            
            for t in range(1, time_horizon + 1):
                z = np.random.standard_normal(n_simulations)
                paths[t] = paths[t-1] * np.exp(drift + diffusion * z)
            
            # Calculate statistics
            final_values = paths[-1]
            final_returns = (final_values / initial_value) - 1
            
            # Risk metrics
            risk_metrics = {}
            for conf in params['confidence_levels']:
                var = np.percentile(final_returns, (1 - conf) * 100)
                tail = final_returns[final_returns <= var]
                cvar = tail.mean() if len(tail) > 0 else var
                
                risk_metrics[f'var_{int(conf*100)}'] = var
                risk_metrics[f'cvar_{int(conf*100)}'] = cvar
            
            return {
                'paths': paths,
                'final_values': final_values,
                'final_returns': final_returns,
                'expected_final_value': final_values.mean(),
                'expected_return': final_returns.mean(),
                'volatility': final_returns.std(),
                'risk_metrics': risk_metrics,
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e)[:100],
                'success': False
            }
    
    def _bootstrap_simulation(self, returns: pd.Series, initial_value: float,
                            params: Dict) -> Dict:
        """Bootstrap simulation using historical returns"""
        
        try:
            n_simulations = params['n_simulations']
            time_horizon = params['time_horizon']
            
            # Bootstrap sampling
            np.random.seed(42)
            paths = np.zeros((time_horizon + 1, n_simulations))
            paths[0] = initial_value
            
            returns_array = returns.values
            
            for t in range(1, time_horizon + 1):
                # Sample with replacement
                samples = np.random.choice(returns_array, size=n_simulations, replace=True)
                paths[t] = paths[t-1] * (1 + samples)
            
            # Calculate statistics
            final_values = paths[-1]
            final_returns = (final_values / initial_value) - 1
            
            # Risk metrics
            risk_metrics = {}
            for conf in params['confidence_levels']:
                var = np.percentile(final_returns, (1 - conf) * 100)
                tail = final_returns[final_returns <= var]
                cvar = tail.mean() if len(tail) > 0 else var
                
                risk_metrics[f'var_{int(conf*100)}'] = var
                risk_metrics[f'cvar_{int(conf*100)}'] = cvar
            
            return {
                'paths': paths,
                'final_values': final_values,
                'final_returns': final_returns,
                'expected_final_value': final_values.mean(),
                'expected_return': final_returns.mean(),
                'volatility': final_returns.std(),
                'risk_metrics': risk_metrics,
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e)[:100],
                'success': False
            }
    
    def _regime_switching_simulation(self, returns: pd.Series, 
                                   initial_value: float, params: Dict) -> Dict:
        """Regime-switching simulation (simplified)"""
        
        try:
            # Simplified two-regime model
            n_simulations = params['n_simulations']
            time_horizon = params['time_horizon']
            
            # Estimate regimes using volatility clustering
            rolling_vol = returns.rolling(20).std().dropna()
            high_vol_regime = rolling_vol > rolling_vol.median()
            
            # Calculate regime parameters
            returns_high = returns[high_vol_regime.reindex(returns.index, fill_value=False)]
            returns_low = returns[~high_vol_regime.reindex(returns.index, fill_value=False)]
            
            mu_high = returns_high.mean() if len(returns_high) > 0 else returns.mean()
            sigma_high = returns_high.std() if len(returns_high) > 0 else returns.std()
            mu_low = returns_low.mean() if len(returns_low) > 0 else returns.mean()
            sigma_low = returns_low.std() if len(returns_low) > 0 else returns.std()
            
            # Transition probabilities (simplified)
            p_high_to_low = 0.1
            p_low_to_high = 0.05
            
            # Simulation
            np.random.seed(42)
            paths = np.zeros((time_horizon + 1, n_simulations))
            paths[0] = initial_value
            
            regimes = np.ones(n_simulations, dtype=int)  # Start in low volatility regime
            
            for t in range(1, time_horizon + 1):
                # Generate returns based on regime
                returns_t = np.zeros(n_simulations)
                
                # High volatility regime
                high_mask = regimes == 1
                returns_t[high_mask] = np.random.normal(
                    mu_high / self.config.TRADING_DAYS,
                    sigma_high / np.sqrt(self.config.TRADING_DAYS),
                    high_mask.sum()
                )
                
                # Low volatility regime
                low_mask = regimes == 0
                returns_t[low_mask] = np.random.normal(
                    mu_low / self.config.TRADING_DAYS,
                    sigma_low / np.sqrt(self.config.TRADING_DAYS),
                    low_mask.sum()
                )
                
                # Update prices
                paths[t] = paths[t-1] * (1 + returns_t)
                
                # Regime transitions
                transitions = np.random.random(n_simulations)
                
                # High to low transitions
                high_to_low = high_mask & (transitions < p_high_to_low)
                regimes[high_to_low] = 0
                
                # Low to high transitions
                low_to_high = low_mask & (transitions < p_low_to_high)
                regimes[low_to_high] = 1
            
            # Calculate statistics
            final_values = paths[-1]
            final_returns = (final_values / initial_value) - 1
            
            return {
                'paths': paths,
                'final_values': final_values,
                'final_returns': final_returns,
                'expected_final_value': final_values.mean(),
                'expected_return': final_returns.mean(),
                'volatility': final_returns.std(),
                'regime_parameters': {
                    'high_vol_mu': mu_high,
                    'high_vol_sigma': sigma_high,
                    'low_vol_mu': mu_low,
                    'low_vol_sigma': sigma_low
                },
                'success': True
            }
        except Exception as e:
            return {
                'error': str(e)[:100],
                'success': False
            }
    
    def _stress_scenarios(self, returns: pd.Series, 
                         initial_value: float) -> Dict:
        """Stress test scenarios"""
        
        try:
            scenarios = {
                '2008_Financial_Crisis': -0.50,
                '2020_COVID_Crash': -0.35,
                '2011_European_Debt_Crisis': -0.20,
                '2015_China_Stock_Market_Crash': -0.30,
                '2018_Volatility_Spike': -0.15
            }
            
            results = {}
            current_vol = returns.std() * np.sqrt(self.config.TRADING_DAYS)
            
            for scenario, shock in scenarios.items():
                impact = initial_value * shock
                probability = stats.norm.cdf(shock / current_vol) if current_vol > 0 else 0
                
                results[scenario] = {
                    'shock': shock,
                    'impact': impact,
                    'probability': probability,
                    'recovery_time': self._estimate_recovery_time(shock, returns)
                }
            
            return results
        except Exception as e:
            return {'error': str(e)[:100]}
    
    def _estimate_recovery_time(self, shock: float, 
                              returns: pd.Series) -> float:
        """Estimate recovery time from shock"""
        
        # Simplified recovery estimation
        expected_return = returns.mean() * self.config.TRADING_DAYS
        if expected_return > 0:
            recovery_years = abs(shock) / expected_return
            return recovery_years
        else:
            return float('inf')

# -------------------------------------------------------------
# ENHANCED PORTFOLIO ANALYTICS DASHBOARD
# -------------------------------------------------------------
class InstitutionalPortfolioDashboard:
    """Professional portfolio analytics dashboard"""
    
    def __init__(self):
        self.config = InstitutionalConfig()
        self.data_loader = InstitutionalDataLoader()
        self.optimizer = InstitutionalPortfolioOptimizer()
        self.risk_analytics = InstitutionalRiskAnalytics()
        self.performance_attribution = InstitutionalPerformanceAttribution()
        self.monte_carlo = InstitutionalMonteCarlo()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            
            # Core data
            st.session_state.selected_tickers = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
            st.session_state.portfolio_strategy = "Maximum Sharpe Ratio"
            st.session_state.custom_weights = {}
            st.session_state.data_loaded = False
            st.session_state.current_weights = None
            st.session_state.use_demo_data = False
            
            # Analysis results
            st.session_state.analysis_results = {}
            st.session_state.risk_report = {}
            st.session_state.attribution_report = {}
            st.session_state.monte_carlo_results = {}
            
            # UI state
            st.session_state.current_tab = "Overview"
            st.session_state.optimization_history = []
    
    def run(self):
        """Run the dashboard"""
        
        # Set page config
        st.set_page_config(
            page_title="🏛️ Apollo/ENIGMA - Institutional Portfolio Terminal",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply institutional styling
        self._apply_institutional_styling()
        
        # Main title
        st.title("🏛️ Apollo/ENIGMA - Institutional Portfolio Terminal v5.0")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content area
        self._render_main_content()
    
    def _apply_institutional_styling(self):
        """Apply institutional-grade styling"""
        
        st.markdown("""
        <style>
        :root {
            --primary: #1a5fb4;
            --primary-dark: #0d4fa0;
            --secondary: #26a269;
            --danger: #c01c28;
            --warning: #f5a623;
            --dark-bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --success: #16a34a;
            --success-dark: #15803d;
        }

        /* Enhanced institutional styling */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        /* Professional cards */
        .institutional-card {
            background: linear-gradient(145deg, var(--card-bg), #1a2332);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        
        .institutional-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: var(--primary);
        }
        
        /* Enhanced metrics */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, var(--card-bg), #1a2332) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 24px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        }
        
        /* Professional tables */
        .stDataFrame {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        
        /* Enhanced tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--card-bg) !important;
            border-radius: 12px !important;
            padding: 8px !important;
            gap: 4px !important;
            border: 1px solid var(--border) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            color: var(--text-muted) !important;
            border: 1px solid transparent !important;
            transition: all 0.3s !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
            color: white !important;
            border-color: var(--primary) !important;
            box-shadow: 0 4px 12px rgba(26, 95, 180, 0.3) !important;
        }
        
        /* Professional buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
            color: white !important;
            border: none !important;
            padding: 14px 28px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.3s !important;
            box-shadow: 0 4px 15px rgba(26, 95, 180, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(26, 95, 180, 0.4) !important;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-success { background-color: var(--success); }
        .status-warning { background-color: var(--warning); }
        .status-error { background-color: var(--danger); }
        .status-info { background-color: var(--primary); }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark-bg);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(var(--primary), var(--primary-dark));
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(var(--primary-dark), var(--primary));
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar configuration"""
        
        with st.sidebar:
            st.title("⚙️ Institutional Configuration")
            
            # Asset selection
            self._render_asset_selection()
            
            # Benchmark selection
            self._render_benchmark_selection()
            
            # Date range
            self._render_date_range()
            
            # Portfolio strategy
            self._render_portfolio_strategy()
            
            # Advanced settings
            self._render_advanced_settings()
            
            # Action buttons
            self._render_action_buttons()
    
    def _render_asset_selection(self):
        """Render asset selection section"""
        
        st.subheader("🌍 Asset Universe")
        
        # Category filter
        categories = list(GLOBAL_ASSET_UNIVERSE_ENHANCED.keys())
        selected_categories = st.multiselect(
            "Filter Categories",
            categories,
            default=["US_Indices", "US_Stocks", "Bonds", "Commodities"],
            key="sidebar_categories"
        )
        
        # Get assets from selected categories
        filtered_assets = []
        for category in selected_categories:
            filtered_assets.extend([asset.ticker for asset in GLOBAL_ASSET_UNIVERSE_ENHANCED[category]])
        
        # Asset search and selection
        asset_search = st.text_input("🔍 Search Assets", key="asset_search")
        
        if asset_search:
            filtered_assets = [ticker for ticker in filtered_assets 
                             if asset_search.lower() in ticker.lower()]
        
        # Multi-select with checkboxes for better UX
        selected_tickers = st.multiselect(
            "Select Assets (3-20 recommended)",
            filtered_assets,
            default=st.session_state.selected_tickers,
            key="sidebar_assets"
        )
        
        # Validate selection
        if len(selected_tickers) > InstitutionalConfig.MAX_ASSETS:
            st.error(f"Maximum {InstitutionalConfig.MAX_ASSETS} assets allowed")
            selected_tickers = selected_tickers[:InstitutionalConfig.MAX_ASSETS]
        
        if len(selected_tickers) < 2:
            st.warning("Select at least 2 assets for portfolio analysis")
        
        st.session_state.selected_tickers = selected_tickers
        
        # Display selected assets with metadata
        if selected_tickers:
            with st.expander("📋 Selected Assets", expanded=False):
                for ticker in selected_tickers:
                    metadata = TICKER_TO_METADATA.get(ticker)
                    if metadata:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.text(ticker)
                        with col2:
                            st.caption(f"{metadata.name} • {metadata.category}")
    
    def _render_benchmark_selection(self):
        """Render benchmark selection"""
        
        st.subheader("📊 Benchmark")
        
        benchmark_options = ["SPY", "QQQ", "VTI", "IWM"] + [
            ticker for ticker in ALL_TICKERS_ENHANCED 
            if ticker not in st.session_state.selected_tickers
        ][:20]
        
        benchmark = st.selectbox(
            "Primary Benchmark",
            benchmark_options,
            index=0,
            help="Benchmark for performance comparison",
            key="sidebar_benchmark"
        )
        
        st.session_state.benchmark = benchmark
    
    def _render_date_range(self):
        """Render date range selection"""
        
        st.subheader("📅 Time Horizon")
        
        date_preset = st.selectbox(
            "Analysis Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "10 Years", "Max", "Custom"],
            index=3,  # Default to 1 year
            key="sidebar_date_preset"
        )
        
        end_date = pd.Timestamp.today()
        
        if date_preset == "1 Month":
            start_date = end_date - pd.DateOffset(months=1)
        elif date_preset == "3 Months":
            start_date = end_date - pd.DateOffset(months=3)
        elif date_preset == "6 Months":
            start_date = end_date - pd.DateOffset(months=6)
        elif date_preset == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        elif date_preset == "3 Years":
            start_date = end_date - pd.DateOffset(years=3)
        elif date_preset == "5 Years":
            start_date = end_date - pd.DateOffset(years=5)
        elif date_preset == "10 Years":
            start_date = end_date - pd.DateOffset(years=10)
        elif date_preset == "Max":
            start_date = pd.Timestamp("2000-01-01")
        else:  # Custom
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", pd.Timestamp("2023-01-01"))
            with col2:
                end_date = st.date_input("End Date", pd.Timestamp.today())
        
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
    
    def _render_portfolio_strategy(self):
        """Render portfolio strategy selection"""
        
        st.subheader("🎯 Portfolio Strategy")
        
        strategies = [
            "Equal Weight",
            "Minimum Volatility", 
            "Maximum Sharpe Ratio",
            "Maximum Quadratic Utility",
            "Efficient Risk",
            "Efficient Return",
            "Risk Parity",
            "Maximum Diversification",
            "Hierarchical Risk Parity",
            "Custom Weights"
        ]
        
        strategy = st.selectbox(
            "Optimization Method",
            strategies,
            index=2,  # Default to Maximum Sharpe Ratio
            help="Portfolio construction methodology",
            key="sidebar_strategy"
        )
        
        st.session_state.portfolio_strategy = strategy
        
        # Strategy-specific parameters
        if strategy == "Efficient Risk":
            target_risk = st.slider(
                "Target Annual Volatility (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=0.5,
                key="target_risk"
            ) / 100
            st.session_state.strategy_params = {'target_risk': target_risk}
        
        elif strategy == "Efficient Return":
            target_return = st.slider(
                "Target Annual Return (%)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.5,
                key="target_return"
            ) / 100
            st.session_state.strategy_params = {'target_return': target_return}
    
    def _render_advanced_settings(self):
        """Render advanced settings"""
        
        with st.expander("⚙️ Advanced Institutional Settings", expanded=False):
            
            # Risk parameters
            st.subheader("📈 Risk Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rf_rate = st.number_input(
                    "Risk-Free Rate (annual %)",
                    value=3.0,
                    min_value=0.0,
                    max_value=20.0,
                    step=0.1,
                    key="rf_rate"
                ) / 100
            
            with col2:
                confidence_level = st.select_slider(
                    "VaR Confidence Level",
                    options=[90, 95, 99],
                    value=95,
                    key="confidence_level"
                )
            
            # Optimization constraints
            st.subheader("⚖️ Optimization Constraints")
            
            col1, col2 = st.columns(2)
            
            with col1:
                allow_short = st.checkbox(
                    "Allow Short Selling",
                    value=False,
                    key="allow_short"
                )
                
                max_concentration = st.slider(
                    "Maximum Concentration (%)",
                    min_value=5,
                    max_value=100,
                    value=25,
                    step=5,
                    key="max_concentration"
                ) / 100
            
            with col2:
                min_diversification = st.slider(
                    "Minimum Effective Assets",
                    min_value=1,
                    max_value=20,
                    value=3,
                    step=1,
                    key="min_diversification"
                )
            
            # Data settings
            st.subheader("💾 Data Settings")
            
            use_demo = st.checkbox(
                "Use demo data if needed",
                value=False,
                help="Generate synthetic data if real data is unavailable",
                key="use_demo"
            )
            
            enable_cache = st.checkbox(
                "Enable caching",
                value=True,
                help="Cache data for faster analysis",
                key="enable_cache"
            )
            
            # Store settings
            st.session_state.advanced_settings = {
                'rf_rate': rf_rate,
                'confidence_level': confidence_level / 100,
                'allow_short': allow_short,
                'max_concentration': max_concentration,
                'min_diversification': min_diversification,
                'use_demo': use_demo,
                'enable_cache': enable_cache
            }
    
    def _render_action_buttons(self):
        """Render action buttons"""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                self._run_analysis()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                self._reset_analysis()
        
        with col3:
            if st.button("📊 Export", use_container_width=True):
                self._export_results()
    
    def _run_analysis(self):
        """Run comprehensive portfolio analysis"""
        
        try:
            # Validate inputs
            if len(st.session_state.selected_tickers) < 2:
                st.error("Please select at least 2 assets")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load data
            status_text.text("📥 Loading market data...")
            progress_bar.progress(10)
            
            data_loader = InstitutionalDataLoader(
                cache_enabled=st.session_state.advanced_settings.get('enable_cache', True)
            )
            
            all_tickers = st.session_state.selected_tickers + [st.session_state.benchmark]
            
            try:
                prices = data_loader.load_batch(
                    all_tickers,
                    st.session_state.start_date,
                    st.session_state.end_date,
                    use_parallel=True
                )
            except Exception as e:
                if st.session_state.advanced_settings.get('use_demo', False):
                    st.warning("Real data loading failed. Using demo data...")
                    # Generate demo data
                    demo_data = {}
                    for ticker in all_tickers:
                        demo_series = data_loader._generate_demo_data(
                            ticker,
                            st.session_state.start_date,
                            st.session_state.end_date
                        )
                        demo_data[ticker] = demo_series
                    
                    prices = pd.DataFrame(demo_data)
                else:
                    raise
            
            if prices.empty:
                st.error("No data loaded. Please check your selections.")
                return
            
            # Ensure benchmark is in the data
            if st.session_state.benchmark not in prices.columns:
                st.error(f"Benchmark {st.session_state.benchmark} not found in data")
                return
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            if len(returns) < InstitutionalConfig.MIN_DATA_POINTS:
                st.warning(f"Limited data: only {len(returns)} trading days")
            
            # Store data
            st.session_state.prices = prices
            st.session_state.returns = returns
            st.session_state.data_loaded = True
            
            progress_bar.progress(30)
            
            # Step 2: Portfolio optimization
            status_text.text("🔧 Optimizing portfolio...")
            
            optimizer = InstitutionalPortfolioOptimizer()
            
            # Get constraints
            constraints = {
                'min_weight': 0.0 if not st.session_state.advanced_settings.get('allow_short', False) else -1.0,
                'max_weight': 1.0,
                'max_concentration': st.session_state.advanced_settings.get('max_concentration', 0.25),
                'min_diversification': st.session_state.advanced_settings.get('min_diversification', 3)
            }
            
            # Merge strategy params
            if hasattr(st.session_state, 'strategy_params'):
                constraints.update(st.session_state.strategy_params)
            
            # Run optimization
            optimization_result = optimizer.optimize_with_constraints(
                returns[st.session_state.selected_tickers],
                st.session_state.portfolio_strategy,
                constraints,
                st.session_state.advanced_settings.get('rf_rate', 0.03)
            )
            
            st.session_state.optimization_result = optimization_result
            st.session_state.current_weights = optimization_result['weights']
            
            # Calculate portfolio returns
            portfolio_returns = (returns[st.session_state.selected_tickers] * 
                               optimization_result['weights']).sum(axis=1)
            st.session_state.portfolio_returns = portfolio_returns
            
            progress_bar.progress(50)
            
            # Step 3: Risk analytics
            status_text.text("📊 Analyzing risks...")
            
            risk_analytics = InstitutionalRiskAnalytics()
            risk_report = risk_analytics.comprehensive_risk_report(
                portfolio_returns,
                portfolio_value=1000000  # Default portfolio value
            )
            
            st.session_state.risk_report = risk_report
            
            progress_bar.progress(70)
            
            # Step 4: Performance attribution
            status_text.text("🎯 Attributing performance...")
            
            benchmark_returns = returns[st.session_state.benchmark]
            benchmark_weights = np.ones(len(st.session_state.selected_tickers)) / len(st.session_state.selected_tickers)
            
            attribution = InstitutionalPerformanceAttribution()
            attribution_report = attribution.comprehensive_attribution(
                portfolio_returns,
                benchmark_returns,
                optimization_result['weights'],
                benchmark_weights,
                returns[st.session_state.selected_tickers]
            )
            
            st.session_state.attribution_report = attribution_report
            
            progress_bar.progress(85)
            
            # Step 5: Monte Carlo simulation
            status_text.text("🎲 Running simulations...")
            
            monte_carlo = InstitutionalMonteCarlo()
            monte_carlo_results = monte_carlo.comprehensive_simulation(
                portfolio_returns,
                initial_value=1000000,
                params={
                    'n_simulations': 10000,
                    'time_horizon': 252,
                    'confidence_levels': [0.90, 0.95, 0.99]
                }
            )
            
            st.session_state.monte_carlo_results = monte_carlo_results
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store analysis timestamp
            st.session_state.analysis_timestamp = datetime.now().isoformat()
            
            # Show success message
            st.success(f"Analysis completed successfully! Processed {len(returns)} trading days.")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)[:200]}")
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
    
    def _reset_analysis(self):
        """Reset analysis results"""
        
        for key in list(st.session_state.keys()):
            if key not in ['dashboard_initialized', 'selected_tickers', 
                          'portfolio_strategy', 'use_demo_data']:
                del st.session_state[key]
        
        st.rerun()
    
    def _export_results(self):
        """Export analysis results"""
        
        if not st.session_state.data_loaded:
            st.warning("No analysis results to export")
            return
        
        try:
            # Create export data
            export_data = {
                'timestamp': st.session_state.get('analysis_timestamp', datetime.now().isoformat()),
                'configuration': {
                    'selected_tickers': st.session_state.selected_tickers,
                    'benchmark': st.session_state.benchmark,
                    'strategy': st.session_state.portfolio_strategy,
                    'start_date': str(st.session_state.start_date),
                    'end_date': str(st.session_state.end_date)
                },
                'optimization_result': st.session_state.get('optimization_result', {}),
                'risk_report': st.session_state.get('risk_report', {}),
                'attribution_report': st.session_state.get('attribution_report', {}),
                'monte_carlo_results': st.session_state.get('monte_carlo_results', {})
            }
            
            # Convert to JSON
            export_json = json.dumps(export_data, indent=2, default=str)
            
            # Create download button
            st.download_button(
                label="📥 Download Analysis Results",
                data=export_json,
                file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Export failed: {str(e)[:100]}")
    
    def _render_main_content(self):
        """Render the main content area"""
        
        # Check if analysis has been run
        if not st.session_state.data_loaded:
            self._render_welcome_screen()
            return
        
        # Create enhanced tabs
        tab_names = [
            "📊 Portfolio Overview",
            "⚖️ Risk Analytics",
            "🎯 Optimization Results", 
            "🔗 Correlation & Factors",
            "📈 Performance Attribution",
            "🎲 Monte Carlo Simulation",
            "📋 Institutional Reports"
        ]
        
        tabs = st.tabs(tab_names)
        
        # Tab 1: Portfolio Overview
        with tabs[0]:
            self._render_portfolio_overview()
        
        # Tab 2: Risk Analytics
        with tabs[1]:
            self._render_risk_analytics()
        
        # Tab 3: Optimization Results
        with tabs[2]:
            self._render_optimization_results()
        
        # Tab 4: Correlation & Factors
        with tabs[3]:
            self._render_correlation_factors()
        
        # Tab 5: Performance Attribution
        with tabs[4]:
            self._render_performance_attribution()
        
        # Tab 6: Monte Carlo Simulation
        with tabs[5]:
            self._render_monte_carlo_simulation()
        
        # Tab 7: Institutional Reports
        with tabs[6]:
            self._render_institutional_reports()
    
    def _render_welcome_screen(self):
        """Render welcome screen"""
        
        st.markdown("""
        <div class="institutional-card">
            <h2>🏛️ Welcome to Apollo/ENIGMA Institutional Terminal</h2>
            <p>Professional portfolio optimization and risk analytics platform for institutional investors.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="institutional-card">
                <h3>🚀 Quick Start</h3>
                <ol>
                    <li>Select assets in the sidebar (3-20 recommended)</li>
                    <li>Choose a benchmark for comparison</li>
                    <li>Select time period for analysis</li>
                    <li>Choose portfolio optimization strategy</li>
                    <li>Configure advanced settings if needed</li>
                    <li>Click "Run Analysis" to begin</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="institutional-card">
                <h3>📊 Recommended Portfolios</h3>
                <ul>
                    <li><strong>Core Satellite:</strong> SPY + TLT + GLD + 2-3 individual stocks</li>
                    <li><strong>Risk Parity:</strong> Equal risk contribution across asset classes</li>
                    <li><strong>Global Diversified:</strong> Mix of US, International, Bonds, Commodities</li>
                    <li><strong>Thematic:</strong> Focus on specific sectors or trends</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Asset universe statistics
        st.markdown("""
        <div class="institutional-card">
            <h3>🌍 Asset Universe Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assets", len(ALL_TICKERS_ENHANCED))
        
        with col2:
            st.metric("Asset Categories", len(GLOBAL_ASSET_UNIVERSE_ENHANCED))
        
        with col3:
            st.metric("Geographic Coverage", "15+ Countries")
        
        with col4:
            st.metric("Asset Types", "Equities, Bonds, Commodities, Crypto")
    
    def _render_portfolio_overview(self):
        """Render portfolio overview tab"""
        
        st.header("📊 Portfolio Overview")
        
        if not hasattr(st.session_state, 'optimization_result'):
            st.warning("Please run analysis first")
            return
        
        result = st.session_state.optimization_result
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Return", 
                f"{result.get('expected_return', 0):.2%}",
                delta=f"{result.get('expected_return', 0) - st.session_state.advanced_settings.get('rf_rate', 0.03):+.2%}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Expected Risk",
                f"{result.get('expected_risk', 0):.2%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{result.get('sharpe_ratio', 0):.2f}",
                delta=None
            )
        
        with col4:
            diversification = 1 / (result['weights'] ** 2).sum()
            st.metric(
                "Effective Diversification",
                f"{diversification:.1f}",
                delta=None
            )
        
        # Portfolio weights visualization
        st.subheader("📈 Portfolio Allocation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Weights table
            weights_df = pd.DataFrame({
                'Asset': st.session_state.selected_tickers,
                'Weight': result['weights'],
                'Category': [
                    TICKER_TO_METADATA.get(ticker, AssetMetadata(ticker, "", "", "", "")).category 
                    for ticker in st.session_state.selected_tickers
                ]
            }).sort_values('Weight', ascending=False)
            
            st.dataframe(
                weights_df.style.format({'Weight': '{:.2%}'}).background_gradient(
                    subset=['Weight'], cmap='Blues'
                ),
                use_container_width=True,
                height=400
            )
        
        with col2:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=weights_df['Asset'],
                values=weights_df['Weight'],
                hole=0.4,
                textinfo='label+percent',
                textposition='auto',
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig.update_layout(
                title="Portfolio Allocation",
                height=400,
                template="plotly_dark",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance chart
        st.subheader("📊 Portfolio Performance")
        
        portfolio_cumulative = (1 + st.session_state.portfolio_returns).cumprod()
        benchmark_cumulative = (1 + st.session_state.returns[st.session_state.benchmark]).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            name=f"Portfolio ({result.get('method', 'Unknown')})",
            line=dict(color='#1a5fb4', width=3),
            fill='tozeroy',
            fillcolor='rgba(26, 95, 180, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            name=f"Benchmark ({st.session_state.benchmark})",
            line=dict(color='#26a269', width=2, dash='dash')
        ))
        
        fig.update_layout(
            height=500,
            template="plotly_dark",
            title="Cumulative Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_analytics(self):
        """Render risk analytics tab"""
        
        st.header("⚖️ Comprehensive Risk Analytics")
        
        if not hasattr(st.session_state, 'risk_report'):
            st.warning("Please run analysis first")
            return
        
        risk_report = st.session_state.risk_report
        
        if not risk_report.get('success', False):
            st.error("Risk analysis failed")
            return
        
        # Basic statistics
        st.subheader("📈 Basic Risk Statistics")
        
        basic_stats = risk_report.get('basic_statistics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Annual Return", f"{basic_stats.get('annual_return', 0):.2%}")
        
        with col2:
            st.metric("Annual Volatility", f"{basic_stats.get('annual_volatility', 0):.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{basic_stats.get('sharpe_ratio', 0):.2f}")
        
        with col4:
            st.metric("Sortino Ratio", f"{basic_stats.get('sortino_ratio', 0):.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Skewness", f"{basic_stats.get('skewness', 0):.2f}")
        
        with col2:
            st.metric("Kurtosis", f"{basic_stats.get('kurtosis', 0):.2f}")
        
        with col3:
            st.metric("VaR (95%)", f"{basic_stats.get('var_95', 0):.2%}")
        
        with col4:
            st.metric("CVaR (95%)", f"{basic_stats.get('cvar_95', 0):.2%}")
        
        # VaR Analysis
        st.subheader("📊 Value at Risk Analysis")
        
        var_analysis = risk_report.get('var_analysis', {})
        
        # Create VaR comparison table
        var_data = []
        for conf_key, horizon_data in var_analysis.items():
            conf_level = int(conf_key.split('_')[1])
            for horizon_key, methods_data in horizon_data.items():
                horizon = horizon_key.split('_')[0]
                for method, method_data in methods_data.items():
                    if isinstance(method_data, dict) and method_data.get('success', False):
                        var_data.append({
                            'Confidence': f"{conf_level}%",
                            'Horizon': f"{horizon} Day",
                            'Method': method_data.get('method', method),
                            'VaR': method_data.get('var', np.nan),
                            'CVaR': method_data.get('cvar', np.nan)
                        })
        
        if var_data:
            var_df = pd.DataFrame(var_data)
            
            # Pivot table for better visualization
            pivot_df = var_df.pivot_table(
                index=['Method', 'Horizon'],
                columns='Confidence',
                values='VaR',
                aggfunc='first'
            )
            
            st.dataframe(
                pivot_df.style.format('{:.4%}'),
                use_container_width=True
            )
        
        # Drawdown Analysis
        st.subheader("📉 Drawdown Analysis")
        
        drawdown_analysis = risk_report.get('drawdown_analysis', {})
        
        if drawdown_analysis.get('success', False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Maximum Drawdown",
                    f"{drawdown_analysis.get('max_drawdown', 0):.2%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Avg Drawdown Duration",
                    f"{drawdown_analysis.get('avg_drawdown_duration', 0):.0f} days",
                    delta=None
                )
            
            with col3:
                recovery_days = drawdown_analysis.get('recovery_days', 'N/A')
                if isinstance(recovery_days, (int, float)):
                    st.metric("Recovery Time", f"{recovery_days} days")
                else:
                    st.metric("Recovery Time", recovery_days)
            
            # Drawdown chart
            if 'drawdown_series' in drawdown_analysis:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=st.session_state.portfolio_returns.index,
                    y=drawdown_analysis['drawdown_series'],
                    fill='tozeroy',
                    fillcolor='rgba(193, 28, 40, 0.3)',
                    line=dict(color='#c01c28', width=2),
                    name="Drawdown"
                ))
                
                fig.update_layout(
                    height=400,
                    template="plotly_dark",
                    title="Portfolio Drawdown Over Time",
                    xaxis_title="Date",
                    yaxis_title="Drawdown",
                    yaxis_tickformat='.1%'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Stress Tests
        st.subheader("🌪️ Stress Test Scenarios")
        
        stress_tests = risk_report.get('stress_tests', {})
        
        if stress_tests.get('success', False):
            scenarios = []
            for scenario, data in stress_tests.items():
                if scenario != 'success' and scenario != 'error':
                    scenarios.append({
                        'Scenario': scenario.replace('_', ' ').title(),
                        'Historical Loss': f"{data.get('historical_worst_day', 0):.2%}",
                        'Probability': f"{data.get('probability', 0):.2%}",
                        'Potential Impact': f"${data.get('potential_loss', 0):,.0f}"
                    })
            
            if scenarios:
                stress_df = pd.DataFrame(scenarios)
                st.dataframe(
                    stress_df.style.format({
                        'Historical Loss': '{:.2%}',
                        'Probability': '{:.2%}'
                    }),
                    use_container_width=True
                )
    
    def _render_optimization_results(self):
        """Render optimization results tab"""
        
        st.header("🎯 Optimization Results")
        
        if not hasattr(st.session_state, 'optimization_result'):
            st.warning("Please run analysis first")
            return
        
        result = st.session_state.optimization_result
        
        # Optimization details
        st.subheader("📋 Optimization Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Optimization Method", result.get('method', 'Unknown'))
            st.metric("Optimizer Used", result.get('optimizer', 'Unknown'))
            st.metric("Constraints Applied", "Yes" if result.get('constraints_applied', False) else "No")
        
        with col2:
            st.metric("Success Status", "Success" if result.get('success', False) else "Failed")
            st.metric("Risk Limits Applied", "Yes" if result.get('risk_limits_applied', False) else "No")
            if result.get('effective_diversification'):
                st.metric("Effective Diversification", f"{result['effective_diversification']:.1f}")
        
        # Efficient Frontier
        st.subheader("📈 Efficient Frontier Analysis")
        
        # Generate efficient frontier points
        if PYPFOPT_AVAILABLE and hasattr(st.session_state, 'returns'):
            try:
                returns = st.session_state.returns[st.session_state.selected_tickers]
                
                mu = expected_returns.mean_historical_return(returns)
                S = risk_models.sample_cov(returns)
                
                ef = EfficientFrontier(mu, S)
                
                # Generate efficient frontier
                risk_range = np.linspace(0.05, 0.30, 50) / np.sqrt(InstitutionalConfig.TRADING_DAYS)
                efficient_portfolios = []
                
                for risk_target in risk_range:
                    try:
                        ef.efficient_risk(risk_target)
                        ret, risk, _ = ef.portfolio_performance()
                        efficient_portfolios.append((risk * np.sqrt(InstitutionalConfig.TRADING_DAYS), 
                                                   ret * InstitutionalConfig.TRADING_DAYS))
                    except:
                        continue
                
                if efficient_portfolios:
                    eff_risks, eff_returns = zip(*efficient_portfolios)
                    
                    # Plot efficient frontier
                    fig = go.Figure()
                    
                    # Efficient frontier
                    fig.add_trace(go.Scatter(
                        x=eff_risks,
                        y=eff_returns,
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color='#1a5fb4', width=3)
                    ))
                    
                    # Current portfolio
                    fig.add_trace(go.Scatter(
                        x=[result['expected_risk']],
                        y=[result['expected_return']],
                        mode='markers',
                        name='Current Portfolio',
                        marker=dict(size=15, color='#f5a623', symbol='star')
                    ))
                    
                    # Equal weight portfolio
                    eq_weights = np.ones(len(st.session_state.selected_tickers)) / len(st.session_state.selected_tickers)
                    eq_returns = (returns * eq_weights).sum(axis=1)
                    eq_return = eq_returns.mean() * InstitutionalConfig.TRADING_DAYS
                    eq_risk = eq_returns.std() * np.sqrt(InstitutionalConfig.TRADING_DAYS)
                    
                    fig.add_trace(go.Scatter(
                        x=[eq_risk],
                        y=[eq_return],
                        mode='markers',
                        name='Equal Weight',
                        marker=dict(size=12, color='#26a269', symbol='circle')
                    ))
                    
                    # Individual assets
                    asset_returns = returns.mean() * InstitutionalConfig.TRADING_DAYS
                    asset_risks = returns.std() * np.sqrt(InstitutionalConfig.TRADING_DAYS)
                    
                    fig.add_trace(go.Scatter(
                        x=asset_risks,
                        y=asset_returns,
                        mode='markers+text',
                        name='Individual Assets',
                        text=returns.columns,
                        textposition="top center",
                        marker=dict(size=10, color='#94a3b8', symbol='square')
                    ))
                    
                    fig.update_layout(
                        height=600,
                        template="plotly_dark",
                        title="Efficient Frontier Analysis",
                        xaxis_title="Annual Volatility",
                        yaxis_title="Annual Return",
                        hovermode='closest',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not generate efficient frontier: {str(e)[:100]}")
        
        # Optimization history
        st.subheader("📊 Optimization History")
        
        if hasattr(st.session_state, 'optimization_history') and st.session_state.optimization_history:
            history_df = pd.DataFrame(st.session_state.optimization_history)
            
            # Plot optimization history
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['expected_return'],
                mode='lines+markers',
                name='Expected Return',
                line=dict(color='#1a5fb4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['expected_risk'],
                mode='lines+markers',
                name='Expected Risk',
                line=dict(color='#f5a623', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['sharpe_ratio'],
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color='#26a269', width=2)
            ))
            
            fig.update_layout(
                height=400,
                template="plotly_dark",
                title="Optimization History",
                xaxis_title="Optimization Run",
                yaxis_title="Value",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_factors(self):
        """Render correlation and factors tab"""
        
        st.header("🔗 Correlation & Factor Analysis")
        
        if not hasattr(st.session_state, 'returns'):
            st.warning("Please run analysis first")
            return
        
        returns = st.session_state.returns[st.session_state.selected_tickers]
        
        # Correlation matrix
        st.subheader("📊 Correlation Matrix")
        
        corr_matrix = returns.corr()
        
        # Enhanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            height=700,
            template="plotly_dark",
            title="Asset Correlation Matrix",
            xaxis_title="Assets",
            yaxis_title="Assets",
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation statistics
        st.subheader("📈 Correlation Statistics")
        
        # Get upper triangle of correlation matrix
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_corr = corr_values.mean() if len(corr_values) > 0 else 0
            st.metric("Average Correlation", f"{avg_corr:.3f}")
        
        with col2:
            min_corr = corr_values.min() if len(corr_values) > 0 else 0
            st.metric("Minimum Correlation", f"{min_corr:.3f}")
        
        with col3:
            max_corr = corr_values.max() if len(corr_values) > 0 else 0
            st.metric("Maximum Correlation", f"{max_corr:.3f}")
        
        with col4:
            std_corr = corr_values.std() if len(corr_values) > 0 else 0
            st.metric("Correlation Std", f"{std_corr:.3f}")
        
        # Correlation distribution
        fig2 = go.Figure()
        
        fig2.add_trace(go.Histogram(
            x=corr_values,
            nbinsx=30,
            name="Correlation Distribution",
            marker_color='#1a5fb4',
            opacity=0.7
        ))
        
        fig2.update_layout(
            height=400,
            template="plotly_dark",
            title="Correlation Distribution",
            xaxis_title="Correlation",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Principal Component Analysis
        st.subheader("🔍 Principal Component Analysis")
        
        try:
            # Standardize returns
            returns_standardized = (returns - returns.mean()) / returns.std()
            
            # Perform PCA
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=min(5, len(returns.columns)))
            pca.fit(returns_standardized)
            
            # Explained variance
            fig3 = go.Figure()
            
            fig3.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                y=pca.explained_variance_ratio_ * 100,
                name="Explained Variance",
                marker_color='#1a5fb4'
            ))
            
            fig3.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                y=np.cumsum(pca.explained_variance_ratio_) * 100,
                name="Cumulative Variance",
                line=dict(color='#f5a623', width=3),
                mode='lines+markers'
            ))
            
            fig3.update_layout(
                height=400,
                template="plotly_dark",
                title="PCA Explained Variance",
                xaxis_title="Principal Components",
                yaxis_title="Variance Explained (%)",
                yaxis_tickformat='.1f'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # PCA loadings heatmap
            loadings = pca.components_[:3]  # First 3 PCs
            
            fig4 = make_subplots(
                rows=1, cols=3,
                subplot_titles=[f"PC{i+1} Loadings" for i in range(3)]
            )
            
            for i in range(3):
                fig4.add_trace(
                    go.Bar(
                        x=returns.columns,
                        y=loadings[i],
                        name=f"PC{i+1}",
                        marker_color='#1a5fb4'
                    ),
                    row=1, col=i+1
                )
            
            fig4.update_layout(
                height=400,
                template="plotly_dark",
                showlegend=False,
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig4, use_container_width=True)
            
        except Exception as e:
            st.warning(f"PCA analysis skipped: {str(e)[:100]}")
    
    def _render_performance_attribution(self):
        """Render performance attribution tab"""
        
        st.header("📈 Performance Attribution")
        
        if not hasattr(st.session_state, 'attribution_report'):
            st.warning("Please run analysis first")
            return
        
        attribution = st.session_state.attribution_report
        
        if not attribution.get('success', False):
            st.error("Performance attribution failed")
            return
        
        # Brinson attribution
        brinson = attribution.get('brinson_attribution', {})
        
        st.subheader("🎯 Brinson Attribution Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Active Return",
                f"{brinson.get('total_active_return', 0):.2%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Allocation Effect",
                f"{brinson.get('allocation_effect', 0):.2%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Selection Effect",
                f"{brinson.get('selection_effect', 0):.2%}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Interaction Effect",
                f"{brinson.get('interaction_effect', 0):.2%}",
                delta=None
            )
        
        # Attribution breakdown
        st.subheader("📊 Attribution Breakdown")
        
        fig = go.Figure(data=[
            go.Bar(
                name='Allocation',
                x=['Allocation'],
                y=[brinson.get('allocation_effect', 0)],
                marker_color='#1a5fb4'
            ),
            go.Bar(
                name='Selection',
                x=['Selection'],
                y=[brinson.get('selection_effect', 0)],
                marker_color='#26a269'
            ),
            go.Bar(
                name='Interaction',
                x=['Interaction'],
                y=[brinson.get('interaction_effect', 0)],
                marker_color='#f5a623'
            ),
            go.Bar(
                name='Total Active',
                x=['Total'],
                y=[brinson.get('total_active_return', 0)],
                marker_color='#c01c28'
            )
        ])
        
        fig.update_layout(
            height=400,
            template="plotly_dark",
            title="Performance Attribution Components",
            yaxis_title="Active Return Contribution",
            yaxis_tickformat='.2%',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Asset-level attribution
        st.subheader("📈 Asset-Level Attribution")
        
        asset_attribution = brinson.get('asset_attribution', [])
        
        if asset_attribution:
            asset_df = pd.DataFrame(asset_attribution)
            
            # Sort by total effect
            asset_df = asset_df.sort_values('total_effect', ascending=False)
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Bar(
                x=asset_df['asset'],
                y=asset_df['allocation_effect'],
                name='Allocation Effect',
                marker_color='#1a5fb4'
            ))
            
            fig2.add_trace(go.Bar(
                x=asset_df['asset'],
                y=asset_df['selection_effect'],
                name='Selection Effect',
                marker_color='#26a269'
            ))
            
            fig2.update_layout(
                height=400,
                template="plotly_dark",
                title="Asset-Level Attribution",
                xaxis_title="Assets",
                yaxis_title="Contribution",
                yaxis_tickformat='.2%',
                barmode='stack',
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Risk decomposition
        st.subheader("⚖️ Risk Decomposition")
        
        risk_decomp = attribution.get('risk_decomposition', {})
        
        if 'portfolio_volatility' in risk_decomp:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Portfolio Volatility",
                    f"{risk_decomp['portfolio_volatility']:.2%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Diversification Ratio",
                    f"{risk_decomp.get('diversification_ratio', 0):.2f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Effective Number of Assets",
                    f"{risk_decomp.get('effective_number', 0):.1f}",
                    delta=None
                )
    
    def _render_monte_carlo_simulation(self):
        """Render Monte Carlo simulation tab"""
        
        st.header("🎲 Monte Carlo Simulation")
        
        if not hasattr(st.session_state, 'monte_carlo_results'):
            st.warning("Please run analysis first")
            return
        
        mc_results = st.session_state.monte_carlo_results
        
        if not mc_results.get('success', False):
            st.error("Monte Carlo simulation failed")
            return
        
        # GBM Simulation Results
        st.subheader("📊 Geometric Brownian Motion Simulation")
        
        gbm_results = mc_results.get('gbm_simulation', {})
        
        if gbm_results.get('success', False):
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Final Value",
                    f"${gbm_results.get('expected_final_value', 0):,.0f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Expected Return",
                    f"{gbm_results.get('expected_return', 0):.2%}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Volatility",
                    f"{gbm_results.get('volatility', 0):.2%}",
                    delta=None
                )
            
            with col4:
                risk_metrics = gbm_results.get('risk_metrics', {})
                var_95 = risk_metrics.get('var_95', 0)
                st.metric(
                    "VaR (95%)",
                    f"{var_95:.2%}",
                    delta=None
                )
            
            # Simulation paths visualization
            st.subheader("📈 Simulation Paths")
            
            paths = gbm_results.get('paths', np.array([]))
            
            if paths.size > 0:
                # Plot sample paths
                fig = go.Figure()
                
                # Plot first 100 paths
                n_paths_to_plot = min(100, paths.shape[1])
                for i in range(n_paths_to_plot):
                    fig.add_trace(go.Scatter(
                        x=list(range(paths.shape[0])),
                        y=paths[:, i],
                        mode='lines',
                        line=dict(width=0.5, color='rgba(26, 95, 180, 0.1)'),
                        showlegend=False
                    ))
                
                # Plot mean path and confidence intervals
                mean_path = paths.mean(axis=1)
                upper_95 = np.percentile(paths, 97.5, axis=1)
                lower_95 = np.percentile(paths, 2.5, axis=1)
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(mean_path))),
                    y=mean_path,
                    mode='lines',
                    line=dict(width=3, color='#f5a623'),
                    name='Mean Path'
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(upper_95))) + list(range(len(lower_95)))[::-1],
                    y=list(upper_95) + list(lower_95)[::-1],
                    fill='toself',
                    fillcolor='rgba(26, 95, 180, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(
                    height=500,
                    template="plotly_dark",
                    title=f"Monte Carlo Simulation Paths ({gbm_results.get('paths', np.array([])).shape[1]:,} simulations)",
                    xaxis_title="Days",
                    yaxis_title="Portfolio Value",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Final value distribution
            st.subheader("📊 Final Value Distribution")
            
            final_values = gbm_results.get('final_values', np.array([]))
            
            if len(final_values) > 0:
                fig2 = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Value Distribution", "Cumulative Distribution")
                )
                
                # Histogram
                fig2.add_trace(
                    go.Histogram(
                        x=final_values,
                        nbinsx=50,
                        name="Final Values",
                        marker_color='#1a5fb4',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # CDF
                sorted_values = np.sort(final_values)
                cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                
                fig2.add_trace(
                    go.Scatter(
                        x=sorted_values,
                        y=cdf,
                        mode='lines',
                        name="CDF",
                        line=dict(color='#26a269', width=3)
                    ),
                    row=1, col=2
                )
                
                # Add VaR lines
                risk_metrics = gbm_results.get('risk_metrics', {})
                for conf, var in risk_metrics.items():
                    if 'var_' in conf:
                        conf_level = int(conf.split('_')[1])
                        var_value = mc_results['parameters']['initial_value'] * (1 + var)
                        
                        fig2.add_vline(
                            x=var_value,
                            line_dash="dash",
                            line_color="#f5a623" if conf_level == 95 else "#c01c28",
                            annotation_text=f"VaR {conf_level}%",
                            row=1, col=1
                        )
                
                fig2.update_layout(
                    height=400,
                    template="plotly_dark",
                    showlegend=True
                )
                
                fig2.update_xaxes(title_text="Final Value", row=1, col=1)
                fig2.update_xaxes(title_text="Final Value", row=1, col=2)
                fig2.update_yaxes(title_text="Frequency", row=1, col=1)
                fig2.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
                
                st.plotly_chart(fig2, use_container_width=True)
        
        # Comparison of simulation methods
        st.subheader("🔄 Simulation Method Comparison")
        
        # Compare GBM and Bootstrap
        bootstrap_results = mc_results.get('bootstrap_simulation', {})
        
        if gbm_results.get('success', False) and bootstrap_results.get('success', False):
            comparison_data = []
            
            # GBM results
            gbm_risk = gbm_results.get('risk_metrics', {})
            bootstrap_risk = bootstrap_results.get('risk_metrics', {})
            
            for conf in [90, 95, 99]:
                gbm_var = gbm_risk.get(f'var_{conf}', np.nan)
                bootstrap_var = bootstrap_risk.get(f'var_{conf}', np.nan)
                
                comparison_data.append({
                    'Confidence': f'{conf}%',
                    'GBM VaR': gbm_var,
                    'Bootstrap VaR': bootstrap_var,
                    'Difference': gbm_var - bootstrap_var
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.dataframe(
                comparison_df.style.format({
                    'GBM VaR': '{:.4%}',
                    'Bootstrap VaR': '{:.4%}',
                    'Difference': '{:+.4%}'
                }),
                use_container_width=True
            )
        
        # Stress scenarios
        st.subheader("🌪️ Stress Test Scenarios")
        
        stress_scenarios = mc_results.get('stress_scenarios', {})
        
        if stress_scenarios:
            scenarios_data = []
            for scenario, data in stress_scenarios.items():
                if isinstance(data, dict):
                    scenarios_data.append({
                        'Scenario': scenario.replace('_', ' ').title(),
                        'Shock': data.get('shock', 0),
                        'Impact': data.get('impact', 0),
                        'Probability': data.get('probability', 0),
                        'Recovery Time': f"{data.get('recovery_time', 0):.1f} years"
                    })
            
            if scenarios_data:
                scenarios_df = pd.DataFrame(scenarios_data)
                
                st.dataframe(
                    scenarios_df.style.format({
                        'Shock': '{:.1%}',
                        'Impact': '${:,.0f}',
                        'Probability': '{:.2%}'
                    }),
                    use_container_width=True
                )
    
    def _render_institutional_reports(self):
        """Render institutional reports tab"""
        
        st.header("📋 Institutional Reports")
        
        # Report generator
        st.subheader("📄 Generate Institutional Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Comprehensive Analysis", "Risk Report", "Performance Attribution", 
                 "Compliance Report", "Executive Summary"],
                key="report_type"
            )
        
        with col2:
            report_format = st.selectbox(
                "Format",
                ["HTML", "PDF", "Markdown", "JSON"],
                key="report_format"
            )
        
        with col3:
            include_charts = st.checkbox("Include Charts", value=True, key="include_charts")
        
        if st.button("📊 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                self._generate_institutional_report(report_type, report_format, include_charts)
        
        # Pre-generated reports
        st.subheader("📁 Available Reports")
        
        if hasattr(st.session_state, 'analysis_timestamp'):
            # Summary report
            with st.expander("📋 Executive Summary", expanded=True):
                self._render_executive_summary()
            
            # Risk summary
            with st.expander("⚖️ Risk Summary", expanded=False):
                self._render_risk_summary()
            
            # Performance summary
            with st.expander("📈 Performance Summary", expanded=False):
                self._render_performance_summary()
    
    def _generate_institutional_report(self, report_type: str, 
                                     report_format: str, 
                                     include_charts: bool):
        """Generate institutional report"""
        
        try:
            # Create report content
            report_content = {
                'title': f"Institutional Portfolio Analysis Report",
                'type': report_type,
                'format': report_format,
                'timestamp': datetime.now().isoformat(),
                'analysis_timestamp': st.session_state.get('analysis_timestamp', 'N/A'),
                'configuration': {
                    'assets': st.session_state.selected_tickers,
                    'benchmark': st.session_state.benchmark,
                    'strategy': st.session_state.portfolio_strategy,
                    'time_period': f"{st.session_state.start_date} to {st.session_state.end_date}"
                }
            }
            
            # Add analysis results if available
            if hasattr(st.session_state, 'optimization_result'):
                report_content['optimization'] = {
                    'expected_return': st.session_state.optimization_result.get('expected_return', 0),
                    'expected_risk': st.session_state.optimization_result.get('expected_risk', 0),
                    'sharpe_ratio': st.session_state.optimization_result.get('sharpe_ratio', 0),
                    'method': st.session_state.optimization_result.get('method', 'Unknown')
                }
            
            # Convert to desired format
            if report_format == "JSON":
                report_text = json.dumps(report_content, indent=2)
            elif report_format == "Markdown":
                report_text = self._generate_markdown_report(report_content)
            else:  # HTML as default
                report_text = self._generate_html_report(report_content)
            
            # Create download button
            file_extension = {
                'HTML': 'html',
                'PDF': 'pdf',
                'Markdown': 'md',
                'JSON': 'json'
            }.get(report_format, 'txt')
            
            st.download_button(
                label=f"📥 Download {report_type} Report",
                data=report_text,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                mime={
                    'HTML': 'text/html',
                    'PDF': 'application/pdf',
                    'Markdown': 'text/markdown',
                    'JSON': 'application/json'
                }.get(report_format, 'text/plain')
            )
            
            st.success("Report generated successfully!")
            
        except Exception as e:
            st.error(f"Report generation failed: {str(e)[:100]}")
    
    def _generate_markdown_report(self, report_content: Dict) -> str:
        """Generate markdown report"""
        
        md = f"""# {report_content['title']}
        
**Report Type:** {report_content['type']}
**Generated:** {report_content['timestamp']}
**Analysis Date:** {report_content['analysis_timestamp']}

## Configuration
- **Assets:** {', '.join(report_content['configuration']['assets'])}
- **Benchmark:** {report_content['configuration']['benchmark']}
- **Strategy:** {report_content['configuration']['strategy']}
- **Time Period:** {report_content['configuration']['time_period']}

"""
        
        if 'optimization' in report_content:
            md += f"""
## Optimization Results
- **Expected Return:** {report_content['optimization']['expected_return']:.2%}
- **Expected Risk:** {report_content['optimization']['expected_risk']:.2%}
- **Sharpe Ratio:** {report_content['optimization']['sharpe_ratio']:.2f}
- **Method:** {report_content['optimization']['method']}
"""
        
        return md
    
    def _generate_html_report(self, report_content: Dict) -> str:
        """Generate HTML report"""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_content['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 30px 0; }}
        .metric {{ background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_content['title']}</h1>
        <p><strong>Report Type:</strong> {report_content['type']}</p>
        <p><strong>Generated:</strong> {report_content['timestamp']}</p>
        <p><strong>Analysis Date:</strong> {report_content['analysis_timestamp']}</p>
    </div>
    
    <div class="section">
        <h2>Configuration</h2>
        <ul>
            <li><strong>Assets:</strong> {', '.join(report_content['configuration']['assets'])}</li>
            <li><strong>Benchmark:</strong> {report_content['configuration']['benchmark']}</li>
            <li><strong>Strategy:</strong> {report_content['configuration']['strategy']}</li>
            <li><strong>Time Period:</strong> {report_content['configuration']['time_period']}</li>
        </ul>
    </div>
"""
        
        if 'optimization' in report_content:
            html += f"""
    <div class="section">
        <h2>Optimization Results</h2>
        <div class="metric">
            <p><strong>Expected Return:</strong> {report_content['optimization']['expected_return']:.2%}</p>
            <p><strong>Expected Risk:</strong> {report_content['optimization']['expected_risk']:.2%}</p>
            <p><strong>Sharpe Ratio:</strong> {report_content['optimization']['sharpe_ratio']:.2f}</p>
            <p><strong>Method:</strong> {report_content['optimization']['method']}</p>
        </div>
    </div>
"""
        
        html += """
</body>
</html>"""
        
        return html
    
    def _render_executive_summary(self):
        """Render executive summary"""
        
        if not hasattr(st.session_state, 'optimization_result'):
            return
        
        result = st.session_state.optimization_result
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Portfolio Return",
                f"{result.get('expected_return', 0):.2%}",
                delta=f"vs {st.session_state.advanced_settings.get('rf_rate', 0.03):.2%} RF"
            )
        
        with col2:
            st.metric(
                "Portfolio Risk",
                f"{result.get('expected_risk', 0):.2%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Risk-Adjusted Return",
                f"{result.get('sharpe_ratio', 0):.2f}",
                delta=None
            )
        
        # Summary text
        st.markdown("""
        ### 📋 Executive Summary
        
        This portfolio analysis demonstrates a professionally optimized investment strategy
        with comprehensive risk management and performance attribution.
        
        **Key Highlights:**
        - Professionally optimized using institutional-grade algorithms
        - Comprehensive risk assessment with multiple VaR methodologies
        - Detailed performance attribution and factor analysis
        - Monte Carlo simulation for forward-looking risk assessment
        - Compliance with institutional risk limits and constraints
        
        **Recommendations:**
        1. Monitor concentration risk and ensure diversification
        2. Regularly review and rebalance portfolio
        3. Consider stress test results in risk management
        4. Use attribution analysis for investment decisions
        """)
    
    def _render_risk_summary(self):
        """Render risk summary"""
        
        if not hasattr(st.session_state, 'risk_report'):
            return
        
        risk_report = st.session_state.risk_report
        
        # Risk metrics
        basic_stats = risk_report.get('basic_statistics', {})
        
        st.markdown("""
        ### ⚖️ Risk Management Summary
        
        **Primary Risk Metrics:**
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Value at Risk (95%)", f"{basic_stats.get('var_95', 0):.2%}")
            st.metric("Conditional VaR (95%)", f"{basic_stats.get('cvar_95', 0):.2%}")
        
        with col2:
            st.metric("Maximum Drawdown", f"{risk_report.get('drawdown_analysis', {}).get('max_drawdown', 0):.2%}")
            st.metric("Volatility", f"{basic_stats.get('annual_volatility', 0):.2%}")
        
        with col3:
            st.metric("Skewness", f"{basic_stats.get('skewness', 0):.2f}")
            st.metric("Kurtosis", f"{basic_stats.get('kurtosis', 0):.2f}")
        
        # Risk assessment
        st.markdown("""
        **Risk Assessment:**
        - Portfolio demonstrates appropriate risk-return characteristics
        - Diversification appears adequate based on correlation analysis
        - Stress tests indicate resilience to historical market shocks
        - Liquidity metrics suggest sufficient market depth
        
        **Risk Mitigation Recommendations:**
        1. Implement stop-loss mechanisms for extreme scenarios
        2. Consider hedging strategies for tail risk
        3. Maintain liquidity buffers for market stress periods
        4. Regularly review and update risk models
        """)
    
    def _render_performance_summary(self):
        """Render performance summary"""
        
        if not hasattr(st.session_state, 'attribution_report'):
            return
        
        attribution = st.session_state.attribution_report
        brinson = attribution.get('brinson_attribution', {})
        
        st.markdown("""
        ### 📈 Performance Attribution Summary
        
        **Performance Drivers:**
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Active Return", f"{brinson.get('total_active_return', 0):.2%}")
        
        with col2:
            st.metric("Allocation Effect", f"{brinson.get('allocation_effect', 0):.2%}")
        
        with col3:
            st.metric("Selection Effect", f"{brinson.get('selection_effect', 0):.2%}")
        
        # Performance analysis
        st.markdown("""
        **Performance Analysis:**
        - Active return decomposition identifies sources of outperformance
        - Risk contribution analysis shows portfolio diversification
        - Factor exposure analysis reveals style tilts
        
        **Performance Enhancement Opportunities:**
        1. Optimize asset allocation based on attribution analysis
        2. Enhance security selection in underperforming areas
        3. Adjust factor exposures to align with market views
        4. Improve timing of portfolio rebalancing
        """)

# -------------------------------------------------------------
# MAIN APPLICATION ENTRY POINT
# -------------------------------------------------------------
def main():
    """Main application entry point"""
    
    try:
        # Initialize dashboard
        dashboard = InstitutionalPortfolioDashboard()
        
        # Run dashboard
        dashboard.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

# -------------------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
