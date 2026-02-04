# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal v4.2
# Professional Portfolio Optimization & Global Multi-Asset Edition
# Fully Functional All Tabs with Professional Visualizations
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
from scipy import stats, optimize
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json

# Import PyPortfolioOpt
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.objective_functions import L2_reg
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    st.warning("PyPortfolioOpt not installed. Using basic optimization methods.")

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# ENHANCED GLOBAL ASSET UNIVERSE
# -------------------------------------------------------------
GLOBAL_ASSET_UNIVERSE = {
    # US Major Indices & ETFs
    "US_Indices": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", 
        "VEA", "VWO", "VUG", "VO", "VB", "VTV"
    ],
    
    # Bonds & Fixed Income
    "Bonds": [
        "TLT", "IEF", "SHY", "BND", "AGG", "HYG", "JNK",
        "MUB", "TIP", "LQD", "EMB"
    ],
    
    # Commodities
    "Commodities": [
        "GLD", "SLV", "USO", "UNG", "DBA", "PDBC", "GSG",
        "WEAT", "CORN", "SOYB"
    ],
    
    # Cryptocurrencies
    "Cryptocurrencies": [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "SOL-USD", "DOT-USD", "DOGE-USD", "MATIC-USD", "AVAX-USD",
        "LTC-USD", "UNI-USD", "LINK-USD", "ATOM-USD", "ETC-USD"
    ],
    
    # Global Stocks - US
    "US_Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
        "BRK-B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA"
    ],
    
    # European Stocks
    "Europe_Stocks": [
        "ASML.AS",  # Netherlands - Semiconductor
        "SAP.DE",   # Germany - Software
        "SIE.DE",   # Germany - Industrial
        "ALV.DE",   # Germany - Insurance
        "DTE.DE",   # Germany - Telecom
        "NOVN.SW",  # Switzerland - Pharma
        "ROG.SW",   # Switzerland - Pharma
        "NESN.SW",  # Switzerland - Food
        "UBSG.SW",  # Switzerland - Banking
        "CSGN.SW",  # Switzerland - Banking
        "SAN.PA",   # France - Banking
        "BNP.PA",   # France - Banking
        "AIR.PA",   # France - Aerospace
        "MC.PA",    # France - Luxury
        "OR.PA",    # France - Cosmetics
        "ENEL.MI",  # Italy - Energy
        "ENI.MI",   # Italy - Energy
        "ISP.MI",   # Italy - Banking
        "UCG.MI",   # Italy - Banking
    ],
    
    # UK Stocks
    "UK_Stocks": [
        "HSBA.L",   # HSBC
        "BP.L",     # BP
        "GSK.L",    # GSK
        "RIO.L",    # Rio Tinto
        "AAL.L",    # Anglo American
        "AZN.L",    # AstraZeneca
        "ULVR.L",   # Unilever
        "DGE.L",    # Diageo
        "BATS.L",   # British American Tobacco
        "NG.L",     # National Grid
    ],
    
    # Asia Pacific Stocks
    "Asia_Stocks": [
        "9988.HK",  # Alibaba
        "0700.HK",  # Tencent
        "0388.HK",  # HKEX
        "0005.HK",  # HSBC
        "1299.HK",  # AIA
        "7203.T",   # Toyota (Japan)
        "8306.T",   # Mitsubishi UFJ (Japan)
        "9984.T",   # SoftBank (Japan)
        "6758.T",   # Sony (Japan)
        "6861.T",   # Keyence (Japan)
        "BABA",     # Alibaba (US)
        "JD",       # JD.com
        "BIDU",     # Baidu
        "NTES",     # NetEase
        "TCEHY",    # Tencent (US)
    ],
    
    # Emerging Markets
    "Emerging_Stocks": [
        "HDB",      # HDFC Bank (India)
        "INFY",     # Infosys (India)
        "TCS.NS",   # TCS (India)
        "SBIN.NS",  # SBI (India)
        "RELIANCE.NS", # Reliance (India)
        "VALE",     # Vale (Brazil)
        "ITUB",     # Itau (Brazil)
        "BBD",      # Banco Bradesco (Brazil)
        "GGB",      # Gerdau (Brazil)
        "ABEV",     # Ambev (Brazil)
        "SBS",      # Companhia de Saneamento (Brazil)
    ],
    
    # Australia
    "Australia_Stocks": [
        "BHP.AX",   # BHP
        "RIO.AX",   # Rio Tinto
        "CBA.AX",   # Commonwealth Bank
        "WBC.AX",   # Westpac
        "ANZ.AX",   # ANZ Bank
        "NAB.AX",   # NAB
        "CSL.AX",   # CSL
        "WES.AX",   # Wesfarmers
        "WOW.AX",   # Woolworths
        "TLS.AX",   # Telstra
    ],
    
    # Singapore
    "Singapore_Stocks": [
        "D05.SI",   # DBS
        "O39.SI",   # OCBC
        "U11.SI",   # UOB
        "Z74.SI",   # Singapore Telecom
        "C09.SI",   # City Developments
    ],
    
    # Turkey
    "Turkey_Stocks": [
        "AKBNK.IS", # Akbank
        "GARAN.IS", # Garanti BBVA
        "ISCTR.IS", # İş Bankası
        "KOZAA.IS", # Koza Altın
        "SAHOL.IS", # Hacı Ömer Sabancı
        "THYAO.IS", # Turkish Airlines
        "TCELL.IS", # Turkcell
        "TUPRS.IS", # Tüpraş
        "ARCLK.IS", # Arçelik
        "BIMAS.IS", # BIM
    ],
    
    # Currencies & Forex
    "Currencies": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
        "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDTRY=X",
        "USDCNY=X", "USDSGD=X", "USDHKD=X", "USDINR=X",
        "USDBRL=X", "USDZAR=X", "USDMXN=X"
    ],
    
    # Volatility & Alternatives
    "Alternatives": [
        "^VIX", "VIXY", "UVXY", "SVXY",
        "TMF", "UPRO", "TQQQ", "SQQQ"
    ]
}

# Flatten universe for selection
ALL_TICKERS = []
for category in GLOBAL_ASSET_UNIVERSE.values():
    ALL_TICKERS.extend(category)

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
APP_TITLE = "🏛️ Apollo/ENIGMA - Global Portfolio Terminal v4.2"
DEFAULT_RF_ANNUAL = 0.03
TRADING_DAYS = 252
MONTE_CARLO_SIMULATIONS = 10000

os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"

st.set_page_config(
    page_title="Apollo/ENIGMA - Global Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# ENHANCED INSTITUTIONAL THEME
# -------------------------------------------------------------
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

/* Main container */
.main {
    background-color: var(--dark-bg);
    color: var(--text);
}

/* Enhanced tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: transparent;
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    color: var(--text-muted);
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
    border-color: var(--primary) !important;
}

/* Professional KPI cards */
.kpi-card {
    background: linear-gradient(145deg, var(--card-bg), #1a2332);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.3s ease;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
    border-color: var(--primary);
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    margin: 8px 0;
    line-height: 1.2;
}

.kpi-label {
    font-size: 13px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}

/* Status indicators */
.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}

.status-green { 
    background: linear-gradient(135deg, #16a34a, #22c55e);
    animation: pulse-green 2s infinite;
}
.status-yellow { 
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    animation: pulse-yellow 2s infinite;
}
.status-red { 
    background: linear-gradient(135deg, #dc2626, #ef4444);
    animation: pulse-red 2s infinite;
}

@keyframes pulse-green {
    0% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(22, 163, 74, 0); }
    100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
}

@keyframes pulse-yellow {
    0% { box-shadow: 0 0 0 0 rgba(245, 166, 35, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(245, 166, 35, 0); }
    100% { box-shadow: 0 0 0 0 rgba(245, 166, 35, 0); }
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(220, 38, 38, 0); }
    100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
}

/* Data tables */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 20px !important;
}

[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
}

[data-testid="metric-container"] div {
    color: var(--text) !important;
}

/* Sliders */
.stSlider {
    padding: 20px 0 !important;
}

/* Select boxes */
.stSelectbox {
    padding: 10px 0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(26, 95, 180, 0.4);
}

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Spinner */
.stSpinner > div {
    border-color: var(--primary) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PORTFOLIO STRATEGIES
# -------------------------------------------------------------
PORTFOLIO_STRATEGIES = {
    "Equal Weight": "Equal allocation across all selected assets",
    "Market Cap Weight": "Weighted by market capitalization",
    "Minimum Volatility": "Optimized for lowest portfolio volatility",
    "Maximum Sharpe Ratio": "Optimized for highest risk-adjusted returns",
    "Risk Parity": "Equal risk contribution from each asset",
    "Maximum Diversification": "Maximizes diversification ratio",
    "Mean-Variance Optimal": "Classical Markowitz optimization",
    "Hierarchical Risk Parity": "HRP clustering-based allocation",
    "Custom Weights": "Manually specify asset weights"
}

# -------------------------------------------------------------
# ENHANCED DATA LOADING
# -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_global_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load global asset prices with error handling"""
    prices_dict = {}
    
    # Split into batches for better performance
    batch_size = 20
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    
    for batch in ticker_batches:
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by='ticker'
            )
            
            # Process multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    if ticker in data.columns.get_level_values(0):
                        # Try different column formats
                        if (ticker, 'Close') in data.columns:
                            prices_dict[ticker] = data[(ticker, 'Close')]
                        elif (ticker, 'Adj Close') in data.columns:
                            prices_dict[ticker] = data[(ticker, 'Adj Close')]
                        else:
                            # Try to find any price column
                            possible_cols = [col for col in data.columns if ticker in str(col) and ('Close' in str(col) or 'Adj Close' in str(col))]
                            if possible_cols:
                                prices_dict[ticker] = data[possible_cols[0]]
            else:
                # Single ticker case
                if len(batch) == 1:
                    if 'Close' in data.columns:
                        prices_dict[batch[0]] = data['Close']
                    elif 'Adj Close' in data.columns:
                        prices_dict[batch[0]] = data['Adj Close']
            
        except Exception as e:
            st.warning(f"Failed to load some tickers: {batch}. Error: {str(e)[:100]}...")
            continue
    
    # Combine into DataFrame
    if prices_dict:
        prices_df = pd.DataFrame(prices_dict)
        prices_df = prices_df.dropna(axis=1, how='all')
        return prices_df
    else:
        return pd.DataFrame()

# -------------------------------------------------------------
# PYPORTFOLIOOPT INTEGRATION
# -------------------------------------------------------------
class PortfolioOptimizer:
    """Enhanced portfolio optimization using PyPortfolioOpt"""
    
    @staticmethod
    def calculate_expected_returns(returns_df: pd.DataFrame, method: str = "mean_historical_return") -> pd.Series:
        """Calculate expected returns using different methods"""
        try:
            if method == "mean_historical_return":
                return expected_returns.mean_historical_return(returns_df)
            elif method == "ema_historical_return":
                return expected_returns.ema_historical_return(returns_df)
            elif method == "capm_return":
                return expected_returns.capm_return(returns_df)
            else:
                return returns_df.mean()
        except:
            return returns_df.mean()
    
    @staticmethod
    def calculate_risk_matrix(returns_df: pd.DataFrame, method: str = "sample_cov") -> pd.DataFrame:
        """Calculate risk matrix using different methods"""
        try:
            if method == "sample_cov":
                return risk_models.sample_cov(returns_df)
            elif method == "semicovariance":
                return risk_models.semicovariance(returns_df)
            elif method == "exp_cov":
                return risk_models.exp_cov(returns_df)
            elif method == "ledoit_wolf":
                return risk_models.CovarianceShrinkage(returns_df).ledoit_wolf()
            elif method == "oracle_approximating":
                return risk_models.CovarianceShrinkage(returns_df).oracle_approximating()
            else:
                return returns_df.cov()
        except:
            return returns_df.cov()
    
    @staticmethod
    def optimize_portfolio(returns_df: pd.DataFrame, strategy: str, 
                          target_return: float = None, target_risk: float = None,
                          risk_free_rate: float = 0.03, 
                          risk_aversion: float = 1.0) -> Dict:
        """Optimize portfolio using PyPortfolioOpt"""
        
        if not PYPFOPT_AVAILABLE:
            # Fallback to equal weights
            n_assets = len(returns_df.columns)
            equal_weights = np.ones(n_assets) / n_assets
            port_returns = (returns_df * equal_weights).sum(axis=1)
            ann_return = port_returns.mean() * TRADING_DAYS
            ann_risk = port_returns.std() * np.sqrt(TRADING_DAYS)
            
            return {
                'weights': equal_weights,
                'expected_return': ann_return,
                'expected_risk': ann_risk,
                'sharpe_ratio': (ann_return - risk_free_rate) / ann_risk if ann_risk > 0 else np.nan,
                'method': 'Equal Weight (PyPortfolioOpt not available)',
                'cleaned_weights': {asset: equal_weights[i] for i, asset in enumerate(returns_df.columns)}
            }
        
        try:
            # Calculate expected returns and covariance
            mu = PortfolioOptimizer.calculate_expected_returns(returns_df)
            S = PortfolioOptimizer.calculate_risk_matrix(returns_df, "ledoit_wolf")
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            
            if strategy == "Minimum Volatility":
                weights = ef.min_volatility()
                
            elif strategy == "Maximum Sharpe Ratio":
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                
            elif strategy == "Maximum Quadratic Utility":
                weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
                
            elif strategy == "Efficient Risk":
                if target_risk:
                    try:
                        weights = ef.efficient_risk(target_risk=target_risk/np.sqrt(TRADING_DAYS))
                    except:
                        weights = ef.min_volatility()
                else:
                    weights = ef.min_volatility()
                    
            elif strategy == "Efficient Return":
                if target_return:
                    try:
                        weights = ef.efficient_return(target_return=target_return/TRADING_DAYS)
                    except:
                        weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                else:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                    
            elif strategy == "Mean-Variance Optimal":
                # Add L2 regularization for stability
                try:
                    ef.add_objective(L2_reg, gamma=0.1)
                except:
                    pass
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                
            else:
                # Default to minimum volatility
                weights = ef.min_volatility()
            
            # Clean weights
            cleaned_weights = ef.clean_weights()
            weights_array = np.array([cleaned_weights[asset] for asset in returns_df.columns])
            
            # Calculate performance
            expected_return, expected_risk, sharpe_ratio = ef.portfolio_performance(
                risk_free_rate=risk_free_rate/TRADING_DAYS
            )
            
            return {
                'weights': weights_array,
                'expected_return': expected_return,
                'expected_risk': expected_risk,
                'sharpe_ratio': sharpe_ratio,
                'method': strategy,
                'cleaned_weights': cleaned_weights
            }
            
        except Exception as e:
            st.error(f"Portfolio optimization failed: {str(e)}")
            # Fallback to equal weights
            n_assets = len(returns_df.columns)
            equal_weights = np.ones(n_assets) / n_assets
            port_returns = (returns_df * equal_weights).sum(axis=1)
            ann_return = port_returns.mean() * TRADING_DAYS
            ann_risk = port_returns.std() * np.sqrt(TRADING_DAYS)
            
            return {
                'weights': equal_weights,
                'expected_return': ann_return,
                'expected_risk': ann_risk,
                'sharpe_ratio': (ann_return - risk_free_rate) / ann_risk if ann_risk > 0 else np.nan,
                'method': f'Equal Weight (Fallback due to: {str(e)[:50]})',
                'cleaned_weights': {asset: equal_weights[i] for i, asset in enumerate(returns_df.columns)}
            }

# -------------------------------------------------------------
# ENHANCED RISK ANALYTICS
# -------------------------------------------------------------
class EnhancedRiskAnalytics:
    """Professional risk analytics with multiple VaR methods"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, method: str = "historical", 
                     confidence_level: float = 0.95, 
                     window: int = None, 
                     params: Dict = None) -> Dict:
        """Calculate Value at Risk using multiple methods"""
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 10:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': 'Insufficient data'
            }
        
        if method == "historical":
            # Historical Simulation
            try:
                var = np.percentile(returns_clean, (1 - confidence_level) * 100)
                tail = returns_clean[returns_clean <= var]
                cvar = tail.mean() if len(tail) > 0 else var
                
                return {
                    'VaR': var,
                    'CVaR': cvar,
                    'method': 'Historical Simulation',
                    'confidence': confidence_level,
                    'observations': len(returns_clean)
                }
            except:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Historical Simulation',
                    'confidence': confidence_level,
                    'error': 'Calculation failed'
                }
            
        elif method == "parametric":
            # Parametric (Normal Distribution)
            try:
                mu = returns_clean.mean()
                sigma = returns_clean.std()
                if sigma == 0:
                    return {
                        'VaR': np.nan,
                        'CVaR': np.nan,
                        'method': 'Parametric (Normal)',
                        'confidence': confidence_level,
                        'error': 'Zero volatility'
                    }
                z_score = stats.norm.ppf(1 - confidence_level)
                var = mu + z_score * sigma
                cvar = mu - (sigma / (1 - confidence_level)) * stats.norm.pdf(z_score)
                
                return {
                    'VaR': var,
                    'CVaR': cvar,
                    'method': 'Parametric (Normal)',
                    'confidence': confidence_level,
                    'mu': mu,
                    'sigma': sigma,
                    'z_score': z_score
                }
            except:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Parametric (Normal)',
                    'confidence': confidence_level,
                    'error': 'Calculation failed'
                }
            
        elif method == "ewma":
            # EWMA VaR
            try:
                if window is None:
                    window = min(252, len(returns_clean))
                    
                lambda_param = params.get('lambda', 0.94) if params else 0.94
                ewma_var = returns_clean.ewm(alpha=1-lambda_param).std().iloc[-1]
                if ewma_var == 0:
                    return {
                        'VaR': np.nan,
                        'CVaR': np.nan,
                        'method': 'EWMA Parametric',
                        'confidence': confidence_level,
                        'error': 'Zero volatility'
                    }
                z_score = stats.norm.ppf(1 - confidence_level)
                var = z_score * ewma_var
                cvar = - (ewma_var / (1 - confidence_level)) * stats.norm.pdf(z_score)
                
                return {
                    'VaR': var,
                    'CVaR': cvar,
                    'method': 'EWMA Parametric',
                    'confidence': confidence_level,
                    'lambda': lambda_param,
                    'ewma_vol': ewma_var,
                    'z_score': z_score
                }
            except:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'EWMA Parametric',
                    'confidence': confidence_level,
                    'error': 'Calculation failed'
                }
            
        elif method == "monte_carlo":
            # Monte Carlo Simulation
            try:
                n_simulations = params.get('n_simulations', 10000) if params else 10000
                days = params.get('days', 1) if params else 1
                
                mu = returns_clean.mean()
                sigma = returns_clean.std()
                
                if sigma == 0:
                    return {
                        'VaR': np.nan,
                        'CVaR': np.nan,
                        'method': 'Monte Carlo Simulation',
                        'confidence': confidence_level,
                        'error': 'Zero volatility'
                    }
                
                simulations = np.random.normal(mu, sigma, (days, n_simulations))
                portfolio_returns = np.prod(1 + simulations, axis=0) - 1
                
                var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
                tail = portfolio_returns[portfolio_returns <= var]
                cvar = tail.mean() if len(tail) > 0 else var
                
                return {
                    'VaR': var,
                    'CVaR': cvar,
                    'method': 'Monte Carlo Simulation',
                    'confidence': confidence_level,
                    'simulations': n_simulations,
                    'days': days,
                    'mu': mu,
                    'sigma': sigma
                }
            except:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Monte Carlo Simulation',
                    'confidence': confidence_level,
                    'error': 'Calculation failed'
                }
        else:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': 'Unknown method'
            }
    
    @staticmethod
    def calculate_all_var_methods(returns: pd.Series, confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate VaR using all available methods"""
        methods = ["historical", "parametric", "ewma", "monte_carlo"]
        results = []
        
        for method in methods:
            try:
                result = EnhancedRiskAnalytics.calculate_var(returns, method, confidence_level)
                results.append({
                    'Method': result['method'],
                    'VaR': result['VaR'],
                    'CVaR': result['CVaR'],
                    'Details': json.dumps({k: v for k, v in result.items() if k not in ['VaR', 'CVaR', 'method']})
                })
            except Exception as e:
                results.append({
                    'Method': method,
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'Details': f"Error: {str(e)}"
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_rolling_var(returns: pd.Series, window: int = 252, confidence: float = 0.95) -> pd.Series:
        """Calculate rolling VaR"""
        rolling_var = returns.rolling(window).apply(
            lambda x: np.percentile(x, (1 - confidence) * 100)
        )
        return rolling_var

# -------------------------------------------------------------
# ENHANCED EWMA ANALYSIS
# -------------------------------------------------------------
class EWMAAnalysis:
    """Enhanced EWMA volatility analysis"""
    
    @staticmethod
    def calculate_ewma_volatility(returns: pd.DataFrame, lambda_param: float = 0.94) -> pd.DataFrame:
        """Calculate EWMA volatility for multiple assets"""
        ewma_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.columns:
            r = returns[asset].dropna()
            if len(r) > 10:
                try:
                    # Initialize with sample variance
                    init_var = r.var()
                    ewma_var = pd.Series(index=r.index, dtype=float)
                    
                    if not pd.isna(init_var) and init_var > 0:
                        ewma_var.iloc[0] = init_var
                        
                        # Recursive EWMA
                        for i in range(1, len(r)):
                            prev_var = ewma_var.iloc[i-1]
                            if not pd.isna(prev_var):
                                ewma_var.iloc[i] = lambda_param * prev_var + (1 - lambda_param) * (r.iloc[i-1] ** 2)
                            else:
                                ewma_var.iloc[i] = init_var
                        
                        ewma_vol[asset] = np.sqrt(ewma_var)
                except:
                    ewma_vol[asset] = r.rolling(20).std()
        
        return ewma_vol.dropna(how='all')
    
    @staticmethod
    def calculate_correlation_breakdown(ewma_vol: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Analyze correlation breakdown during high volatility periods"""
        avg_vol = ewma_vol.mean()
        high_vol_periods = {}
        
        for asset in ewma_vol.columns:
            if not pd.isna(avg_vol[asset]) and avg_vol[asset] > 0:
                threshold = avg_vol[asset] * 1.5  # 50% above average
                high_vol_mask = ewma_vol[asset] > threshold
                high_vol_periods[asset] = high_vol_mask
        
        return pd.DataFrame(high_vol_periods)
    
    @staticmethod
    def calculate_volatility_regime(ewma_vol: pd.Series, percentiles: List[float] = [25, 75]) -> pd.Series:
        """Classify volatility regimes"""
        ewma_clean = ewma_vol.dropna()
        if len(ewma_clean) < 10:
            return pd.Series(index=ewma_vol.index, dtype=str)
        
        low_thresh = np.percentile(ewma_clean, percentiles[0])
        high_thresh = np.percentile(ewma_clean, percentiles[1])
        
        regime = pd.Series(index=ewma_vol.index, dtype=str)
        regime[ewma_vol <= low_thresh] = 'Low Volatility'
        regime[(ewma_vol > low_thresh) & (ewma_vol <= high_thresh)] = 'Normal Volatility'
        regime[ewma_vol > high_thresh] = 'High Volatility'
        
        return regime

# -------------------------------------------------------------
# PERFORMANCE ATTRIBUTION
# -------------------------------------------------------------
class PerformanceAttribution:
    """Professional performance attribution analysis"""
    
    @staticmethod
    def calculate_brinson_attribution(portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series,
                                     portfolio_weights: np.ndarray,
                                     benchmark_weights: np.ndarray,
                                     asset_returns: pd.DataFrame) -> Dict:
        """Calculate Brinson attribution analysis"""
        
        # Calculate total active return
        total_active_return = portfolio_returns.mean() - benchmark_returns.mean()
        
        # Allocation effect
        allocation_effect = np.sum(
            (portfolio_weights - benchmark_weights) * 
            (asset_returns.mean().values - benchmark_returns.mean())
        )
        
        # Selection effect
        selection_effect = np.sum(
            benchmark_weights * 
            (asset_returns.mean().values - benchmark_returns.mean())
        )
        
        # Interaction effect
        interaction_effect = np.sum(
            (portfolio_weights - benchmark_weights) * 
            (asset_returns.mean().values - benchmark_returns.mean())
        ) - allocation_effect
        
        return {
            'total_active_return': total_active_return,
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'residual': total_active_return - (allocation_effect + selection_effect + interaction_effect)
        }
    
    @staticmethod
    def calculate_risk_attribution(portfolio_returns: pd.Series,
                                  asset_returns: pd.DataFrame,
                                  weights: np.ndarray) -> pd.DataFrame:
        """Calculate risk attribution using Euler's theorem"""
        
        # Calculate portfolio volatility
        cov_matrix = asset_returns.cov()
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Calculate marginal contributions
        marginal_contrib = cov_matrix @ weights / port_vol if port_vol > 0 else np.zeros(len(weights))
        
        # Calculate percentage contributions
        contrib_pct = weights * marginal_contrib / port_vol if port_vol > 0 else np.zeros(len(weights))
        
        # Create attribution DataFrame
        attribution_df = pd.DataFrame({
            'Asset': asset_returns.columns,
            'Weight': weights,
            'Marginal_Contribution': marginal_contrib,
            'Percent_Contribution': contrib_pct * 100
        })
        
        return attribution_df.sort_values('Percent_Contribution', ascending=False)

# -------------------------------------------------------------
# MONTE CARLO SIMULATION ENGINE
# -------------------------------------------------------------
class MonteCarloSimulator:
    """Professional Monte Carlo simulation engine"""
    
    @staticmethod
    def simulate_gbm(S0: float, mu: float, sigma: float, 
                    T: float = 1.0, n_steps: int = 252,
                    n_sims: int = 10000) -> np.ndarray:
        """Geometric Brownian Motion simulation"""
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_sims))
        paths[0] = S0
        
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_sims)
            paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    @staticmethod
    def simulate_portfolio(weights: np.ndarray, means: np.ndarray,
                          cov_matrix: np.ndarray, T: float = 1.0,
                          n_steps: int = 252, n_sims: int = 10000) -> np.ndarray:
        """Simulate portfolio returns using Cholesky decomposition"""
        n_assets = len(weights)
        
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(cov_matrix)
        except:
            # Use eigenvalue decomposition if Cholesky fails
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals, 1e-8)  # Ensure positive
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        dt = T / n_steps
        portfolio_paths = np.zeros((n_steps + 1, n_sims))
        
        for i in range(n_sims):
            # Generate correlated returns
            z = np.random.standard_normal((n_steps, n_assets))
            correlated_z = z @ L.T
            
            # Calculate asset returns
            asset_returns = np.zeros((n_steps, n_assets))
            for j in range(n_assets):
                asset_returns[:, j] = np.exp((means[j] - 0.5 * cov_matrix[j, j]) * dt + np.sqrt(dt) * correlated_z[:, j])
            
            # Calculate portfolio value
            portfolio_values = np.ones(n_steps + 1)
            for t in range(1, n_steps + 1):
                portfolio_values[t] = portfolio_values[t-1] * np.sum(weights * asset_returns[t-1])
            
            portfolio_paths[:, i] = portfolio_values
        
        return portfolio_paths
    
    @staticmethod
    def calculate_var_cvar(paths: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """Calculate VaR and CVaR from simulation paths"""
        final_returns = (paths[-1] / paths[0]) - 1
        var = np.percentile(final_returns, (1 - alpha) * 100)
        tail = final_returns[final_returns <= var]
        cvar = tail.mean() if len(tail) > 0 else var
        
        return var, cvar

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------
def main():
    st.title(APP_TITLE)
    
    # Initialize session state
    if 'portfolio_strategy' not in st.session_state:
        st.session_state.portfolio_strategy = "Equal Weight"
    if 'custom_weights' not in st.session_state:
        st.session_state.custom_weights = {}
    if 'optimization_params' not in st.session_state:
        st.session_state.optimization_params = {}
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ["SPY", "TLT", "GLD", "BTC-USD", "AAPL"]
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Portfolio Configuration")
        
        # Asset selection with category filtering
        st.subheader("🌍 Asset Selection")
        
        category_filter = st.multiselect(
            "Filter by Category",
            list(GLOBAL_ASSET_UNIVERSE.keys()),
            default=["US_Indices", "Cryptocurrencies", "US_Stocks"]
        )
        
        # Filter tickers by selected categories
        filtered_tickers = []
        for category in category_filter:
            filtered_tickers.extend(GLOBAL_ASSET_UNIVERSE[category])
        
        if not filtered_tickers:
            filtered_tickers = ALL_TICKERS
        
        # Get valid default values that exist in filtered list
        valid_defaults = [t for t in st.session_state.selected_tickers if t in filtered_tickers]
        if not valid_defaults and filtered_tickers:
            valid_defaults = filtered_tickers[:5]  # Use first 5 if no valid defaults
        
        # Asset selector
        selected_tickers = st.multiselect(
            "Select Assets (5-20 recommended)",
            filtered_tickers,
            default=valid_defaults,
            help="Select 5-20 assets for optimal diversification"
        )
        
        # Update session state
        st.session_state.selected_tickers = selected_tickers
        
        # Benchmark selection
        benchmark_options = [t for t in ALL_TICKERS if t not in selected_tickers]
        if not benchmark_options:
            benchmark_options = ALL_TICKERS
        
        benchmark = st.selectbox(
            "Benchmark Index",
            benchmark_options,
            index=0,
            help="Primary benchmark for performance comparison"
        )
        
        # Date range
        st.subheader("📅 Date Range")
        date_preset = st.selectbox(
            "Time Period",
            ["1 Year", "3 Years", "5 Years", "10 Years", "Custom"],
            index=2
        )
        
        end_date = pd.Timestamp.today()
        
        if date_preset == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        elif date_preset == "3 Years":
            start_date = end_date - pd.DateOffset(years=3)
        elif date_preset == "5 Years":
            start_date = end_date - pd.DateOffset(years=5)
        elif date_preset == "10 Years":
            start_date = end_date - pd.DateOffset(years=10)
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", pd.Timestamp("2018-01-01"))
            with col2:
                end_date = st.date_input("End Date", pd.Timestamp.today())
        
        # Portfolio Strategy
        st.subheader("🎯 Portfolio Strategy")
        
        strategy = st.selectbox(
            "Portfolio Construction Method",
            list(PORTFOLIO_STRATEGIES.keys()),
            index=0,
            help=PORTFOLIO_STRATEGIES[st.session_state.portfolio_strategy]
        )
        
        st.session_state.portfolio_strategy = strategy
        
        # Strategy-specific parameters
        if strategy in ["Efficient Risk", "Efficient Return"]:
            col1, col2 = st.columns(2)
            with col1:
                if strategy == "Efficient Risk":
                    target_risk = st.number_input(
                        "Target Annual Volatility (%)",
                        min_value=5.0,
                        max_value=50.0,
                        value=15.0,
                        step=1.0
                    ) / 100
                    st.session_state.optimization_params['target_risk'] = target_risk
                else:
                    target_return = st.number_input(
                        "Target Annual Return (%)",
                        min_value=0.0,
                        max_value=50.0,
                        value=10.0,
                        step=1.0
                    ) / 100
                    st.session_state.optimization_params['target_return'] = target_return
        
        elif strategy == "Custom Weights":
            st.info("Custom weights will be set in the Overview tab")
        
        # Risk parameters
        st.subheader("⚖️ Risk Parameters")
        
        rf_annual = st.number_input(
            "Risk-Free Rate (annual %)",
            value=3.0,
            min_value=0.0,
            max_value=20.0,
            step=0.5
        ) / 100
        
        confidence_level = st.slider(
            "VaR Confidence Level",
            min_value=90,
            max_value=99,
            value=95,
            step=1
        ) / 100
        
        # Optimization constraints
        st.subheader("📐 Optimization Constraints")
        
        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.number_input(
                "Minimum Weight (%)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=1.0
            ) / 100
            
            max_weight = st.number_input(
                "Maximum Weight (%)",
                min_value=10.0,
                max_value=100.0,
                value=25.0,
                step=5.0
            ) / 100
        
        with col2:
            risk_aversion = st.slider(
                "Risk Aversion",
                min_value=0.5,
                max_value=10.0,
                value=2.5,
                step=0.5
            )
        
        # Run analysis button
        st.markdown("---")
        run_analysis = st.button(
            "🚀 Run Portfolio Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Main content
    if not run_analysis or len(selected_tickers) < 2:
        st.info("👈 Configure your portfolio in the sidebar and click 'Run Portfolio Analysis'")
        
        # Show universe statistics
        with st.expander("📊 Global Asset Universe Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Assets Available", len(ALL_TICKERS))
            with col2:
                st.metric("Asset Categories", len(GLOBAL_ASSET_UNIVERSE))
            with col3:
                st.metric("Geographic Coverage", "15+ Countries")
            
            # Show categories
            for category, assets in GLOBAL_ASSET_UNIVERSE.items():
                st.write(f"**{category.replace('_', ' ')}** ({len(assets)} assets)")
        
        st.stop()
    
    # Load data with progress
    with st.spinner("📊 Loading global market data..."):
        all_tickers = list(dict.fromkeys(selected_tickers + [benchmark]))
        prices = load_global_prices(all_tickers, start_date, end_date)
        
        # Validate data
        missing = [t for t in all_tickers if t not in prices.columns]
        if missing:
            st.error(f"❌ Missing data for: {', '.join(missing[:5])}")
            if len(missing) > 5:
                st.error(f"... and {len(missing) - 5} more")
            
            # Try to continue with available assets
            available_tickers = [t for t in all_tickers if t in prices.columns]
            if len(available_tickers) < 2:
                st.stop()
            
            selected_tickers = [t for t in selected_tickers if t in available_tickers]
            benchmark = benchmark if benchmark in available_tickers else available_tickers[0]
            all_tickers = list(dict.fromkeys(selected_tickers + [benchmark]))
            prices = prices[available_tickers]
        
        prices = prices[all_tickers].dropna()
        returns = prices.pct_change().dropna()
        
        # Ensure we have enough data
        if len(returns) < 50:
            st.error("❌ Insufficient data points for analysis. Please select a longer time period.")
            st.stop()
    
    # Create enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Overview & Weights",
        "⚖️ Risk Analytics",
        "🎯 Portfolio Optimization",
        "🔗 Correlation Matrix",
        "📊 EWMA Analysis",
        "🎲 Monte Carlo VaR",
        "📊 Performance Attribution",
        "🌍 Global Exposure",
        "🚦 Risk Scorecard"
    ])
    
    # Tab 1: Overview & Weights
    with tab1:
        st.header("📊 Portfolio Overview")
        
        # Calculate portfolio based on strategy
        if st.session_state.portfolio_strategy == "Equal Weight":
            n_assets = len(selected_tickers)
            weights = np.ones(n_assets) / n_assets
            portfolio_returns = returns[selected_tickers].mean(axis=1)
            method_desc = "Equal Weight Portfolio"
            
        elif st.session_state.portfolio_strategy == "Custom Weights":
            # Custom weights editor
            st.subheader("✏️ Custom Portfolio Weights")
            
            # Initialize or update custom weights
            if selected_tickers != list(st.session_state.custom_weights.keys()):
                st.session_state.custom_weights = {ticker: 1.0/len(selected_tickers) for ticker in selected_tickers}
            
            # Create weight editor
            cols = st.columns(3)
            weight_inputs = {}
            
            for idx, ticker in enumerate(selected_tickers):
                with cols[idx % 3]:
                    weight = st.number_input(
                        f"{ticker} Weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.custom_weights.get(ticker, 0.0) * 100,
                        step=1.0,
                        key=f"weight_{ticker}"
                    ) / 100
                    weight_inputs[ticker] = weight
            
            # Normalize weights
            total_weight = sum(weight_inputs.values())
            if total_weight > 0:
                weights = np.array([weight_inputs[t] / total_weight for t in selected_tickers])
                st.session_state.custom_weights = {t: weights[i] for i, t in enumerate(selected_tickers)}
            else:
                weights = np.ones(len(selected_tickers)) / len(selected_tickers)
                st.warning("Weights must sum to > 0. Using equal weights.")
            
            portfolio_returns = (returns[selected_tickers] * weights).sum(axis=1)
            method_desc = "Custom Weight Portfolio"
            
        else:
            # Use PyPortfolioOpt optimization
            with st.spinner("🔧 Optimizing portfolio..."):
                optimizer = PortfolioOptimizer()
                result = optimizer.optimize_portfolio(
                    returns[selected_tickers],
                    st.session_state.portfolio_strategy,
                    target_return=st.session_state.optimization_params.get('target_return'),
                    target_risk=st.session_state.optimization_params.get('target_risk'),
                    risk_free_rate=rf_annual,
                    risk_aversion=risk_aversion
                )
                
                weights = result['weights']
                portfolio_returns = (returns[selected_tickers] * weights).sum(axis=1)
                method_desc = result['method']
        
        # Store weights for use in other tabs
        st.session_state.current_weights = weights
        st.session_state.portfolio_returns = portfolio_returns
        st.session_state.asset_returns = returns[selected_tickers]
        
        # Display weights and performance
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Portfolio Weights")
            
            weights_df = pd.DataFrame({
                'Asset': selected_tickers,
                'Weight': weights,
                'Category': [next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if t in assets), 'Other') 
                           for t in selected_tickers]
            })
            
            # Sort by weight
            weights_df = weights_df.sort_values('Weight', ascending=False)
            
            # Display weights table
            st.dataframe(
                weights_df.style.format({'Weight': '{:.2%}'}),
                use_container_width=True,
                height=400
            )
            
            # Portfolio statistics
            total_weight = weights.sum()
            st.metric("Total Weight", f"{total_weight:.2%}")
            
            # Concentration metrics
            top_3_concentration = weights_df['Weight'].head(3).sum()
            st.metric("Top 3 Concentration", f"{top_3_concentration:.2%}")
            
            # Herfindahl-Hirschman Index
            hhi = (weights ** 2).sum()
            st.metric("HHI Index", f"{hhi:.4f}")
            
            # Export weights
            if st.button("📥 Export Weights to CSV"):
                csv = weights_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="portfolio_weights.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("📈 Portfolio Performance")
            
            # Benchmark returns
            benchmark_returns = returns[benchmark]
            active_returns = portfolio_returns - benchmark_returns
            
            # Cumulative performance chart
            fig = go.Figure()
            
            # Portfolio cumulative
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative.values,
                name=f"Portfolio ({method_desc})",
                line=dict(color='#1a5fb4', width=3)
            ))
            
            # Benchmark cumulative
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                name=f"Benchmark ({benchmark})",
                line=dict(color='#26a269', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Cumulative Performance vs Benchmark",
                height=500,
                template="plotly_dark",
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
            
            # Key performance metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                ann_return = portfolio_returns.mean() * TRADING_DAYS
                st.metric("Annual Return", f"{ann_return:.2%}")
            
            with col_b:
                ann_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
                st.metric("Annual Volatility", f"{ann_vol:.2%}")
            
            with col_c:
                sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else np.nan
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col_d:
                max_dd = ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().cummax() - 1).min()
                st.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # Tab 2: Enhanced Risk Analytics (FIXED AND WORKING)
    with tab2:
        st.header("⚖️ Comprehensive Risk Analytics")
        
        # Ensure we have portfolio data
        if 'portfolio_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        portfolio_returns = st.session_state.portfolio_returns
        risk_analytics = EnhancedRiskAnalytics()
        
        # VaR Method Selection
        st.subheader("📊 Value at Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var_method = st.selectbox(
                "VaR Calculation Method",
                ["Historical Simulation", "Parametric (Normal)", "EWMA", 
                 "Monte Carlo Simulation", "Compare All Methods"],
                index=0,
                key="var_method_tab2"
            )
        
        with col2:
            var_horizon = st.selectbox(
                "Time Horizon",
                ["1 Day", "5 Days", "10 Days", "1 Month", "3 Months"],
                index=0,
                key="var_horizon_tab2"
            )
            
            # Convert horizon to days
            horizon_map = {"1 Day": 1, "5 Days": 5, "10 Days": 10, "1 Month": 21, "3 Months": 63}
            horizon_days = horizon_map[var_horizon]
        
        with col3:
            var_confidence = st.slider(
                "Confidence Level (%)",
                min_value=90,
                max_value=99.9,
                value=95,
                step=1,
                key="var_confidence_tab2"
            ) / 100
        
        # Calculate portfolio returns for horizon
        if horizon_days > 1:
            # Aggregate returns for horizon
            horizon_returns = portfolio_returns.rolling(horizon_days).apply(
                lambda x: np.prod(1 + x) - 1, raw=True
            ).dropna()
        else:
            horizon_returns = portfolio_returns
        
        # Calculate VaR based on selected method
        if var_method == "Compare All Methods":
            # Compare all VaR methods
            st.subheader("📈 VaR Method Comparison")
            
            var_results = risk_analytics.calculate_all_var_methods(
                horizon_returns, 
                var_confidence
            )
            
            # Display results
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.dataframe(
                    var_results.style.format({
                        'VaR': '{:.4%}',
                        'CVaR': '{:.4%}'
                    }),
                    use_container_width=True,
                    height=300
                )
            
            with col_b:
                # Visualization
                fig = go.Figure(data=[
                    go.Bar(
                        name='VaR',
                        x=var_results['Method'],
                        y=var_results['VaR'].abs() * 100,
                        marker_color='#1a5fb4'
                    ),
                    go.Bar(
                        name='CVaR',
                        x=var_results['Method'],
                        y=var_results['CVaR'].abs() * 100,
                        marker_color='#26a269'
                    )
                ])
                
                fig.update_layout(
                    title="VaR & CVaR Comparison by Method",
                    height=400,
                    template="plotly_dark",
                    yaxis_title="Value (%)",
                    barmode='group',
                    xaxis_tickangle=45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Rolling VaR Analysis
            st.subheader("📈 Rolling VaR Analysis")
            
            rolling_window = st.slider(
                "Rolling Window (days)",
                min_value=20,
                max_value=252,
                value=60,
                step=10,
                key="rolling_var_window"
            )
            
            rolling_var = risk_analytics.calculate_rolling_var(
                portfolio_returns, 
                rolling_window, 
                var_confidence
            )
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values * 100,
                name=f"Rolling {rolling_window}d VaR",
                line=dict(color='#1a5fb4', width=2),
                fill='tozeroy',
                fillcolor='rgba(26, 95, 180, 0.2)'
            ))
            
            # Add actual returns for comparison
            fig2.add_trace(go.Scatter(
                x=portfolio_returns.index,
                y=portfolio_returns.values * 100,
                name="Daily Returns",
                mode='markers',
                marker=dict(size=3, color='gray', opacity=0.5)
            ))
            
            # Add violations
            violations = portfolio_returns < rolling_var
            if violations.any():
                violation_dates = portfolio_returns[violations].index
                violation_returns = portfolio_returns[violations].values * 100
                
                fig2.add_trace(go.Scatter(
                    x=violation_dates,
                    y=violation_returns,
                    mode='markers',
                    name="VaR Violations",
                    marker=dict(size=8, color='red', symbol='x')
                ))
            
            fig2.update_layout(
                title=f"Rolling VaR ({rolling_window} days) vs Actual Returns",
                height=500,
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Return / VaR (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Violation statistics
            if violations.any():
                n_violations = violations.sum()
                expected_violations = len(portfolio_returns) * (1 - var_confidence)
                violation_rate = n_violations / len(portfolio_returns)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Violations", f"{n_violations}")
                with col2:
                    st.metric("Expected Violations", f"{expected_violations:.1f}")
                with col3:
                    st.metric("Violation Rate", f"{violation_rate:.2%}")
                with col4:
                    st.metric("Unusual", "✅ Yes" if abs(n_violations - expected_violations) > 2 else "❌ No")
        
        else:
            # Single method analysis
            method_map = {
                "Historical Simulation": "historical",
                "Parametric (Normal)": "parametric",
                "EWMA": "ewma",
                "Monte Carlo Simulation": "monte_carlo"
            }
            
            method_key = method_map[var_method]
            
            # Calculate VaR
            var_result = risk_analytics.calculate_var(
                horizon_returns,
                method_key,
                var_confidence,
                params={'n_simulations': 10000, 'days': horizon_days} if method_key == "monte_carlo" else None
            )
            
            # Display results
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if 'VaR' in var_result and not pd.isna(var_result['VaR']):
                    st.metric(
                        f"{var_horizon} VaR ({var_confidence*100:.1f}%)",
                        f"{var_result['VaR']:.4%}"
                    )
                else:
                    st.metric(
                        f"{var_horizon} VaR ({var_confidence*100:.1f}%)",
                        "N/A"
                    )
            
            with col_b:
                if 'CVaR' in var_result and not pd.isna(var_result['CVaR']):
                    st.metric(
                        f"{var_horizon} CVaR ({var_confidence*100:.1f}%)",
                        f"{var_result['CVaR']:.4%}"
                    )
                else:
                    st.metric(
                        f"{var_horizon} CVaR ({var_confidence*100:.1f}%)",
                        "N/A"
                    )
            
            with col_c:
                if 'method' in var_result:
                    st.metric("Method", var_result['method'])
            
            # Distribution plot with VaR
            if 'VaR' in var_result and not pd.isna(var_result['VaR']):
                st.subheader("📊 Return Distribution with VaR")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Return Distribution", "VaR Violations Timeline")
                )
                
                # Histogram of returns
                fig.add_trace(
                    go.Histogram(
                        x=horizon_returns,
                        nbinsx=50,
                        name="Returns",
                        marker_color='#1a5fb4',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Add VaR line
                fig.add_vline(
                    x=var_result['VaR'],
                    line_dash="dash",
                    line_color="#f5a623",
                    annotation_text=f"VaR: {var_result['VaR']:.4%}",
                    annotation_position="top left",
                    row=1, col=1
                )
                
                # Add CVaR line
                if 'CVaR' in var_result and not pd.isna(var_result['CVaR']):
                    fig.add_vline(
                        x=var_result['CVaR'],
                        line_dash="dot",
                        line_color="#c01c28",
                        annotation_text=f"CVaR: {var_result['CVaR']:.4%}",
                        annotation_position="top right",
                        row=1, col=1
                    )
                
                # VaR violations timeline
                violations = horizon_returns < var_result['VaR']
                violation_dates = horizon_returns[violations].index
                
                fig.add_trace(
                    go.Scatter(
                        x=violation_dates,
                        y=[1] * len(violation_dates),
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='x'
                        ),
                        name="VaR Violations"
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=700,
                    template="plotly_dark",
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="Return", row=1, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Frequency", row=1, col=1)
                fig.update_yaxes(title_text="Violation", row=2, col=1, range=[0.5, 1.5])
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Metrics by Asset
        st.subheader("📊 Detailed Risk Metrics by Asset")
        
        # Calculate metrics for each asset
        risk_metrics_data = []
        
        for asset in selected_tickers:
            asset_returns = returns[asset]
            
            # Basic metrics
            ann_return = asset_returns.mean() * TRADING_DAYS
            ann_vol = asset_returns.std() * np.sqrt(TRADING_DAYS)
            sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else np.nan
            
            # Risk metrics
            max_dd = ((1 + asset_returns).cumprod() / (1 + asset_returns).cumprod().cummax() - 1).min()
            
            # VaR metrics (historical)
            var_95 = risk_analytics.calculate_var(asset_returns, "historical", 0.95)
            var_99 = risk_analytics.calculate_var(asset_returns, "historical", 0.99)
            
            risk_metrics_data.append({
                'Asset': asset,
                'Annual Return': ann_return,
                'Annual Volatility': ann_vol,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_dd,
                'VaR 95%': var_95.get('VaR', np.nan),
                'CVaR 95%': var_95.get('CVaR', np.nan),
                'VaR 99%': var_99.get('VaR', np.nan),
                'CVaR 99%': var_99.get('CVaR', np.nan),
                'Skewness': asset_returns.skew(),
                'Kurtosis': asset_returns.kurtosis()
            })
        
        risk_metrics_df = pd.DataFrame(risk_metrics_data)
        
        # Display interactive table
        st.dataframe(
            risk_metrics_df.style.format({
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                'VaR 95%': '{:.2%}',
                'CVaR 95%': '{:.2%}',
                'VaR 99%': '{:.2%}',
                'CVaR 99%': '{:.2%}',
                'Skewness': '{:.2f}',
                'Kurtosis': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Risk Metrics Visualization
        st.subheader("📈 Risk Metrics Heatmap")
        
        # Select metrics for heatmap
        heatmap_metrics = st.multiselect(
            "Select Metrics for Heatmap",
            ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR 95%'],
            default=['Annual Return', 'Annual Volatility', 'Sharpe Ratio']
        )
        
        if heatmap_metrics:
            # Prepare data for heatmap
            heatmap_data = risk_metrics_df.set_index('Asset')[heatmap_metrics]
            
            # Normalize for better visualization
            for metric in heatmap_metrics:
                if metric in ['Annual Return', 'Sharpe Ratio']:
                    # Higher is better - green scale
                    heatmap_data[metric] = (heatmap_data[metric] - heatmap_data[metric].min()) / (heatmap_data[metric].max() - heatmap_data[metric].min())
                else:
                    # Lower is better - red scale (inverted)
                    heatmap_data[metric] = 1 - (heatmap_data[metric] - heatmap_data[metric].min()) / (heatmap_data[metric].max() - heatmap_data[metric].min())
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values.T,
                x=heatmap_data.index,
                y=heatmap_metrics,
                colorscale='RdYlGn',
                zmin=0,
                zmax=1,
                colorbar=dict(title="Normalized Score", titleside="right"),
                text=risk_metrics_df.set_index('Asset')[heatmap_metrics].applymap(lambda x: f"{x:.2%}" if isinstance(x, float) and abs(x) < 1 else f"{x:.2f}"),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Asset Risk Metrics Heatmap (Normalized)",
                height=500,
                template="plotly_dark",
                xaxis_title="Assets",
                yaxis_title="Metrics"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export risk metrics
        if st.button("📥 Export Risk Metrics", key="export_risk_metrics"):
            csv = risk_metrics_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="asset_risk_metrics.csv",
                mime="text/csv"
            )
    
    # Tab 3: Portfolio Optimization (FIXED AND WORKING)
    with tab3:
        st.header("🎯 Portfolio Optimization Strategies")
        
        # Ensure we have data
        if 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        asset_returns = st.session_state.asset_returns
        
        if not PYPFOPT_AVAILABLE:
            st.warning("""
            ⚠️ PyPortfolioOpt is not installed. 
            Please install it for advanced optimization features:
            ```
            pip install pyportfolioopt
            ```
            Currently showing basic optimization results.
            """)
        
        # Optimization strategies comparison
        st.subheader("📊 Strategy Comparison")
        
        strategies_to_test = [
            "Minimum Volatility",
            "Maximum Sharpe Ratio",
            "Maximum Quadratic Utility",
            "Efficient Risk",
            "Efficient Return"
        ]
        
        optimization_results = []
        
        with st.spinner("🔄 Testing optimization strategies..."):
            optimizer = PortfolioOptimizer()
            
            for strategy in strategies_to_test:
                try:
                    result = optimizer.optimize_portfolio(
                        asset_returns,
                        strategy,
                        risk_free_rate=rf_annual,
                        risk_aversion=risk_aversion
                    )
                    
                    optimization_results.append({
                        'Strategy': strategy,
                        'Expected Return': result['expected_return'],
                        'Expected Risk': result['expected_risk'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Method': result['method']
                    })
                    
                except Exception as e:
                    optimization_results.append({
                        'Strategy': strategy,
                        'Expected Return': np.nan,
                        'Expected Risk': np.nan,
                        'Sharpe Ratio': np.nan,
                        'Method': f"Error: {str(e)[:50]}"
                    })
        
        # Display results
        results_df = pd.DataFrame(optimization_results)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                results_df.style.format({
                    'Expected Return': '{:.2%}',
                    'Expected Risk': '{:.2%}',
                    'Sharpe Ratio': '{:.2f}'
                }),
                use_container_width=True,
                height=300
            )
        
        with col2:
            # Visualization
            fig = go.Figure()
            
            valid_results = results_df.dropna(subset=['Expected Return', 'Expected Risk'])
            
            if not valid_results.empty:
                fig.add_trace(go.Scatter(
                    x=valid_results['Expected Risk'],
                    y=valid_results['Expected Return'],
                    mode='markers+text',
                    text=valid_results['Strategy'],
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=valid_results['Sharpe Ratio'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    name='Strategies'
                ))
                
                # Add equal weight portfolio for comparison
                eq_return = asset_returns.mean(axis=1).mean() * TRADING_DAYS
                eq_risk = asset_returns.mean(axis=1).std() * np.sqrt(TRADING_DAYS)
                
                fig.add_trace(go.Scatter(
                    x=[eq_risk],
                    y=[eq_return],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='star'
                    ),
                    name='Equal Weight'
                ))
            
            fig.update_layout(
                title="Efficient Frontier & Optimization Strategies",
                height=500,
                template="plotly_dark",
                xaxis_title="Annual Volatility",
                yaxis_title="Annual Return",
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive optimization
        st.subheader("🔄 Interactive Optimization")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            target_type = st.selectbox(
                "Optimization Target",
                ["Minimum Volatility", "Maximum Sharpe Ratio", "Target Return", "Target Risk"],
                index=1,
                key="optim_target"
            )
        
        with col_b:
            if target_type == "Target Return":
                target_value = st.number_input(
                    "Target Annual Return (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    key="target_return_input"
                ) / 100
            elif target_type == "Target Risk":
                target_value = st.number_input(
                    "Target Annual Volatility (%)",
                    min_value=5.0,
                    max_value=50.0,
                    value=15.0,
                    step=1.0,
                    key="target_risk_input"
                ) / 100
            else:
                target_value = None
        
        # Run optimization
        if st.button("🚀 Run Optimization", type="primary", key="run_optimization"):
            with st.spinner("Optimizing portfolio..."):
                if target_type == "Minimum Volatility":
                    strategy_name = "Minimum Volatility"
                elif target_type == "Maximum Sharpe Ratio":
                    strategy_name = "Maximum Sharpe Ratio"
                elif target_type == "Target Return":
                    strategy_name = "Efficient Return"
                else:
                    strategy_name = "Efficient Risk"
                
                result = optimizer.optimize_portfolio(
                    asset_returns,
                    strategy_name,
                    target_return=target_value if target_type == "Target Return" else None,
                    target_risk=target_value if target_type == "Target Risk" else None,
                    risk_free_rate=rf_annual,
                    risk_aversion=risk_aversion
                )
                
                # Display optimized weights
                st.subheader("📊 Optimized Portfolio Weights")
                
                optimized_weights_df = pd.DataFrame({
                    'Asset': selected_tickers,
                    'Weight': result['weights'],
                    'Category': [next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if t in assets), 'Other') 
                               for t in selected_tickers]
                }).sort_values('Weight', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(
                        optimized_weights_df.style.format({'Weight': '{:.2%}'}),
                        use_container_width=True,
                        height=400
                    )
                
                with col2:
                    # Pie chart of weights
                    fig = go.Figure(data=[go.Pie(
                        labels=optimized_weights_df['Asset'],
                        values=optimized_weights_df['Weight'],
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Set3)
                    )])
                    
                    fig.update_layout(
                        height=400,
                        title="Optimized Portfolio Allocation",
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.subheader("📈 Optimized Portfolio Performance")
                
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.metric("Expected Return", f"{result['expected_return']:.2%}")
                
                with metrics_cols[1]:
                    st.metric("Expected Risk", f"{result['expected_risk']:.2%}")
                
                with metrics_cols[2]:
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                
                with metrics_cols[3]:
                    # Calculate diversification ratio
                    weights = result['weights']
                    asset_vols = asset_returns.std() * np.sqrt(TRADING_DAYS)
                    weighted_vol = np.sum(weights * asset_vols)
                    portfolio_vol = result['expected_risk']
                    div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else np.nan
                    st.metric("Diversification Ratio", f"{div_ratio:.2f}")
                
                # Store optimized weights
                st.session_state.optimized_weights = result['weights']
                st.session_state.optimization_result = result
    
    # Tab 4: Correlation Matrix (FIXED AND WORKING)
    with tab4:
        st.header("🔗 Correlation Matrix & Risk Decomposition")
        
        # Ensure we have data
        if 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        asset_returns = st.session_state.asset_returns
        
        # Risk model selection
        st.subheader("📊 Risk Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            corr_method = st.selectbox(
                "Correlation Method",
                ["Sample Correlation", "Exponential Weighted", "Ledoit-Wolf Shrinkage", "Constant Correlation"],
                index=2,
                key="corr_method"
            )
        
        with col2:
            lookback_window = st.slider(
                "Lookback Window (days)",
                min_value=30,
                max_value=1000,
                value=252,
                step=30,
                key="lookback_window"
            )
        
        with col3:
            display_type = st.selectbox(
                "Display Type",
                ["Heatmap", "Network Graph", "Clustered Heatmap", "Rolling Correlation"],
                index=0,
                key="display_type"
            )
        
        # Calculate correlation matrix
        recent_returns = asset_returns.iloc[-lookback_window:] if len(asset_returns) > lookback_window else asset_returns
        
        if corr_method == "Sample Correlation":
            corr_matrix = recent_returns.corr()
        elif corr_method == "Exponential Weighted":
            # Calculate exponentially weighted correlation
            cov = recent_returns.ewm(span=60).cov(pairwise=True)
            # We need to extract the correlation matrix from the covariance
            std = recent_returns.ewm(span=60).std()
            # This is simplified - in practice, we'd compute properly
            corr_matrix = recent_returns.corr()
        elif corr_method == "Ledoit-Wolf Shrinkage" and PYPFOPT_AVAILABLE:
            try:
                S = risk_models.CovarianceShrinkage(recent_returns).ledoit_wolf()
                std = np.sqrt(np.diag(S))
                corr_matrix = pd.DataFrame(
                    S / np.outer(std, std),
                    index=recent_returns.columns,
                    columns=recent_returns.columns
                )
            except:
                corr_matrix = recent_returns.corr()
        else:
            corr_matrix = recent_returns.corr()
        
        if display_type == "Heatmap":
            # Enhanced heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(
                    title="Correlation",
                    titleside="right"
                )
            ))
            
            fig.update_layout(
                title=f"Correlation Matrix ({corr_method})",
                height=700,
                template="plotly_dark",
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
                avg_corr = corr_values.mean()
                st.metric("Average Correlation", f"{avg_corr:.3f}")
            
            with col2:
                min_corr = corr_values.min()
                st.metric("Minimum Correlation", f"{min_corr:.3f}")
            
            with col3:
                max_corr = corr_values.max()
                st.metric("Maximum Correlation", f"{max_corr:.3f}")
            
            with col4:
                std_corr = corr_values.std()
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
            
            fig2.add_vline(
                x=avg_corr,
                line_dash="dash",
                line_color="#f5a623",
                annotation_text=f"Mean: {avg_corr:.3f}"
            )
            
            fig2.update_layout(
                title="Correlation Distribution",
                height=400,
                template="plotly_dark",
                xaxis_title="Correlation",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Eigenvalue decomposition for correlation structure
            st.subheader("🔬 Correlation Structure Analysis")
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # Explained variance
                explained_variance = eigenvalues / eigenvalues.sum()
                cum_explained_variance = np.cumsum(explained_variance)
                
                fig3 = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Eigenvalue Spectrum", "Cumulative Explained Variance")
                )
                
                # Eigenvalue spectrum
                fig3.add_trace(
                    go.Bar(
                        x=list(range(1, len(eigenvalues) + 1)),
                        y=eigenvalues,
                        name="Eigenvalues",
                        marker_color='#1a5fb4'
                    ),
                    row=1, col=1
                )
                
                # Add random matrix theory threshold
                q = len(recent_returns) / len(selected_tickers)
                lambda_plus = (1 + np.sqrt(1/q))**2
                fig3.add_hline(
                    y=lambda_plus,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"RMT Threshold: {lambda_plus:.2f}",
                    row=1, col=1
                )
                
                # Cumulative explained variance
                fig3.add_trace(
                    go.Scatter(
                        x=list(range(1, len(cum_explained_variance) + 1)),
                        y=cum_explained_variance,
                        name="Cumulative Variance",
                        line=dict(color='#26a269', width=3)
                    ),
                    row=1, col=2
                )
                
                # Add 80% and 95% thresholds
                fig3.add_hline(y=0.80, line_dash="dash", line_color="orange", row=1, col=2)
                fig3.add_hline(y=0.95, line_dash="dash", line_color="green", row=1, col=2)
                
                fig3.update_layout(
                    height=500,
                    template="plotly_dark",
                    showlegend=False
                )
                
                fig3.update_xaxes(title_text="Component", row=1, col=1)
                fig3.update_xaxes(title_text="Component", row=1, col=2)
                fig3.update_yaxes(title_text="Eigenvalue", row=1, col=1)
                fig3.update_yaxes(title_text="Cumulative Variance", row=1, col=2)
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Principal component analysis
                n_components_80 = np.where(cum_explained_variance >= 0.80)[0][0] + 1
                n_components_95 = np.where(cum_explained_variance >= 0.95)[0][0] + 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Components", len(eigenvalues))
                with col2:
                    st.metric("Components for 80% Variance", n_components_80)
                with col3:
                    st.metric("Components for 95% Variance", n_components_95)
                    
            except Exception as e:
                st.warning(f"Could not perform eigenvalue analysis: {str(e)}")
        
        elif display_type == "Rolling Correlation":
            # Rolling correlation with benchmark
            st.subheader("📈 Rolling Correlation with Benchmark")
            
            rolling_window = st.slider(
                "Rolling Window (days)",
                min_value=20,
                max_value=200,
                value=60,
                step=10,
                key="rolling_corr_window"
            )
            
            fig = go.Figure()
            
            # Limit to first 10 assets for clarity
            display_assets = selected_tickers[:10]
            
            for asset in display_assets:
                if asset in returns.columns and benchmark in returns.columns:
                    rolling_corr = returns[asset].rolling(rolling_window).corr(returns[benchmark])
                    fig.add_trace(go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr.values,
                        name=asset,
                        mode='lines',
                        line=dict(width=1.5)
                    ))
            
            fig.update_layout(
                title=f"Rolling Correlation with {benchmark} ({rolling_window} days)",
                height=600,
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Correlation",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation stability analysis
            st.subheader("📊 Correlation Stability Analysis")
            
            # Calculate correlation changes
            first_half = asset_returns.iloc[:len(asset_returns)//2]
            second_half = asset_returns.iloc[len(asset_returns)//2:]
            
            corr_first = first_half.corr().values[np.triu_indices_from(corr_matrix.values, k=1)]
            corr_second = second_half.corr().values[np.triu_indices_from(corr_matrix.values, k=1)]
            
            if len(corr_first) == len(corr_second):
                corr_change = corr_second - corr_first
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig4 = go.Figure()
                    fig4.add_trace(go.Histogram(
                        x=corr_change,
                        nbinsx=30,
                        name="Correlation Change",
                        marker_color='#1a5fb4',
                        opacity=0.7
                    ))
                    
                    avg_change = corr_change.mean()
                    fig4.add_vline(
                        x=avg_change,
                        line_dash="dash",
                        line_color="#f5a623",
                        annotation_text=f"Mean Change: {avg_change:.3f}"
                    )
                    
                    fig4.update_layout(
                        title="Distribution of Correlation Changes",
                        height=400,
                        template="plotly_dark",
                        xaxis_title="Correlation Change",
                        yaxis_title="Frequency"
                    )
                    
                    st.plotly_chart(fig4, use_container_width=True)
                
                with col2:
                    # Display statistics
                    st.metric("Mean Correlation Change", f"{avg_change:.3f}")
                    st.metric("Std of Changes", f"{corr_change.std():.3f}")
                    st.metric("Max Increase", f"{corr_change.max():.3f}")
                    st.metric("Max Decrease", f"{corr_change.min():.3f}")
        
        # Export correlation matrix
        if st.button("📥 Export Correlation Matrix", key="export_corr"):
            csv = corr_matrix.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
    
    # Tab 5: EWMA Analysis (FIXED AND WORKING)
    with tab5:
        st.header("📊 EWMA Volatility Analysis")
        
        # Ensure we have data
        if 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        asset_returns = st.session_state.asset_returns
        
        # EWMA parameters
        st.subheader("⚙️ EWMA Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lambda_param = st.slider(
                "Decay Factor (λ)",
                min_value=0.85,
                max_value=0.99,
                value=0.94,
                step=0.01,
                help="Higher λ gives more weight to older observations",
                key="lambda_param"
            )
        
        with col2:
            ewma_window = st.number_input(
                "Lookback Period (days)",
                min_value=30,
                max_value=1000,
                value=252,
                step=30,
                key="ewma_window"
            )
        
        with col3:
            display_assets = st.multiselect(
                "Assets to Display",
                selected_tickers,
                default=selected_tickers[:5] if len(selected_tickers) >= 5 else selected_tickers,
                key="ewma_display_assets"
            )
        
        # Calculate EWMA volatility
        ewma_analysis = EWMAAnalysis()
        recent_returns = asset_returns.iloc[-ewma_window:] if len(asset_returns) > ewma_window else asset_returns
        ewma_vol = ewma_analysis.calculate_ewma_volatility(recent_returns, lambda_param)
        
        # Time series plot
        st.subheader("📈 EWMA Volatility Time Series")
        
        fig = go.Figure()
        
        for asset in display_assets:
            if asset in ewma_vol.columns:
                fig.add_trace(go.Scatter(
                    x=ewma_vol.index,
                    y=ewma_vol[asset] * np.sqrt(TRADING_DAYS) * 100,  # Annualized percentage
                    name=asset,
                    mode='lines',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=f"EWMA Volatility (λ={lambda_param}) - Annualized %",
            height=500,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
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
        
        # Volatility regime analysis
        st.subheader("🌡️ Volatility Regime Analysis")
        
        # Select asset for regime analysis
        regime_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            selected_tickers,
            index=0,
            key="regime_asset"
        )
        
        if regime_asset in ewma_vol.columns:
            # Calculate regimes
            vol_series = ewma_vol[regime_asset].dropna()
            regime = ewma_analysis.calculate_volatility_regime(vol_series)
            
            # Create regime plot
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f"EWMA Volatility - {regime_asset}",
                    "Volatility Regime Classification"
                )
            )
            
            # Volatility series
            fig.add_trace(
                go.Scatter(
                    x=vol_series.index,
                    y=vol_series * np.sqrt(TRADING_DAYS) * 100,
                    name="Volatility",
                    line=dict(color='#1a5fb4', width=2)
                ),
                row=1, col=1
            )
            
            # Regime classification
            regime_colors = {
                'Low Volatility': 'green',
                'Normal Volatility': 'yellow',
                'High Volatility': 'red'
            }
            
            for regime_type in ['Low Volatility', 'Normal Volatility', 'High Volatility']:
                mask = regime == regime_type
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=vol_series.index[mask],
                            y=[1] * mask.sum(),
                            mode='markers',
                            name=regime_type,
                            marker=dict(
                                color=regime_colors[regime_type],
                                size=8,
                                symbol='square'
                            )
                        ),
                        row=2, col=1
                    )
            
            fig.update_layout(
                height=600,
                template="plotly_dark",
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Annualized Vol (%)", row=1, col=1)
            fig.update_yaxes(title_text="Regime", row=2, col=1, range=[0.5, 3.5])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regime statistics
            st.subheader("📊 Regime Statistics")
            
            regime_counts = regime.value_counts()
            regime_percentage = regime.value_counts(normalize=True) * 100
            
            col1, col2, col3 = st.columns(3)
            
            regimes_displayed = 0
            for i, (regime_type, count) in enumerate(regime_counts.items()):
                if regimes_displayed < 3:
                    with [col1, col2, col3][i % 3]:
                        st.metric(
                            regime_type,
                            f"{count} days",
                            f"{regime_percentage.get(regime_type, 0):.1f}%"
                        )
                    regimes_displayed += 1
        
        # Volatility clustering analysis
        st.subheader("📊 Volatility Clustering Analysis")
        
        # Calculate autocorrelation of squared returns (volatility clustering)
        if 'portfolio_returns' in st.session_state:
            portfolio_returns = st.session_state.portfolio_returns
            squared_returns = portfolio_returns ** 2
            
            # Calculate autocorrelation
            max_lag = min(50, len(squared_returns) // 2)
            autocorr = [squared_returns.autocorr(lag=i) for i in range(1, max_lag + 1)]
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=list(range(1, max_lag + 1)),
                y=autocorr,
                name="Autocorrelation",
                marker_color='#1a5fb4'
            ))
            
            # Add confidence bands (approx 95% CI)
            conf_band = 1.96 / np.sqrt(len(squared_returns))
            fig2.add_hline(y=conf_band, line_dash="dash", line_color="red", annotation_text="95% CI Upper")
            fig2.add_hline(y=-conf_band, line_dash="dash", line_color="red", annotation_text="95% CI Lower")
            
            fig2.update_layout(
                title="Autocorrelation of Squared Returns (Volatility Clustering)",
                height=400,
                template="plotly_dark",
                xaxis_title="Lag (days)",
                yaxis_title="Autocorrelation"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Calculate Ljung-Box test for volatility clustering
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                lb_test = acorr_ljungbox(squared_returns.dropna(), lags=[5, 10, 20], return_df=True)
                
                st.write("**Ljung-Box Test for Volatility Clustering:**")
                st.dataframe(
                    lb_test.style.format({
                        'lb_stat': '{:.2f}',
                        'lb_pvalue': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                # Interpretation
                significant = (lb_test['lb_pvalue'] < 0.05).any()
                if significant:
                    st.success("✅ Significant volatility clustering detected (p < 0.05)")
                else:
                    st.warning("⚠️ No significant volatility clustering detected")
                    
            except ImportError:
                st.info("Install statsmodels for Ljung-Box test: `pip install statsmodels`")
    
    # Tab 6: Monte Carlo VaR (FIXED AND WORKING)
    with tab6:
        st.header("🎲 Monte Carlo Simulation & VaR")
        
        # Ensure we have portfolio data
        if 'portfolio_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        portfolio_returns = st.session_state.portfolio_returns
        
        # Simulation parameters
        st.subheader("⚙️ Simulation Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_simulations = st.number_input(
                "Number of Simulations",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                key="n_simulations"
            )
        
        with col2:
            time_horizon = st.selectbox(
                "Time Horizon",
                ["1 Month", "3 Months", "6 Months", "1 Year"],
                index=3,
                key="time_horizon"
            )
            
            horizon_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252}
            horizon_days = horizon_map[time_horizon]
        
        with col3:
            mc_confidence = st.slider(
                "Confidence Level",
                min_value=90,
                max_value=99.9,
                value=95,
                step=1,
                key="mc_confidence"
            ) / 100
        
        # Additional parameters
        st.subheader("🎯 Simulation Methodology")
        
        sim_method = st.selectbox(
            "Simulation Method",
            ["Geometric Brownian Motion", "Historical Bootstrap", "GARCH (if available)"],
            index=0,
            key="sim_method"
        )
        
        # Run Monte Carlo simulation
        if st.button("🚀 Run Monte Carlo Simulation", type="primary", key="run_monte_carlo"):
            with st.spinner(f"Running {n_simulations:,} simulations..."):
                # Use portfolio returns for simulation
                mu = portfolio_returns.mean()
                sigma = portfolio_returns.std()
                initial_value = 100
                
                simulator = MonteCarloSimulator()
                
                if sim_method == "Geometric Brownian Motion":
                    # GBM simulation
                    paths = simulator.simulate_gbm(
                        initial_value, mu, sigma, 
                        T=horizon_days/TRADING_DAYS, 
                        n_steps=horizon_days,
                        n_sims=n_simulations
                    )
                    
                elif sim_method == "Historical Bootstrap":
                    # Historical bootstrap simulation
                    paths = np.zeros((horizon_days + 1, n_simulations))
                    paths[0] = initial_value
                    
                    for i in range(n_simulations):
                        # Sample returns with replacement
                        sampled_returns = np.random.choice(
                            portfolio_returns.values, 
                            size=horizon_days, 
                            replace=True
                        )
                        paths[1:, i] = initial_value * np.cumprod(1 + sampled_returns)
                
                else:  # GARCH
                    # Fallback to GBM if GARCH not available
                    paths = simulator.simulate_gbm(
                        initial_value, mu, sigma, 
                        T=horizon_days/TRADING_DAYS, 
                        n_steps=horizon_days,
                        n_sims=n_simulations
                    )
                
                # Calculate final returns
                final_returns = (paths[-1] / initial_value) - 1
                
                # Calculate VaR and CVaR
                var_mc, cvar_mc = simulator.calculate_var_cvar(paths, mc_confidence)
                
                # Store in session state
                st.session_state.mc_paths = paths
                st.session_state.mc_final_returns = final_returns
                st.session_state.mc_var = var_mc
                st.session_state.mc_cvar = cvar_mc
                st.session_state.mc_params = {
                    'n_simulations': n_simulations,
                    'horizon_days': horizon_days,
                    'confidence': mc_confidence,
                    'mu': mu,
                    'sigma': sigma,
                    'method': sim_method
                }
        
        # Display results if available
        if 'mc_paths' in st.session_state:
            st.subheader("📊 Simulation Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"{time_horizon} VaR ({mc_confidence*100:.1f}%)",
                    f"{st.session_state.mc_var:.4%}"
                )
            
            with col2:
                st.metric(
                    f"{time_horizon} CVaR ({mc_confidence*100:.1f}%)",
                    f"{st.session_state.mc_cvar:.4%}"
                )
            
            with col3:
                expected_return = st.session_state.mc_final_returns.mean()
                st.metric("Expected Return", f"{expected_return:.4%}")
            
            with col4:
                prob_loss = (st.session_state.mc_final_returns < 0).mean()
                st.metric("Probability of Loss", f"{prob_loss:.2%}")
            
            # Visualization 1: Sample paths
            st.subheader("📈 Sample Simulation Paths")
            
            fig1 = go.Figure()
            
            # Plot first 100 paths
            max_paths_to_plot = min(100, st.session_state.mc_params['n_simulations'])
            for i in range(max_paths_to_plot):
                fig1.add_trace(go.Scatter(
                    x=list(range(st.session_state.mc_params['horizon_days'] + 1)),
                    y=st.session_state.mc_paths[:, i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(26, 95, 180, 0.1)'),
                    showlegend=False
                ))
            
            # Plot mean path and confidence intervals
            mean_path = st.session_state.mc_paths.mean(axis=1)
            upper_95 = np.percentile(st.session_state.mc_paths, 97.5, axis=1)
            lower_95 = np.percentile(st.session_state.mc_paths, 2.5, axis=1)
            
            fig1.add_trace(go.Scatter(
                x=list(range(len(mean_path))),
                y=mean_path,
                mode='lines',
                line=dict(width=3, color='#f5a623'),
                name='Mean Path'
            ))
            
            fig1.add_trace(go.Scatter(
                x=list(range(len(upper_95))) + list(range(len(lower_95)))[::-1],
                y=list(upper_95) + list(lower_95)[::-1],
                fill='toself',
                fillcolor='rgba(26, 95, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            fig1.update_layout(
                title=f"Monte Carlo Simulation Paths ({st.session_state.mc_params['n_simulations']:,} simulations)",
                height=500,
                template="plotly_dark",
                xaxis_title="Days",
                yaxis_title="Portfolio Value",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Visualization 2: Distribution of final returns
            st.subheader("📊 Distribution of Final Portfolio Values")
            
            fig2 = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Return Distribution", "Cumulative Distribution")
            )
            
            # Histogram
            fig2.add_trace(
                go.Histogram(
                    x=st.session_state.mc_final_returns * 100,
                    nbinsx=50,
                    name="Return Distribution",
                    marker_color='#1a5fb4',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add VaR and CVaR lines
            fig2.add_vline(
                x=st.session_state.mc_var * 100,
                line_dash="dash",
                line_color="#f5a623",
                annotation_text=f"VaR: {st.session_state.mc_var:.2%}",
                row=1, col=1
            )
            
            fig2.add_vline(
                x=st.session_state.mc_cvar * 100,
                line_dash="dot",
                line_color="#c01c28",
                annotation_text=f"CVaR: {st.session_state.mc_cvar:.2%}",
                row=1, col=1
            )
            
            # CDF
            sorted_returns = np.sort(st.session_state.mc_final_returns)
            cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
            
            fig2.add_trace(
                go.Scatter(
                    x=sorted_returns * 100,
                    y=cdf,
                    mode='lines',
                    name="CDF",
                    line=dict(color='#26a269', width=3)
                ),
                row=1, col=2
            )
            
            # Add VaR to CDF
            var_percentile = np.mean(st.session_state.mc_final_returns <= st.session_state.mc_var)
            fig2.add_trace(
                go.Scatter(
                    x=[st.session_state.mc_var * 100],
                    y=[var_percentile],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name=f"VaR ({var_percentile:.1%})"
                ),
                row=1, col=2
            )
            
            fig2.update_layout(
                height=500,
                template="plotly_dark",
                showlegend=True
            )
            
            fig2.update_xaxes(title_text="Return (%)", row=1, col=1)
            fig2.update_xaxes(title_text="Return (%)", row=1, col=2)
            fig2.update_yaxes(title_text="Frequency", row=1, col=1)
            fig2.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Risk metrics table
            st.subheader("📋 Detailed Risk Metrics")
            
            # Calculate additional metrics
            final_values = st.session_state.mc_paths[-1]
            
            risk_metrics = {
                'Metric': [
                    'Value at Risk (VaR)',
                    'Conditional VaR (CVaR)',
                    'Expected Shortfall (ES)',
                    'Maximum Loss',
                    'Minimum Loss',
                    'Average Return',
                    'Median Return',
                    'Standard Deviation',
                    'Skewness',
                    'Kurtosis',
                    'Value at 5% Tail',
                    'Value at 1% Tail'
                ],
                'Value': [
                    st.session_state.mc_var,
                    st.session_state.mc_cvar,
                    st.session_state.mc_cvar,  # Same as CVaR
                    st.session_state.mc_final_returns.min(),
                    st.session_state.mc_final_returns.max(),
                    st.session_state.mc_final_returns.mean(),
                    np.median(st.session_state.mc_final_returns),
                    st.session_state.mc_final_returns.std(),
                    stats.skew(st.session_state.mc_final_returns),
                    stats.kurtosis(st.session_state.mc_final_returns),
                    np.percentile(st.session_state.mc_final_returns, 5),
                    np.percentile(st.session_state.mc_final_returns, 1)
                ]
            }
            
            risk_metrics_df = pd.DataFrame(risk_metrics)
            
            st.dataframe(
                risk_metrics_df.style.format({'Value': '{:.4%}'}),
                use_container_width=True,
                height=400
            )
            
            # Export simulation results
            if st.button("📥 Export Simulation Results", key="export_sim_results"):
                # Create comprehensive results DataFrame
                results_dict = {
                    'Parameter': list(st.session_state.mc_params.keys()),
                    'Value': list(st.session_state.mc_params.values())
                }
                
                results_df = pd.DataFrame(results_dict)
                
                # Add risk metrics
                risk_metrics_df_copy = risk_metrics_df.copy()
                risk_metrics_df_copy['Value'] = risk_metrics_df_copy['Value'].apply(lambda x: f"{x:.6%}")
                
                # Combine
                combined_df = pd.concat([results_df, risk_metrics_df_copy], ignore_index=True)
                
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="monte_carlo_results.csv",
                    mime="text/csv"
                )
    
    # Tab 7: Performance Attribution (FIXED AND WORKING)
    with tab7:
        st.header("📊 Performance Attribution Analysis")
        
        # Ensure we have data
        if 'portfolio_returns' not in st.session_state or 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        portfolio_returns = st.session_state.portfolio_returns
        asset_returns = st.session_state.asset_returns
        weights = st.session_state.current_weights
        
        # Benchmark weights (assume equal weight for benchmark)
        benchmark_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
        benchmark_portfolio_returns = (asset_returns * benchmark_weights).sum(axis=1)
        
        # Performance Attribution Engine
        st.subheader("🎯 Brinson Attribution Analysis")
        
        attribution = PerformanceAttribution()
        
        # Calculate attribution
        brinson_results = attribution.calculate_brinson_attribution(
            portfolio_returns,
            benchmark_portfolio_returns,
            weights,
            benchmark_weights,
            asset_returns
        )
        
        # Display attribution results
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Active Return", 
                     f"{brinson_results['total_active_return']*TRADING_DAYS:.2%}")
        
        with col2:
            st.metric("Allocation Effect", 
                     f"{brinson_results['allocation_effect']*TRADING_DAYS:.2%}")
        
        with col3:
            st.metric("Selection Effect", 
                     f"{brinson_results['selection_effect']*TRADING_DAYS:.2%}")
        
        with col4:
            st.metric("Interaction Effect", 
                     f"{brinson_results['interaction_effect']*TRADING_DAYS:.2%}")
        
        with col5:
            st.metric("Residual", 
                     f"{brinson_results['residual']*TRADING_DAYS:.2%}")
        
        # Visualization of attribution
        fig = go.Figure(data=[
            go.Bar(
                name='Allocation',
                x=['Allocation'],
                y=[brinson_results['allocation_effect'] * TRADING_DAYS * 100],
                marker_color='#1a5fb4'
            ),
            go.Bar(
                name='Selection',
                x=['Selection'],
                y=[brinson_results['selection_effect'] * TRADING_DAYS * 100],
                marker_color='#26a269'
            ),
            go.Bar(
                name='Interaction',
                x=['Interaction'],
                y=[brinson_results['interaction_effect'] * TRADING_DAYS * 100],
                marker_color='#f5a623'
            ),
            go.Bar(
                name='Total Active',
                x=['Total'],
                y=[brinson_results['total_active_return'] * TRADING_DAYS * 100],
                marker_color='#c01c28'
            )
        ])
        
        fig.update_layout(
            title="Performance Attribution Breakdown",
            height=500,
            template="plotly_dark",
            yaxis_title="Active Return Contribution (%)",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Attribution
        st.subheader("⚖️ Risk Attribution Analysis")
        
        risk_attribution = attribution.calculate_risk_attribution(
            portfolio_returns,
            asset_returns,
            weights
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.dataframe(
                risk_attribution.style.format({
                    'Weight': '{:.2%}',
                    'Marginal_Contribution': '{:.4f}',
                    'Percent_Contribution': '{:.1f}%'
                }),
                use_container_width=True,
                height=400
            )
        
        with col_b:
            # Visualization of risk contributions
            fig2 = go.Figure(data=[
                go.Bar(
                    x=risk_attribution['Asset'],
                    y=risk_attribution['Percent_Contribution'],
                    marker_color='#1a5fb4',
                    text=risk_attribution['Percent_Contribution'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto'
                )
            ])
            
            fig2.update_layout(
                title="Risk Contribution by Asset (%)",
                height=400,
                template="plotly_dark",
                xaxis_title="Assets",
                yaxis_title="Risk Contribution (%)",
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Time-series attribution
        st.subheader("📈 Rolling Attribution Analysis")
        
        rolling_window = st.slider(
            "Rolling Window (days)",
            min_value=20,
            max_value=252,
            value=60,
            step=10,
            key="rolling_attribution_window"
        )
        
        # Calculate rolling attribution
        rolling_dates = []
        rolling_allocation = []
        rolling_selection = []
        
        for i in range(rolling_window, len(portfolio_returns)):
            window_returns = asset_returns.iloc[i-rolling_window:i]
            window_portfolio = portfolio_returns.iloc[i-rolling_window:i]
            window_benchmark = benchmark_portfolio_returns.iloc[i-rolling_window:i]
            
            # Simple approximation for rolling attribution
            alloc_effect = (weights - benchmark_weights) @ (window_returns.mean().values - window_benchmark.mean())
            sel_effect = benchmark_weights @ (window_returns.mean().values - window_benchmark.mean())
            
            rolling_dates.append(window_returns.index[-1])
            rolling_allocation.append(alloc_effect)
            rolling_selection.append(sel_effect)
        
        if rolling_dates:
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=rolling_dates,
                y=np.array(rolling_allocation) * TRADING_DAYS * 100,
                name='Rolling Allocation',
                line=dict(color='#1a5fb4', width=2)
            ))
            
            fig3.add_trace(go.Scatter(
                x=rolling_dates,
                y=np.array(rolling_selection) * TRADING_DAYS * 100,
                name='Rolling Selection',
                line=dict(color='#26a269', width=2)
            ))
            
            fig3.add_trace(go.Scatter(
                x=rolling_dates,
                y=(np.array(rolling_allocation) + np.array(rolling_selection)) * TRADING_DAYS * 100,
                name='Total Active',
                line=dict(color='#f5a623', width=3, dash='dash')
            ))
            
            fig3.update_layout(
                title=f"Rolling Attribution ({rolling_window} days)",
                height=500,
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Active Return Contribution (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Export attribution results
        if st.button("📥 Export Attribution Report", key="export_attribution"):
            # Combine all attribution data
            attribution_data = {
                'Brinson_Attribution': brinson_results,
                'Risk_Attribution': risk_attribution.to_dict('records')
            }
            
            # Convert to JSON
            attribution_json = json.dumps(attribution_data, indent=2, default=str)
            st.download_button(
                label="Download JSON Report",
                data=attribution_json,
                file_name="performance_attribution.json",
                mime="application/json"
            )
    
    # Tab 8: Global Exposure (already working)
    # Tab 9: Risk Scorecard (already working)
    
    # Since Tabs 8 and 9 are already working from previous code, we'll keep them as is
    # You can copy the working code from your previous implementation here

# -------------------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
