# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal v4.0
# Professional Portfolio Optimization & Global Multi-Asset Edition
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
from scipy import stats
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
APP_TITLE = "🏛️ Apollo/ENIGMA - Global Portfolio Terminal v4.0"
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
                        close_col = (ticker, 'Close') if ('Close', ticker) not in data.columns else (ticker, 'Close')
                        if close_col in data.columns:
                            prices_dict[ticker] = data[close_col]
                        else:
                            # Try to find any close price column
                            possible_cols = [col for col in data.columns if ticker in str(col) and 'Close' in str(col)]
                            if possible_cols:
                                prices_dict[ticker] = data[possible_cols[0]]
            else:
                # Single ticker case
                if len(batch) == 1 and 'Close' in data.columns:
                    prices_dict[batch[0]] = data['Close']
            
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
        if method == "mean_historical_return":
            return expected_returns.mean_historical_return(returns_df)
        elif method == "ema_historical_return":
            return expected_returns.ema_historical_return(returns_df)
        elif method == "capm_return":
            return expected_returns.capm_return(returns_df)
        else:
            return returns_df.mean()
    
    @staticmethod
    def calculate_risk_matrix(returns_df: pd.DataFrame, method: str = "sample_cov") -> pd.DataFrame:
        """Calculate risk matrix using different methods"""
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
            return {
                'weights': equal_weights,
                'expected_return': float(returns_df.mean().dot(equal_weights) * TRADING_DAYS),
                'expected_risk': float(np.sqrt(equal_weights.T @ returns_df.cov() @ equal_weights * TRADING_DAYS)),
                'sharpe_ratio': np.nan,
                'method': 'Equal Weight (PyPortfolioOpt not available)'
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
                    weights = ef.efficient_risk(target_risk=target_risk/np.sqrt(TRADING_DAYS))
                else:
                    weights = ef.min_volatility()
                    
            elif strategy == "Efficient Return":
                if target_return:
                    weights = ef.efficient_return(target_return=target_return/TRADING_DAYS)
                else:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                    
            elif strategy == "Mean-Variance Optimal":
                # Add L2 regularization for stability
                ef.add_objective(L2_reg, gamma=0.1)
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
            return {
                'weights': equal_weights,
                'expected_return': float(returns_df.mean().dot(equal_weights) * TRADING_DAYS),
                'expected_risk': float(np.sqrt(equal_weights.T @ returns_df.cov() @ equal_weights * TRADING_DAYS)),
                'sharpe_ratio': np.nan,
                'method': f'Equal Weight (Fallback due to: {str(e)[:50]})'
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
        
        if method == "historical":
            # Historical Simulation
            var = np.percentile(returns_clean, (1 - confidence_level) * 100)
            cvar = returns_clean[returns_clean <= var].mean()
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Historical Simulation',
                'confidence': confidence_level,
                'observations': len(returns_clean)
            }
            
        elif method == "parametric":
            # Parametric (Normal Distribution)
            mu = returns_clean.mean()
            sigma = returns_clean.std()
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
            
        elif method == "ewma":
            # EWMA VaR
            if window is None:
                window = 252
                
            lambda_param = params.get('lambda', 0.94) if params else 0.94
            ewma_var = returns_clean.ewm(alpha=1-lambda_param).std().iloc[-1]
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
            
        elif method == "monte_carlo":
            # Monte Carlo Simulation
            n_simulations = params.get('n_simulations', 10000) if params else 10000
            days = params.get('days', 1) if params else 1
            
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            
            simulations = np.random.normal(mu, sigma, (days, n_simulations))
            portfolio_returns = np.prod(1 + simulations, axis=0) - 1
            
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            
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
            
        elif method == "garch":
            # GARCH VaR (simplified)
            try:
                from arch import arch_model
                
                # Fit GARCH(1,1) model
                am = arch_model(returns_clean * 100, vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                
                # Forecast
                forecasts = res.forecast(horizon=1)
                conditional_vol = np.sqrt(forecasts.variance.iloc[-1].values[0]) / 100
                
                z_score = stats.norm.ppf(1 - confidence_level)
                var = z_score * conditional_vol
                cvar = - (conditional_vol / (1 - confidence_level)) * stats.norm.pdf(z_score)
                
                return {
                    'VaR': var,
                    'CVaR': cvar,
                    'method': 'GARCH(1,1)',
                    'confidence': confidence_level,
                    'conditional_vol': conditional_vol,
                    'z_score': z_score,
                    'garch_params': res.params.to_dict()
                }
                
            except ImportError:
                # Fallback to parametric
                return EnhancedRiskAnalytics.calculate_var(returns_clean, "parametric", confidence_level)
    
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
    def backtest_var(returns: pd.Series, var_series: pd.Series, confidence_level: float = 0.95) -> Dict:
        """Backtest VaR estimates"""
        violations = returns < var_series
        n_violations = violations.sum()
        expected_violations = len(returns) * (1 - confidence_level)
        
        # Kupiec test
        if n_violations > 0:
            LR = -2 * np.log(((1 - confidence_level) ** (len(returns) - n_violations) * 
                            confidence_level ** n_violations) / 
                           ((1 - n_violations/len(returns)) ** (len(returns) - n_violations) * 
                            (n_violations/len(returns)) ** n_violations))
            p_value = 1 - stats.chi2.cdf(LR, 1)
        else:
            LR = 0
            p_value = 1.0
        
        return {
            'total_observations': len(returns),
            'violations': int(n_violations),
            'violation_rate': n_violations / len(returns),
            'expected_violations': expected_violations,
            'unexpected_violations': n_violations - expected_violations,
            'kupiec_LR': LR,
            'kupiec_p_value': p_value,
            'test_passed': p_value > 0.05  # 95% confidence
        }

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
            if len(r) > 0:
                # Initialize with sample variance
                init_var = r.var()
                ewma_var = pd.Series(index=r.index, dtype=float)
                ewma_var.iloc[0] = init_var
                
                # Recursive EWMA
                for i in range(1, len(r)):
                    ewma_var.iloc[i] = lambda_param * ewma_var.iloc[i-1] + (1 - lambda_param) * (r.iloc[i-1] ** 2)
                
                ewma_vol[asset] = np.sqrt(ewma_var)
        
        return ewma_vol.dropna(how='all')
    
    @staticmethod
    def calculate_correlation_breakdown(ewma_vol: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Analyze correlation breakdown during high volatility periods"""
        avg_vol = ewma_vol.mean()
        high_vol_periods = {}
        
        for asset in ewma_vol.columns:
            threshold = avg_vol[asset] * 1.5  # 50% above average
            high_vol_mask = ewma_vol[asset] > threshold
            high_vol_periods[asset] = high_vol_mask
        
        return pd.DataFrame(high_vol_periods)
    
    @staticmethod
    def calculate_volatility_regime(ewma_vol: pd.Series, percentiles: List[float] = [25, 75]) -> pd.Series:
        """Classify volatility regimes"""
        low_thresh = np.percentile(ewma_vol.dropna(), percentiles[0])
        high_thresh = np.percentile(ewma_vol.dropna(), percentiles[1])
        
        regime = pd.Series(index=ewma_vol.index, dtype=str)
        regime[ewma_vol <= low_thresh] = 'Low Volatility'
        regime[(ewma_vol > low_thresh) & (ewma_vol <= high_thresh)] = 'Normal Volatility'
        regime[ewma_vol > high_thresh] = 'High Volatility'
        
        return regime

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
        
        # Asset selector
        selected_tickers = st.multiselect(
            "Select Assets (5-20 recommended)",
            filtered_tickers,
            default=["SPY", "TLT", "GLD", "BTC-USD", "AAPL"],
            help="Select 5-20 assets for optimal diversification"
        )
        
        # Benchmark selection
        benchmark = st.selectbox(
            "Benchmark Index",
            [t for t in ALL_TICKERS if t not in selected_tickers],
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
            st.stop()
        
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
        "📉 Performance Attribution",
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
    
    # Tab 2: Enhanced Risk Analytics
    with tab2:
        st.header("⚖️ Comprehensive Risk Analytics")
        
        # VaR Method Selection
        st.subheader("📊 Value at Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var_method = st.selectbox(
                "VaR Calculation Method",
                ["Historical Simulation", "Parametric (Normal)", "EWMA", 
                 "Monte Carlo Simulation", "GARCH (if available)", "Compare All Methods"],
                index=0
            )
        
        with col2:
            var_horizon = st.selectbox(
                "Time Horizon",
                ["1 Day", "5 Days", "10 Days", "1 Month", "3 Months"],
                index=0
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
                step=1
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
        risk_analytics = EnhancedRiskAnalytics()
        
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
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Backtesting
            st.subheader("🔍 VaR Backtesting")
            
            # Use historical VaR for backtesting
            historical_var = risk_analytics.calculate_var(
                horizon_returns, 
                "historical", 
                var_confidence
            )
            
            # Create VaR series for backtesting
            var_series = pd.Series(
                index=horizon_returns.index, 
                data=historical_var['VaR']
            )
            
            backtest_results = risk_analytics.backtest_var(
                horizon_returns, 
                var_series, 
                var_confidence
            )
            
            # Display backtest results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Violations", 
                    f"{backtest_results['violations']}",
                    f"{backtest_results['unexpected_violations']:+.0f}"
                )
            
            with col2:
                st.metric(
                    "Violation Rate", 
                    f"{backtest_results['violation_rate']:.2%}",
                    f"Expected: {(1-var_confidence):.2%}"
                )
            
            with col3:
                st.metric(
                    "Kupiec Test p-value", 
                    f"{backtest_results['kupiec_p_value']:.4f}"
                )
            
            with col4:
                status = "✅ Passed" if backtest_results['test_passed'] else "❌ Failed"
                st.metric("Backtest Result", status)
        
        else:
            # Single method analysis
            method_map = {
                "Historical Simulation": "historical",
                "Parametric (Normal)": "parametric",
                "EWMA": "ewma",
                "Monte Carlo Simulation": "monte_carlo",
                "GARCH (if available)": "garch"
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
                st.metric(
                    f"{var_horizon} VaR ({var_confidence*100:.1f}%)",
                    f"{var_result['VaR']:.4%}"
                )
            
            with col_b:
                st.metric(
                    f"{var_horizon} CVaR ({var_confidence*100:.1f}%)",
                    f"{var_result['CVaR']:.4%}"
                )
            
            with col_c:
                if 'method' in var_result:
                    st.metric("Method", var_result['method'])
            
            # Distribution plot with VaR
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
                'VaR 95%': var_95['VaR'],
                'CVaR 95%': var_95['CVaR'],
                'VaR 99%': var_99['VaR'],
                'CVaR 99%': var_99['CVaR'],
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
        
        # Export risk metrics
        if st.button("📥 Export Risk Metrics"):
            csv = risk_metrics_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="asset_risk_metrics.csv",
                mime="text/csv"
            )
    
    # Tab 3: Portfolio Optimization
    with tab3:
        st.header("🎯 Portfolio Optimization Strategies")
        
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
                        returns[selected_tickers],
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
            
            fig.add_trace(go.Scatter(
                x=results_df['Expected Risk'],
                y=results_df['Expected Return'],
                mode='markers+text',
                text=results_df['Strategy'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=results_df['Sharpe Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Strategies'
            ))
            
            # Add equal weight portfolio for comparison
            eq_return = returns[selected_tickers].mean(axis=1).mean() * TRADING_DAYS
            eq_risk = returns[selected_tickers].mean(axis=1).std() * np.sqrt(TRADING_DAYS)
            
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
                index=1
            )
        
        with col_b:
            if target_type == "Target Return":
                target_value = st.number_input(
                    "Target Annual Return (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0
                ) / 100
            elif target_type == "Target Risk":
                target_value = st.number_input(
                    "Target Annual Volatility (%)",
                    min_value=5.0,
                    max_value=50.0,
                    value=15.0,
                    step=1.0
                ) / 100
            else:
                target_value = None
        
        # Run optimization
        if st.button("🚀 Run Optimization", type="primary"):
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
                    returns[selected_tickers],
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
                    asset_vols = returns[selected_tickers].std() * np.sqrt(TRADING_DAYS)
                    weighted_vol = np.sum(weights * asset_vols)
                    portfolio_vol = result['expected_risk']
                    div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else np.nan
                    st.metric("Diversification Ratio", f"{div_ratio:.2f}")
    
    # Tab 4: Correlation Matrix (Enhanced)
    with tab4:
        st.header("🔗 Correlation Matrix & Risk Decomposition")
        
        # Risk model selection
        st.subheader("📊 Risk Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            corr_method = st.selectbox(
                "Correlation Method",
                ["Sample Correlation", "Exponential Weighted", "Ledoit-Wolf Shrinkage", "Constant Correlation"],
                index=2
            )
        
        with col2:
            lookback_window = st.slider(
                "Lookback Window (days)",
                min_value=30,
                max_value=1000,
                value=252,
                step=30
            )
        
        with col3:
            display_type = st.selectbox(
                "Display Type",
                ["Heatmap", "Network Graph", "Clustered Heatmap", "Rolling Correlation"],
                index=0
            )
        
        # Calculate correlation matrix
        recent_returns = returns[selected_tickers].iloc[-lookback_window:]
        
        if corr_method == "Sample Correlation":
            corr_matrix = recent_returns.corr()
        elif corr_method == "Exponential Weighted":
            corr_matrix = recent_returns.ewm(span=60).corr().iloc[-len(selected_tickers):, -len(selected_tickers):]
        elif corr_method == "Ledoit-Wolf Shrinkage" and PYPFOPT_AVAILABLE:
            try:
                S = risk_models.CovarianceShrinkage(recent_returns).ledoit_wolf()
                std = np.sqrt(np.diag(S))
                corr_matrix = pd.DataFrame(
                    S / np.outer(std, std),
                    index=selected_tickers,
                    columns=selected_tickers
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
                yaxis_title="Assets"
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
        
        elif display_type == "Network Graph":
            # Network visualization
            import networkx as nx
            
            # Create graph from correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for asset in selected_tickers:
                G.add_node(asset)
            
            # Add edges with correlation as weight
            for i in range(len(selected_tickers)):
                for j in range(i+1, len(selected_tickers)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.3:  # Only show significant correlations
                        G.add_edge(
                            selected_tickers[i], 
                            selected_tickers[j],
                            weight=abs(corr),
                            color='green' if corr > 0 else 'red'
                        )
            
            # Create network plot
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            edge_trace = []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=edge[2]['weight']*5, color=edge[2]['color']),
                    hoverinfo='none',
                    mode='lines'
                )
                edge_trace.append(trace)
            
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=20,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                )
            )
            
            fig = go.Figure(data=edge_trace + [node_trace])
            
            fig.update_layout(
                title="Correlation Network Graph",
                height=700,
                template="plotly_dark",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif display_type == "Rolling Correlation":
            # Rolling correlation with benchmark
            st.subheader("📈 Rolling Correlation with Benchmark")
            
            rolling_window = st.slider(
                "Rolling Window (days)",
                min_value=20,
                max_value=200,
                value=60,
                step=10
            )
            
            fig = go.Figure()
            
            for asset in selected_tickers[:10]:  # Limit to first 10 for clarity
                rolling_corr = returns[asset].rolling(rolling_window).corr(returns[benchmark])
                fig.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    name=asset,
                    mode='lines'
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
    
    # Tab 5: Enhanced EWMA Analysis
    with tab5:
        st.header("📊 EWMA Volatility Analysis")
        
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
                help="Higher λ gives more weight to older observations"
            )
        
        with col2:
            ewma_window = st.number_input(
                "Lookback Period (days)",
                min_value=30,
                max_value=1000,
                value=252,
                step=30
            )
        
        with col3:
            display_assets = st.multiselect(
                "Assets to Display",
                selected_tickers,
                default=selected_tickers[:5]
            )
        
        # Calculate EWMA volatility
        ewma_analysis = EWMAAnalysis()
        recent_returns = returns[selected_tickers].iloc[-ewma_window:]
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
            index=0
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
            
            for i, (regime_type, count) in enumerate(regime_counts.items()):
                with [col1, col2, col3][i % 3]:
                    st.metric(
                        regime_type,
                        f"{count} days",
                        f"{regime_percentage.get(regime_type, 0):.1f}%"
                    )
        
        # EWMA correlation breakdown
        st.subheader("🔗 EWMA Correlation During High Volatility")
        
        # Identify high volatility periods
        vol_breakdown = ewma_analysis.calculate_correlation_breakdown(ewma_vol)
        
        if not vol_breakdown.empty:
            # Calculate correlation during high vs normal volatility
            high_vol_correlations = []
            normal_vol_correlations = []
            
            for asset in selected_tickers[:5]:  # Limit to 5 for clarity
                if asset in vol_breakdown.columns:
                    high_vol_mask = vol_breakdown[asset]
                    normal_vol_mask = ~high_vol_mask
                    
                    # Calculate average correlation during high volatility
                    if high_vol_mask.any():
                        high_vol_returns = recent_returns.loc[high_vol_mask, asset]
                        high_vol_corr = high_vol_returns.corr(recent_returns.loc[high_vol_mask, benchmark])
                        high_vol_correlations.append(high_vol_corr)
                    
                    # Calculate average correlation during normal volatility
                    if normal_vol_mask.any():
                        normal_vol_returns = recent_returns.loc[normal_vol_mask, asset]
                        normal_vol_corr = normal_vol_returns.corr(recent_returns.loc[normal_vol_mask, benchmark])
                        normal_vol_correlations.append(normal_vol_corr)
            
            # Create comparison chart
            fig = go.Figure(data=[
                go.Bar(
                    name='High Volatility Periods',
                    x=selected_tickers[:5],
                    y=high_vol_correlations,
                    marker_color='red'
                ),
                go.Bar(
                    name='Normal Volatility Periods',
                    x=selected_tickers[:5],
                    y=normal_vol_correlations,
                    marker_color='green'
                )
            ])
            
            fig.update_layout(
                title="Correlation with Benchmark: High vs Normal Volatility",
                height=500,
                template="plotly_dark",
                xaxis_title="Assets",
                yaxis_title="Correlation",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Monte Carlo VaR (Enhanced)
    with tab6:
        st.header("🎲 Monte Carlo Simulation & VaR")
        
        # Simulation parameters
        st.subheader("⚙️ Simulation Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_simulations = st.number_input(
                "Number of Simulations",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000
            )
        
        with col2:
            time_horizon = st.selectbox(
                "Time Horizon",
                ["1 Month", "3 Months", "6 Months", "1 Year"],
                index=3
            )
            
            horizon_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252}
            horizon_days = horizon_map[time_horizon]
        
        with col3:
            mc_confidence = st.slider(
                "Confidence Level",
                min_value=90,
                max_value=99.9,
                value=95,
                step=1
            ) / 100
        
        # Run Monte Carlo simulation
        if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Running {n_simulations:,} simulations..."):
                # Use portfolio returns for simulation
                mu = portfolio_returns.mean()
                sigma = portfolio_returns.std()
                initial_value = 100
                
                # Generate random paths
                dt = 1 / TRADING_DAYS
                paths = np.zeros((horizon_days, n_simulations))
                paths[0] = initial_value
                
                # GBM simulation
                for t in range(1, horizon_days):
                    z = np.random.standard_normal(n_simulations)
                    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                
                # Calculate final returns
                final_returns = (paths[-1] / initial_value) - 1
                
                # Calculate VaR and CVaR
                var_mc = np.percentile(final_returns, (1 - mc_confidence) * 100)
                cvar_mc = final_returns[final_returns <= var_mc].mean()
                
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
                    'sigma': sigma
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
            for i in range(min(100, st.session_state.mc_params['n_simulations'])):
                fig1.add_trace(go.Scatter(
                    x=list(range(st.session_state.mc_params['horizon_days'])),
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
                    'Kurtosis'
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
                    stats.kurtosis(st.session_state.mc_final_returns)
                ]
            }
            
            risk_metrics_df = pd.DataFrame(risk_metrics)
            
            st.dataframe(
                risk_metrics_df.style.format({'Value': '{:.4%}'}),
                use_container_width=True,
                height=400
            )
            
            # Export simulation results
            if st.button("📥 Export Simulation Results"):
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
    
    # Tab 7: Performance Attribution (Shortened for space)
    with tab7:
        st.header("📉 Performance Attribution")
        st.info("Performance attribution analysis would go here...")
    
    # Tab 8: Global Exposure
    with tab8:
        st.header("🌍 Global Portfolio Exposure")
        
        # Calculate geographical exposure
        geo_exposure = {}
        
        for ticker in selected_tickers:
            # Determine geography based on ticker pattern or known mapping
            if ticker in GLOBAL_ASSET_UNIVERSE['US_Stocks'] + GLOBAL_ASSET_UNIVERSE['US_Indices']:
                region = 'United States'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Europe_Stocks']:
                region = 'Europe'
            elif ticker in GLOBAL_ASSET_UNIVERSE['UK_Stocks']:
                region = 'United Kingdom'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Asia_Stocks']:
                region = 'Asia Pacific'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Emerging_Stocks']:
                region = 'Emerging Markets'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Australia_Stocks']:
                region = 'Australia'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Singapore_Stocks']:
                region = 'Singapore'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Turkey_Stocks']:
                region = 'Turkey'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Cryptocurrencies']:
                region = 'Cryptocurrency'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Bonds']:
                region = 'Fixed Income'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Commodities']:
                region = 'Commodities'
            elif ticker in GLOBAL_ASSET_UNIVERSE['Currencies']:
                region = 'Currencies'
            else:
                region = 'Other'
            
            # Add weight to region
            idx = selected_tickers.index(ticker)
            weight = weights[idx]
            geo_exposure[region] = geo_exposure.get(region, 0) + weight
        
        # Create exposure DataFrame
        exposure_df = pd.DataFrame({
            'Region': list(geo_exposure.keys()),
            'Weight': list(geo_exposure.values())
        }).sort_values('Weight', ascending=False)
        
        # Display exposure
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=exposure_df['Region'],
                    y=exposure_df['Weight'] * 100,
                    marker_color='#1a5fb4',
                    text=exposure_df['Weight'].apply(lambda x: f"{x:.1%}"),
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Geographical Exposure",
                height=500,
                template="plotly_dark",
                xaxis_title="Region",
                yaxis_title="Weight (%)",
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=exposure_df['Region'],
                values=exposure_df['Weight'],
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig.update_layout(
                title="Portfolio Allocation by Region",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sector exposure (simplified)
        st.subheader("🏭 Sector Exposure")
        
        # Simplified sector mapping
        sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'ASML.AS'],
            'Financials': ['JPM', 'HSBA.L', 'D05.SI', 'AKBNK.IS'],
            'Healthcare': ['JNJ', 'GSK.L', 'NOVN.SW'],
            'Energy': ['BP.L', 'ENEL.MI', 'USO'],
            'Consumer': ['PG', 'ULVR.L', 'BIMAS.IS'],
            'Industrials': ['CAT', 'SIEGn.DE'],
            'Materials': ['BHP.AX', 'RIO.L'],
            'Real Estate': ['PLD', 'C09.SI'],
            'Utilities': ['NGG', 'A2A.MI'],
            'Cryptocurrency': ['BTC-USD', 'ETH-USD'],
            'Fixed Income': ['TLT', 'IEF', 'BND'],
            'Commodities': ['GLD', 'SLV']
        }
        
        sector_exposure = {}
        
        for sector, tickers_in_sector in sector_mapping.items():
            sector_weight = 0
            for ticker in selected_tickers:
                if ticker in tickers_in_sector:
                    idx = selected_tickers.index(ticker)
                    sector_weight += weights[idx]
            
            if sector_weight > 0:
                sector_exposure[sector] = sector_weight
        
        sector_df = pd.DataFrame({
            'Sector': list(sector_exposure.keys()),
            'Weight': list(sector_exposure.values())
        }).sort_values('Weight', ascending=False)
        
        st.dataframe(
            sector_df.style.format({'Weight': '{:.2%}'}),
            use_container_width=True,
            height=300
        )
    
    # Tab 9: Risk Scorecard
    with tab9:
        st.header("🚦 Comprehensive Risk Scorecard")
        
        # Calculate all risk metrics
        risk_analytics = EnhancedRiskAnalytics()
        
        # Portfolio metrics
        ann_return = portfolio_returns.mean() * TRADING_DAYS
        ann_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else np.nan
        
        max_dd = ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().cummax() - 1).min()
        
        # VaR metrics
        var_95 = risk_analytics.calculate_var(portfolio_returns, "historical", 0.95)
        var_99 = risk_analytics.calculate_var(portfolio_returns, "historical", 0.99)
        
        # Active risk
        active_returns = portfolio_returns - returns[benchmark]
        tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
        information_ratio = active_returns.mean() * TRADING_DAYS / tracking_error if tracking_error > 0 else np.nan
        
        # Concentration metrics
        hhi = (weights ** 2).sum()
        top_3_concentration = np.sort(weights)[-3:].sum()
        
        # Create scorecard
        scorecard_data = [
            ("Annual Return", ann_return, "≥", (0.10, 0.05), True, "Higher is better"),
            ("Sharpe Ratio", sharpe, "≥", (1.0, 0.5), True, "Higher is better"),
            ("Annual Volatility", ann_vol, "≤", (0.15, 0.25), False, "Lower is better"),
            ("Max Drawdown", abs(max_dd), "≤", (0.15, 0.25), False, "Lower is better"),
            ("Tracking Error", tracking_error, "≤", (0.05, 0.10), False, "Lower is better"),
            ("Information Ratio", information_ratio, "≥", (0.5, 0.2), True, "Higher is better"),
            ("VaR 95%", abs(var_95['VaR']), "≤", (0.03, 0.06), False, "Lower is better"),
            ("CVaR 95%", abs(var_95['CVaR']), "≤", (0.05, 0.10), False, "Lower is better"),
            ("HHI Index", hhi, "≤", (0.10, 0.20), False, "Lower is better"),
            ("Top 3 Concentration", top_3_concentration, "≤", (0.50, 0.70), False, "Lower is better"),
        ]
        
        # Threshold configuration
        st.subheader("⚙️ Scorecard Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Green Threshold Adjustment (%)", value=0.0, step=1.0, key="threshold_adj")
        
        with col2:
            score_display = st.selectbox(
                "Display Format",
                ["Traffic Light", "Numeric Score", "Both"],
                index=0
            )
        
        # Calculate scores
        scores = []
        for name, value, comparator, (green_thresh, yellow_thresh), reverse, description in scorecard_data:
            if np.isnan(value):
                score = 0
                status = "gray"
            else:
                if reverse:
                    if value >= green_thresh:
                        score = 3
                        status = "green"
                    elif value >= yellow_thresh:
                        score = 2
                        status = "yellow"
                    else:
                        score = 1
                        status = "red"
                else:
                    if value <= green_thresh:
                        score = 3
                        status = "green"
                    elif value <= yellow_thresh:
                        score = 2
                        status = "yellow"
                    else:
                        score = 1
                        status = "red"
            
            scores.append(score)
            
            # Format value for display
            if name in ["Annual Return", "Annual Volatility", "Max Drawdown", "Tracking Error", 
                       "VaR 95%", "CVaR 95%", "Top 3 Concentration"]:
                display_value = f"{value:.2%}"
            elif name == "HHI Index":
                display_value = f"{value:.4f}"
            else:
                display_value = f"{value:.2f}"
            
            # Display row
            col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
            
            with col_a:
                st.write(f"**{name}**")
                st.caption(description)
            
            with col_b:
                st.write(display_value)
            
            with col_c:
                if score_display in ["Traffic Light", "Both"]:
                    color_map = {"green": "🟢", "yellow": "🟡", "red": "🔴", "gray": "⚫"}
                    st.write(color_map[status])
            
            with col_d:
                if score_display in ["Numeric Score", "Both"]:
                    st.write(f"{score}/3")
            
            st.markdown("---")
        
        # Overall score
        overall_score = np.mean(scores)
        overall_status = "green" if overall_score >= 2.5 else "yellow" if overall_score >= 1.5 else "red"
        
        st.subheader("📊 Overall Risk Score")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Score", f"{overall_score:.1f}/3.0")
        
        with col2:
            status_text = {"green": "Low Risk", "yellow": "Medium Risk", "red": "High Risk"}
            st.metric("Risk Level", status_text[overall_status])
        
        with col3:
            st.metric("Assessment", "✅ Good" if overall_score >= 2.0 else "⚠️ Needs Improvement")
        
        # Risk breakdown
        st.subheader("📈 Risk Breakdown")
        
        categories = {
            "Return Metrics": ["Annual Return", "Sharpe Ratio"],
            "Risk Metrics": ["Annual Volatility", "Max Drawdown", "VaR 95%", "CVaR 95%"],
            "Active Risk": ["Tracking Error", "Information Ratio"],
            "Concentration": ["HHI Index", "Top 3 Concentration"]
        }
        
        for category, metrics in categories.items():
            # Find indices of metrics in this category
            metric_indices = [i for i, (name, _, _, _, _, _) in enumerate(scorecard_data) if name in metrics]
            if metric_indices:
                category_score = np.mean([scores[i] for i in metric_indices])
                
                col_a, col_b, col_c = st.columns([2, 1, 3])
                
                with col_a:
                    st.write(f"**{category}**")
                
                with col_b:
                    st.write(f"{category_score:.1f}/3.0")
                
                with col_c:
                    # Progress bar
                    progress = category_score / 3
                    st.progress(progress)
        
        # Export scorecard
        if st.button("📥 Export Risk Scorecard"):
            # Create scorecard DataFrame
            scorecard_df = pd.DataFrame([
                {
                    'Metric': name,
                    'Value': value,
                    'Threshold Green': green_thresh,
                    'Threshold Yellow': yellow_thresh,
                    'Score': score,
                    'Status': status,
                    'Description': description
                }
                for (name, value, _, (green_thresh, yellow_thresh), _, description), score, status in 
                zip(scorecard_data, scores, [s["status"] for s in scores])
            ])
            
            # Add overall metrics
            overall_row = pd.DataFrame([{
                'Metric': 'OVERALL_SCORE',
                'Value': overall_score,
                'Threshold Green': 2.5,
                'Threshold Yellow': 1.5,
                'Score': overall_score,
                'Status': overall_status,
                'Description': 'Overall risk assessment score'
            }])
            
            scorecard_df = pd.concat([scorecard_df, overall_row], ignore_index=True)
            
            csv = scorecard_df.to_csv(index=False)
            st.download_button(
                label="Download Scorecard CSV",
                data=csv,
                file_name="risk_scorecard.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
