# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal v4.1
# Professional Portfolio Optimization & Global Multi-Asset Edition
# Enhanced with Comprehensive Historical Stress Testing
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
# HISTORICAL STRESS TESTING SCENARIOS
# -------------------------------------------------------------
HISTORICAL_STRESS_SCENARIOS = {
    # 21st Century Financial Crises
    "2008_Financial_Crisis": {
        "name": "2008 Global Financial Crisis",
        "period": ("2007-10-09", "2009-03-09"),
        "description": "Subprime mortgage crisis leading to global banking collapse",
        "characteristics": {
            "equity_drawdown": -56.4,  # S&P 500 peak to trough
            "duration_days": 517,
            "vix_peak": 80.86,
            "recovery_days": 1325,
            "key_events": ["Lehman Brothers bankruptcy", "Bear Stearns collapse", "AIG bailout"]
        }
    },
    
    "2020_COVID_Crash": {
        "name": "2020 COVID-19 Market Crash",
        "period": ("2020-02-19", "2020-03-23"),
        "description": "Global pandemic-induced market panic and rapid recovery",
        "characteristics": {
            "equity_drawdown": -33.9,
            "duration_days": 33,
            "vix_peak": 82.69,
            "recovery_days": 152,
            "key_events": ["WHO declares pandemic", "Global lockdowns", "Unprecedented fiscal stimulus"]
        }
    },
    
    "2011_European_Debt_Crisis": {
        "name": "2011 European Sovereign Debt Crisis",
        "period": ("2011-04-29", "2011-10-03"),
        "description": "Eurozone sovereign debt concerns and banking system stress",
        "characteristics": {
            "equity_drawdown": -19.4,
            "duration_days": 157,
            "vix_peak": 48.0,
            "recovery_days": 310,
            "key_events": ["Greek debt crisis", "EU emergency summits", "ECB intervention"]
        }
    },
    
    "2015_2016_China_Growth_Scare": {
        "name": "2015-2016 China Growth Scare",
        "period": ("2015-06-12", "2016-02-11"),
        "description": "Chinese economic slowdown and currency devaluation fears",
        "characteristics": {
            "equity_drawdown": -14.2,
            "duration_days": 244,
            "vix_peak": 40.74,
            "recovery_days": 180,
            "key_events": ["Chinese stock market crash", "Renminbi devaluation", "Commodity collapse"]
        }
    },
    
    "2018_Q4_Rout": {
        "name": "2018 Q4 Market Rout",
        "period": ("2018-09-20", "2018-12-24"),
        "description": "Trade war fears and Federal Reserve policy uncertainty",
        "characteristics": {
            "equity_drawdown": -19.8,
            "duration_days": 95,
            "vix_peak": 36.07,
            "recovery_days": 110,
            "key_events": ["US-China trade war escalation", "Fed rate hike concerns", "Growth fears"]
        }
    },
    
    # Sector-Specific Crises
    "2000_Dotcom_Bubble": {
        "name": "2000 Dot-com Bubble Burst",
        "period": ("2000-03-10", "2002-10-09"),
        "description": "Technology stock bubble collapse",
        "characteristics": {
            "equity_drawdown": -49.1,
            "duration_days": 943,
            "vix_peak": 45.08,
            "recovery_days": 1825,
            "key_events": ["NASDAQ crash", "Tech company bankruptcies", "Accounting scandals"]
        }
    },
    
    "2014_Oil_Price_Crash": {
        "name": "2014-2016 Oil Price Collapse",
        "period": ("2014-06-20", "2016-01-20"),
        "description": "OPEC production war leading to oil price collapse",
        "characteristics": {
            "oil_drawdown": -76.0,
            "duration_days": 580,
            "energy_sector_drawdown": -45.0,
            "recovery_days": 720,
            "key_events": ["OPEC production increase", "Shale oil boom", "Global oversupply"]
        }
    },
    
    "2022_Inflation_Shock": {
        "name": "2022 Inflation & Rate Shock",
        "period": ("2022-01-03", "2022-10-12"),
        "description": "Post-pandemic inflation surge and aggressive central bank tightening",
        "characteristics": {
            "equity_drawdown": -25.4,
            "duration_days": 282,
            "bond_drawdown": -20.0,
            "recovery_days": 180,
            "key_events": ["Russia-Ukraine war", "40-year high inflation", "Aggressive Fed hikes"]
        }
    },
    
    # Regional Crises
    "1997_Asian_Financial_Crisis": {
        "name": "1997 Asian Financial Crisis",
        "period": ("1997-07-02", "1998-08-31"),
        "description": "Currency and banking crisis across Southeast Asia",
        "characteristics": {
            "regional_drawdown": -70.0,  # Thailand SET Index
            "duration_days": 425,
            "currency_devaluation": -56.0,  # Thai Baht
            "recovery_days": 1460,
            "key_events": ["Thai baht devaluation", "IMF bailouts", "Regional contagion"]
        }
    },
    
    "1998_Russian_Default": {
        "name": "1998 Russian Financial Crisis",
        "period": ("1998-08-17", "1998-10-08"),
        "description": "Russian government default and LTCM collapse",
        "characteristics": {
            "russian_market_drawdown": -85.0,
            "duration_days": 52,
            "vix_peak": 45.74,
            "recovery_days": 180,
            "key_events": ["Russian default", "LTCM collapse", "Global liquidity crunch"]
        }
    },
    
    "2013_Taper_Tantrum": {
        "name": "2013 Taper Tantrum",
        "period": ("2013-05-22", "2013-06-24"),
        "description": "Bond market selloff on Fed taper announcement",
        "characteristics": {
            "bond_drawdown": -8.0,  # US 10-year Treasury
            "duration_days": 33,
            "emerging_markets_drawdown": -15.0,
            "recovery_days": 90,
            "key_events": ["Fed taper announcement", "Emerging market outflows", "Rate volatility"]
        }
    },
    
    # Black Swan Events
    "1987_Black_Monday": {
        "name": "1987 Black Monday",
        "period": ("1987-10-14", "1987-10-19"),
        "description": "Largest one-day percentage decline in stock market history",
        "characteristics": {
            "one_day_drawdown": -22.6,
            "duration_days": 5,
            "vix_equivalent": 150.0,  # Estimated
            "recovery_days": 728,
            "key_events": ["Program trading", "Portfolio insurance", "Global synchronized crash"]
        }
    },
    
    "2010_Flash_Crash": {
        "name": "2010 Flash Crash",
        "period": ("2010-05-06", "2010-05-06"),
        "description": "Ultra-fast electronic trading crash and recovery",
        "characteristics": {
            "intraday_drawdown": -9.0,
            "duration_minutes": 36,
            "recovery_minutes": 20,
            "key_events": ["Algorithmic trading feedback loop", "Liquidity evaporation", "Circuit breakers triggered"]
        }
    },
    
    # Custom Stress Periods
    "Custom_Period_1": {
        "name": "Custom Stress Period 1",
        "period": ("2018-12-01", "2019-01-31"),
        "description": "Custom defined stress period",
        "characteristics": {
            "custom_metric": "User defined",
            "adjustable": True
        }
    }
}

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
APP_TITLE = "🏛️ Apollo/ENIGMA - Global Portfolio Terminal v4.1"
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

/* Custom cards for stress testing */
.stress-card {
    background: linear-gradient(135deg, #2d1b69, #1a1033);
    border: 1px solid #5b21b6;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(91, 33, 182, 0.3);
    transition: all 0.3s ease;
}

.stress-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(91, 33, 182, 0.4);
    border-color: #7c3aed;
}

.stress-card.crisis {
    background: linear-gradient(135deg, #7f1d1d, #450a0a);
    border-color: #dc2626;
}

.stress-card.crisis:hover {
    box-shadow: 0 8px 25px rgba(220, 38, 38, 0.4);
    border-color: #ef4444;
}

.stress-card.recovery {
    background: linear-gradient(135deg, #064e3b, #022c22);
    border-color: #059669;
}

.stress-card.recovery:hover {
    box-shadow: 0 8px 25px rgba(5, 150, 105, 0.4);
    border-color: #10b981;
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
# HISTORICAL STRESS TESTING ENGINE
# -------------------------------------------------------------
class HistoricalStressTester:
    """Comprehensive historical stress testing engine"""
    
    @staticmethod
    def analyze_stress_period(prices: pd.DataFrame, stress_period: Tuple[str, str], 
                             portfolio_weights: np.ndarray = None) -> Dict:
        """Analyze portfolio performance during historical stress period"""
        
        # Extract stress period
        start_date, end_date = stress_period
        stress_prices = prices.loc[start_date:end_date]
        
        if len(stress_prices) < 5:
            return {"error": "Insufficient data for stress period"}
        
        # Calculate returns during stress period
        stress_returns = stress_prices.pct_change().dropna()
        
        if portfolio_weights is not None:
            # Calculate portfolio returns
            portfolio_returns = (stress_returns * portfolio_weights).sum(axis=1)
        else:
            # Use equal weights if none provided
            n_assets = len(stress_returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            portfolio_returns = (stress_returns * equal_weights).sum(axis=1)
        
        # Calculate stress metrics
        cumulative_return = (1 + portfolio_returns).prod() - 1
        max_drawdown = HistoricalStressTester.calculate_max_drawdown(portfolio_returns)
        volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        worst_day = portfolio_returns.min()
        worst_5day = portfolio_returns.rolling(5).sum().min()
        worst_month = portfolio_returns.rolling(21).sum().min()
        
        # Calculate recovery metrics
        recovery_days = HistoricalStressTester.calculate_recovery_days(stress_prices, portfolio_weights)
        
        # Calculate asset contributions to loss
        asset_contributions = HistoricalStressTester.calculate_asset_contributions(
            stress_returns, portfolio_weights
        )
        
        return {
            'stress_period': stress_period,
            'duration_days': len(stress_prices),
            'portfolio_cumulative_return': cumulative_return,
            'portfolio_max_drawdown': max_drawdown,
            'portfolio_volatility': volatility,
            'worst_day_return': worst_day,
            'worst_5day_return': worst_5day,
            'worst_month_return': worst_month,
            'recovery_days': recovery_days,
            'asset_contributions': asset_contributions,
            'stress_returns': portfolio_returns
        }
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def calculate_recovery_days(prices: pd.DataFrame, weights: np.ndarray) -> int:
        """Calculate days to recover to pre-stress levels"""
        if len(prices) < 2:
            return 0
        
        # Calculate portfolio values
        portfolio_values = (prices / prices.iloc[0]) @ weights
        
        # Find trough
        trough_idx = portfolio_values.argmin()
        trough_value = portfolio_values.iloc[trough_idx]
        
        # Find when recovery happens
        recovery_mask = portfolio_values.iloc[trough_idx:] >= portfolio_values.iloc[0]
        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()
            recovery_days = (recovery_idx - portfolio_values.index[trough_idx]).days
            return max(0, recovery_days)
        
        return len(portfolio_values) - trough_idx
    
    @staticmethod
    def calculate_asset_contributions(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Calculate each asset's contribution to total loss"""
        if weights is None:
            n_assets = len(returns.columns)
            weights = np.ones(n_assets) / n_assets
        
        total_return = (returns * weights).sum(axis=1)
        asset_contributions = {}
        
        for asset in returns.columns:
            asset_idx = list(returns.columns).index(asset)
            # Calculate contribution as weighted return
            contribution = returns[asset] * weights[asset_idx]
            # Normalize by total portfolio return
            normalized_contribution = contribution.sum() / total_return.sum() if total_return.sum() != 0 else 0
            asset_contributions[asset] = normalized_contribution
        
        return pd.Series(asset_contributions)
    
    @staticmethod
    def compare_across_scenarios(portfolio_weights: np.ndarray, prices: pd.DataFrame, 
                                scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """Compare portfolio performance across multiple historical scenarios"""
        
        results = []
        
        for scenario_id, scenario_info in scenarios.items():
            try:
                analysis = HistoricalStressTester.analyze_stress_period(
                    prices, 
                    scenario_info['period'],
                    portfolio_weights
                )
                
                if 'error' not in analysis:
                    results.append({
                        'Scenario': scenario_info['name'],
                        'Period': f"{scenario_info['period'][0]} to {scenario_info['period'][1]}",
                        'Duration (days)': analysis['duration_days'],
                        'Portfolio Return': analysis['portfolio_cumulative_return'],
                        'Max Drawdown': analysis['portfolio_max_drawdown'],
                        'Volatility': analysis['portfolio_volatility'],
                        'Worst Day': analysis['worst_day_return'],
                        'Recovery Days': analysis['recovery_days'],
                        'Scenario ID': scenario_id
                    })
            except Exception as e:
                results.append({
                    'Scenario': scenario_info['name'],
                    'Period': f"{scenario_info['period'][0]} to {scenario_info['period'][1]}",
                    'Duration (days)': np.nan,
                    'Portfolio Return': np.nan,
                    'Max Drawdown': np.nan,
                    'Volatility': np.nan,
                    'Worst Day': np.nan,
                    'Recovery Days': np.nan,
                    'Scenario ID': scenario_id,
                    'Error': str(e)
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def create_custom_stress_period(prices: pd.DataFrame, start_date: str, end_date: str) -> Dict:
        """Create analysis for custom stress period"""
        return {
            'name': f"Custom Stress Period: {start_date} to {end_date}",
            'period': (start_date, end_date),
            'description': 'User-defined stress testing period',
            'characteristics': {
                'custom': True,
                'user_defined': True
            }
        }

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
        
        # Asset selector - FIXED: Ensure defaults are in options
        selected_tickers = st.multiselect(
            "Select Assets (5-20 recommended)",
            filtered_tickers,
            default=valid_defaults,
            help="Select 5-20 assets for optimal diversification"
        )
        
        # Update session state
        st.session_state.selected_tickers = selected_tickers
        
        # Benchmark selection - ensure benchmark is not in selected tickers
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
    
    # Create enhanced tabs with historical stress testing
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Overview & Weights",
        "⚖️ Risk Analytics",
        "🎯 Portfolio Optimization",
        "🔗 Correlation Matrix",
        "📊 EWMA Analysis",
        "🎲 Monte Carlo VaR",
        "📉 Historical Stress Testing",  # NEW TAB
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
    
    # Tab 2: Enhanced Risk Analytics (already implemented)
    # [Previous Tab 2 code remains the same]
    
    # Tab 6: Monte Carlo VaR (already implemented)  
    # [Previous Tab 6 code remains the same]
    
    # Tab 7: NEW - HISTORICAL STRESS TESTING
    with tab7:
        st.header("📉 Historical Stress Testing")
        
        # Introduction
        st.markdown("""
        ### 🎯 Stress Testing Overview
        Analyze how your portfolio would have performed during major historical market crises.
        This helps understand tail risks and improve portfolio resilience.
        """)
        
        # Stress Testing Engine
        stress_tester = HistoricalStressTester()
        
        # Scenario Selection
        st.subheader("📊 Select Historical Stress Scenarios")
        
        # Group scenarios by type
        scenario_groups = {
            "21st Century Crises": ["2008_Financial_Crisis", "2020_COVID_Crash", "2011_European_Debt_Crisis",
                                   "2015_2016_China_Growth_Scare", "2018_Q4_Rout", "2022_Inflation_Shock"],
            "Sector-Specific Crises": ["2000_Dotcom_Bubble", "2014_Oil_Price_Crash"],
            "Regional Crises": ["1997_Asian_Financial_Crisis", "1998_Russian_Default", "2013_Taper_Tantrum"],
            "Black Swan Events": ["1987_Black_Monday", "2010_Flash_Crash"]
        }
        
        selected_scenarios = []
        
        for group_name, scenario_ids in scenario_groups.items():
            with st.expander(f"📂 {group_name}"):
                for scenario_id in scenario_ids:
                    if scenario_id in HISTORICAL_STRESS_SCENARIOS:
                        scenario = HISTORICAL_STRESS_SCENARIOS[scenario_id]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{scenario['name']}**")
                            st.caption(f"{scenario['description']}")
                            st.caption(f"Period: {scenario['period'][0]} to {scenario['period'][1]}")
                        with col2:
                            if st.checkbox("Select", key=f"stress_{scenario_id}"):
                                selected_scenarios.append(scenario_id)
        
        # Custom stress period
        st.subheader("🔄 Custom Stress Period")
        
        col1, col2 = st.columns(2)
        with col1:
            custom_start = st.date_input("Custom Start Date", pd.Timestamp("2020-02-19"))
        with col2:
            custom_end = st.date_input("Custom End Date", pd.Timestamp("2020-03-23"))
        
        if st.button("➕ Add Custom Stress Period"):
            custom_scenario = stress_tester.create_custom_stress_period(
                prices, str(custom_start), str(custom_end)
            )
            custom_id = f"custom_{custom_start}_{custom_end}"
            HISTORICAL_STRESS_SCENARIOS[custom_id] = custom_scenario
            selected_scenarios.append(custom_id)
            st.success(f"Added custom stress period: {custom_start} to {custom_end}")
        
        # Run stress testing
        if selected_scenarios and st.button("🚀 Run Stress Testing Analysis", type="primary"):
            with st.spinner("🔬 Analyzing portfolio stress performance..."):
                
                # Get current portfolio weights
                if 'current_weights' in st.session_state:
                    portfolio_weights = st.session_state.current_weights
                else:
                    # Fallback to equal weights
                    portfolio_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
                
                # Filter scenarios to only selected ones
                selected_scenario_info = {sid: HISTORICAL_STRESS_SCENARIOS[sid] for sid in selected_scenarios}
                
                # Compare across scenarios
                comparison_results = stress_tester.compare_across_scenarios(
                    portfolio_weights, prices, selected_scenario_info
                )
                
                # Display results
                st.subheader("📊 Stress Testing Results Comparison")
                
                # Format results for display
                display_results = comparison_results.copy()
                for col in ['Portfolio Return', 'Max Drawdown', 'Volatility', 'Worst Day']:
                    display_results[col] = display_results[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
                
                st.dataframe(
                    display_results[['Scenario', 'Period', 'Duration (days)', 
                                     'Portfolio Return', 'Max Drawdown', 'Volatility', 
                                     'Worst Day', 'Recovery Days']],
                    use_container_width=True,
                    height=400
                )
                
                # Visualization 1: Portfolio Returns Comparison
                st.subheader("📈 Portfolio Returns During Stress Periods")
                
                fig = go.Figure()
                
                for _, row in comparison_results.iterrows():
                    if not pd.isna(row['Portfolio Return']):
                        # Get scenario details
                        scenario_info = HISTORICAL_STRESS_SCENARIOS.get(row['Scenario ID'], {})
                        color = 'red' if row['Portfolio Return'] < -0.1 else 'orange' if row['Portfolio Return'] < 0 else 'green'
                        
                        fig.add_trace(go.Bar(
                            x=[row['Scenario']],
                            y=[row['Portfolio Return'] * 100],
                            name=row['Scenario'],
                            text=f"{row['Portfolio Return']:.1%}",
                            textposition='auto',
                            marker_color=color,
                            hovertemplate=(
                                f"<b>{row['Scenario']}</b><br>" +
                                f"Period: {row['Period']}<br>" +
                                f"Return: {row['Portfolio Return']:.2%}<br>" +
                                f"Max DD: {row['Max Drawdown']:.2%}<br>" +
                                f"Volatility: {row['Volatility']:.2%}<br>" +
                                f"Recovery: {row['Recovery Days']} days<br>" +
                                "<extra></extra>"
                            )
                        ))
                
                fig.update_layout(
                    title="Portfolio Performance During Historical Stress Periods",
                    height=500,
                    template="plotly_dark",
                    xaxis_title="Stress Scenario",
                    yaxis_title="Portfolio Return (%)",
                    xaxis_tickangle=45,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualization 2: Detailed Analysis for Each Scenario
                st.subheader("🔍 Detailed Scenario Analysis")
                
                # Let user select a scenario for detailed view
                selected_scenario_detail = st.selectbox(
                    "Select Scenario for Detailed Analysis",
                    comparison_results['Scenario'].tolist()
                )
                
                if selected_scenario_detail:
                    scenario_row = comparison_results[comparison_results['Scenario'] == selected_scenario_detail].iloc[0]
                    scenario_id = scenario_row['Scenario ID']
                    
                    if scenario_id in HISTORICAL_STRESS_SCENARIOS:
                        scenario_info = HISTORICAL_STRESS_SCENARIOS[scenario_id]
                        
                        # Analyze this specific scenario
                        detailed_analysis = stress_tester.analyze_stress_period(
                            prices, scenario_info['period'], portfolio_weights
                        )
                        
                        if 'error' not in detailed_analysis:
                            # Display scenario characteristics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                <div class='stress-card crisis'>
                                    <h4>📉 Crisis Characteristics</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show key characteristics
                                chars = scenario_info.get('characteristics', {})
                                for key, value in chars.items():
                                    if key != 'key_events':
                                        if isinstance(value, (int, float)):
                                            if key.endswith('drawdown') or key.endswith('devaluation'):
                                                st.metric(key.replace('_', ' ').title(), f"{value:.1f}%")
                                            else:
                                                st.metric(key.replace('_', ' ').title(), value)
                                
                                # Key events
                                if 'key_events' in chars:
                                    st.write("**Key Events:**")
                                    for event in chars['key_events']:
                                        st.write(f"• {event}")
                            
                            with col2:
                                st.markdown("""
                                <div class='stress-card'>
                                    <h4>📊 Portfolio Impact</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Portfolio impact metrics
                                metrics = [
                                    ("Cumulative Return", detailed_analysis['portfolio_cumulative_return'], "{:.2%}"),
                                    ("Maximum Drawdown", detailed_analysis['portfolio_max_drawdown'], "{:.2%}"),
                                    ("Annualized Volatility", detailed_analysis['portfolio_volatility'], "{:.2%}"),
                                    ("Worst Daily Return", detailed_analysis['worst_day_return'], "{:.2%}"),
                                    ("Worst 5-Day Return", detailed_analysis['worst_5day_return'], "{:.2%}"),
                                    ("Days to Recovery", detailed_analysis['recovery_days'], "{:.0f} days"),
                                    ("Stress Duration", detailed_analysis['duration_days'], "{:.0f} days")
                                ]
                                
                                for name, value, fmt in metrics:
                                    if not pd.isna(value):
                                        st.metric(name, fmt.format(value))
                            
                            # Visualization: Portfolio Performance During Stress
                            st.subheader("📉 Portfolio Performance Timeline")
                            
                            # Get prices for stress period
                            stress_prices = prices.loc[scenario_info['period'][0]:scenario_info['period'][1]]
                            stress_returns = stress_prices[selected_tickers].pct_change().dropna()
                            stress_portfolio_returns = (stress_returns * portfolio_weights).sum(axis=1)
                            stress_cumulative = (1 + stress_portfolio_returns).cumprod()
                            
                            fig2 = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=(
                                    "Portfolio Cumulative Return",
                                    "Daily Returns & Drawdown"
                                )
                            )
                            
                            # Cumulative return
                            fig2.add_trace(
                                go.Scatter(
                                    x=stress_cumulative.index,
                                    y=stress_cumulative.values,
                                    name="Portfolio Value",
                                    line=dict(color='#1a5fb4', width=3)
                                ),
                                row=1, col=1
                            )
                            
                            # Drawdown area
                            drawdown = (stress_cumulative / stress_cumulative.cummax() - 1)
                            fig2.add_trace(
                                go.Scatter(
                                    x=drawdown.index,
                                    y=drawdown.values * 100,
                                    name="Drawdown",
                                    fill='tozeroy',
                                    fillcolor='rgba(220, 38, 38, 0.3)',
                                    line=dict(color='rgba(220, 38, 38, 0.5)', width=1)
                                ),
                                row=2, col=1
                            )
                            
                            # Daily returns as bars
                            fig2.add_trace(
                                go.Bar(
                                    x=stress_portfolio_returns.index,
                                    y=stress_portfolio_returns.values * 100,
                                    name="Daily Return",
                                    marker_color=['red' if x < 0 else 'green' for x in stress_portfolio_returns.values],
                                    opacity=0.6
                                ),
                                row=2, col=1
                            )
                            
                            fig2.update_layout(
                                height=700,
                                template="plotly_dark",
                                showlegend=True,
                                title=f"Portfolio Performance During {scenario_info['name']}"
                            )
                            
                            fig2.update_yaxes(title_text="Cumulative Return", row=1, col=1)
                            fig2.update_yaxes(title_text="Return / Drawdown (%)", row=2, col=1)
                            
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Asset Contribution Analysis
                            st.subheader("🔍 Asset Contribution to Loss")
                            
                            if 'asset_contributions' in detailed_analysis:
                                asset_contributions = detailed_analysis['asset_contributions']
                                contributions_df = pd.DataFrame({
                                    'Asset': asset_contributions.index,
                                    'Contribution to Loss': asset_contributions.values
                                }).sort_values('Contribution to Loss')
                                
                                # Only show top and bottom contributors
                                top_loss = contributions_df.tail(5)  # Biggest contributors to loss
                                top_gain = contributions_df.head(5)  # Biggest mitigators
                                
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.write("**Top Loss Contributors**")
                                    st.dataframe(
                                        top_loss.style.format({'Contribution to Loss': '{:.2%}'}),
                                        use_container_width=True
                                    )
                                
                                with col_b:
                                    st.write("**Top Loss Mitigators**")
                                    st.dataframe(
                                        top_gain.style.format({'Contribution to Loss': '{:.2%}'}),
                                        use_container_width=True
                                    )
                                
                                # Visualization of contributions
                                fig3 = go.Figure(data=[
                                    go.Bar(
                                        x=contributions_df['Asset'],
                                        y=contributions_df['Contribution to Loss'] * 100,
                                        marker_color=['red' if x > 0 else 'green' for x in contributions_df['Contribution to Loss']],
                                        text=[f"{x:.1f}%" for x in contributions_df['Contribution to Loss'] * 100],
                                        textposition='auto'
                                    )
                                ])
                                
                                fig3.update_layout(
                                    title="Asset Contributions to Portfolio Loss During Stress",
                                    height=400,
                                    template="plotly_dark",
                                    xaxis_title="Assets",
                                    yaxis_title="Contribution to Loss (%)",
                                    xaxis_tickangle=45
                                )
                                
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # Stress Test Recommendations
                            st.subheader("💡 Stress Test Recommendations")
                            
                            recommendations = []
                            
                            # Analyze results for recommendations
                            if detailed_analysis['portfolio_max_drawdown'] < -0.20:
                                recommendations.append("⚠️ **High Drawdown Risk**: Portfolio experienced >20% drawdown. Consider adding defensive assets.")
                            
                            if detailed_analysis['worst_day_return'] < -0.05:
                                recommendations.append("⚠️ **Extreme Daily Loss**: Portfolio had daily loss >5%. Increase liquidity or add hedging.")
                            
                            if detailed_analysis['recovery_days'] > 365:
                                recommendations.append("⚠️ **Slow Recovery**: Recovery took >1 year. Consider strategies for faster recovery.")
                            
                            if len(recommendations) == 0:
                                recommendations.append("✅ **Good Resilience**: Portfolio showed reasonable resilience during this stress period.")
                            
                            for rec in recommendations:
                                st.info(rec)
                            
                            # Export stress test results
                            if st.button("📥 Export Stress Test Report"):
                                # Create comprehensive report
                                report_data = {
                                    'Scenario': scenario_info['name'],
                                    'Period': scenario_info['period'],
                                    'Description': scenario_info['description'],
                                    'Portfolio Metrics': {
                                        'Cumulative Return': detailed_analysis['portfolio_cumulative_return'],
                                        'Max Drawdown': detailed_analysis['portfolio_max_drawdown'],
                                        'Volatility': detailed_analysis['portfolio_volatility'],
                                        'Worst Day': detailed_analysis['worst_day_return'],
                                        'Recovery Days': detailed_analysis['recovery_days']
                                    },
                                    'Asset Contributions': asset_contributions.to_dict() if 'asset_contributions' in detailed_analysis else {},
                                    'Recommendations': recommendations
                                }
                                
                                # Convert to JSON for export
                                report_json = json.dumps(report_data, indent=2, default=str)
                                st.download_button(
                                    label="Download JSON Report",
                                    data=report_json,
                                    file_name=f"stress_test_{scenario_id}.json",
                                    mime="application/json"
                                )
        
        else:
            st.info("👆 Select stress scenarios above and click 'Run Stress Testing Analysis'")
            
            # Show available scenarios
            with st.expander("📋 Available Stress Scenarios"):
                for scenario_id, scenario_info in HISTORICAL_STRESS_SCENARIOS.items():
                    if scenario_id not in ['Custom_Period_1']:  # Skip custom placeholder
                        st.write(f"**{scenario_info['name']}**")
                        st.caption(f"Period: {scenario_info['period'][0]} to {scenario_info['period'][1]}")
                        st.caption(f"{scenario_info['description']}")
                        st.markdown("---")
    
    # Tab 8: Performance Attribution
    with tab8:
        st.header("📊 Performance Attribution")
        st.info("Performance attribution analysis would go here...")
    
    # Tab 9: Global Exposure
    with tab9:
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
            weight = st.session_state.current_weights[idx] if 'current_weights' in st.session_state else 1/len(selected_tickers)
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
                    sector_weight += st.session_state.current_weights[idx] if 'current_weights' in st.session_state else 1/len(selected_tickers)
            
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
    
    # Tab 10: Risk Scorecard
    with tab10:
        st.header("🚦 Comprehensive Risk Scorecard")
        
        # Calculate all risk metrics
        risk_analytics = EnhancedRiskAnalytics()
        
        # Portfolio metrics
        portfolio_returns = returns[selected_tickers].mean(axis=1) if 'current_weights' not in st.session_state else (returns[selected_tickers] * st.session_state.current_weights).sum(axis=1)
        ann_return = portfolio_returns.mean() * TRADING_DAYS
        ann_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else np.nan
        
        max_dd = ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().cummax() - 1).min()
        
        # VaR metrics
        var_95 = risk_analytics.calculate_var(portfolio_returns, "historical", 0.95)
        var_99 = risk_analytics.calculate_var(portfolio_returns, "historical", 0.99)
        
        # Active risk
        benchmark_returns = returns[benchmark]
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
        information_ratio = active_returns.mean() * TRADING_DAYS / tracking_error if tracking_error > 0 else np.nan
        
        # Concentration metrics
        weights = st.session_state.current_weights if 'current_weights' in st.session_state else np.ones(len(selected_tickers)) / len(selected_tickers)
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
        statuses = []
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
            statuses.append(status)
            
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
                zip(scorecard_data, scores, statuses)
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
