"""
===============================================================================
PROJECT 9: Water Stress & Agricultural Commodity Prices
===============================================================================
RESEARCH QUESTION:
    Do water stress indicators predict agricultural commodity price movements?
METHOD:
    Time series analysis — correlation, Granger causality, VAR
DATA:
    Yahoo Finance (commodity ETFs), World Bank water indicators
===============================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

print("STEP 1: Downloading commodity and water-related asset data...")

tickers = {
    'DBA':'Agriculture ETF','WEAT':'Wheat ETF','CORN':'Corn ETF',
    'PHO':'Water Utilities ETF','CGW':'Global Water ETF','XLU':'Utilities ETF'
}

prices = {}
for t in tickers:
    df = yf.download(t, start='2018-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if not df.empty:
        prices[t] = df['Close']
        print(f"  {t} ({tickers[t]}): {len(df)} obs")

prices = pd.DataFrame(prices).dropna()
returns = np.log(prices/prices.shift(1)).dropna() * 100
prices.to_csv('data/prices.csv')
returns.to_csv('data/returns.csv')

print(f"\nSTEP 2: Computing correlations and Granger causality...")

# Rolling correlations: water ETFs vs agriculture
corr_results = []
water = ['PHO','CGW']
agri = ['DBA','WEAT','CORN']
for w in water:
    for a in agri:
        if w in returns.columns and a in returns.columns:
            r = returns[w].corr(returns[a])
            corr_results.append({'Water_Asset':w, 'Agri_Asset':a, 'Correlation':round(r,4)})
            
            # Granger causality
            try:
                gc = grangercausalitytests(returns[[a, w]].dropna(), maxlag=5, verbose=False)
                min_p = min([gc[lag][0]['ssr_ftest'][1] for lag in gc])
                corr_results[-1]['Granger_p'] = round(min_p, 4)
                corr_results[-1]['Granger_sig'] = 'Yes' if min_p < 0.05 else 'No'
            except:
                pass

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv('output/tables/correlation_granger.csv', index=False)
print(corr_df.to_string(index=False))

print(f"\nSTEP 3: VAR model...")
var_cols = [c for c in ['PHO','DBA','WEAT'] if c in returns.columns]
if len(var_cols) >= 2:
    model = VAR(returns[var_cols].dropna())
    result = model.fit(maxlags=5, ic='aic')
    print(f"  VAR({result.k_ar}) fitted")
    
    irf = result.irf(20)
    fig, axes = plt.subplots(len(var_cols), len(var_cols), figsize=(14, 10))
    for i in range(len(var_cols)):
        for j in range(len(var_cols)):
            ax = axes[i][j] if len(var_cols) > 1 else axes
            irfs = irf.irfs[:, i, j]
            ax.plot(irfs, 'steelblue', lw=2)
            ax.axhline(0, color='black', lw=0.5)
            ax.fill_between(range(len(irfs)), irfs, alpha=0.2, color='steelblue')
            ax.set_title(f'{var_cols[j]}→{var_cols[i]}', fontsize=9, fontweight='bold')
    plt.suptitle('Impulse Response Functions', fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/figures/fig1_irf.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"\nSTEP 4: Creating visualizations...")

# Normalized prices
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
norm = (prices/prices.iloc[0])*100
for t in agri:
    if t in norm.columns:
        axes[0].plot(norm.index, norm[t], label=tickers.get(t,t), lw=1.2)
axes[0].set_title('Agricultural Commodity ETFs (Normalized)', fontweight='bold')
axes[0].legend(); axes[0].set_ylabel('Normalized Price (100=start)')

for t in water:
    if t in norm.columns:
        axes[1].plot(norm.index, norm[t], label=tickers.get(t,t), lw=1.2)
axes[1].set_title('Water Infrastructure ETFs (Normalized)', fontweight='bold')
axes[1].legend(); axes[1].set_ylabel('Normalized Price (100=start)')
plt.tight_layout()
plt.savefig('output/figures/fig2_prices.png', dpi=150, bbox_inches='tight')
plt.close()

# Rolling correlations
fig, ax = plt.subplots(figsize=(14, 5))
if 'PHO' in returns.columns and 'DBA' in returns.columns:
    rc = returns['PHO'].rolling(60).corr(returns['DBA'])
    ax.plot(rc.index, rc, color='steelblue', lw=1)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('Rolling 60-day Correlation: Water (PHO) vs Agriculture (DBA)', fontweight='bold')
    ax.set_ylabel('Correlation')
plt.tight_layout()
plt.savefig('output/figures/fig3_rolling_corr.png', dpi=150, bbox_inches='tight')
plt.close()

# Full correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(returns.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, linewidths=1)
ax.set_title('Return Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig4_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
