"""
core/indicators.py — Shared, market-agnostic indicator computation.

Used by both US (S&P 1500) and India (NSE) live screeners, and intended for
reuse by the universal backtest engine. All market-specific behaviour is
driven by the `config` dict passed in — no hardcoded tickers, currencies, or
thresholds.

Required config keys:
    sma_short, sma_long, rsi_period, vol_lookback, adv_period,
    cmf_period, adv_divisor

`adv_divisor` converts (volume * close) into the unit used for the ADV
threshold:
    - US:    1e6  (USD millions)
    - India: 1e7  (INR crore)

mcap_data: dict {ticker: market_cap_in_same_unit_as_adv} OR a precomputed
pandas DataFrame (same shape as `close`) — pass whichever is more natural for
the calling screener. If a dict is given, it's broadcast across all dates
(US-style live mcap snapshot). If a DataFrame is given, it's used as-is
(NSE-style shares_outstanding * close, which varies by date).
"""

import numpy as np
import pandas as pd


def compute_indicators(raw_data, mcap_data, screen_tickers, config):
    """
    raw_data: dict-like with 'Close', 'Volume', 'High', 'Low' DataFrames
              (as returned by yf.download for multiple tickers)
    mcap_data: dict {ticker: mcap} for a broadcast snapshot, OR a DataFrame
               of the same shape as `close` for a per-date mcap matrix
    screen_tickers: list of tickers to compute indicators for
    config: dict of strategy parameters (see module docstring)

    Returns a dict of DataFrames: close, volume, high, low, sma_short,
    sma_long, rank_score, rsi, ann_vol, adv, high_52w, cmf, mcap
    """
    SMA_SHORT    = config['sma_short']
    SMA_LONG     = config['sma_long']
    RSI_PERIOD   = config['rsi_period']
    VOL_LOOKBACK = config['vol_lookback']
    ADV_PERIOD   = config['adv_period']
    CMF_PERIOD   = config['cmf_period']
    ADV_DIVISOR  = config['adv_divisor']

    available = [t for t in screen_tickers if t in raw_data['Close'].columns]
    print(f'   {len(available)} tickers in data ({len(screen_tickers)-len(available)} missing)')

    close  = raw_data['Close'][available].copy().astype(float)
    volume = raw_data['Volume'][available].copy().astype(float)
    high   = raw_data['High'][available].copy().astype(float)
    low    = raw_data['Low'][available].copy().astype(float)
    print(f'   Shape: {close.shape}')

    # Forward-fill up to 3 trading days: some tickers lag behind mega-caps
    # in yfinance's batch response, which would otherwise show as NaN on the
    # anchor screen date despite having ample trading history. Volume is
    # filled with 0 (no trading that day) so ADV/CMF aren't inflated.
    close  = close.ffill(limit=3)
    high   = high.ffill(limit=3)
    low    = low.ffill(limit=3)
    volume = volume.fillna(0)

    print('   [1/8] SMA21 / SMA200...', end=' ', flush=True)
    sma_short = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    sma_long  = close.rolling(SMA_LONG,  min_periods=SMA_LONG).mean()
    print('✓')

    print('   [2/8] Rank score...', end=' ', flush=True)
    rank_score = sma_short / sma_long.replace(0, np.nan)
    print('✓')

    print('   [3/8] RSI 14...', end=' ', flush=True)
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    rsi      = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
    print('✓')

    print('   [4/8] Annualised volatility...', end=' ', flush=True)
    log_ret = np.log(close / close.shift(1))
    ann_vol = log_ret.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std() * np.sqrt(252)
    print('✓')

    print('   [5/8] ADV...', end=' ', flush=True)
    adv = ((volume * close) / ADV_DIVISOR).rolling(ADV_PERIOD, min_periods=ADV_PERIOD).mean()
    print('✓')

    print('   [6/8] 52W high...', end=' ', flush=True)
    high_52w = high.rolling(252, min_periods=100).max()
    print('✓')

    print('   [7/8] CMF...', end=' ', flush=True)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    # Circuit/flat days (high==low): MFM is undefined by the standard formula.
    # On these days there's no intra-day range, but direction is unambiguous —
    # an upper-circuit is maximal buying pressure (+1), lower-circuit is
    # maximal selling pressure (-1), unchanged is neutral (0). This keeps the
    # value in the same [-1,1] scale as normal MFM so the threshold stays meaningful.
    circuit_mfm = np.sign(close - close.shift(1))
    mfm = mfm.where(mfm.notna(), circuit_mfm)
    mfv = mfm * volume
    cmf = mfv.rolling(CMF_PERIOD, min_periods=CMF_PERIOD).sum() / \
          volume.rolling(CMF_PERIOD, min_periods=CMF_PERIOD).sum().replace(0, np.nan)
    print('✓')

    print('   [8/8] MCap matrix...', end=' ', flush=True)
    if isinstance(mcap_data, pd.DataFrame):
        # Per-date mcap matrix already computed by caller (e.g. shares * close)
        mcap_mat = mcap_data.reindex(columns=close.columns)
    else:
        # Broadcast snapshot dict across all dates
        mcap_arr = np.array([float(mcap_data.get(t, 0)) for t in close.columns], dtype=float)
        mcap_arr[mcap_arr == 0] = np.nan
        mcap_mat = pd.DataFrame(
            np.tile(mcap_arr[np.newaxis, :], (len(close), 1)),
            index=close.index, columns=close.columns
        )
    print('✓')

    return dict(
        close=close, volume=volume, high=high, low=low,
        sma_short=sma_short, sma_long=sma_long, rank_score=rank_score,
        rsi=rsi, ann_vol=ann_vol, adv=adv, high_52w=high_52w, cmf=cmf, mcap=mcap_mat
    )
