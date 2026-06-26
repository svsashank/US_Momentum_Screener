"""
core/screener_engine.py — Shared, market-agnostic screening and ranking logic.

Used by both US and India live screeners. All thresholds come from `config`,
already expressed in the correct unit for that market (USD millions for US,
INR crore for India) — no unit conversion happens here.

Required config keys:
    anchors          list of ticker symbols to try (in order) for anchoring
                     the screen date, e.g. ['AAPL','MSFT','SPY'] or
                     ['^NSEI','RELIANCE.NS','TCS.NS']
    min_mcap         minimum market cap (same unit as mcap matrix)
    min_adv          minimum ADV (same unit as adv matrix)
    max_volatility
    rsi_threshold
    max_from_high
    cmf_threshold
    portfolio_size
    sma_buffer       price must be >= SMA21 * (1 - sma_buffer), default 0.05
"""

import pandas as pd


def find_screen_date(ind, anchors):
    """
    Anchor on the first available highly-liquid ticker that has data for the
    most recent trading day — avoids picking a stale date just because some
    smaller/illiquid tickers lag in yfinance's batch response.
    """
    available_anchors = [t for t in anchors if t in ind['close'].columns]
    if available_anchors:
        anchor_close = ind['close'][available_anchors[0]]
        valid_dates = anchor_close.dropna().index
        if len(valid_dates):
            return valid_dates[-1]

    # Fallback: relaxed coverage threshold (50% instead of 80%)
    non_null  = ind['close'].notna().sum(axis=1)
    threshold = len(ind['close'].columns) * 0.50
    return non_null[non_null >= threshold].index[-1]


def run_screen(ind, config):
    MIN_MCAP       = config['min_mcap']
    MIN_ADV        = config['min_adv']
    MAX_VOLATILITY = config['max_volatility']
    RSI_THRESHOLD  = config['rsi_threshold']
    MAX_FROM_HIGH  = config['max_from_high']
    CMF_THRESHOLD  = config['cmf_threshold']
    PORTFOLIO_SIZE = config['portfolio_size']
    SMA_BUFFER     = config.get('sma_buffer', 0.05)   # price must be >= SMA21 × (1 - buffer)
    anchors        = config['anchors']

    screen_date = find_screen_date(ind, anchors)
    idx = ind['close'].index.get_indexer([screen_date], method='ffill')[0]

    close_row  = ind['close'].iloc[idx]
    sma_s_row  = ind['sma_short'].iloc[idx]
    sma_l_row  = ind['sma_long'].iloc[idx]
    rank_row   = ind['rank_score'].iloc[idx]
    rsi_row    = ind['rsi'].iloc[idx]
    vol_row    = ind['ann_vol'].iloc[idx]
    adv_row    = ind['adv'].iloc[idx]
    high52_row = ind['high_52w'].iloc[idx]
    cmf_row    = ind['cmf'].iloc[idx]
    mcap_row   = ind['mcap'].iloc[idx]

    valid  = close_row.notna() & sma_l_row.notna() & sma_s_row.notna()
    print(f'   [diag] screen_date={screen_date.date()}, valid_after_ffill={int(valid.sum())}/{len(valid)}, '
          f'close_nan={int(close_row.isna().sum())}, sma200_nan={int(sma_l_row.isna().sum())}, '
          f'sma21_nan={int(sma_s_row.isna().sum())}')

    m_mcap = mcap_row.ge(MIN_MCAP).fillna(False)
    m_adv  = adv_row.ge(MIN_ADV)
    m_vol  = vol_row.le(MAX_VOLATILITY)
    m_rsi  = rsi_row.ge(RSI_THRESHOLD)
    m_sma  = close_row.ge(sma_s_row.mul(1 - SMA_BUFFER))
    m_high = close_row.ge(high52_row.mul(1 - MAX_FROM_HIGH))
    m_cmf  = cmf_row.ge(CMF_THRESHOLD)
    passed = valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & m_high & m_cmf

    rejections = {
        'no_data'   : int((~valid).sum()),
        'mcap'      : int((valid & ~m_mcap).sum()),
        'adv'       : int((valid & m_mcap & ~m_adv).sum()),
        'volatility': int((valid & m_mcap & m_adv & ~m_vol).sum()),
        'rsi'       : int((valid & m_mcap & m_adv & m_vol & ~m_rsi).sum()),
        'sma'       : int((valid & m_mcap & m_adv & m_vol & m_rsi & ~m_sma).sum()),
        'high52w'   : int((valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & ~m_high).sum()),
        'cmf'       : int((valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & m_high & ~m_cmf).sum()),
    }

    print(f'\n── Rejection waterfall ──────────')
    for k, v in rejections.items():
        print(f'   {k:<12}: {v}')

    # ── Full universe: all valid tickers with indicators + per-filter pass flags ──
    all_tickers = valid[valid].index.tolist()
    universe_df = pd.DataFrame({
        'ticker'        : all_tickers,
        'price'         : close_row[all_tickers].values,
        'rank_score'    : rank_row[all_tickers].values,
        'rsi'           : rsi_row[all_tickers].values,
        'volatility_pct': vol_row[all_tickers].values * 100,
        'adv_m'         : adv_row[all_tickers].values,
        'mcap_m'        : mcap_row[all_tickers].values,
        'pct_from_high' : (close_row[all_tickers].values / high52_row[all_tickers].values - 1) * 100,
        'cmf'           : cmf_row[all_tickers].values,
        'sma21'         : sma_s_row[all_tickers].values,
        'sma200'        : sma_l_row[all_tickers].values,
        # Per-filter pass flags (cumulative waterfall order)
        'p_mcap'        : m_mcap[all_tickers].values,
        'p_adv'         : m_adv[all_tickers].values,
        'p_vol'         : m_vol[all_tickers].values,
        'p_rsi'         : m_rsi[all_tickers].values,
        'p_sma'         : m_sma[all_tickers].values,
        'p_high'        : m_high[all_tickers].values,
        'p_cmf'         : m_cmf[all_tickers].values,
        'passes_all'    : passed[all_tickers].values,
    }).sort_values('rank_score', ascending=False).reset_index(drop=True)
    universe_df.index += 1

    if not passed.any():
        return pd.DataFrame(), pd.DataFrame(), universe_df, rejections, screen_date

    pt     = passed[passed].index.tolist()
    result = universe_df[universe_df['ticker'].isin(pt)].copy().reset_index(drop=True)
    result.index += 1

    top15       = result.head(PORTFOLIO_SIZE).copy()
    all_passing = result.copy()

    print(f'\n✅ Screen date  : {screen_date.date()}')
    print(f'   Universe     : {len(universe_df)} (valid data)')
    print(f'   Passing      : {len(all_passing)}')
    print(f'   Top {PORTFOLIO_SIZE}       : {len(top15)}')
    print(f'\n🏆 TOP {len(top15)}:')
    print(top15[['ticker', 'price', 'rank_score', 'rsi', 'adv_m', 'cmf']].to_string())

    return top15, all_passing, universe_df, rejections, screen_date
