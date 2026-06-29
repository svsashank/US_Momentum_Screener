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

    # Near-miss tolerance: each threshold relaxed by 10% of its own value
    NEAR_MISS_TOL  = 0.10
    NM_MIN_MCAP    = MIN_MCAP    * (1 - NEAR_MISS_TOL)
    NM_MIN_ADV     = MIN_ADV     * (1 - NEAR_MISS_TOL)
    NM_MAX_VOL     = MAX_VOLATILITY * (1 + NEAR_MISS_TOL)
    NM_RSI         = RSI_THRESHOLD  * (1 - NEAR_MISS_TOL)
    NM_SMA_BUFFER  = SMA_BUFFER  * (1 + NEAR_MISS_TOL)   # slightly wider gap allowed
    NM_MAX_HIGH    = MAX_FROM_HIGH  * (1 + NEAR_MISS_TOL)
    NM_CMF         = CMF_THRESHOLD  * (1 - NEAR_MISS_TOL)

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

    # ── Strict filter masks ────────────────────────────────────────────────────
    m_mcap = mcap_row.ge(MIN_MCAP).fillna(False)
    m_adv  = adv_row.ge(MIN_ADV)
    m_vol  = vol_row.le(MAX_VOLATILITY)
    m_rsi  = rsi_row.ge(RSI_THRESHOLD)
    m_sma  = close_row.ge(sma_s_row.mul(1 - SMA_BUFFER))
    m_high = close_row.ge(high52_row.mul(1 - MAX_FROM_HIGH))
    m_cmf  = cmf_row.ge(CMF_THRESHOLD)
    passed = valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & m_high & m_cmf

    # ── Near-miss filter masks (relaxed by 10% of each threshold) ─────────────
    nm_mcap = mcap_row.ge(NM_MIN_MCAP).fillna(False)
    nm_adv  = adv_row.ge(NM_MIN_ADV)
    nm_vol  = vol_row.le(NM_MAX_VOL)
    nm_rsi  = rsi_row.ge(NM_RSI)
    nm_sma  = close_row.ge(sma_s_row.mul(1 - NM_SMA_BUFFER))
    nm_high = close_row.ge(high52_row.mul(1 - NM_MAX_HIGH))
    nm_cmf  = cmf_row.ge(NM_CMF)

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

    # Per-ticker near-miss detection:
    # A ticker is a near-miss candidate if it passes all relaxed filters
    # but fails exactly 1 strict filter (and that 1 failure is within 10% tolerance).
    # We encode which filter it misses as a string label.
    filter_pairs = [
        ('mcap', m_mcap, nm_mcap),
        ('adv',  m_adv,  nm_adv),
        ('vol',  m_vol,  nm_vol),
        ('rsi',  m_rsi,  nm_rsi),
        ('sma',  m_sma,  nm_sma),
        ('high', m_high, nm_high),
        ('cmf',  m_cmf,  nm_cmf),
    ]

    # For each valid ticker: count strict failures and relaxed failures
    # near_miss = passes_all_relaxed AND exactly_1_strict_failure
    is_near_miss_map  = {}
    near_miss_filter_map = {}
    for t in all_tickers:
        if passed[t]:
            is_near_miss_map[t]     = False
            near_miss_filter_map[t] = None
            continue
        strict_fails  = [name for name, sm, _ in filter_pairs if not sm[t]]
        relaxed_fails = [name for name, _, rm in filter_pairs if not rm[t]]
        if len(strict_fails) == 1 and len(relaxed_fails) == 0:
            is_near_miss_map[t]     = True
            near_miss_filter_map[t] = strict_fails[0]
        else:
            is_near_miss_map[t]     = False
            near_miss_filter_map[t] = None

    universe_df = pd.DataFrame({
        'ticker'          : all_tickers,
        'price'           : close_row[all_tickers].values,
        'rank_score'      : rank_row[all_tickers].values,
        'rsi'             : rsi_row[all_tickers].values,
        'volatility_pct'  : vol_row[all_tickers].values * 100,
        'adv_m'           : adv_row[all_tickers].values,
        'mcap_m'          : mcap_row[all_tickers].values,
        'pct_from_high'   : (close_row[all_tickers].values / high52_row[all_tickers].values - 1) * 100,
        'cmf'             : cmf_row[all_tickers].values,
        'sma21'           : sma_s_row[all_tickers].values,
        'sma200'          : sma_l_row[all_tickers].values,
        # Per-filter strict pass flags
        'p_mcap'          : m_mcap[all_tickers].values,
        'p_adv'           : m_adv[all_tickers].values,
        'p_vol'           : m_vol[all_tickers].values,
        'p_rsi'           : m_rsi[all_tickers].values,
        'p_sma'           : m_sma[all_tickers].values,
        'p_high'          : m_high[all_tickers].values,
        'p_cmf'           : m_cmf[all_tickers].values,
        'passes_all'      : passed[all_tickers].values,
        # Near-miss fields
        'is_near_miss'    : [is_near_miss_map[t]     for t in all_tickers],
        'near_miss_filter': [near_miss_filter_map[t] if near_miss_filter_map[t] is not None else None for t in all_tickers],
    }).sort_values('rank_score', ascending=False).reset_index(drop=True)
    universe_df.index += 1

    if not passed.any() and not any(is_near_miss_map.values()):
        return pd.DataFrame(), pd.DataFrame(), universe_df, pd.DataFrame(), rejections, screen_date

    # ── Build Top N and Hold Zone: walk universe by rank ─────────────────────────
    # Include each stock if strict pass OR near-miss (within top 50 by rank).
    # Walk continues to HOLD_ZONE_SIZE to define the anti-whipsaw hold buffer.
    # A higher-ranked near-miss always beats a lower-ranked strict pass.
    HOLD_ZONE_SIZE = config.get('hold_zone_size', 25)

    top_n_rows    = []   # top PORTFOLIO_SIZE eligible stocks
    hold_zone_rows = []  # top HOLD_ZONE_SIZE eligible stocks
    n_promoted    = 0

    for rank_position, row in universe_df.iterrows():
        # rank_position is 1-based (universe_df.index starts at 1)
        if len(hold_zone_rows) >= HOLD_ZONE_SIZE:
            break
        eligible = row['passes_all'] or (row['is_near_miss'] and rank_position <= 50)
        if not eligible:
            continue
        hold_zone_rows.append(row)
        if len(top_n_rows) < PORTFOLIO_SIZE:
            top_n_rows.append(row)
            if row['is_near_miss']:
                n_promoted += 1

    top_n_df      = pd.DataFrame(top_n_rows).reset_index(drop=True)
    top_n_df.index += 1
    hold_zone_df  = pd.DataFrame(hold_zone_rows).reset_index(drop=True)
    hold_zone_df.index += 1

    all_passing = universe_df[universe_df['passes_all']].copy().reset_index(drop=True)
    all_passing.index += 1

    top15 = top_n_df.copy()

    n_strict = len(top15) - n_promoted
    n_cash   = PORTFOLIO_SIZE - len(top15)

    print(f'\n✅ Screen date  : {screen_date.date()}')
    print(f'   Universe     : {len(universe_df)} (valid data)')
    print(f'   Passing      : {len(all_passing)} (strict)')
    print(f'   Near-miss promoted: {n_promoted}')
    print(f'   Hold zone    : {len(hold_zone_df)} (top {HOLD_ZONE_SIZE})')
    print(f'   Cash slots   : {n_cash}')
    print(f'   Top {PORTFOLIO_SIZE}       : {len(top15)}')
    print(f'\n🏆 TOP {len(top15)}:')
    print(top15[['ticker', 'price', 'rank_score', 'rsi', 'adv_m', 'cmf', 'is_near_miss', 'near_miss_filter']].to_string())

    return top15, all_passing, universe_df, hold_zone_df, rejections, screen_date
