"""
Momentum Live Screener — GitHub Actions version
S&P 1500 · Fresh fetch each run · Supabase push
Translated from Momentum_Live_Screener_v1.ipynb (Cells 1–10)
"""

import os, json, time, math, warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from supabase import create_client

warnings.filterwarnings('ignore')

# ── Credentials (from GitHub Actions secrets) ─────────────────────────────────
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

# ── Strategy parameters ────────────────────────────────────────────────────────
UNIVERSE_NAME  = 'sp1500'
PORTFOLIO_SIZE = 15

MIN_MCAP_USD   = 500_000_000    # $500M
MIN_ADV_USD    = 10_000_000     # $10M/day
MAX_VOLATILITY = 0.75           # 75% ann. vol
RSI_THRESHOLD  = 50
MAX_FROM_HIGH  = 0.25           # within 25% of 52W high

SMA_SHORT      = 21
SMA_LONG       = 200
RSI_PERIOD     = 14
VOL_LOOKBACK   = 63
ADV_PERIOD     = 63

# ── Universe ticker list (committed to repo) ───────────────────────────────────
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), 'sp1500.json')


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load universe
# ──────────────────────────────────────────────────────────────────────────────
def load_universe():
    with open(UNIVERSE_FILE) as f:
        tickers = json.load(f)
    print(f'✅ Universe loaded: {len(tickers)} tickers')
    return tickers


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Fetch OHLCV (fresh each run, ~260 trading days)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_ohlcv(tickers, lookback_days=400, batch_size=100):
    end_date   = datetime.today().date()
    start_date = end_date - timedelta(days=lookback_days)
    start_str  = start_date.strftime('%Y-%m-%d')
    end_str    = end_date.strftime('%Y-%m-%d')

    print(f'⏳ Fetching OHLCV: {start_str} → {end_str} for {len(tickers)} tickers...')

    batches  = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    all_data = []

    for i, batch in enumerate(batches):
        pct = (i + 1) / len(batches) * 100
        print(f'  [{pct:5.1f}%] Batch {i+1}/{len(batches)}...', end=' ', flush=True)
        try:
            data = yf.download(
                batch, start=start_str, end=end_str,
                auto_adjust=True, progress=False, threads=True
            )
            if not data.empty:
                all_data.append(data)
                print('✓')
            else:
                print('empty')
        except Exception as e:
            print(f'⚠ ({str(e)[:60]})')
        # Rate limit courtesy pause
        time.sleep(0.5 if (i + 1) % 5 != 0 else 2)

    if not all_data:
        raise RuntimeError('No OHLCV data fetched — aborting.')

    raw = pd.concat(all_data, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.sort_index(inplace=True)

    available = raw['Close'].columns.tolist()
    print(f'✅ OHLCV fetched: {len(available)} stocks, {raw.index[0].date()} → {raw.index[-1].date()}')
    return raw, available


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Fetch market cap
# ──────────────────────────────────────────────────────────────────────────────
def fetch_mcap(tickers):
    print(f'⏳ Fetching market cap for {len(tickers)} stocks...')
    mcap_data = {}

    # Detect working yfinance method
    test = yf.Ticker('AAPL')
    getter = None
    for method_name, fn in [
        ('fast_info.market_cap', lambda t: t.fast_info.market_cap),
        ('info.marketCap',       lambda t: t.info.get('marketCap')),
        ('shares_x_price',       lambda t: t.fast_info.shares * t.fast_info.last_price),
    ]:
        try:
            val = fn(test)
            if val and val > 1e9:
                print(f'   MCap method: {method_name} (AAPL: ${val/1e9:.0f}B)')
                getter = fn
                break
        except:
            continue

    if getter is None:
        print('⚠ All MCap methods failed — MCap filter will pass all stocks')
        return {}

    for i, ticker in enumerate(tickers):
        try:
            val = getter(yf.Ticker(ticker))
            if val and val > 0:
                mcap_data[ticker] = float(val)
        except:
            pass
        if (i + 1) % 200 == 0:
            print(f'   {i+1}/{len(tickers)} — found {len(mcap_data)}')
            time.sleep(0.5)

    pct = len(mcap_data) / len(tickers) * 100
    print(f'✅ MCap data: {len(mcap_data)}/{len(tickers)} ({pct:.1f}% coverage)')
    return mcap_data


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Compute indicators (vectorised)
# ──────────────────────────────────────────────────────────────────────────────
def compute_indicators(raw_data, mcap_data, screen_tickers):
    available = [t for t in screen_tickers if t in raw_data['Close'].columns]
    missing   = len(screen_tickers) - len(available)
    if missing > 0:
        print(f'   ⚠ {missing} universe tickers not in data (likely delisted)')

    close  = raw_data['Close'][available].copy().astype(float)
    volume = raw_data['Volume'][available].copy().astype(float)
    high   = raw_data['High'][available].copy().astype(float)

    print(f'   Shape: {close.shape}')

    print(f'   [1/7] SMA{SMA_SHORT} / SMA{SMA_LONG}...', end=' ', flush=True)
    sma_short_df = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    sma_long_df  = close.rolling(SMA_LONG,  min_periods=SMA_LONG).mean()
    print('✓')

    print(f'   [2/7] Rank score (SMA{SMA_SHORT}/SMA{SMA_LONG})...', end=' ', flush=True)
    rank_score = sma_short_df / sma_long_df.replace(0, np.nan)
    print('✓')

    print('   [3/7] RSI 14...', end=' ', flush=True)
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rsi      = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
    print('✓')

    print('   [4/7] Annualised volatility...', end=' ', flush=True)
    log_ret = np.log(close / close.shift(1))
    ann_vol = log_ret.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std() * np.sqrt(252)
    print('✓')

    print('   [5/7] ADV...', end=' ', flush=True)
    daily_value = (volume * close) / 1e6   # USD millions/day
    adv         = daily_value.rolling(ADV_PERIOD, min_periods=ADV_PERIOD).mean()
    print('✓')

    print('   [6/7] 52W rolling high...', end=' ', flush=True)
    high_52w = high.rolling(252, min_periods=100).max()
    print('✓')

    print('   [7/7] Market cap matrix...', end=' ', flush=True)
    mcap_arr = np.array(
        [float(mcap_data.get(t, 0)) for t in close.columns],
        dtype=float
    )
    mcap_arr[mcap_arr == 0] = np.nan
    # numpy broadcasting — avoids pandas multiply() alignment bug
    mcap_mat = pd.DataFrame(
        np.tile(mcap_arr[np.newaxis, :] / 1e6, (len(close), 1)),
        index=close.index, columns=close.columns
    )
    for chk in ['AAPL', 'MSFT', 'NVDA']:
        if chk in close.columns:
            val = mcap_mat[chk].iloc[-1]
            if not np.isnan(val):
                print(f'\n   {chk}: ${val:,.0f}M', end=' ')
    print('\n✓')

    return {
        'close'      : close,
        'volume'     : volume,
        'high'       : high,
        'sma_short'  : sma_short_df,
        'sma_long'   : sma_long_df,
        'rank_score' : rank_score,
        'rsi'        : rsi,
        'ann_vol'    : ann_vol,
        'adv'        : adv,
        'high_52w'   : high_52w,
        'mcap'       : mcap_mat,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Screen
# ──────────────────────────────────────────────────────────────────────────────
def find_screen_date(indicators):
    """Last date where >= 80% of stocks have price data (avoids yfinance 1-day lag)."""
    non_null  = indicators['close'].notna().sum(axis=1)
    threshold = len(indicators['close'].columns) * 0.80
    return non_null[non_null >= threshold].index[-1]


def run_screen(indicators):
    screen_date = find_screen_date(indicators)
    idx = indicators['close'].index.get_indexer([screen_date], method='ffill')[0]

    close_row  = indicators['close'].iloc[idx]
    sma_s_row  = indicators['sma_short'].iloc[idx]
    sma_l_row  = indicators['sma_long'].iloc[idx]
    rank_row   = indicators['rank_score'].iloc[idx]
    rsi_row    = indicators['rsi'].iloc[idx]
    vol_row    = indicators['ann_vol'].iloc[idx]
    adv_row    = indicators['adv'].iloc[idx]
    high52_row = indicators['high_52w'].iloc[idx]
    mcap_row   = indicators['mcap'].iloc[idx]

    min_mcap_m = MIN_MCAP_USD / 1e6
    min_adv_m  = MIN_ADV_USD  / 1e6

    valid  = close_row.notna() & sma_l_row.notna() & sma_s_row.notna()
    m_mcap = mcap_row.ge(min_mcap_m).fillna(False)
    m_adv  = adv_row.ge(min_adv_m)
    m_vol  = vol_row.le(MAX_VOLATILITY)
    m_rsi  = rsi_row.ge(RSI_THRESHOLD)
    m_sma  = close_row.gt(sma_s_row)
    m_high = close_row.ge(high52_row.mul(1 - MAX_FROM_HIGH))
    passed = valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & m_high

    rejections = {
        'no_data'   : int((~valid).sum()),
        'mcap'      : int((valid & ~m_mcap).sum()),
        'adv'       : int((valid & m_mcap & ~m_adv).sum()),
        'volatility': int((valid & m_mcap & m_adv & ~m_vol).sum()),
        'rsi'       : int((valid & m_mcap & m_adv & m_vol & ~m_rsi).sum()),
        'sma'       : int((valid & m_mcap & m_adv & m_vol & m_rsi & ~m_sma).sum()),
        'high52w'   : int((valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & ~m_high).sum()),
    }

    print(f'\n── Rejection waterfall ──')
    for k, v in rejections.items():
        print(f'   {k:<12}: {v}')

    if not passed.any():
        return pd.DataFrame(), pd.DataFrame(), rejections, screen_date

    pt = passed[passed].index.tolist()
    result = pd.DataFrame({
        'ticker'         : pt,
        'price'          : close_row[pt].values,
        'rank_score'     : rank_row[pt].values,
        'rsi'            : rsi_row[pt].values,
        'volatility_pct' : vol_row[pt].values * 100,
        'adv_m'          : adv_row[pt].values,
        'mcap_m'         : mcap_row[pt].values,
        'pct_from_high'  : (close_row[pt].values / high52_row[pt].values - 1) * 100,
        'sma21'          : sma_s_row[pt].values,
        'sma200'         : sma_l_row[pt].values,
    })
    result = result.dropna(subset=['rank_score'])
    result = result.sort_values('rank_score', ascending=False).reset_index(drop=True)
    result.index += 1

    top15      = result.head(PORTFOLIO_SIZE).copy()
    all_passing = result.copy()

    print(f'\n✅ Screen date: {screen_date.date()}')
    print(f'   Passing: {len(all_passing)} | Top {PORTFOLIO_SIZE}: {len(top15)}')
    print(f'\n🏆 TOP {len(top15)}:')
    print(top15[['ticker', 'price', 'rank_score', 'rsi', 'adv_m']].to_string())

    return top15, all_passing, rejections, screen_date


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Supabase push
# ──────────────────────────────────────────────────────────────────────────────
def clean_for_json(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    return val


def df_to_records(df):
    return [{k: clean_for_json(v) for k, v in row.items()} for _, row in df.iterrows()]


def push_to_supabase(supabase, top15, all_passing, rejections, screen_date):
    # screen_runs row
    filters_payload = {
        'universe'      : UNIVERSE_NAME,
        'portfolio_size': PORTFOLIO_SIZE,
        'min_mcap_usd'  : MIN_MCAP_USD,
        'min_adv_usd'   : MIN_ADV_USD,
        'max_vol'       : MAX_VOLATILITY,
        'rsi_threshold' : RSI_THRESHOLD,
        'max_from_high' : MAX_FROM_HIGH,
        'sma_short'     : SMA_SHORT,
        'sma_long'      : SMA_LONG,
        'rejections'    : rejections,
    }
    row = {
        'run_date'   : str(screen_date.date()),
        'universe'   : UNIVERSE_NAME,
        'top15'      : df_to_records(top15.reset_index())      if not top15.empty       else [],
        'all_passing': df_to_records(all_passing.reset_index()) if not all_passing.empty else [],
        'filters'    : filters_payload,
        'run_status' : 'complete',
    }
    resp   = supabase.table('screen_runs').insert(row).execute()
    run_id = resp.data[0]['id'] if resp.data else None
    print(f'✅ screen_runs → inserted (id: {run_id})')

    # stock_snapshots upsert
    if not all_passing.empty:
        top_tickers  = set(top15['ticker'].tolist()) if not top15.empty else set()
        top_idx_map  = {r['ticker']: int(i) for i, r in top15.reset_index().iterrows()} if not top15.empty else {}

        rows = []
        for _, r in all_passing.iterrows():
            t = r['ticker']
            rows.append({
                'ticker'     : t,
                'price'      : clean_for_json(r['price']),
                'sma21'      : clean_for_json(r['sma21']),
                'sma200'     : clean_for_json(r['sma200']),
                'rank_score' : clean_for_json(r['rank_score']),
                'rsi14'      : clean_for_json(r['rsi']),
                'adv20'      : clean_for_json(r['adv_m']),
                'ann_vol'    : clean_for_json(r['volatility_pct']),
                'high52w'    : clean_for_json(
                    r['price'] / (1 + r['pct_from_high'] / 100)
                    if r['pct_from_high'] is not None else None
                ),
                'passes_all' : True,
                'in_top15'   : t in top_tickers,
                'top15_rank' : top_idx_map.get(t),
                'updated_at' : datetime.utcnow().isoformat(),
            })

        BATCH = 200
        total = 0
        for i in range(0, len(rows), BATCH):
            supabase.table('stock_snapshots').upsert(
                rows[i:i+BATCH], on_conflict='ticker'
            ).execute()
            total += len(rows[i:i+BATCH])
        print(f'✅ stock_snapshots → {total} rows upserted')

    return run_id


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print('=' * 60)
    print('  MOMENTUM LIVE SCREENER — GitHub Actions')
    print(f'  {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
    print('=' * 60)

    # Init Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print('✅ Supabase connected')

    # Run pipeline
    tickers              = load_universe()
    raw_data, available  = fetch_ohlcv(tickers)
    screen_tickers       = [t for t in tickers if t in available]
    mcap_data            = fetch_mcap(available)

    print('\n⏳ Computing indicators...')
    indicators           = compute_indicators(raw_data, mcap_data, screen_tickers)

    print('\n⏳ Running screen...')
    top15, all_passing, rejections, screen_date = run_screen(indicators)

    print('\n📤 Pushing to Supabase...')
    run_id = push_to_supabase(supabase, top15, all_passing, rejections, screen_date)

    elapsed = (time.time() - start_time) / 60
    print(f'\n✅ Done in {elapsed:.1f} min — run_id: {run_id}')


if __name__ == '__main__':
    main()
