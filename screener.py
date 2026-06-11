"""
Momentum Live Screener — GitHub Actions version
S&P 1500 · Fresh fetch each run · Supabase push
"""

import os, json, time, math, warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from supabase import create_client

warnings.filterwarnings('ignore')

# ── Credentials ───────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

# ── Strategy parameters ───────────────────────────────────────────────────────
UNIVERSE_NAME  = 'sp1500'
PORTFOLIO_SIZE = 15
MIN_MCAP_USD   = 500_000_000
MIN_ADV_USD    = 10_000_000
MAX_VOLATILITY = 0.75
RSI_THRESHOLD  = 50
MAX_FROM_HIGH  = 0.25
SMA_SHORT      = 21
SMA_LONG       = 200
RSI_PERIOD     = 14
VOL_LOOKBACK   = 63
ADV_PERIOD     = 63
CMF_PERIOD     = 20
CMF_THRESHOLD  = 0.1

UNIVERSE_FILE  = os.path.join(os.path.dirname(__file__), 'sp1500.json')


# ── Step 1: Load universe ─────────────────────────────────────────────────────
def load_universe():
    with open(UNIVERSE_FILE) as f:
        tickers = json.load(f)
    print(f'✅ Universe: {len(tickers)} tickers')
    return tickers


# ── Step 2: Fetch OHLCV with retries ─────────────────────────────────────────
def fetch_batch_with_retry(batch, start_str, end_str, max_retries=3):
    """Download one batch, retrying up to max_retries times on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                batch, start=start_str, end=end_str,
                auto_adjust=True, progress=False, threads=True,
                timeout=30
            )
            if not data.empty:
                return data
            else:
                print(f'empty (attempt {attempt})', end=' ')
        except Exception as e:
            print(f'⚠ attempt {attempt}: {str(e)[:60]}', end=' ')
            if attempt < max_retries:
                time.sleep(5 * attempt)   # back-off: 5s, 10s, 15s
    return None


def fetch_ohlcv(tickers, lookback_days=400, batch_size=100):
    end_date   = datetime.today().date()
    start_date = end_date - timedelta(days=lookback_days)
    start_str  = start_date.strftime('%Y-%m-%d')
    end_str    = end_date.strftime('%Y-%m-%d')

    print(f'⏳ Fetching OHLCV {start_str} → {end_str} for {len(tickers)} tickers...')

    batches   = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    all_data  = []
    failed    = 0

    for i, batch in enumerate(batches):
        pct = (i + 1) / len(batches) * 100
        print(f'  [{pct:5.1f}%] Batch {i+1}/{len(batches)}...', end=' ', flush=True)

        data = fetch_batch_with_retry(batch, start_str, end_str)
        if data is not None:
            all_data.append(data)
            print('✓')
        else:
            failed += 1
            print('FAILED — skipping')

        # Pause every 5 batches to avoid rate limiting
        if (i + 1) % 5 == 0:
            time.sleep(3)
        else:
            time.sleep(0.5)

    if not all_data:
        raise RuntimeError('No OHLCV data fetched at all — aborting.')

    print(f'\n🔧 Merging {len(all_data)} batches ({failed} failed)...')
    raw = pd.concat(all_data, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.sort_index(inplace=True)

    available = raw['Close'].columns.tolist()
    print(f'✅ OHLCV: {len(available)} stocks, {raw.index[0].date()} → {raw.index[-1].date()}')
    return raw, available


# ── Step 3: Fetch market cap ──────────────────────────────────────────────────
def fetch_mcap(tickers):
    """
    Fetch market cap via yfinance fast_info.
    Falls back gracefully — missing mcap means stock fails the $500M filter,
    which is conservative (won't include unknown small caps).
    """
    print(f'⏳ Fetching market cap for {len(tickers)} stocks...')
    mcap_data = {}

    # Test which method works
    getter = None
    test   = yf.Ticker('AAPL')
    for name, fn in [
        ('fast_info.market_cap', lambda t: t.fast_info.market_cap),
        ('info.marketCap',       lambda t: t.info.get('marketCap')),
        ('shares_x_price',       lambda t: t.fast_info.shares * t.fast_info.last_price),
    ]:
        try:
            val = fn(test)
            if val and val > 1e9:
                print(f'   Method: {name} (AAPL: ${val/1e9:.0f}B)')
                getter = fn
                break
        except:
            continue

    if getter is None:
        print('⚠ All MCap methods failed — MCap filter disabled for this run')
        return {}

    for i, ticker in enumerate(tickers):
        for attempt in range(3):
            try:
                val = getter(yf.Ticker(ticker))
                if val and val > 0:
                    mcap_data[ticker] = float(val)
                break
            except:
                time.sleep(1)
        if (i + 1) % 200 == 0:
            print(f'   {i+1}/{len(tickers)} done — {len(mcap_data)} found')
            time.sleep(1)

    pct = len(mcap_data) / len(tickers) * 100 if tickers else 0
    print(f'✅ MCap: {len(mcap_data)}/{len(tickers)} ({pct:.0f}% coverage)')
    return mcap_data


# ── Step 4: Compute indicators ────────────────────────────────────────────────
def compute_indicators(raw_data, mcap_data, screen_tickers):
    available = [t for t in screen_tickers if t in raw_data['Close'].columns]
    print(f'   {len(available)} tickers in data ({len(screen_tickers)-len(available)} missing)')

    close  = raw_data['Close'][available].copy().astype(float)
    volume = raw_data['Volume'][available].copy().astype(float)
    high   = raw_data['High'][available].copy().astype(float)
    low    = raw_data['Low'][available].copy().astype(float)
    print(f'   Shape: {close.shape}')

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
    adv = ((volume * close) / 1e6).rolling(ADV_PERIOD, min_periods=ADV_PERIOD).mean()
    print('✓')

    print('   [6/8] 52W high...', end=' ', flush=True)
    high_52w = high.rolling(252, min_periods=100).max()
    print('✓')

    print('   [7/8] CMF...', end=' ', flush=True)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    cmf = mfv.rolling(CMF_PERIOD, min_periods=CMF_PERIOD).sum() / \
          volume.rolling(CMF_PERIOD, min_periods=CMF_PERIOD).sum().replace(0, np.nan)
    print('✓')

    print('   [8/8] MCap matrix...', end=' ', flush=True)
    mcap_arr = np.array([float(mcap_data.get(t, 0)) for t in close.columns], dtype=float)
    mcap_arr[mcap_arr == 0] = np.nan
    mcap_mat = pd.DataFrame(
        np.tile(mcap_arr[np.newaxis, :] / 1e6, (len(close), 1)),
        index=close.index, columns=close.columns
    )
    for chk in ['AAPL', 'MSFT', 'NVDA']:
        if chk in close.columns:
            val = mcap_mat[chk].iloc[-1]
            if not np.isnan(val):
                print(f'\n     {chk}: ${val:,.0f}M', end=' ')
    print('\n✓')

    return dict(
        close=close, volume=volume, high=high, low=low,
        sma_short=sma_short, sma_long=sma_long, rank_score=rank_score,
        rsi=rsi, ann_vol=ann_vol, adv=adv, high_52w=high_52w, cmf=cmf, mcap=mcap_mat
    )


# ── Step 5: Screen ────────────────────────────────────────────────────────────
def find_screen_date(ind):
    # Anchor on a highly-liquid mega-cap that is virtually guaranteed to have
    # data for any completed trading day (avoids picking a stale date just
    # because some smaller/illiquid tickers lag in yfinance's batch response).
    anchors = [t for t in ('AAPL', 'MSFT', 'SPY') if t in ind['close'].columns]
    if anchors:
        anchor_close = ind['close'][anchors[0]]
        valid_dates = anchor_close.dropna().index
        if len(valid_dates):
            return valid_dates[-1]

    # Fallback: relaxed coverage threshold (50% instead of 80%)
    non_null  = ind['close'].notna().sum(axis=1)
    threshold = len(ind['close'].columns) * 0.50
    return non_null[non_null >= threshold].index[-1]


def run_screen(ind):
    screen_date = find_screen_date(ind)
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
    m_mcap = mcap_row.ge(MIN_MCAP_USD / 1e6).fillna(False)
    m_adv  = adv_row.ge(MIN_ADV_USD / 1e6)
    m_vol  = vol_row.le(MAX_VOLATILITY)
    m_rsi  = rsi_row.ge(RSI_THRESHOLD)
    m_sma  = close_row.gt(sma_s_row)
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

    if not passed.any():
        return pd.DataFrame(), pd.DataFrame(), rejections, screen_date

    pt     = passed[passed].index.tolist()
    result = pd.DataFrame({
        'ticker'        : pt,
        'price'         : close_row[pt].values,
        'rank_score'    : rank_row[pt].values,
        'rsi'           : rsi_row[pt].values,
        'volatility_pct': vol_row[pt].values * 100,
        'adv_m'         : adv_row[pt].values,
        'mcap_m'        : mcap_row[pt].values,
        'pct_from_high' : (close_row[pt].values / high52_row[pt].values - 1) * 100,
        'cmf'           : cmf_row[pt].values,
        'sma21'         : sma_s_row[pt].values,
        'sma200'        : sma_l_row[pt].values,
    }).dropna(subset=['rank_score']).sort_values('rank_score', ascending=False).reset_index(drop=True)
    result.index += 1

    top15       = result.head(PORTFOLIO_SIZE).copy()
    all_passing = result.copy()

    print(f'\n✅ Screen date  : {screen_date.date()}')
    print(f'   Passing      : {len(all_passing)}')
    print(f'   Top {PORTFOLIO_SIZE}       : {len(top15)}')
    print(f'\n🏆 TOP {len(top15)}:')
    print(top15[['ticker', 'price', 'rank_score', 'rsi', 'adv_m', 'cmf']].to_string())

    return top15, all_passing, rejections, screen_date


# ── Step 6: Push to Supabase ──────────────────────────────────────────────────
def clean(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, np.integer):  return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return val

def to_records(df):
    return [{k: clean(v) for k, v in row.items()} for _, row in df.iterrows()]

def push(supabase, top15, all_passing, rejections, screen_date):
    row = {
        'run_date'   : str(screen_date.date()),
        'universe'   : UNIVERSE_NAME,
        'top15'      : to_records(top15.reset_index())       if not top15.empty       else [],
        'all_passing': to_records(all_passing.reset_index()) if not all_passing.empty else [],
        'filters'    : {
            'universe': UNIVERSE_NAME, 'portfolio_size': PORTFOLIO_SIZE,
            'min_mcap_usd': MIN_MCAP_USD, 'min_adv_usd': MIN_ADV_USD,
            'max_vol': MAX_VOLATILITY, 'rsi_threshold': RSI_THRESHOLD,
            'max_from_high': MAX_FROM_HIGH, 'sma_short': SMA_SHORT,
            'sma_long': SMA_LONG, 'cmf_period': CMF_PERIOD, 'cmf_threshold': CMF_THRESHOLD,
            'rejections': rejections,
        },
        'run_status' : 'complete',
        'triggered_at': datetime.utcnow().isoformat(),
    }
    resp   = supabase.table('screen_runs').insert(row).execute()
    run_id = resp.data[0]['id'] if resp.data else None
    print(f'✅ screen_runs → id: {run_id}')

    if not all_passing.empty:
        top_set     = set(top15['ticker']) if not top15.empty else set()
        top_idx_map = {r['ticker']: int(i) for i, r in top15.reset_index().iterrows()}
        rows = []
        for _, r in all_passing.iterrows():
            t = r['ticker']
            rows.append({
                'ticker'    : t,
                'price'     : clean(r['price']),
                'sma21'     : clean(r['sma21']),
                'sma200'    : clean(r['sma200']),
                'rank_score': clean(r['rank_score']),
                'rsi14'     : clean(r['rsi']),
                'adv20'     : clean(r['adv_m']),
                'ann_vol'   : clean(r['volatility_pct']),
                'cmf'       : clean(r['cmf']),
                'high52w'   : clean(r['price'] / (1 + r['pct_from_high']/100)) if r['pct_from_high'] is not None else None,
                'passes_all': True,
                'in_top15'  : t in top_set,
                'top15_rank': top_idx_map.get(t),
                'updated_at': datetime.utcnow().isoformat(),
            })
        total = 0
        for i in range(0, len(rows), 200):
            supabase.table('stock_snapshots').upsert(rows[i:i+200], on_conflict='ticker').execute()
            total += min(200, len(rows) - i)
        print(f'✅ stock_snapshots → {total} upserted')

    return run_id


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print('='*60)
    print('  MOMENTUM LIVE SCREENER — GitHub Actions')
    print(f'  {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
    print('='*60)

    supabase             = create_client(SUPABASE_URL, SUPABASE_KEY)
    tickers              = load_universe()
    raw, available       = fetch_ohlcv(tickers)
    screen_tickers       = [t for t in tickers if t in available]
    mcap                 = fetch_mcap(available)

    print('\n⏳ Computing indicators...')
    ind                  = compute_indicators(raw, mcap, screen_tickers)

    print('\n⏳ Running screen...')
    top15, all_passing, rejections, screen_date = run_screen(ind)

    print('\n📤 Pushing to Supabase...')
    run_id = push(supabase, top15, all_passing, rejections, screen_date)

    print(f'\n✅ Done in {(time.time()-t0)/60:.1f} min — run_id: {run_id}')

if __name__ == '__main__':
    main()
