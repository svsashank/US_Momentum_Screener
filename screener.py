"""
Momentum Live Screener — GitHub Actions version
S&P 1500 · Fresh fetch each run · Supabase push
Uses shared core/ modules (indicators, screener_engine, data_fetcher).
"""

import os, json, time, math, warnings
import numpy as np
import yfinance as yf
from datetime import datetime
from supabase import create_client

from core.data_fetcher import fetch_ohlcv
from core.indicators import compute_indicators
from core.screener_engine import run_screen

warnings.filterwarnings('ignore')

# ── Credentials ───────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

# ── Strategy parameters (config.json + optional GUI overrides via SCREEN_PARAMS)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

# GUI sends overrides as a single JSON string (avoids dispatch 10-property limit)
_params_raw = os.environ.get('SCREEN_PARAMS', '')
if _params_raw:
    try:
        _decoded = json.loads(_params_raw)
        _params = json.loads(_decoded) if isinstance(_decoded, str) else _decoded
        _int_keys = ['portfolio_size', 'sma_short', 'sma_long', 'rsi_period',
                     'vol_lookback', 'adv_period', 'cmf_period']
        for k, v in _params.items():
            if v is not None and v != '' and k in CONFIG:
                CONFIG[k] = int(v) if k in _int_keys else float(v)
        print(f'⚙ SCREEN_PARAMS applied: {list(_params.keys())}')
    except Exception as e:
        print(f'⚠ SCREEN_PARAMS parse error: {e} — using config.json defaults')

UNIVERSE_NAME  = CONFIG['universe_name']
PORTFOLIO_SIZE = CONFIG['portfolio_size']
LOOKBACK_DAYS  = CONFIG['lookback_days']

UNIVERSE_FILE  = os.path.join(os.path.dirname(__file__), 'sp1500.json')


# ── Step 1: Load universe ─────────────────────────────────────────────────────
def load_universe():
    with open(UNIVERSE_FILE) as f:
        tickers = json.load(f)
    print(f'✅ Universe: {len(tickers)} tickers')
    return tickers


# ── Step 2: Fetch market cap (USD, in millions) ────────────────────────────────
def fetch_mcap(tickers):
    """
    Fetch market cap via yfinance fast_info, expressed in USD millions to
    match config['min_mcap'] / config['adv_divisor'].
    Falls back gracefully — missing mcap means stock fails the mcap filter,
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
                    mcap_data[ticker] = float(val) / 1e6  # USD millions
                break
            except:
                time.sleep(1)
        if (i + 1) % 200 == 0:
            print(f'   {i+1}/{len(tickers)} done — {len(mcap_data)} found')
            time.sleep(1)

    pct = len(mcap_data) / len(tickers) * 100 if tickers else 0
    print(f'✅ MCap: {len(mcap_data)}/{len(tickers)} ({pct:.0f}% coverage)')
    return mcap_data


# ── Push to Supabase ──────────────────────────────────────────────────────────
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

def push(supabase, top15, all_passing, all_universe, rejections, screen_date):
    row = {
        'run_date'    : str(screen_date.date()),
        'universe'    : UNIVERSE_NAME,
        'top15'       : to_records(top15.reset_index())        if not top15.empty        else [],
        'all_passing' : to_records(all_passing.reset_index())  if not all_passing.empty  else [],
        'all_universe': to_records(all_universe.reset_index()) if not all_universe.empty else [],
        'filters'     : {
            'universe': UNIVERSE_NAME, 'portfolio_size': PORTFOLIO_SIZE,
            'min_mcap_usd_m': CONFIG['min_mcap'], 'min_adv_usd_m': CONFIG['min_adv'],
            'max_vol': CONFIG['max_volatility'], 'rsi_threshold': CONFIG['rsi_threshold'],
            'max_from_high': CONFIG['max_from_high'], 'sma_short': CONFIG['sma_short'],
            'sma_long': CONFIG['sma_long'], 'cmf_period': CONFIG['cmf_period'],
            'cmf_threshold': CONFIG['cmf_threshold'],
            'rejections': rejections,
        },
        'run_status'  : 'complete',
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
    raw, available       = fetch_ohlcv(tickers, lookback_days=LOOKBACK_DAYS)
    screen_tickers       = [t for t in tickers if t in available]
    mcap                 = fetch_mcap(available)

    print('\n⏳ Computing indicators...')
    ind                  = compute_indicators(raw, mcap, screen_tickers, CONFIG)

    print('\n⏳ Running screen...')
    top15, all_passing, all_universe, rejections, screen_date = run_screen(ind, CONFIG)

    print('\n📤 Pushing to Supabase...')
    run_id = push(supabase, top15, all_passing, all_universe, rejections, screen_date)

    print(f'\n✅ Done in {(time.time()-t0)/60:.1f} min — run_id: {run_id}')

if __name__ == '__main__':
    main()
