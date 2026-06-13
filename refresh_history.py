"""
refresh_history.py — Fetch multi-year OHLCV for the full universe and store
it in Supabase Storage (parquet), merging with any existing stored history.

This replaces the Google Drive pkl cache. Run periodically (e.g. monthly)
via GitHub Actions — NOT on every live screen run (which uses a short
lookback fetched fresh each time via core/data_fetcher).

The backtest engine reads from this stored history via core/history_store,
avoiding a multi-year refetch on every backtest run.
"""

import os, json, time, warnings
from datetime import datetime
from supabase import create_client

from core.data_fetcher import fetch_ohlcv
from core.history_store import (
    load_history, save_history, merge_history, raw_multiindex_to_fields
)

warnings.filterwarnings('ignore')

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

UNIVERSE_NAME = CONFIG['universe_name']
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), 'sp1500.json')

# Years of history to fetch on a full (first-time) refresh. Subsequent runs
# only need to cover the gap since last_updated, but we still fetch this
# full window each time for simplicity — merge_history() handles dedup, and
# yfinance fetches for ~1500 tickers over a few years complete within the
# Actions timeout (see batch performance in live screener runs).
HISTORY_YEARS = int(os.environ.get('HISTORY_YEARS', '3'))


def main():
    t0 = time.time()
    print('='*60)
    print('  OHLCV HISTORY REFRESH — Supabase Storage')
    print(f'  Universe: {UNIVERSE_NAME}')
    print(f'  {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
    print('='*60)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    with open(UNIVERSE_FILE) as f:
        tickers = json.load(f)
    print(f'Universe: {len(tickers)} tickers, fetching {HISTORY_YEARS} years')

    raw, available = fetch_ohlcv(tickers, lookback_days=HISTORY_YEARS * 365)
    fresh = raw_multiindex_to_fields(raw)

    print('\nLoading existing stored history (if any)...')
    existing = load_history(supabase, UNIVERSE_NAME)
    if existing is not None:
        print(f'   Existing: {existing["Close"].shape[1]} tickers, '
              f'{existing["Close"].index[0].date()} -> {existing["Close"].index[-1].date()}')
    else:
        print('   No existing history — first run')

    print('\nMerging...')
    merged = merge_history(existing, fresh)

    print('\nSaving to Supabase Storage...')
    save_history(supabase, UNIVERSE_NAME, merged)

    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
