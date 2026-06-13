"""
core/data_fetcher.py — Shared OHLCV fetch logic with batching and retries.

Identical between US and India screeners already; extracted here so both
(and the future backtest engine) use one implementation.
"""

import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


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
    """
    Fetch OHLCV for a list of tickers with batching, retries, and rate-limit
    pauses. The +2 day end-date buffer avoids yfinance's exclusive-end-date
    clipping the latest session.

    Returns (raw_data, available_tickers).
    """
    end_date   = datetime.today().date() + timedelta(days=2)
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
