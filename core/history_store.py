"""
core/history_store.py — Persist/retrieve multi-year OHLCV history as parquet
in Supabase Storage. Replaces the Google Drive pkl cache.

Storage layout (bucket 'ohlcv-history'):
    {universe_name}/close.parquet
    {universe_name}/high.parquet
    {universe_name}/low.parquet
    {universe_name}/volume.parquet
    {universe_name}/meta.json   -- {'last_updated': ..., 'tickers': [...], 'date_range': [...]}

Each parquet file is a DataFrame: index = date, columns = tickers.
Storing fields separately (rather than one wide MultiIndex file) keeps
read/write simple and avoids MultiIndex<->parquet quirks.
"""

import io
import json
import pandas as pd

BUCKET = 'ohlcv-history'


def _field_path(universe_name, field):
    return f'{universe_name}/{field}.parquet'


def _meta_path(universe_name):
    return f'{universe_name}/meta.json'


def ensure_bucket(supabase):
    """Create the ohlcv-history bucket if it doesn't exist (idempotent)."""
    try:
        buckets = supabase.storage.list_buckets()
        names = [b.name for b in buckets] if buckets else []
        if BUCKET not in names:
            supabase.storage.create_bucket(BUCKET, options={'public': False})
            print(f"✅ Created Supabase Storage bucket '{BUCKET}'")
    except Exception as e:
        # Bucket may already exist with a race, or create_bucket signature
        # may differ across supabase-py versions — log and continue, the
        # upload call below will fail loudly if the bucket truly is missing.
        print(f"⚠ ensure_bucket: {e}")


def _df_to_parquet_bytes(df):
    buf = io.BytesIO()
    df.to_parquet(buf, engine='pyarrow', compression='snappy')
    return buf.getvalue()


def _parquet_bytes_to_df(data):
    buf = io.BytesIO(data)
    return pd.read_parquet(buf, engine='pyarrow')


def load_history(supabase, universe_name):
    """
    Load stored OHLCV history for `universe_name` from Supabase Storage.

    Returns a dict {'Close': df, 'High': df, 'Low': df, 'Volume': df} in the
    same shape as fetch_ohlcv's raw['Close'] etc, or None if nothing is
    stored yet (first run).
    """
    fields = ['Close', 'High', 'Low', 'Volume']
    result = {}
    for field in fields:
        path = _field_path(universe_name, field.lower())
        try:
            data = supabase.storage.from_(BUCKET).download(path)
            result[field] = _parquet_bytes_to_df(data)
        except Exception as e:
            print(f"⚠ No stored history for {universe_name}/{field.lower()} ({e})")
            return None

    for df in result.values():
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

    return result


def load_meta(supabase, universe_name):
    try:
        data = supabase.storage.from_(BUCKET).download(_meta_path(universe_name))
        return json.loads(data.decode('utf-8'))
    except Exception:
        return None


def save_history(supabase, universe_name, history_dict):
    """
    Save {'Close': df, 'High': df, 'Low': df, 'Volume': df} to Supabase
    Storage as parquet, overwriting any existing data for this universe.
    """
    ensure_bucket(supabase)

    for field, df in history_dict.items():
        path = _field_path(universe_name, field.lower())
        payload = _df_to_parquet_bytes(df)
        try:
            supabase.storage.from_(BUCKET).upload(
                path, payload,
                file_options={'content-type': 'application/octet-stream',
                               'x-upsert': 'true'}
            )
        except Exception as e:
            print(f"⚠ Upload failed for {path}: {e}")
            raise

    close_df = history_dict['Close']
    meta = {
        'last_updated': pd.Timestamp.utcnow().isoformat(),
        'n_tickers': len(close_df.columns),
        'tickers': close_df.columns.tolist(),
        'date_range': [str(close_df.index[0].date()), str(close_df.index[-1].date())],
        'n_rows': len(close_df),
    }
    try:
        supabase.storage.from_(BUCKET).upload(
            _meta_path(universe_name),
            json.dumps(meta).encode('utf-8'),
            file_options={'content-type': 'application/json', 'x-upsert': 'true'}
        )
    except Exception as e:
        print(f"⚠ Meta upload failed: {e}")

    sizes = {f: len(_df_to_parquet_bytes(d)) for f, d in history_dict.items()}
    total_mb = sum(sizes.values()) / 1e6
    print(f"✅ Saved {universe_name} history: {meta['n_tickers']} tickers, "
          f"{meta['n_rows']} rows, {meta['date_range'][0]} → {meta['date_range'][1]} "
          f"({total_mb:.1f} MB)")
    return meta


def merge_history(existing, fresh):
    """
    Merge freshly-fetched OHLCV (dict of DataFrames, e.g. from fetch_ohlcv's
    raw split by field) into existing stored history. New dates/tickers are
    added; overlapping dates are updated from `fresh` (handles
    restatements/corrections). Returns the merged dict.

    If `existing` is None, returns `fresh` unchanged.
    """
    if existing is None:
        return fresh

    merged = {}
    for field in ['Close', 'High', 'Low', 'Volume']:
        old_df = existing.get(field)
        new_df = fresh.get(field)
        if old_df is None:
            merged[field] = new_df
            continue
        if new_df is None:
            merged[field] = old_df
            continue

        # Union of columns (tickers), union of index (dates).
        # combine_first: values from new_df take precedence where both exist.
        combined = new_df.combine_first(old_df)
        combined = combined.reindex(sorted(combined.columns), axis=1)
        combined.sort_index(inplace=True)
        merged[field] = combined

    return merged


def raw_multiindex_to_fields(raw):
    """
    Convert a yfinance-style MultiIndex DataFrame (raw['Close'], raw['High'],
    etc. accessible via top-level column index) into the flat
    {'Close': df, 'High': df, ...} dict used by this module.
    """
    return {field: raw[field].copy() for field in ['Close', 'High', 'Low', 'Volume']
            if field in raw.columns.get_level_values(0)}


def fields_to_raw_multiindex(fields_dict):
    """Inverse of raw_multiindex_to_fields — rebuild a MultiIndex DataFrame
    suitable for core.indicators.compute_indicators (which expects
    raw_data['Close'], raw_data['High'], etc.)."""
    return pd.concat(fields_dict, axis=1)
