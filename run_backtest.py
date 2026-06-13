"""
run_backtest.py — On-demand backtest runner (GitHub Actions).

Reads OHLCV history from Supabase Storage (populated by refresh_history.py),
computes indicators once, runs core.backtest_engine over a configurable
date range / parameter set, and writes results to the `backtest_runs`
Supabase table for the GUI to display.

All parameters are overridable via environment variables (set by the
repository_dispatch payload from the GUI). Falls back to config.json /
sensible defaults if not provided.
"""

import os, json, time, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from supabase import create_client

from core.history_store import load_history, fields_to_raw_multiindex
from core.indicators import compute_indicators
from core.backtest_engine import get_rebalance_dates, run_backtest, compute_performance_stats

warnings.filterwarnings('ignore')

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_FILE) as f:
    BASE_CONFIG = json.load(f)

UNIVERSE_NAME = BASE_CONFIG['universe_name']


def env_override(config, key, env_name, cast=float):
    """Override a config value from an environment variable if set."""
    val = os.environ.get(env_name)
    if val is not None and val != '':
        config[key] = cast(val)
    return config


def build_config():
    """Build the run config: base config.json + env-var overrides for
    everything the GUI exposes as configurable."""
    cfg = dict(BASE_CONFIG)

    # Screening parameters
    env_override(cfg, 'portfolio_size', 'BT_PORTFOLIO_SIZE', int)
    env_override(cfg, 'min_mcap', 'BT_MIN_MCAP', float)
    env_override(cfg, 'min_adv', 'BT_MIN_ADV', float)
    env_override(cfg, 'max_volatility', 'BT_MAX_VOLATILITY', float)
    env_override(cfg, 'rsi_threshold', 'BT_RSI_THRESHOLD', float)
    env_override(cfg, 'max_from_high', 'BT_MAX_FROM_HIGH', float)
    env_override(cfg, 'cmf_threshold', 'BT_CMF_THRESHOLD', float)
    env_override(cfg, 'sma_short', 'BT_SMA_SHORT', int)
    env_override(cfg, 'sma_long', 'BT_SMA_LONG', int)

    # Backtest mechanics
    env_override(cfg, 'cost_buy', 'BT_COST_BUY', float)
    env_override(cfg, 'cost_sell', 'BT_COST_SELL', float)
    cfg['cash_mode'] = os.environ.get('BT_CASH_MODE', cfg.get('cash_mode', 'partial'))
    env_override(cfg, 'min_stocks_to_invest', 'BT_MIN_STOCKS_TO_INVEST', int)
    env_override(cfg, 'retention_rank', 'BT_RETENTION_RANK', int)
    env_override(cfg, 'risk_free_rate', 'BT_RISK_FREE_RATE', float)

    return cfg


def clean(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(val, (pd.Timestamp,)):
        return str(val.date())
    return val


def df_to_records(df):
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.reset_index().iterrows():
        out.append({k: clean(v) for k, v in row.items()})
    return out


def main():
    t0 = time.time()
    print('='*60)
    print('  BACKTEST RUNNER — GitHub Actions')
    print(f'  {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
    print('='*60)

    config = build_config()

    start_date = os.environ.get('BT_START_DATE')   # 'YYYY-MM-DD' or empty
    end_date   = os.environ.get('BT_END_DATE')      # 'YYYY-MM-DD' or empty
    rebalance_type = os.environ.get('BT_REBALANCE_TYPE', 'monthly')
    initial_capital = float(os.environ.get('BT_INITIAL_CAPITAL', '1000000'))

    print(f'Config overrides applied. Rebalance: {rebalance_type}, '
          f'capital: {initial_capital:,.0f}, cash_mode: {config.get("cash_mode")}')

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print('\nLoading OHLCV history from Supabase Storage...')
    history = load_history(supabase, UNIVERSE_NAME)
    if history is None:
        raise RuntimeError(
            f'No stored history found for universe "{UNIVERSE_NAME}". '
            f'Run refresh_history.py first.'
        )
    print(f'   {history["Close"].shape[1]} tickers, '
          f'{history["Close"].index[0].date()} -> {history["Close"].index[-1].date()}')

    raw = fields_to_raw_multiindex(history)
    tickers = history['Close'].columns.tolist()

    # mcap: backtest uses a flat snapshot per ticker is not meaningful across
    # years, so for now treat min_mcap as effectively disabled in backtests
    # by supplying a large constant mcap for every ticker (passes filter).
    # TODO: store point-in-time mcap (shares outstanding history) for a
    # fully accurate backtest mcap filter.
    mcap_data = {t: max(config['min_mcap'] * 10, 1e6) for t in tickers}

    print('\nComputing indicators over full history...')
    ind = compute_indicators(raw, mcap_data, tickers, config)

    full_start = ind['close'].index[0]
    full_end   = ind['close'].index[-1]
    bt_start = pd.Timestamp(start_date) if start_date else full_start
    bt_end   = pd.Timestamp(end_date) if end_date else full_end
    bt_start = max(bt_start, full_start)
    bt_end   = min(bt_end, full_end)

    print(f'\nBacktest window: {bt_start.date()} -> {bt_end.date()} ({rebalance_type})')
    rebalance_dates = get_rebalance_dates(bt_start, bt_end, ind['close'].index, rebalance_type)
    print(f'{len(rebalance_dates)} rebalance dates')

    if len(rebalance_dates) < 2:
        raise RuntimeError('Fewer than 2 rebalance dates in the selected window — '
                            'widen the date range.')

    portfolio_df, trades_df, snapshots_df = run_backtest(
        ind, config, rebalance_dates, initial_capital, verbose=True
    )

    stats = compute_performance_stats(portfolio_df, rebalance_type=rebalance_type,
                                       risk_free_rate=config.get('risk_free_rate', 0.0))

    print('\nPerformance:')
    for k, v in stats.items():
        print(f'   {k}: {v}')

    row = {
        'universe': UNIVERSE_NAME,
        'start_date': str(bt_start.date()),
        'end_date': str(bt_end.date()),
        'rebalance_type': rebalance_type,
        'initial_capital': initial_capital,
        'config': {k: clean(v) for k, v in config.items() if not k.startswith('_')},
        'stats': {k: clean(v) for k, v in stats.items()},
        'portfolio_curve': df_to_records(portfolio_df.reset_index().rename(columns={'index': 'date'})),
        'snapshots': df_to_records(snapshots_df.reset_index().drop(columns=['top_picks'], errors='ignore')),
        'trades': df_to_records(trades_df),
        'run_status': 'complete',
        'triggered_at': datetime.utcnow().isoformat(),
    }

    print('\nPushing results to Supabase...')
    resp = supabase.table('backtest_runs').insert(row).execute()
    run_id = resp.data[0]['id'] if resp.data else None
    print(f'backtest_runs -> id: {run_id}')

    print(f'\nDone in {(time.time()-t0)/60:.1f} min — run_id: {run_id}')


if __name__ == '__main__':
    main()
