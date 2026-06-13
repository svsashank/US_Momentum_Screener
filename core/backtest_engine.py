"""
core/backtest_engine.py — Universal momentum strategy backtester.

Built on top of core/indicators.py and core/screener_engine.py so the
backtest uses EXACTLY the same indicator computation and screening logic as
the live screeners. This eliminates the historical divergence where the
backtest and live pipelines reimplemented the funnel separately.

Usage pattern:
    1. Fetch OHLCV for the full universe over the full backtest period
       (one fetch, vectorized indicator computation over the whole history —
       this is the "2-5 minute" approach vs. the old per-ticker loops).
    2. compute_indicators() once over the whole date range.
    3. run_backtest() walks rebalance dates, applies run_screen() at each
       date, and simulates portfolio turnover.

Point-in-time universe membership (e.g. S&P 500 historical constituents) is
handled by passing a `membership` function/lookup — at each rebalance date,
only tickers that were actually in the universe on that date are eligible.
If no membership data is provided, the full static universe is used for
every date (acceptable for India where point-in-time membership data isn't
yet available).
"""

import pandas as pd
import numpy as np

from core.screener_engine import run_screen


def get_rebalance_dates(date_index, cadence='monthly'):
    """
    Return a list of rebalance dates from `date_index` (a DatetimeIndex of
    trading days), spaced according to `cadence`.

    cadence: 'monthly' (last trading day of each month),
             'quarterly' (last trading day of each quarter),
             'weekly' (last trading day of each week)
    """
    df = pd.Series(date_index, index=date_index)
    if cadence == 'monthly':
        groups = df.groupby([date_index.year, date_index.month])
    elif cadence == 'quarterly':
        groups = df.groupby([date_index.year, date_index.quarter])
    elif cadence == 'weekly':
        groups = df.groupby([date_index.year, date_index.isocalendar().week])
    else:
        raise ValueError(f"Unknown cadence: {cadence}")
    return sorted(g.iloc[-1] for _, g in groups)


def _ind_slice_up_to(ind, date):
    """
    Return a copy of the indicator dict sliced to only include data up to
    (and including) `date`. This is what makes run_screen() see only
    "past and present" data at each rebalance date — no lookahead bias.
    """
    sliced = {}
    for key, df in ind.items():
        sliced[key] = df.loc[:date]
    return sliced


def run_backtest(ind, config, rebalance_dates, membership=None,
                  initial_capital=1_000_000, retention_rank=25):
    """
    Walk through `rebalance_dates`, applying the screening funnel (via
    core.screener_engine.run_screen) at each date using only data available
    up to that date (no lookahead).

    ind: full-history indicator dict from core.indicators.compute_indicators,
         computed once over the entire backtest period for the full universe
    config: same config dict used by the live screener (thresholds, anchors,
            portfolio_size, etc.)
    rebalance_dates: list of pd.Timestamp, output of get_rebalance_dates()
    membership: optional dict {date: set_of_tickers} or callable(date) ->
                set_of_tickers, for point-in-time universe filtering.
                If None, all tickers in `ind['close'].columns` are eligible
                on every date.
    initial_capital: starting portfolio value
    retention_rank: HOLD if a held stock still passes the funnel and its
                    rank <= retention_rank, else SELL (matches live
                    portfolio_engine retention buffer rule)

    Returns a dict with:
        history: DataFrame indexed by rebalance date, columns include
                 portfolio_value, n_holdings, turnover
        holdings_log: dict {date: list of tickers held after rebalance}
        screen_log: dict {date: (top15_df, all_passing_df, rejections)}
    """
    portfolio_size = config['portfolio_size']

    portfolio_value = initial_capital
    current_holdings = {}  # ticker -> weight (equal-weight assumed)

    history_rows = []
    holdings_log = {}
    screen_log = {}

    for i, date in enumerate(rebalance_dates):
        ind_slice = _ind_slice_up_to(ind, date)

        # Apply point-in-time universe filter if provided
        if membership is not None:
            if callable(membership):
                eligible = membership(date)
            else:
                eligible = membership.get(date, set(ind['close'].columns))
            cols = [c for c in ind_slice['close'].columns if c in eligible]
            ind_slice = {k: v[cols] for k, v in ind_slice.items()}

        top15, all_passing, rejections, screen_date = run_screen(ind_slice, config)
        screen_log[date] = (top15, all_passing, rejections)

        passing_tickers = set(all_passing['ticker']) if not all_passing.empty else set()
        rank_map = {r['ticker']: idx for idx, r in all_passing.reset_index().iterrows()} \
            if not all_passing.empty else {}

        new_holdings = {}

        # Retention buffer: keep currently-held stocks that still pass and
        # rank within retention_rank
        retained = []
        for t in current_holdings:
            if t in passing_tickers and rank_map.get(t, 999) <= retention_rank:
                retained.append(t)

        # Fill remaining slots from top15, skipping already-retained tickers
        slots_remaining = portfolio_size - len(retained)
        fill_candidates = [t for t in top15['ticker'].tolist()] if not top15.empty else []
        fill = [t for t in fill_candidates if t not in retained][:slots_remaining]

        target_tickers = retained + fill

        if target_tickers:
            weight = 1.0 / len(target_tickers)
            new_holdings = {t: weight for t in target_tickers}

        # Compute portfolio return since last rebalance using the held
        # tickers' price change over the period (only after the first
        # rebalance — first period has no prior holdings to value)
        if i > 0 and current_holdings:
            prev_date = rebalance_dates[i-1]
            period_return = 0.0
            for t, w in current_holdings.items():
                if t in ind['close'].columns:
                    try:
                        p0 = ind['close'][t].loc[:prev_date].iloc[-1]
                        p1 = ind['close'][t].loc[:date].iloc[-1]
                        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                            period_return += w * (p1 / p0 - 1)
                    except (IndexError, KeyError):
                        continue
            portfolio_value *= (1 + period_return)

        turnover = len(set(new_holdings) - set(current_holdings)) / max(len(new_holdings), 1) \
            if new_holdings else 0.0

        history_rows.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'n_holdings': len(new_holdings),
            'turnover': turnover,
            'n_passing': len(all_passing),
        })
        holdings_log[date] = list(new_holdings.keys())
        current_holdings = new_holdings

    history = pd.DataFrame(history_rows).set_index('date')
    return {'history': history, 'holdings_log': holdings_log, 'screen_log': screen_log}


def compute_performance_stats(history):
    """
    Compute CAGR, annualised volatility, Sharpe (rf=0), and max drawdown
    from a backtest history DataFrame (output of run_backtest()['history']).
    """
    values = history['portfolio_value']
    returns = values.pct_change().dropna()

    n_periods = len(values)
    years = (values.index[-1] - values.index[0]).days / 365.25
    total_return = values.iloc[-1] / values.iloc[0]
    cagr = total_return ** (1 / years) - 1 if years > 0 else np.nan

    # Infer periods/year from spacing for annualisation
    periods_per_year = n_periods / years if years > 0 else 12

    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan

    running_max = values.cummax()
    drawdown = values / running_max - 1
    max_drawdown = drawdown.min()

    return {
        'cagr': cagr,
        'total_return': total_return - 1,
        'ann_volatility': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'years': years,
    }
