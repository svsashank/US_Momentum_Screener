"""
core/backtest_engine.py — Universal momentum strategy backtester.

Built on core/screener_engine.run_screen() so the backtest funnel is
IDENTICAL to the live screeners (same filters, same CMF, same ranking).
Portfolio mechanics (share counts, cash modes, transaction costs, retention
buffer) follow Momentum_Backtest_Universal_v3.ipynb, generalized to be
config-driven for both US and India.

Required config keys (in addition to the screener_engine keys):
    cost_buy            fractional transaction cost on buys (e.g. 0.0008)
    cost_sell           fractional transaction cost on sells
    cash_mode           'partial' or 'full_or_cash'
    min_stocks_to_invest  only used when cash_mode == 'full_or_cash'
    retention_rank      HOLD if a held stock passes funnel and rank <=
                         retention_rank (set to 0 to disable — i.e. sell
                         anything not in the fresh top-N, matching the
                         original notebook's behaviour)
    risk_free_rate      annual %, used to accrue returns on uninvested cash
"""

import pandas as pd
import numpy as np

from core.screener_engine import run_screen


# ── Rebalance dates ─────────────────────────────────────────────────────────
def get_rebalance_dates(start, end, price_index, rebalance_type='monthly'):
    """
    Return a DatetimeIndex of rebalance dates (first trading day of each
    month, or first trading day of each quarter) within [start, end].
    """
    in_range = price_index[(price_index >= start) & (price_index <= end)]

    if rebalance_type == 'quarterly':
        grouped = in_range.to_frame().groupby([in_range.year, in_range.month]).first()
        starts  = grouped[grouped.index.get_level_values(1).isin([1, 4, 7, 10])].values.flatten()
    else:
        starts = in_range.to_frame().groupby([in_range.year, in_range.month]).first().values.flatten()

    return pd.DatetimeIndex(starts)


def _ind_slice_up_to(ind, date):
    """Slice every indicator DataFrame to data up to and including `date` —
    prevents lookahead bias when screening at a historical rebalance date."""
    return {k: v.loc[:date] for k, v in ind.items()}


# ── Main backtest loop ───────────────────────────────────────────────────────
def run_backtest(ind, config, rebalance_dates, initial_capital,
                 verbose=True):
    """
    Walk through `rebalance_dates`, applying core.screener_engine.run_screen
    at each date (no lookahead — only data up to that date is visible), and
    simulating a real portfolio with share counts, transaction costs, and
    configurable cash handling.

    ind: full-history indicator dict from core.indicators.compute_indicators,
         computed once over the entire backtest period
    config: same config dict used by the live screener, plus the backtest
            keys listed in the module docstring
    rebalance_dates: DatetimeIndex, output of get_rebalance_dates()
    initial_capital: starting portfolio value (in the market's currency)

    Returns (portfolio_df, trades_df, snapshots_df):
        portfolio_df: DataFrame indexed by date, column 'value'
        trades_df:    DataFrame of all BUY/SELL/SELL_CASH/SELL_CAP/TRIM trades
        snapshots_df: DataFrame indexed by date with per-period detail
                      (stocks_screened, slots_used, cash_slots, in_cash,
                      cash_balance, top_picks, rejection counts)
    """
    portfolio_size       = config['portfolio_size']
    cost_buy             = config.get('cost_buy', 0.0)
    cost_sell            = config.get('cost_sell', 0.0)
    cash_mode            = config.get('cash_mode', 'partial')
    min_stocks_to_invest = config.get('min_stocks_to_invest', portfolio_size)
    retention_rank       = config.get('retention_rank', 0)
    risk_free_rate       = config.get('risk_free_rate', 0.0)

    daily_cash_return = (1 + risk_free_rate / 100) ** (1 / 252) - 1

    capital          = float(initial_capital)
    holdings         = {}   # ticker -> {'shares': int, 'cost_price': float}
    prev_rebal_date  = None
    cash_periods     = 0

    portfolio_values, snapshots, trade_log = [], [], []
    all_trading_days = ind['close'].index

    for i, rebal_date in enumerate(rebalance_dates):
        label = rebal_date.strftime('%d %b %Y')

        # Accrue cash return on uninvested capital since last rebalance
        if prev_rebal_date is not None and capital > 0:
            days = all_trading_days[(all_trading_days > prev_rebal_date) &
                                     (all_trading_days <= rebal_date)]
            capital *= (1 + daily_cash_return) ** len(days)

        # Value current holdings at today's prices
        idx = ind['close'].index.get_indexer([rebal_date], method='ffill')[0]
        price_row = ind['close'].iloc[idx]
        portfolio_value = capital
        for ticker, pos in holdings.items():
            p = price_row.get(ticker, np.nan)
            if pd.notna(p):
                portfolio_value += pos['shares'] * p

        # Screen using only data up to this date — no lookahead
        ind_slice = _ind_slice_up_to(ind, rebal_date)
        top15, all_passing, rejections, screen_date = run_screen(ind_slice, config)
        n_passed = len(all_passing)

        passing_tickers = set(all_passing['ticker']) if not all_passing.empty else set()
        rank_map = {r['ticker']: idx_ for idx_, r in all_passing.reset_index().iterrows()} \
            if not all_passing.empty else {}

        # ── CASH MODE: full_or_cash -> liquidate everything if too few pass ──
        if cash_mode == 'full_or_cash' and n_passed < min_stocks_to_invest:
            status = f'FULL CASH ({n_passed}/{min_stocks_to_invest})'
            for ticker in list(holdings.keys()):
                p = price_row.get(ticker, np.nan)
                if pd.notna(p):
                    capital += holdings[ticker]['shares'] * p * (1 - cost_sell)
                    trade_log.append({'date': rebal_date, 'ticker': ticker,
                                       'action': 'SELL_CASH', 'price': p,
                                       'shares': holdings[ticker]['shares']})
            holdings   = {}
            in_cash    = True
            slots_used = 0
            cash_periods += 1
            if verbose:
                print(f'{label} [{i+1}/{len(rebalance_dates)}] {status}')

        else:
            # ── Determine target tickers (retention buffer + top-up) ──────
            if retention_rank > 0:
                retained = [t for t in holdings
                            if t in passing_tickers and rank_map.get(t, 999) <= retention_rank]
                slots_remaining = portfolio_size - len(retained)
                fill_candidates = top15['ticker'].tolist() if not top15.empty else []
                fill = [t for t in fill_candidates if t not in retained][:max(slots_remaining, 0)]
                target = retained + fill
            else:
                # Original notebook behaviour: target = fresh top-N passing
                slots_available = min(n_passed, portfolio_size)
                target = top15.head(slots_available)['ticker'].tolist() if not top15.empty else []

            slots_used = len(target)
            cash_slots = portfolio_size - slots_used
            in_cash    = False

            status = (f'{n_passed} passed -> {slots_used} stocks + {cash_slots} cash slots'
                      if cash_slots > 0 else f'{n_passed} passed -> {slots_used} selected')
            if verbose:
                print(f'{label} [{i+1}/{len(rebalance_dates)}] {status} | '
                      f'adv={rejections["adv"]:2d} vol={rejections["volatility"]:2d} '
                      f'rsi={rejections["rsi"]:2d} cmf={rejections.get("cmf", 0):2d}')

            # Sell exits -- holdings no longer in target
            for ticker in [t for t in list(holdings.keys()) if t not in target]:
                p = price_row.get(ticker, np.nan)
                if pd.notna(p):
                    capital += holdings[ticker]['shares'] * p * (1 - cost_sell)
                    trade_log.append({'date': rebal_date, 'ticker': ticker,
                                       'action': 'SELL', 'price': p,
                                       'shares': holdings[ticker]['shares']})
                del holdings[ticker]

            # Equal allocation per slot; unfilled slots remain as cash
            value_per_slot = portfolio_value / portfolio_size

            # Trim overweight existing positions back to value_per_slot
            for ticker in [t for t in target if t in holdings]:
                p = price_row.get(ticker, np.nan)
                if pd.isna(p):
                    continue
                excess = holdings[ticker]['shares'] * p - value_per_slot
                if excess > p:
                    trim = int(excess / p)
                    if trim > 0:
                        capital += trim * p * (1 - cost_sell)
                        holdings[ticker]['shares'] -= trim
                        trade_log.append({'date': rebal_date, 'ticker': ticker,
                                           'action': 'TRIM', 'price': p, 'shares': trim})

            # Buy new entrants
            for ticker in [t for t in target if t not in holdings]:
                p = price_row.get(ticker, np.nan)
                if pd.isna(p):
                    continue
                alloc = value_per_slot * (1 - cost_buy)
                shares_to_buy = int(alloc / p)
                if shares_to_buy > 0:
                    cost = shares_to_buy * p * (1 + cost_buy)
                    if cost <= capital:
                        capital -= cost
                        holdings[ticker] = {'shares': shares_to_buy, 'cost_price': p}
                        trade_log.append({'date': rebal_date, 'ticker': ticker,
                                           'action': 'BUY', 'price': p,
                                           'shares': shares_to_buy})

        portfolio_values.append({'date': rebal_date, 'value': portfolio_value})
        snapshots.append({
            'date'           : rebal_date,
            'portfolio_value': portfolio_value,
            'stocks_screened': n_passed,
            'slots_used'     : slots_used,
            'cash_slots'     : portfolio_size - slots_used,
            'in_cash'        : in_cash,
            'cash_balance'   : capital,
            'top_picks'      : list(holdings.keys()),
            'rej_adv'        : rejections.get('adv', 0),
            'rej_vol'        : rejections.get('volatility', 0),
            'rej_rsi'        : rejections.get('rsi', 0),
            'rej_mcap'       : rejections.get('mcap', 0),
            'rej_cmf'        : rejections.get('cmf', 0),
        })
        prev_rebal_date = rebal_date

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
    trades_df    = pd.DataFrame(trade_log)
    snapshots_df = pd.DataFrame(snapshots).set_index('date')

    if verbose:
        total = len(snapshots_df)
        full_cash = cash_periods
        partial   = int(((snapshots_df['cash_slots'] > 0) & (~snapshots_df['in_cash'])).sum())
        fully_inv = total - full_cash - partial
        print(f'\nBacktest complete: {total} periods | '
              f'fully invested {fully_inv} ({fully_inv/total*100:.0f}%) | '
              f'partial cash {partial} ({partial/total*100:.0f}%) | '
              f'full cash {full_cash} ({full_cash/total*100:.0f}%)')

    return portfolio_df, trades_df, snapshots_df


# ── Performance metrics ──────────────────────────────────────────────────────
def compute_performance_stats(portfolio_df, rebalance_type='monthly', risk_free_rate=0.0):
    """
    Compute CAGR, total return, annualised volatility, Sharpe, max drawdown,
    and win rate from a backtest's portfolio_df (output of run_backtest()).
    """
    values  = portfolio_df['value']
    returns = values.pct_change().dropna()

    n_years = (values.index[-1] - values.index[0]).days / 365.25
    total_return = values.iloc[-1] / values.iloc[0] - 1
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    periods_per_year = 12 if rebalance_type == 'monthly' else 4
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe  = (cagr - risk_free_rate / 100) / ann_vol if ann_vol > 0 else np.nan

    drawdown = (values - values.cummax()) / values.cummax()
    max_drawdown = drawdown.min()
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else np.nan

    return {
        'cagr': cagr,
        'total_return': total_return,
        'ann_volatility': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'years': n_years,
    }
