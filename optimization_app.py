# ===== Parameter Sweep: Lookback-Months Optimization (with per-ticker bounds) =====
# For each lookback in LOOKBACK_MONTHS_LIST:
#   - Monthly walk-forward: optimize on past X months, hold for next month
#   - Compute MONTHLY EXCESS returns:  (Π(1+r_port_daily) / Π(1+rf_daily)) - 1
# Output: prints a table of avg, sd, return/sd by lookback and the best lookback

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional

# ---------- Configuration ----------
INPUT_DIR = Path("etf_downloads")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dynamically discover available ETFs (Excel files) in INPUT_DIR

def discover_etf_files(input_dir: Path = INPUT_DIR) -> List[str]:
    """Return a list of .xlsx filenames found under INPUT_DIR (non-recursive)."""
    if not input_dir.exists():
        return []
    return sorted([p.name for p in input_dir.glob("*.xlsx") if p.is_file()])

FILES = discover_etf_files(INPUT_DIR)

# Per-ticker bounds (inclusive). Omit a ticker to default to (0.0, 1.0).
LOWER_EPS = 1e-6
TICKER_BOUNDS = {
    # examples:
    # "SPY": (0.0, 1.0),
    "SGOV": (0.24, 0.25),
    "BTC-USD": (0, 0.05),  # emulate 0 < w < 0.05
}

# Backtest span
BACKTEST_START = "2023-01-01"   # first rebalance month (weights applied starting this month)
BACKTEST_END   = None           # e.g. "2025-08-20" or None = latest available

# Parameter sweep (months)
LOOKBACK_MONTHS_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# Window sanity
MIN_WINDOW_DAYS = 30            # require at least this many daily obs in a lookback window

# Risk-free settings
RF_ANNUAL = 0.0                # fallback annualized RF (e.g., 4%)
USE_RF_CSV = True               # if 'risk_free_daily.csv' exists with columns: date, rf_daily

# Score weights (relative, do not need to sum to 10)
W_RET  = 2
W_VOL  = 0
W_LOSS = 8

# Score shape parameter
K_RET    = 0.05                  # softness for return sigmoid (annual units, e.g., 0.05 = 5%)
EPS_LOSS = 1e-12                 # last-resort epsilon baseline

# ---------- IO Helpers ----------
def robust_load_prices_one(path: Path) -> pd.Series:
    """Load 'adj_close' from Excel 'Prices' sheet. Accept 'date' or first column as dates."""
    df = pd.read_excel(path, sheet_name="Prices")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip().lower() for c in df.columns]

    # pick a date column
    cand = None
    for name in ["date", "datetime", "time", "index"]:
        if name in df.columns:
            cand = name
            break
    if cand is None:
        cand = df.columns[0]

    df[cand] = pd.to_datetime(df[cand], errors="coerce")
    if df[cand].isna().all():
        raise ValueError(f"Could not parse dates in {path.name}. Columns: {df.columns.tolist()[:8]}")
    df = df.set_index(cand).sort_index()

    # find adjusted close
    for a in ["adj_close", "adj close", "adjusted close", "adjusted_close"]:
        if a in df.columns:
            adj_col = a
            break
    else:
        raise KeyError(f"'adj_close' not found in {path.name}. Columns: {df.columns.tolist()[:8]}")

    s = df[adj_col].astype(float).dropna()
    s.name = path.stem
    return s

def load_prices(files):
    series = []
    for fname in files:
        p = INPUT_DIR / fname
        if not p.exists():
            print(f"Note: missing file {p}, skipping.")
            continue
        series.append(robust_load_prices_one(p))
    if not series:
        raise RuntimeError("No input price series found.")
    # Keep only dates present in ALL series; strips BTC-only weekends automatically
    px = pd.concat(series, axis=1, join="inner").sort_index()
    return px

def load_risk_free_series(index: pd.DatetimeIndex, fallback_annual: float, use_csv=True) -> pd.Series:
    if use_csv and Path("risk_free_daily.csv").exists():
        rf = pd.read_csv("risk_free_daily.csv")
        rf["date"] = pd.to_datetime(rf["date"])
        rf = rf.set_index("date").sort_index()
        rf_daily = rf["rf_daily"].astype(float)
        rf_daily = rf_daily.reindex(index).fillna(method="ffill").fillna(0.0)
        return rf_daily
    return pd.Series(fallback_annual / 252.0, index=index)

# ---------- Metrics (window-level) ----------
def window_metrics(weights, rets_window, rf_window):
    """
    On a daily window (rets_window: DataFrame, rf_window: Series), compute:
      - Return_ann  = mean(excess_daily) * 252
      - Vol_ann     = std(daily) * sqrt(252)
      - LossMass    = sum of |r_t| for negative raw daily returns
    """
    port_raw = pd.Series(rets_window.values @ weights, index=rets_window.index, name="port_ret")
    port_excess = port_raw - rf_window.reindex(rets_window.index).fillna(0.0)

    if len(port_excess) < MIN_WINDOW_DAYS:
        return np.nan, np.nan, np.nan

    mu_excess_daily = port_excess.mean()
    sd_daily = port_excess.std(ddof=1)
    return_ann = float(mu_excess_daily * 252.0)
    vol_ann    = float(sd_daily * np.sqrt(252.0))
    loss_mass  = float((-port_raw.clip(upper=0.0)).sum())
    return return_ann, vol_ann, loss_mass

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def objective_on_window(weights, rets_window, rf_window, k_ret, w_ret, w_vol, w_loss):
    """
    Normalized score objective on this window, using EW baselines from the SAME window.
    Return NEGATIVE so scipy.minimize can maximize our score.
    """
    n = rets_window.shape[1]
    w_eq = np.full(n, 1.0/n)

    # Baselines from EW
    ret_eq, vol_eq, loss_eq = window_metrics(w_eq, rets_window, rf_window)
    return_ref = 0.0 if not np.isfinite(ret_eq) else ret_eq
    c_vol  = EPS_LOSS if (not np.isfinite(vol_eq) or vol_eq <= 0) else vol_eq
    c_loss = EPS_LOSS if (not np.isfinite(loss_eq) or loss_eq <= 0) else loss_eq

    # Portfolio window metrics
    ret_ann, vol_ann, loss_mass = window_metrics(weights, rets_window, rf_window)
    if not (np.isfinite(ret_ann) and np.isfinite(vol_ann) and np.isfinite(loss_mass)):
        return 1e6

    ret_score  = sigmoid((ret_ann - return_ref) / max(k_ret, 1e-6))
    vol_score  = c_vol  / (c_vol  + max(vol_ann, 0.0))
    loss_score = c_loss / (c_loss + max(loss_mass, 0.0))

    score = w_ret*ret_score + w_vol*vol_score + w_loss*loss_score
    return -float(score)

# ---------- Bounds helpers ----------
def build_bounds_for_tickers(tickers: List[str], overrides: Optional[Dict[str, Tuple[float, float]]] = None):
    """
    Create bounds aligned with 'tickers', using TICKER_BOUNDS where present,
    else default to (0.0, 1.0). Also check feasibility:
      sum(lower_i) <= 1 <= sum(upper_i)
    """
    bnds = []
    sum_lo, sum_hi = 0.0, 0.0
    for t in tickers:
        if overrides is not None and t in overrides:
            lo, hi = overrides[t]
        else:
            lo, hi = TICKER_BOUNDS.get(t, (0.0, 1.0))
        lo = max(0.0, float(lo))
        hi = min(1.0, float(hi))
        if hi < lo:
            raise ValueError(f"Invalid bounds for {t}: ({lo}, {hi})")
        bnds.append((lo, hi))
        sum_lo += lo
        sum_hi += hi
    if sum_lo > 1.0 + 1e-9:
        raise ValueError(f"Lower bounds sum to {sum_lo:.4f} > 1. Infeasible.")
    if sum_hi < 1.0 - 1e-9:
        raise ValueError(f"Upper bounds sum to {sum_hi:.4f} < 1. Infeasible.")
    return bnds

def optimize_weights_on_window(rets_window, rf_window, tickers, W_RET=W_RET, W_VOL=W_VOL, W_LOSS=W_LOSS, bounds_by_ticker: Optional[Dict[str, Tuple[float,float]]] = None):
    n = rets_window.shape[1]
    w0 = np.full(n, 1.0/n)
    bounds = build_bounds_for_tickers(tickers, overrides=bounds_by_ticker)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(
        objective_on_window,
        w0,
        args=(rets_window, rf_window, K_RET, W_RET, W_VOL, W_LOSS),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 400, "ftol": 1e-9, "disp": False},
    )
    if not res.success:
        print("WARNING: Optimizer did not fully converge on a window:", res.message)
    # Clip to bounds then renormalize just in case of tiny numerical drift
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    w = np.clip(res.x, lo, hi)
    s = w.sum()
    return w / (s if s > 0 else 1.0)

# ---------- Backtest Core for a Given Lookback ----------
def run_backtest_for_lookback(
    rets_all,
    rf_daily_all,
    start_dt,
    lookback_months,
    tickers,
    W_RET=W_RET,
    W_VOL=W_VOL,
    W_LOSS=W_LOSS,
    bounds_by_ticker: Optional[Dict[str, Tuple[float, float]]] = None,
    collect_weights: bool = False,
):
    """
    Returns a DataFrame of monthly EXCESS returns for the strategy with this lookback.
    Columns: ['date','excess_return'] where 'date' is the rebalance month (start of hold).
    """
    # Rebalance on first trading day each month >= start_dt
    rets_in_test = rets_all.loc[start_dt:]
    month_starts = rets_in_test.groupby([rets_in_test.index.year, rets_in_test.index.month]).apply(lambda x: x.index.min())
    month_starts = pd.to_datetime(month_starts.values)

    # Check global feasibility of bounds once for this lookback across the selected tickers
    try:
        _ = build_bounds_for_tickers(tickers, overrides=bounds_by_ticker)
    except ValueError as e:
        # Propagate a clear error so the UI can show it
        raise ValueError(f"Bounds infeasible for selected assets: {e}") from e

    monthly_rows = []
    monthly_weights_rows = [] if collect_weights else None

    for i, reb_date in enumerate(month_starts):
        # Lookback window: last lookback_months months ending day before reb_date
        lb_end = reb_date - pd.Timedelta(days=1)
        lb_start = lb_end - pd.DateOffset(months=lookback_months) + pd.Timedelta(days=1)

        rets_win = rets_all.loc[lb_start:lb_end]
        if len(rets_win) < MIN_WINDOW_DAYS:
            continue
        rf_win = rf_daily_all.loc[rets_win.index]

        # Optimize on window with UI-provided weights and bounds; if it fails, skip this month
        try:
            w_opt = optimize_weights_on_window(
                rets_window=rets_win,
                rf_window=rf_win,
                tickers=tickers,
                W_RET=W_RET,
                W_VOL=W_VOL,
                W_LOSS=W_LOSS,
                bounds_by_ticker=bounds_by_ticker,
            )
        except Exception as opt_err:
            # Warn to console and skip this month
            print(f"Skipping month starting {reb_date.date()} due to optimization error: {opt_err}")
            continue
        if collect_weights:
            row = {"date": reb_date}
            for t_name, w_val in zip(tickers, w_opt):
                row[t_name] = float(w_val)
            monthly_weights_rows.append(row)

        # Holding period: from reb_date to day before next month start (or last available date)
        if i < len(month_starts) - 1:
            hold_end = month_starts[i+1] - pd.Timedelta(days=1)
        else:
            hold_end = rets_all.index.max()

        hold_rets = rets_all.loc[reb_date:hold_end]
        if hold_rets.empty:
            continue
        port_daily = hold_rets.values @ w_opt

        # Excess monthly return via compounding relative to RF
        rf_hold = rf_daily_all.loc[hold_rets.index].fillna(0.0)
        gross_port = float(np.prod(1.0 + port_daily))
        gross_rf   = float(np.prod(1.0 + rf_hold.values))
        excess_m = gross_port / gross_rf - 1.0

        monthly_rows.append({"date": reb_date, "excess_return": excess_m})

    monthly_df = pd.DataFrame(monthly_rows).sort_values("date").reset_index(drop=True)
    if collect_weights:
        weights_df = pd.DataFrame(monthly_weights_rows)
        if not weights_df.empty:
            weights_df = weights_df.sort_values("date").set_index("date")
        return monthly_df, weights_df
    return monthly_df


def sweep_lookbacks(
    rets_all: pd.DataFrame,
    rf_daily_all: pd.Series,
    start_dt: pd.Timestamp,
    lookbacks: List[int],
    tickers: List[str],
    W_RET: float,
    W_VOL: float,
    W_LOSS: float,
    bounds_by_ticker: Optional[Dict[str, Tuple[float, float]]] = None,
    return_details: bool = False,
) -> Tuple[pd.DataFrame, Optional[int]]:
    summary_rows = []
    details = {"monthly_weights_by_lb": {}} if return_details else None
    for lb in lookbacks:
        if return_details:
            monthly_df, weights_df = run_backtest_for_lookback(
                rets_all, rf_daily_all, start_dt, lb, tickers,
                W_RET=W_RET, W_VOL=W_VOL, W_LOSS=W_LOSS,
                bounds_by_ticker=bounds_by_ticker,
                collect_weights=True,
            )
            details["monthly_weights_by_lb"][int(lb)] = weights_df
        else:
            monthly_df = run_backtest_for_lookback(
                rets_all, rf_daily_all, start_dt, lb, tickers,
                W_RET=W_RET, W_VOL=W_VOL, W_LOSS=W_LOSS,
                bounds_by_ticker=bounds_by_ticker,
            )
        if monthly_df.empty:
            avg_ex = np.nan; sd_ex = np.nan; ratio = np.nan; n = 0
        else:
            avg_ex = float(monthly_df["excess_return"].mean())
            sd_ex  = float(monthly_df["excess_return"].std(ddof=1))
            ratio  = (avg_ex / sd_ex) if (np.isfinite(sd_ex) and sd_ex > 0) else np.nan
            n = int(len(monthly_df))
        summary_rows.append({
            "lookback_months": lb,
            "avg_monthly_excess": avg_ex,
            "sd_monthly_excess": sd_ex,
            "return_sd_ratio": ratio,
            "n_periods": n,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("lookback_months").reset_index(drop=True)
    best_lb = None
    if summary_df["return_sd_ratio"].notna().any():
        best_idx = summary_df["return_sd_ratio"].idxmax()
        best_lb = int(summary_df.loc[best_idx, "lookback_months"]) if np.isfinite(summary_df.loc[best_idx, "return_sd_ratio"]) else None
    if return_details:
        return summary_df, best_lb, details
    return summary_df, best_lb

def optimize_latest_with_best_lookback(
    px: pd.DataFrame,
    start_dt: pd.Timestamp,
    lookbacks: List[int],
    W_RET: float,
    W_VOL: float,
    W_LOSS: float,
    bounds_by_ticker: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.Series, dict, pd.DataFrame]:
    rets_all = px.pct_change().dropna(how="any")
    rf_daily_all = load_risk_free_series(rets_all.index, RF_ANNUAL, USE_RF_CSV)
    tickers = list(rets_all.columns)
    summary_df, best_lb = sweep_lookbacks(rets_all, rf_daily_all, start_dt, lookbacks, tickers, W_RET, W_VOL, W_LOSS, bounds_by_ticker)
    if best_lb is None:
        raise RuntimeError("No valid lookback produced a finite return/sd ratio.")

    # Identify latest rebalance month in test span
    rets_in_test = rets_all.loc[start_dt:]
    month_starts = rets_in_test.groupby([rets_in_test.index.year, rets_in_test.index.month]).apply(lambda x: x.index.min())
    month_starts = pd.to_datetime(month_starts.values)
    if len(month_starts) == 0:
        raise RuntimeError("Not enough data to identify a rebalance month.")
    reb_date = month_starts[-1]

    # Build lookback window
    lb_end = reb_date - pd.Timedelta(days=1)
    lb_start = lb_end - pd.DateOffset(months=best_lb) + pd.Timedelta(days=1)
    rets_win = rets_all.loc[lb_start:lb_end]
    rf_win = rf_daily_all.loc[rets_win.index]

    w_opt = optimize_weights_on_window(
        rets_window=rets_win,
        rf_window=rf_win,
        tickers=tickers,
        W_RET=W_RET,
        W_VOL=W_VOL,
        W_LOSS=W_LOSS,
        bounds_by_ticker=bounds_by_ticker,
    )

    weights = pd.Series(w_opt, index=tickers, name="weight")

    # Simple stats: annualized return/vol from the hold period following the last rebalance month
    # (re-using monthly approach for consistency is left to the caller; we return sweep summary separately)
    stats = {"best_lookback": int(best_lb)}
    return weights, stats, summary_df

def optimize_as_of_with_best_lookback(
    px: pd.DataFrame,
    start_dt: pd.Timestamp,
    lookbacks: List[int],
    as_of_date: pd.Timestamp,
    W_RET: float,
    W_VOL: float,
    W_LOSS: float,
    bounds_by_ticker: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.Series, dict, pd.DataFrame]:
    """Optimize weights for a lookback that ends at the last available trading day on or before `as_of_date`.
    The best lookback (by return/sd) is selected via a historical sweep starting at `start_dt`.
    Returns (weights_series, stats_dict, sweep_summary_df).
    """
    # Compute full daily returns and clip to as_of_date inclusively
    rets_all = px.pct_change().dropna(how="any")
    as_of_date = pd.to_datetime(as_of_date)
    rets_thru_asof = rets_all.loc[:as_of_date]
    if rets_thru_asof.empty:
        raise RuntimeError("No return data available on or before the selected 'as of' date.")

    # Risk-free series over full index (aligned later to the window)
    rf_daily_all = load_risk_free_series(rets_all.index, RF_ANNUAL, USE_RF_CSV)
    tickers = list(rets_all.columns)

    # Sweep to choose best lookback using full history from start_dt
    summary_df, best_lb, details = sweep_lookbacks(
        rets_all, rf_daily_all, start_dt, lookbacks, tickers,
        W_RET, W_VOL, W_LOSS, bounds_by_ticker,
        return_details=True,
    )
    if best_lb is None:
        raise RuntimeError("No valid lookback produced a finite return/sd ratio.")

    # Identify the last available trading day on/before as_of_date
    lb_end = rets_thru_asof.index.max()

    # Build lookback window ending at lb_end
    lb_start = lb_end - pd.DateOffset(months=best_lb) + pd.Timedelta(days=1)
    rets_win = rets_all.loc[lb_start:lb_end]
    if len(rets_win) < MIN_WINDOW_DAYS:
        raise RuntimeError(
            f"Insufficient data in lookback window ending {lb_end.date()} for best lookback {best_lb} months."
        )
    rf_win = rf_daily_all.loc[rets_win.index]

    # Optimize on the lookback window
    w_opt = optimize_weights_on_window(
        rets_window=rets_win,
        rf_window=rf_win,
        tickers=tickers,
        W_RET=W_RET,
        W_VOL=W_VOL,
        W_LOSS=W_LOSS,
        bounds_by_ticker=bounds_by_ticker,
    )

    weights = pd.Series(w_opt, index=tickers, name="weight")
    stats = {
        "best_lookback": int(best_lb),
        "as_of_date": pd.to_datetime(lb_end).date().isoformat(),
        "monthly_weights": details.get("monthly_weights_by_lb", {}).get(int(best_lb)) if details else None,
    }
    return weights, stats, summary_df



# ---------- Main ----------
def main():
    files = discover_etf_files(INPUT_DIR)
    if not files:
        raise RuntimeError(f"No .xlsx files found in {INPUT_DIR.resolve()}")

    # Load prices panel
    px = load_prices(files)
    if BACKTEST_END is not None:
        px = px.loc[:pd.to_datetime(BACKTEST_END)]
    px = px.ffill()

    # Prepare earliest data needed across ALL lookbacks
    start_dt = pd.to_datetime(BACKTEST_START)
    max_lb = max(LOOKBACK_MONTHS_LIST)
    first_needed = start_dt - pd.DateOffset(months=max_lb) - pd.Timedelta(days=5)
    px = px.loc[first_needed:]

    # Daily returns & RF
    rets_all = px.pct_change().dropna(how="any")
    rf_daily_all = load_risk_free_series(rets_all.index, RF_ANNUAL, USE_RF_CSV)
    tickers = list(rets_all.columns)

    # Sweep lookbacks
    summary_df, best_lb = sweep_lookbacks(rets_all, rf_daily_all, start_dt, LOOKBACK_MONTHS_LIST, tickers, W_RET, W_VOL, W_LOSS)

    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
    print("\nReturn / SD by lookback (using monthly EXCESS returns):")
    print(summary_df)

    if best_lb is not None:
        print(f"\nBest lookback by return/sd ratio: {best_lb} months")
    else:
        print("\nNo valid lookback produced a finite return/sd ratio.")

if __name__ == "__main__":
    main()
