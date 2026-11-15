"""Forecasting + ingredient logic for the pizza demand app.

Pipeline:
  sales (date, pizza, qty)  ->  daily totals  ->  features (weekday, holiday, rain, lags)
  -> Gradient Boosting forecast of total pizzas per day (7-day horizon)
  -> split into pizza varieties using recent sales mix
  -> multiply by recipes -> ingredient quantities
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

HOLIDAYS_MD = {(1, 1), (2, 14), (10, 31), (12, 24), (12, 25), (12, 31)}
FEATURES = ["dow", "month", "is_weekend", "holiday", "rainy",
            "lag7", "lag14", "lag21", "lag28", "mean4", "trend7"]
MIN_DAYS = 90
MAX_HORIZON = 7  # all lag features are >= 7 days, so up to 7 days ahead needs no recursion


def default_holiday(d: pd.Timestamp) -> int:
    return int((d.month, d.day) in HOLIDAYS_MD)


# ---------------------------------------------------------------- data prep
def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"date", "pizza", "qty"} - set(df.columns)
    if missing:
        raise ValueError(f"Sales file is missing columns: {sorted(missing)}. Need: date, pizza, qty")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0).clip(lower=0)
    return out


def build_daily(sales: pd.DataFrame) -> pd.DataFrame:
    total = sales.groupby("date")["qty"].sum()
    idx = pd.date_range(total.index.min(), total.index.max(), freq="D")
    daily = pd.DataFrame(index=idx)
    daily["total"] = total.reindex(idx).fillna(0)
    if "holiday" in sales.columns:
        daily["holiday"] = sales.groupby("date")["holiday"].max().reindex(idx).fillna(0).astype(int)
    else:
        daily["holiday"] = [default_holiday(d) for d in idx]
    if "rainy" in sales.columns:
        daily["rainy"] = sales.groupby("date")["rainy"].max().reindex(idx).fillna(0).astype(int)
    else:
        daily["rainy"] = 0
    return daily


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["dow"] = df.index.dayofweek
    f["month"] = df.index.month
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["holiday"] = df["holiday"]
    f["rainy"] = df["rainy"]
    for k in (7, 14, 21, 28):
        f[f"lag{k}"] = df["total"].shift(k)
    f["mean4"] = f[["lag7", "lag14", "lag21", "lag28"]].mean(axis=1, skipna=False)  # naive baseline
    f["trend7"] = df["total"].shift(7).rolling(7).mean()
    return f


def _model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=0)


def _scores(actual: pd.Series, pred: pd.Series) -> dict:
    err = actual - pred
    return {"MAE": float(err.abs().mean()),
            "MAPE %": float((err.abs() / actual.replace(0, np.nan)).mean() * 100),
            "WAPE %": float(err.abs().sum() / actual.sum() * 100)}


# ---------------------------------------------------------------- backtest
def backtest(daily: pd.DataFrame, test_days: int = 28) -> dict:
    """Hold out the last `test_days`, train on everything before, compare to naive baseline."""
    if len(daily) < MIN_DAYS:
        raise ValueError(f"Need at least {MIN_DAYS} days of sales history (found {len(daily)}).")
    f = make_features(daily)
    valid = f.dropna().index
    cut = daily.index[-test_days]
    train_idx, test_idx = valid[valid < cut], valid[valid >= cut]
    model = _model().fit(f.loc[train_idx, FEATURES], daily.loc[train_idx, "total"])
    frame = pd.DataFrame({
        "actual": daily.loc[test_idx, "total"],
        "model": np.clip(model.predict(f.loc[test_idx, FEATURES]), 0, None),
        "baseline": f.loc[test_idx, "mean4"],
    })
    rel = ((frame["actual"] - frame["model"]) / frame["model"].replace(0, np.nan)).dropna()
    return {
        "frame": frame,
        "scores": {"Naive baseline (avg of same weekday, last 4 wks)": _scores(frame["actual"], frame["baseline"]),
                   "Gradient Boosting model": _scores(frame["actual"], frame["model"])},
        "rel_q80": float(max(rel.quantile(0.80), 0.0)),
        "rel_q10": float(rel.quantile(0.10)),
        "rel_q90": float(rel.quantile(0.90)),
        "importance": pd.Series(model.feature_importances_, index=FEATURES).sort_values(),
    }


# ---------------------------------------------------------------- forecast
def forecast(daily: pd.DataFrame, horizon: int = 7, rainy: dict | None = None,
             holiday: dict | None = None, rel_lo: float = -0.12, rel_hi: float = 0.12) -> pd.DataFrame:
    horizon = min(horizon, MAX_HORIZON)
    rainy, holiday = rainy or {}, holiday or {}
    f = make_features(daily)
    valid = f.dropna().index
    model = _model().fit(f.loc[valid, FEATURES], daily.loc[valid, "total"])

    future = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    ext = pd.concat([daily, pd.DataFrame({
        "total": np.nan,
        "holiday": [int(holiday.get(d, default_holiday(d))) for d in future],
        "rainy": [int(rainy.get(d, 0)) for d in future],
    }, index=future)])
    fx = make_features(ext).loc[future]
    pred = np.clip(model.predict(fx[FEATURES]), 0, None)
    return pd.DataFrame({
        "pred": pred,
        "low": pred * (1 + rel_lo),
        "high": pred * (1 + rel_hi),
        "baseline": fx["mean4"],
        "holiday": ext.loc[future, "holiday"],
        "rainy": ext.loc[future, "rainy"],
    }, index=future)


# ---------------------------------------------------------------- pizza mix + ingredients
def pizza_mix(sales: pd.DataFrame, days: int = 56) -> pd.DataFrame:
    """Share of each pizza in total sales, for weekdays and weekends (last `days` days)."""
    s = sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=days)].copy()
    s["kind"] = np.where(s["date"].dt.dayofweek >= 5, "weekend", "weekday")
    mix = s.pivot_table(index="pizza", columns="kind", values="qty", aggfunc="sum").fillna(0)
    for c in ("weekday", "weekend"):
        if c not in mix:
            mix[c] = mix.get("weekday", mix.get("weekend"))
    return mix / mix.sum()


def split_by_mix(total: float, date: pd.Timestamp, mix: pd.DataFrame) -> pd.Series:
    return mix["weekend" if date.dayofweek >= 5 else "weekday"] * total


def round_half_up(x: float) -> int:
    """Shop-style rounding (138.5 -> 139). Python's round()/format use banker's rounding (138.5 -> 138)."""
    return int(np.floor(float(x) + 0.5))


def allocate(total: int, shares: pd.Series) -> pd.Series:
    """Split a whole number of pizzas across varieties by share (largest-remainder method).
    The parts are whole numbers and always add up exactly to `total`."""
    shares = shares.clip(lower=0).fillna(0)
    if total <= 0 or shares.sum() == 0:
        return pd.Series(0, index=shares.index, dtype="int64")
    raw = shares / shares.sum() * total
    out = np.floor(raw).astype("int64")
    leftover = int(total - out.sum())
    top = (raw - out).sort_values(ascending=False, kind="stable").index[:leftover]
    out.loc[top] += 1
    return out


def plan_pizzas(pred: float, date: pd.Timestamp, mix: pd.DataFrame) -> pd.Series:
    """Whole-pizza plan for a day: rounded forecast split across the menu by the recent mix."""
    return allocate(round_half_up(pred), mix["weekend" if date.dayofweek >= 5 else "weekday"])


def ingredient_need(counts: pd.Series, recipes: pd.DataFrame) -> pd.Series:
    """counts: pizzas per variety; recipes: pizza x ingredient grams  ->  grams per ingredient."""
    return counts.reindex(recipes.index).fillna(0) @ recipes


def ingredient_grams(counts: pd.Series, recipes: pd.DataFrame) -> pd.Series:
    """Whole grams per ingredient (recipes may hold decimals)."""
    return ingredient_need(counts, recipes).round().astype("int64")


def to_10g_units(grams):
    """Grams -> units of 10 g (= 0.01 kg), rounded half-up with exact integer maths."""
    return (grams + 5) // 10


def with_buffer_units(units, buffer_pct: int):
    """Add a whole-percent buffer to 10 g units, rounded half-up with exact integer maths."""
    return (units * (100 + int(buffer_pct)) + 50) // 100


def history_ingredients(sales: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    """date x ingredient grams. Days with no sales rows count as 0, matching build_daily()."""
    wide = sales.pivot_table(index="date", columns="pizza", values="qty", aggfunc="sum").fillna(0)
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(index=full, columns=recipes.index, fill_value=0)
    return wide @ recipes


# ---------------------------------------------------------------- business impact
POLICY_AVG = "Fixed: average day (last 28-day mean + 10%)"
POLICY_PEAK = "Fixed: peak day (90th percentile of last 28 days)"
POLICY_FCST = "Forecast-driven (this app)"
WASTE, STOCKOUT_DAYS, LOST, FILL = ("Waste (% of pizzas prepared)", "Stock-out days (of period)",
                                    "Pizzas lost to stock-outs", "Fill rate %")


def _policy_stats(order: pd.Series, actual: pd.Series) -> dict:
    waste = (order - actual).clip(lower=0)
    short = (actual - order).clip(lower=0)
    return {WASTE: waste.sum() / order.sum() * 100,
            STOCKOUT_DAYS: int((actual > order).sum()),
            LOST: float(short.sum()),
            FILL: (1 - short.sum() / actual.sum()) * 100}


def simulate_policies(bt_frame: pd.DataFrame, daily: pd.DataFrame, buffer: float) -> pd.DataFrame:
    actual = bt_frame["actual"]
    prev = daily["total"].shift(1)
    policies = {
        POLICY_AVG: prev.rolling(28).mean().reindex(actual.index) * 1.10,
        POLICY_PEAK: prev.rolling(28).quantile(0.9).reindex(actual.index),
        POLICY_FCST: bt_frame["model"] * (1 + buffer),
    }
    return pd.DataFrame({k: _policy_stats(v, actual) for k, v in policies.items()}).T
