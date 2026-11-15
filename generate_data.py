"""Generate realistic synthetic daily pizza sales for a 10-pizza menu.

Patterns baked in (so the forecasting model has something real to learn):
  * Weekday vs weekend: Fri is busy, Sat/Sun are the peak, Mon-Wed are quiet
  * Slow growth over the year (+10%) and mild seasonality
  * Holiday boost (+25%) and rainy-day boost (+10%, delivery orders)
  * Weekend mix tilts toward Pepperoni / Chicken pizzas
  * Random day-to-day noise

Run:  python generate_data.py          -> writes sales.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PIZZAS = {  # name: (base popularity, weekend tilt)
    "Margherita": (0.15, 0.90),
    "Pepperoni": (0.14, 1.15),
    "Veggie Supreme": (0.11, 0.95),
    "Paneer Tikka": (0.12, 1.00),
    "Chicken BBQ": (0.10, 1.20),
    "Farmhouse": (0.09, 1.00),
    "Corn & Cheese": (0.08, 0.90),
    "Mushroom Delight": (0.06, 0.85),
    "Hawaiian": (0.07, 1.00),
    "Spicy Jalapeno Chicken": (0.08, 1.15),
}
DOW_BASE = {0: 55, 1: 52, 2: 58, 3: 66, 4: 105, 5: 150, 6: 138}  # Mon..Sun
HOLIDAYS_MD = {(1, 1), (2, 14), (10, 31), (12, 24), (12, 25), (12, 31)}


def generate(days: int = 365, seed: int = 42, end: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end_date = pd.Timestamp(end) if end else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    names = list(PIZZAS)
    base_w = np.array([PIZZAS[n][0] for n in names])
    tilt = np.array([PIZZAS[n][1] for n in names])

    rows = []
    for i, d in enumerate(dates):
        rain_p = 0.35 if d.month in (6, 7, 8, 9) else 0.12
        rainy = int(rng.random() < rain_p)
        holiday = int((d.month, d.day) in HOLIDAYS_MD)
        mean = (
            DOW_BASE[d.dayofweek]
            * (1 + 0.10 * i / days)
            * (1 + 0.06 * np.sin(2 * np.pi * (d.month - 3) / 12))
            * (1.25 if holiday else 1.0)
            * (1.10 if rainy else 1.0)
        )
        total = max(0, int(round(mean * rng.lognormal(0, 0.08))))
        w = base_w * (tilt if d.dayofweek >= 5 else 1.0)
        counts = rng.multinomial(total, w / w.sum())
        for n, q in zip(names, counts):
            rows.append((d.date().isoformat(), n, int(q), rainy, holiday))
    return pd.DataFrame(rows, columns=["date", "pizza", "qty", "rainy", "holiday"])


if __name__ == "__main__":
    df = generate()
    df.to_csv("sales.csv", index=False)
    print(f"Wrote sales.csv: {len(df):,} rows, {df['date'].nunique()} days, {df['pizza'].nunique()} pizzas")
