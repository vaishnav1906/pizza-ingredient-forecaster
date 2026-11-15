"""Hand-checkable maths tests.   Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import forecast_lib as fl  # noqa: E402
from generate_data import generate  # noqa: E402

RECIPES = pd.read_csv(ROOT / "recipes.csv").set_index("Pizza")


@pytest.fixture(scope="module")
def sample():
    sales = fl.clean_sales(generate(days=365, seed=42, end="2026-11-14"))
    daily = fl.build_daily(sales)
    return sales, daily, fl.backtest(daily)


# ---------------------------------------------------------------- rounding
@pytest.mark.parametrize("x, want", [(138.5, 139), (137.5, 138), (2.4999, 2), (0.5, 1), (0, 0), (160.49, 160)])
def test_round_half_up(x, want):
    assert fl.round_half_up(x) == want


def test_allocate_sums_exactly_and_is_proportional():
    shares = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    for total in range(0, 300):
        out = fl.allocate(total, shares)
        assert out.sum() == total
        assert (out >= 0).all()
        assert (abs(out - shares * total) < 1).all()  # never more than one pizza off its exact share
    assert fl.allocate(10, shares).to_dict() == {"A": 5, "B": 3, "C": 2}
    assert fl.allocate(0, shares).sum() == 0
    assert fl.allocate(5, pd.Series({"A": 0.0, "B": 0.0})).sum() == 0


# ---------------------------------------------------------------- ingredients
def test_ingredient_grams_by_hand():
    counts = pd.Series({"Margherita": 2, "Pepperoni": 1})
    g = fl.ingredient_grams(counts, RECIPES)
    assert g["Dough"] == 3 * 250
    assert g["Mozzarella"] == 2 * 120 + 120
    assert g["Pizza Sauce"] == 3 * 60
    assert g["Tomato"] == 2 * 20 and g["Basil"] == 2 * 5 and g["Pepperoni"] == 50
    assert g["Paneer"] == 0


def test_ten_gram_units_and_buffer_are_exact():
    grams = pd.Series([34500, 16015, 16014, 5, 4, 0])
    units = fl.to_10g_units(grams)
    assert units.tolist() == [3450, 1602, 1601, 1, 0, 0]  # half-up to the nearest 10 g
    assert fl.with_buffer_units(units, 10).tolist() == [3795, 1762, 1761, 1, 0, 0]
    assert fl.with_buffer_units(units, 0).tolist() == units.tolist()
    # "With buffer" always equals the shown "Needed" x (1 + buffer), rounded half-up to 0.01 kg
    for pct in range(0, 31, 5):
        for u in range(0, 5000):
            want = int(np.floor(u * (100 + pct) / 100 + 0.5 + 1e-9))
            assert fl.with_buffer_units(u, pct) == want


def test_every_plan_matches_headline_and_dough(sample):
    sales, daily, bt = sample
    fc = fl.forecast(daily, 7, rel_lo=bt["rel_q10"], rel_hi=bt["rel_q90"])
    mix = fl.pizza_mix(sales)
    for d in fc.index:
        plan = fl.plan_pizzas(fc.loc[d, "pred"], d, mix)
        assert plan.sum() == fl.round_half_up(fc.loc[d, "pred"])  # chart bars add up to the headline number
        g = fl.ingredient_grams(plan, RECIPES)
        assert g["Dough"] == 250 * plan.sum()  # every pizza uses 250 g dough
        assert g["Mozzarella"] == (plan * RECIPES["Mozzarella"]).sum()


# ---------------------------------------------------------------- mix / history
def test_pizza_mix_shares_sum_to_one(sample):
    mix = fl.pizza_mix(sample[0])
    assert np.allclose(mix.sum(), 1.0)
    assert (mix >= 0).all().all()


def test_history_counts_missing_days_as_zero():
    sales = fl.clean_sales(pd.DataFrame({"date": ["2026-01-01", "2026-01-03"],
                                         "pizza": ["Margherita", "Margherita"], "qty": [4, 2]}))
    h = fl.history_ingredients(sales, RECIPES)
    assert len(h) == 3 and h["Dough"].tolist() == [1000, 0, 500]
    assert fl.build_daily(sales)["total"].tolist() == [4, 0, 2]


# ---------------------------------------------------------------- model
def test_features_never_look_at_the_last_6_days():
    idx = pd.date_range("2026-01-01", periods=60)
    daily = pd.DataFrame({"total": np.arange(60.0), "holiday": 0, "rainy": 0}, index=idx)
    before = fl.make_features(daily).iloc[-1]
    changed = daily.copy()
    changed.iloc[-7:, 0] = 9999  # tamper with the target day and the 6 days before it
    after = fl.make_features(changed).iloc[-1]
    assert before.equals(after)


def test_baseline_is_average_of_same_weekday_last_4_weeks(sample):
    _, daily, bt = sample
    d = bt["frame"].index[0]
    manual = np.mean([daily.loc[d - pd.Timedelta(days=7 * k), "total"] for k in (1, 2, 3, 4)])
    assert bt["frame"].loc[d, "baseline"] == pytest.approx(manual)


def test_backtest_holdout_and_scores(sample):
    _, daily, bt = sample
    fr = bt["frame"]
    assert len(fr) == 28 and fr.index[-1] == daily.index[-1]
    mae = (fr["actual"] - fr["model"]).abs().mean()
    assert bt["scores"]["Gradient Boosting model"]["MAE"] == pytest.approx(mae)
    assert mae < bt["scores"]["Naive baseline (avg of same weekday, last 4 wks)"]["MAE"]
    assert bt["rel_q10"] <= bt["rel_q80"] <= bt["rel_q90"]


def test_forecast_range_contains_prediction(sample):
    _, daily, bt = sample
    fc = fl.forecast(daily, 7, rel_lo=bt["rel_q10"], rel_hi=bt["rel_q90"])
    assert len(fc) == 7 and fc.index[0] == daily.index[-1] + pd.Timedelta(days=1)
    assert ((fc["low"] <= fc["pred"]) & (fc["pred"] <= fc["high"])).all()
    rainy = fl.forecast(daily, 7, rainy={fc.index[0]: 1})
    assert rainy["pred"].iloc[0] > fc["pred"].iloc[0]  # rain raises demand in this data


# ---------------------------------------------------------------- business impact
def test_policy_stats_by_hand():
    s = fl._policy_stats(pd.Series([10.0, 10.0]), pd.Series([8.0, 12.0]))
    assert s[fl.WASTE] == pytest.approx(2 / 20 * 100)
    assert s[fl.STOCKOUT_DAYS] == 1
    assert s[fl.LOST] == 2
    assert s[fl.FILL] == pytest.approx((1 - 2 / 20) * 100)


def test_simulation_policy_definitions(sample):
    _, daily, bt = sample
    fr = bt["frame"]
    sim = fl.simulate_policies(fr, daily, 0.10)
    d = fr.index[0]
    prev28 = daily["total"].loc[d - pd.Timedelta(days=28): d - pd.Timedelta(days=1)]
    assert len(prev28) == 28
    avg_order = pd.Series(prev28.mean() * 1.10, index=[d])
    assert fl._policy_stats(avg_order, fr["actual"].loc[[d]])[fl.LOST] == pytest.approx(
        max(fr.loc[d, "actual"] - prev28.mean() * 1.10, 0))
    assert sim.loc[fl.POLICY_FCST, fl.WASTE] < sim.loc[fl.POLICY_PEAK, fl.WASTE]
    assert sim.loc[fl.POLICY_FCST, fl.LOST] < sim.loc[fl.POLICY_AVG, fl.LOST]
