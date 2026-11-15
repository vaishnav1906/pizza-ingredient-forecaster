"""Pizza Ingredient Forecaster - Streamlit app.   Run:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import forecast_lib as fl
from generate_data import generate

HERE = Path(__file__).parent
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Pizza Ingredient Forecaster", page_icon="🍕", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_sample() -> pd.DataFrame:
    path = HERE / "sales.csv"
    return pd.read_csv(path) if path.exists() else generate()


@st.cache_data
def load_recipes() -> pd.DataFrame:
    return pd.read_csv(HERE / "recipes.csv").set_index("Pizza")


@st.cache_data
def run_backtest(daily: pd.DataFrame) -> dict:
    return fl.backtest(daily)


@st.cache_data
def run_forecast(daily: pd.DataFrame, rainy: tuple, holiday: tuple, lo: float, hi: float) -> pd.DataFrame:
    return fl.forecast(daily, 7, dict(rainy), dict(holiday), lo, hi)


st.title("🍕 Pizza Ingredient Forecaster")
st.caption("Forecast how many pizzas you'll sell, then turn that into exactly what to order.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    source = st.radio("Sales data", ["Sample data (synthetic)", "Upload my own CSV"])
    raw = load_sample()
    if source.startswith("Upload"):
        up = st.file_uploader("CSV with columns: date, pizza, qty", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
        else:
            st.info("Using sample data until you upload a file.")
    try:
        sales = fl.clean_sales(raw)
        daily = fl.build_daily(sales)
        bt = run_backtest(daily)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    suggested = min(max(round(bt["rel_q80"] * 20) / 20, 0.05), 0.25)
    buffer_pct = st.slider("Safety buffer", 0, 30, int(round(suggested * 100)), 5, format="%d%%",
                           help="Extra stock on top of the forecast. Default comes from the model's "
                                "backtest error, so you're covered on ~80% of days.")
    buffer = buffer_pct / 100
    st.caption(f"Data: {daily.index.min():%d %b %Y} to {daily.index.max():%d %b %Y} "
               f"({len(daily)} days, {sales['pizza'].nunique()} pizzas)")

tabs = st.tabs(["Order list", "7-day outlook", "Average daily needs",
                "Model accuracy", "Business impact", "Data & recipes"])

# ------------------------------------------------------------------ recipes (rendered first so every tab uses edits)
with tabs[5]:
    st.subheader("Recipes (grams per pizza)")
    st.caption("Edit any cell to match your real recipes. Every tab updates automatically.")
    rec_in = load_recipes().reset_index()
    rec_out = st.data_editor(rec_in, disabled=["Pizza"], hide_index=True, use_container_width=True, key="recipes")
    recipes = rec_out.set_index("Pizza").apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    unknown = set(sales["pizza"]) - set(recipes.index)
    if unknown:
        st.warning(f"No recipe for: {', '.join(sorted(unknown))}. Add them to recipes.csv; they're ignored for now.")

    st.subheader("Sales history")
    c1, c2 = st.columns([2, 1])
    fig = px.line(daily.reset_index(names="date"), x="date", y="total", title="Pizzas sold per day")
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    c1.plotly_chart(fig, use_container_width=True)
    c2.dataframe(sales.tail(12).assign(date=sales["date"].tail(12).dt.date), hide_index=True,
                 use_container_width=True)
    st.markdown("**Using your own data?** Upload a CSV with `date, pizza, qty` (one row per pizza per day). "
                "Optional columns `rainy` and `holiday` (0/1) improve accuracy. Need at least 90 days.")

# ------------------------------------------------------------------ forecast inputs
future_dates = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=7)
with st.expander("Adjust conditions for the next 7 days (rain / holiday)"):
    cond = pd.DataFrame({"Date": future_dates.date, "Day": [DAYS[d.dayofweek] for d in future_dates],
                         "Rainy": False, "Holiday": [bool(fl.default_holiday(d)) for d in future_dates]})
    cond = st.data_editor(cond, disabled=["Date", "Day"], hide_index=True, key="cond")
rainy = tuple((d, int(v)) for d, v in zip(future_dates, cond["Rainy"]))
holiday = tuple((d, int(v)) for d, v in zip(future_dates, cond["Holiday"]))

fc = run_forecast(daily, rainy, holiday, bt["rel_q10"], bt["rel_q90"])
mix = fl.pizza_mix(sales)
# Whole-pizza plan per day (sums exactly to the rounded forecast), then exact gram maths.
plans = {d: fl.plan_pizzas(fc.loc[d, "pred"], d, mix) for d in fc.index}
need_units = pd.DataFrame({d: fl.to_10g_units(fl.ingredient_grams(p, recipes)) for d, p in plans.items()}).T
need_units = need_units.loc[:, need_units.sum() > 0]  # day x ingredient, in 10 g units (0.01 kg)
KG = st.column_config.NumberColumn(format="%.2f")

# ------------------------------------------------------------------ tab 1: order list
with tabs[0]:
    if unknown:
        st.warning(f"No recipe for {', '.join(sorted(unknown))}, so ingredient totals below are understated. "
                   "Add the recipe in the Data & recipes tab.")
    labels = {d: f"{DAYS[d.dayofweek]} {d:%d %b}" for d in fc.index}
    day = st.selectbox("Forecast day", list(fc.index), format_func=labels.get)
    row = fc.loc[day]
    total = int(plans[day].sum())  # == round_half_up(forecast)
    last_same = daily["total"].get(day - pd.Timedelta(days=7))
    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast pizzas", f"{total}",
              None if last_same is None else f"{total - int(last_same):+d} vs last {DAYS[day.dayofweek]}")
    m2.metric("Likely range", f"{fl.round_half_up(row['low'])} - {fl.round_half_up(row['high'])}")
    m3.metric("Safety buffer", f"{buffer_pct}%")

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(f"**Pizza mix** ({total} pizzas)")
        figm = px.bar(plans[day].sort_values(), orientation="h", labels={"value": "Pizzas", "pizza": ""})
        figm.update_layout(showlegend=False, height=380, margin=dict(t=10, b=10))
        st.plotly_chart(figm, use_container_width=True)
    with right:
        st.markdown("**Ingredients to order** (enter what you already have in stock)")
        units = need_units.loc[day]
        units = units[units > 0]
        base = pd.DataFrame({"Ingredient": units.index,
                             "Needed (kg)": units.values / 100,
                             "With buffer (kg)": fl.with_buffer_units(units, buffer_pct).values / 100,
                             "In stock (kg)": 0.0})
        edited = st.data_editor(
            base, hide_index=True, use_container_width=True,
            disabled=["Ingredient", "Needed (kg)", "With buffer (kg)"], key=f"stock_{day:%Y%m%d}",
            column_config={"Needed (kg)": KG, "With buffer (kg)": KG,
                           "In stock (kg)": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.2f")})
        stock = pd.to_numeric(edited["In stock (kg)"], errors="coerce").fillna(0).clip(lower=0)
        edited["To order (kg)"] = (edited["With buffer (kg)"] - stock).clip(lower=0).round(2)
        st.dataframe(edited[["Ingredient", "To order (kg)"]], hide_index=True, use_container_width=True,
                     column_config={"To order (kg)": KG})
        st.download_button("Download order list (CSV)", edited.to_csv(index=False),
                           f"order_{day:%Y%m%d}.csv", "text/csv")

# ------------------------------------------------------------------ tab 2: 7-day outlook
with tabs[1]:
    totals = [int(plans[d].sum()) for d in fc.index]
    lows = [fl.round_half_up(v) for v in fc["low"]]
    highs = [fl.round_half_up(v) for v in fc["high"]]
    figf = go.Figure()
    figf.add_bar(x=[labels[d] for d in fc.index], y=totals,
                 error_y=dict(type="data", symmetric=False,
                              array=[h - t for h, t in zip(highs, totals)],
                              arrayminus=[t - lo for t, lo in zip(totals, lows)]),
                 name="Forecast")
    figf.update_layout(title=f"Forecast pizzas per day (week total {sum(totals)}; bars show likely range)",
                       height=380, yaxis_title="Pizzas", margin=dict(t=50, b=10))
    st.plotly_chart(figf, use_container_width=True)
    st.markdown("**Ingredients needed per day (kg, before buffer)**")
    wk = need_units.copy()
    wk.index = [labels[d] for d in wk.index]
    wk.loc["WEEK TOTAL"] = wk.sum()  # exact sum of the cells shown
    st.dataframe((wk / 100).T, use_container_width=True,
                 column_config={c: KG for c in wk.index})

# ------------------------------------------------------------------ tab 3: average daily needs
with tabs[2]:
    hist = fl.history_ingredients(sales, recipes)
    hist = hist.reindex(daily.index[-56:], fill_value=0) / 1000  # exactly 8 of each weekday
    by_dow = hist.groupby(hist.index.dayofweek).mean()
    by_dow.index = [DAYS[i] for i in by_dow.index]
    by_dow = by_dow.loc[:, by_dow.sum() > 0.0005]
    st.markdown("**Average kg needed per day, by weekday** (last 8 weeks of sales)")
    figh = px.imshow(by_dow.round(1), text_auto=True, aspect="auto", color_continuous_scale="Oranges")
    figh.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(figh, use_container_width=True)

    groups = {"Weekday (Mon-Thu)": by_dow.loc[["Mon", "Tue", "Wed", "Thu"]].mean(),
              "Friday": by_dow.loc["Fri"], "Saturday": by_dow.loc["Sat"], "Sunday": by_dow.loc["Sun"]}
    summary = pd.DataFrame(groups)
    weekday_avg = summary["Weekday (Mon-Thu)"]
    ratio = summary[["Saturday", "Sunday"]].mean(axis=1) / weekday_avg.where(weekday_avg > 0)
    summary = summary.round(1)
    summary["Weekend vs weekday"] = ratio.map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}x")
    st.markdown("**Typical day by type (kg)**")
    st.dataframe(summary, use_container_width=True)
    avg_pizzas = daily["total"].tail(56).groupby(daily.index[-56:].dayofweek).mean()
    r = fl.round_half_up
    st.caption(f"Average pizzas sold: Mon-Thu {r(avg_pizzas.loc[0:3].mean())}/day, Fri {r(avg_pizzas[4])}, "
               f"Sat {r(avg_pizzas[5])}, Sun {r(avg_pizzas[6])}")

# ------------------------------------------------------------------ tab 4: model accuracy
with tabs[3]:
    scores = pd.DataFrame(bt["scores"]).T
    base_mae, model_mae = scores["MAE"].iloc[0], scores["MAE"].iloc[1]
    st.markdown(f"Tested on the **last 28 days** the model never saw. "
                f"It cut average error by **{(1 - model_mae / base_mae) * 100:.0f}%** vs the naive method.")
    st.dataframe(scores.round(1), use_container_width=True)
    fr = bt["frame"].rename(columns={"actual": "Actual", "model": "Model", "baseline": "Naive baseline"})
    figa = px.line(fr, labels={"value": "Pizzas", "index": "", "variable": ""})
    figa.update_layout(height=360, margin=dict(t=10, b=10))
    st.plotly_chart(figa, use_container_width=True)
    figi = px.bar(bt["importance"], orientation="h", labels={"value": "Importance", "index": ""})
    figi.update_layout(title="What drives the forecast", showlegend=False, height=330, margin=dict(t=40, b=10))
    st.plotly_chart(figi, use_container_width=True)

# ------------------------------------------------------------------ tab 5: business impact
with tabs[4]:
    st.markdown("Replaying the last 28 days: what if the owner had ordered using each method?")
    sim = fl.simulate_policies(bt["frame"], daily, buffer).round(1)
    st.dataframe(sim, use_container_width=True)
    avg, peak, fcst = sim.loc[fl.POLICY_AVG], sim.loc[fl.POLICY_PEAK], sim.loc[fl.POLICY_FCST]

    def change(name, a, b, unit=""):
        verb = "cuts" if b < a else ("raises" if b > a else "keeps")
        return f"{verb} {name} from {a:g}{unit} to {b:g}{unit}" if verb != "keeps" else f"keeps {name} at {a:g}{unit}"

    st.info(f"Versus prepping for a peak day, forecasting {change('waste', peak[fl.WASTE], fcst[fl.WASTE], '%')}. "
            f"Versus prepping for an average day, it {change('lost pizzas', avg[fl.LOST], fcst[fl.LOST])}.")
    st.caption("Waste = pizzas' worth of ingredients prepared but not used that day (assumes leftovers aren't "
               "carried over). Lost = demand above what was prepared. Move the safety buffer to see the trade-off.")
