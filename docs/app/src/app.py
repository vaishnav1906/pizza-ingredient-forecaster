"""Pizza Ingredient Forecaster - Streamlit app.   Run:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecast_lib as fl
import ui
from generate_data import generate

HERE = Path(__file__).parent
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURE_NAMES = {"dow": "Day of week", "month": "Month", "is_weekend": "Weekend", "holiday": "Holiday",
                 "rainy": "Rain", "lag7": "Sales 1 week ago", "lag14": "Sales 2 weeks ago",
                 "lag21": "Sales 3 weeks ago", "lag28": "Sales 4 weeks ago",
                 "mean4": "Same-weekday 4-week average", "trend7": "Recent weekly trend"}

st.set_page_config(page_title="Slice Planner", page_icon="🍕", layout="wide")
ui.inject_css()
T = ui.tokens()
KG = st.column_config.NumberColumn(format="%.2f")


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


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    ui.sidebar_brand()
    ui.section("Sales data")
    source = st.radio("Sales data", ["Sample data (synthetic)", "Upload my own CSV"], label_visibility="collapsed")
    raw, using_upload = load_sample(), False
    if source.startswith("Upload"):
        up = st.file_uploader("CSV with columns: date, pizza, qty", type="csv")
        if up is not None:
            raw, using_upload = pd.read_csv(up), True
        else:
            st.info("Showing sample data until you upload a file.", icon=":material/info:")
    try:
        sales = fl.clean_sales(raw)
        daily = fl.build_daily(sales)
        bt = run_backtest(daily)
    except ValueError as e:
        st.error(str(e), icon=":material/error:")
        st.stop()
    st.caption(f"{daily.index.min():%d %b %Y} to {daily.index.max():%d %b %Y} · "
               f"{len(daily)} days · {sales['pizza'].nunique()} pizzas")

    st.divider()
    ui.section("Safety buffer")
    suggested = min(max(round(bt["rel_q80"] * 20) / 20, 0.05), 0.25)
    buffer_pct = st.slider("Safety buffer", 0, 30, int(round(suggested * 100)), 5, format="%d%%",
                           label_visibility="collapsed",
                           help="Extra stock on top of the forecast. The default comes from the model's "
                                "backtest error, so you're covered on about 8 in 10 days.")
    buffer = buffer_pct / 100

    st.divider()
    ui.section("Next 7 days", "tick rain or holidays")
    future_dates = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=7)
    cond = pd.DataFrame({"Day": [f"{DAYS[d.dayofweek]} {d.day}" for d in future_dates],
                         "Rainy": False, "Holiday": [bool(fl.default_holiday(d)) for d in future_dates]})
    cond = st.data_editor(cond, disabled=["Day"], hide_index=True, width="stretch", key="cond",
                          column_config={"Day": st.column_config.TextColumn("Day", width="small"),
                                         "Rainy": st.column_config.CheckboxColumn("Rain", width="small"),
                                         "Holiday": st.column_config.CheckboxColumn("Holiday", width="small")})
rainy = tuple((d, int(v)) for d, v in zip(future_dates, cond["Rainy"]))
holiday = tuple((d, int(v)) for d, v in zip(future_dates, cond["Holiday"]))

# ------------------------------------------------------------------ hero
mape = bt["scores"]["Gradient Boosting model"]["MAPE %"]
ui.hero("Your sales data" if using_upload else "Sample data · synthetic",
        "Pizza Ingredient Forecaster",
        "Forecast how many pizzas you'll sell, then turn that into exactly what to order.",
        [("Data", f"{daily.index.min():%b %Y} to {daily.index.max():%b %Y}"),
         ("Menu", f"{sales['pizza'].nunique()} pizzas"),
         ("Forecast error", f"{mape:.1f}%"),
         ("Planning from", f"{DAYS[future_dates[0].dayofweek]} {future_dates[0]:%d %b}")])

tabs = st.tabs([":material/receipt_long: Order list", ":material/calendar_month: 7-day outlook",
                ":material/grid_view: Average needs", ":material/insights: Model accuracy",
                ":material/savings: Business impact", ":material/menu_book: Data & recipes"])

# ------------------------------------------------------------------ recipes (rendered first so every tab uses edits)
with tabs[5]:
    with st.container(border=True):
        ui.section("Recipes", "grams of each ingredient per pizza · edit any cell, every tab updates")
        rec_in = load_recipes().reset_index()
        rec_out = st.data_editor(rec_in, disabled=["Pizza"], hide_index=True, width="stretch", key="recipes")
    recipes = rec_out.set_index("Pizza").apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    unknown = set(sales["pizza"]) - set(recipes.index)
    if unknown:
        st.warning(f"No recipe for: {', '.join(sorted(unknown))}. Add them to recipes.csv; they're ignored for now.",
                   icon=":material/warning:")

    with st.container(border=True):
        ui.section("Sales history", "pizzas sold per day")
        hist_fig = go.Figure()
        hist_fig.add_scatter(x=daily.index, y=daily["total"], name="Daily", mode="lines",
                             line=dict(color=T["axis"], width=1),
                             hovertemplate="%{x|%a %d %b %Y}: %{y:.0f} pizzas<extra></extra>")
        hist_fig.add_scatter(x=daily.index, y=daily["total"].rolling(7).mean(), name="7-day average", mode="lines",
                             line=dict(color=T["brand"], width=2),
                             hovertemplate="%{x|%d %b %Y}: %{y:.1f} avg<extra></extra>")
        ui.show(ui.style(hist_fig, 300, legend=True))
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1, st.container(border=True):
        ui.section("Latest rows")
        st.dataframe(sales.tail(10).assign(date=sales["date"].tail(10).dt.date), hide_index=True, width="stretch")
    with c2, st.container(border=True):
        ui.section("Use your own data")
        st.markdown("Upload a CSV with **`date, pizza, qty`**, one row per pizza per day, from the sidebar.\n\n"
                    "- At least **90 days** of history\n- Optional **`rainy`** and **`holiday`** columns (0/1) "
                    "improve accuracy\n- Edit the recipes above to match your menu")

# ------------------------------------------------------------------ forecast + plans
fc = run_forecast(daily, rainy, holiday, bt["rel_q10"], bt["rel_q90"])
mix = fl.pizza_mix(sales)
# Whole-pizza plan per day (sums exactly to the rounded forecast), then exact gram maths.
plans = {d: fl.plan_pizzas(fc.loc[d, "pred"], d, mix) for d in fc.index}
need_units = pd.DataFrame({d: fl.to_10g_units(fl.ingredient_grams(p, recipes)) for d, p in plans.items()}).T
need_units = need_units.loc[:, need_units.sum() > 0]  # day x ingredient, in 10 g units (0.01 kg)
labels = {d: f"{DAYS[d.dayofweek]} {d:%d %b}" for d in fc.index}
totals = {d: int(plans[d].sum()) for d in fc.index}
lows = {d: fl.round_half_up(fc.loc[d, "low"]) for d in fc.index}
highs = {d: fl.round_half_up(fc.loc[d, "high"]) for d in fc.index}

# ------------------------------------------------------------------ tab 1: order list
with tabs[0]:
    if unknown:
        st.warning(f"No recipe for {', '.join(sorted(unknown))}, so ingredient totals below are understated. "
                   "Add the recipe in the Data & recipes tab.", icon=":material/warning:")
    day = st.segmented_control("Forecast day", list(fc.index), format_func=lambda d: labels[d],
                               default=fc.index[0], key=f"day_{fc.index[0]:%Y%m%d}",
                               label_visibility="collapsed") or fc.index[0]
    total = totals[day]
    last_same = daily["total"].get(day - pd.Timedelta(days=7))
    diff = None if last_same is None else total - int(last_same)
    units = need_units.loc[day]
    units = units[units > 0]
    buffered = fl.with_buffer_units(units, buffer_pct)
    top_item = "Dough" if "Dough" in buffered.index else buffered.idxmax()
    ui.kpi_row([
        ui.kpi("Forecast pizzas", f"{total}", f"vs last {DAYS[day.dayofweek]}" if diff is not None else "",
               pill="" if diff is None else ("no change" if diff == 0 else f"{'▲' if diff > 0 else '▼'} {abs(diff)}"),
               accent=True),
        ui.kpi("Likely range", f"{lows[day]}–{highs[day]}", "8 in 10 days land in this range"),
        ui.kpi("Safety buffer", f"{buffer_pct}%", "extra stock on top of the forecast"),
        ui.kpi(f"{top_item} to prep", f"{buffered[top_item] / 100:.2f}", f"incl. buffer, for {total} pizzas", unit="kg"),
    ])

    left, right = st.columns([1, 1.3], gap="medium")
    with left, st.container(border=True):
        ui.section("Pizza mix", f"{total} pizzas from the recent sales mix")
        p = plans[day].sort_values()
        fig = go.Figure(go.Bar(x=p.values, y=p.index, orientation="h", marker_color=T["brand"],
                               text=p.values, textposition="outside", cliponaxis=False,
                               textfont=dict(color=T["text2"]),
                               hovertemplate="%{y}: %{x} pizzas<extra></extra>"))
        fig = ui.style(fig, 420)
        fig.update_yaxes(showgrid=False, tickfont=dict(color=T["text2"]), ticklabelstandoff=8)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        ui.show(fig)
    with right, st.container(border=True):
        ui.section("Stock check", "kg · type what's already in the kitchen")
        base = pd.DataFrame({"Ingredient": units.index,
                             "Needed (kg)": units.values / 100,
                             "With buffer (kg)": buffered.values / 100,
                             "In stock (kg)": 0.0})
        edited = st.data_editor(
            base, hide_index=True, width="stretch", height=422,
            disabled=["Ingredient", "Needed (kg)", "With buffer (kg)"], key=f"stock_{day:%Y%m%d}",
            column_config={"Ingredient": st.column_config.TextColumn("Ingredient", width="small"),
                           "Needed (kg)": st.column_config.NumberColumn("Needed", format="%.2f", width="small"),
                           "With buffer (kg)": st.column_config.NumberColumn(f"+{buffer_pct}% buffer", format="%.2f",
                                                                             width="small"),
                           "In stock (kg)": st.column_config.NumberColumn("In stock ✎", min_value=0.0, step=0.01,
                                                                          format="%.2f", width="small")})
    stock = pd.to_numeric(edited["In stock (kg)"], errors="coerce").fillna(0).clip(lower=0)
    edited["To order (kg)"] = (edited["With buffer (kg)"] - stock).clip(lower=0).round(2)

    with st.container(border=True):
        h1, h2 = st.columns([3, 1], vertical_alignment="center")
        with h1:
            ui.section("Shopping list", f"{labels[day]} · includes the {buffer_pct}% buffer, minus your stock")
        with h2:
            st.download_button("Download CSV", edited.to_csv(index=False), f"order_{day:%Y%m%d}.csv", "text/csv",
                               icon=":material/download:", width="stretch", type="primary")
        shop = edited[["Ingredient", "To order (kg)"]].sort_values("To order (kg)", ascending=False)
        st.dataframe(shop, hide_index=True, width="stretch",
                     column_config={"To order (kg)": st.column_config.ProgressColumn(
                         "To order (kg)", format="%.2f kg", min_value=0.0,
                         max_value=float(max(shop["To order (kg)"].max(), 0.01)))})

# ------------------------------------------------------------------ tab 2: 7-day outlook
with tabs[1]:
    week_total = sum(totals.values())
    busiest = max(totals, key=totals.get)
    quietest = min(totals, key=totals.get)
    dough_week = need_units["Dough"].sum() / 100 if "Dough" in need_units else None
    ui.kpi_row([
        ui.kpi("Week total", f"{week_total}", "pizzas over the next 7 days", accent=True),
        ui.kpi("Busiest day", labels[busiest], f"{totals[busiest]} pizzas"),
        ui.kpi("Quietest day", labels[quietest], f"{totals[quietest]} pizzas"),
        ui.kpi("Dough for the week", f"{dough_week:.2f}" if dough_week is not None else "-",
               "before buffer", unit="kg" if dough_week is not None else ""),
    ])
    with st.container(border=True):
        ui.section("Pizzas per day", "bars show the forecast, whiskers the likely range")
        xs = [labels[d] for d in fc.index]
        ys = [totals[d] for d in fc.index]
        fig = go.Figure(go.Bar(
            x=xs, y=ys, marker_color=T["brand"], text=ys, textposition="inside", insidetextanchor="end",
            textfont=dict(color="#ffffff", size=13), textangle=0,
            customdata=[[lows[d], highs[d]] for d in fc.index],
            error_y=dict(type="data", symmetric=False, color=T["err"], thickness=1.2, width=5,
                         array=[highs[d] - totals[d] for d in fc.index],
                         arrayminus=[totals[d] - lows[d] for d in fc.index]),
            hovertemplate="%{x}<br><b>%{y} pizzas</b><br>likely %{customdata[0]}–%{customdata[1]}<extra></extra>"))
        fig = ui.style(fig, 360)
        fig.update_yaxes(title="Pizzas", rangemode="tozero")
        fig.update_xaxes(showgrid=False, tickfont=dict(color=T["text2"]))
        ui.show(fig)
    with st.container(border=True):
        ui.section("Ingredients per day", "kg before buffer · week total is the exact sum of the days")
        wk = need_units.copy()
        wk.index = [labels[d] for d in wk.index]
        wk.loc["Week total"] = wk.sum()
        st.dataframe((wk / 100).T, width="stretch", column_config={c: KG for c in wk.index})

# ------------------------------------------------------------------ tab 3: average daily needs
with tabs[2]:
    hist = fl.history_ingredients(sales, recipes)
    hist = hist.reindex(daily.index[-56:], fill_value=0) / 1000  # exactly 8 of each weekday
    by_dow = hist.groupby(hist.index.dayofweek).mean()
    by_dow.index = [DAYS[i] for i in by_dow.index]
    by_dow = by_dow.loc[:, by_dow.sum() > 0.0005]
    avg_pizzas = daily["total"].tail(56).groupby(daily.index[-56:].dayofweek).mean()
    r = fl.round_half_up
    wkday = avg_pizzas.loc[0:3].mean()
    ui.kpi_row([
        ui.kpi("Mon–Thu", f"{r(wkday)}", "average pizzas per day"),
        ui.kpi("Friday", f"{r(avg_pizzas[4])}", f"{avg_pizzas[4] / wkday:.1f}x a weekday"),
        ui.kpi("Saturday", f"{r(avg_pizzas[5])}", f"{avg_pizzas[5] / wkday:.1f}x a weekday", accent=True),
        ui.kpi("Sunday", f"{r(avg_pizzas[6])}", f"{avg_pizzas[6] / wkday:.1f}x a weekday"),
    ])
    with st.container(border=True):
        ui.section("Average kg needed, by weekday",
                   "last 8 weeks · numbers are kg, shading compares each ingredient with its own busiest day")
        steps = T["seq"]
        share = by_dow / by_dow.max().where(by_dow.max() > 0)  # 0-1 within each ingredient
        fig = go.Figure(go.Heatmap(
            z=share.values, x=list(by_dow.columns), y=list(by_dow.index), zmin=0, zmax=1,
            colorscale=[[i / (len(steps) - 1), c] for i, c in enumerate(steps)],
            text=by_dow.map(lambda v: f"{v:.1f}").values, texttemplate="%{text}", textfont=dict(size=12),
            customdata=by_dow.values, xgap=2, ygap=2, showscale=False,
            hovertemplate="%{y} · %{x}<br><b>%{customdata:.2f} kg</b><extra></extra>"))
        fig = ui.style(fig, 360)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=T["text2"]))
        fig.update_xaxes(showgrid=False, tickangle=-35, tickfont=dict(color=T["text2"]))
        ui.show(fig)

    groups = {"Mon–Thu": by_dow.loc[["Mon", "Tue", "Wed", "Thu"]].mean(),
              "Friday": by_dow.loc["Fri"], "Saturday": by_dow.loc["Sat"], "Sunday": by_dow.loc["Sun"]}
    summary = pd.DataFrame(groups)
    weekday_avg = summary["Mon–Thu"]
    ratio = summary[["Saturday", "Sunday"]].mean(axis=1) / weekday_avg.where(weekday_avg > 0)
    summary = summary.round(1)
    summary["Weekend vs weekday"] = ratio.map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}x")
    with st.container(border=True):
        ui.section("Typical day by type", "kg per day")
        st.dataframe(summary, width="stretch")

# ------------------------------------------------------------------ tab 4: model accuracy
with tabs[3]:
    scores = pd.DataFrame(bt["scores"]).T
    base_s, model_s = scores.iloc[0], scores.iloc[1]
    ui.kpi_row([
        ui.kpi("Model error", f"{model_s['MAPE %']:.1f}%", f"average miss of {model_s['MAE']:.1f} pizzas/day", accent=True),
        ui.kpi("Naive method error", f"{base_s['MAPE %']:.1f}%", f"average miss of {base_s['MAE']:.1f} pizzas/day"),
        ui.kpi("Improvement", f"{(1 - model_s['MAE'] / base_s['MAE']) * 100:.0f}%", "less error than the naive method"),
        ui.kpi("Test window", "28", "recent days the model never saw", unit="days"),
    ])
    with st.container(border=True):
        ui.section("Forecast vs what actually sold", "last 28 days, held out from training")
        fr = bt["frame"]
        fig = go.Figure()
        series = [("Actual", "actual", T["s1"], "solid"), ("Model", "model", T["s2"], "solid"),
                  ("Naive baseline", "baseline", T["s3"], "dash")]
        for name, col, color, dash in series:
            fig.add_scatter(x=fr.index, y=fr[col], name=name, mode="lines", line=dict(color=color, width=2, dash=dash),
                            hovertemplate=f"{name}: %{{y:.0f}}<extra></extra>")
        fig = ui.style(fig, 360, legend=True)
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(showgrid=False, showspikes=True, spikecolor=T["axis"], spikethickness=1, spikedash="solid",
                         hoverformat="%a %d %b")
        fig.update_yaxes(title="Pizzas")
        ui.show(fig)
        with st.expander("Show as a table"):
            st.dataframe(fr.rename(columns={"actual": "Actual", "model": "Model", "baseline": "Naive baseline"})
                         .rename_axis("Date").reset_index().assign(Date=lambda x: x["Date"].dt.date),
                         hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.0f")
                                        for c in ("Actual", "Model", "Naive baseline")})
    with st.container(border=True):
        ui.section("What drives the forecast", "share of the model's decisions")
        imp = bt["importance"].rename(index=FEATURE_NAMES)
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=T["brand"],
                               text=[f"{v:.0%}" for v in imp.values], textposition="outside", cliponaxis=False,
                               textfont=dict(color=T["text2"]),
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig = ui.style(fig, 380)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showgrid=False, tickfont=dict(color=T["text2"]), ticklabelstandoff=8)
        ui.show(fig)

# ------------------------------------------------------------------ tab 5: business impact
with tabs[4]:
    st.markdown(f"Replaying the **last 28 days**: what if the owner had ordered each way? "
                f"Forecast-driven uses your **{buffer_pct}%** safety buffer.")
    sim = fl.simulate_policies(bt["frame"], daily, buffer).round(1)
    avg, peak, fcst = sim.loc[fl.POLICY_AVG], sim.loc[fl.POLICY_PEAK], sim.loc[fl.POLICY_FCST]

    def stats(s):
        return [("Waste", f"{s[fl.WASTE]:.1f}%"), ("Stock-out days", f"{s[fl.STOCKOUT_DAYS]:.0f} of 28"),
                ("Pizzas lost", f"{s[fl.LOST]:.0f}"), ("Fill rate", f"{s[fl.FILL]:.1f}%")]

    ui.policy_cards([
        {"name": "Prep for an average day", "rule": "Last 28-day average + 10%, every day", "stats": stats(avg)},
        {"name": "Prep for a peak day", "rule": "90th percentile of the last 28 days, every day", "stats": stats(peak)},
        {"name": "Forecast-driven", "rule": f"This app's daily forecast + {buffer_pct}% buffer",
         "stats": stats(fcst), "best": True},
    ])

    def change(name, a, b, fmt):
        a, b = fmt(a), fmt(b)  # compare the same values the cards show
        return f"keeps {name} at {a}" if a == b else (
            f"{'cuts' if float(b.rstrip('%')) < float(a.rstrip('%')) else 'raises'} {name} from {a} to {b}")

    pct, whole = (lambda v: f"{v:.1f}%"), (lambda v: f"{v:.0f}")
    st.info(f"Versus prepping for a peak day, forecasting {change('waste', peak[fl.WASTE], fcst[fl.WASTE], pct)}. "
            f"Versus prepping for an average day, it {change('lost pizzas', avg[fl.LOST], fcst[fl.LOST], whole)}.",
            icon=":material/lightbulb:")
    st.caption("Waste = pizzas' worth of ingredients prepared but not used that day (assumes leftovers aren't "
               "carried over). Lost = demand above what was prepared. Move the safety buffer to see the trade-off.")
    with st.expander("Show as a table"):
        st.dataframe(sim, width="stretch")
