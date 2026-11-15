"""Look & feel: design tokens, CSS, HTML cards and a shared Plotly style (light + dark)."""
from html import escape

import plotly.graph_objects as go
import streamlit as st

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Chart colours come from a validated categorical palette (slots 1-3 pass colour-blind checks in both modes).
TOKENS = {
    "light": dict(
        card="#ffffff", border="rgba(11,11,11,0.10)", border_solid="#e4e1d8",
        text1="#0b0b0b", text2="#52514e", muted="#898781", grid="#ecebe5", axis="#c3c2b7",
        brand="#eb6834", brand_text="#c9431b", brand_soft="rgba(235,104,52,0.10)", amber_soft="rgba(237,161,0,0.10)",
        s1="#2a78d6", s2="#eb6834", s3="#1baf7a", err="#898781",
        seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        shadow="0 1px 2px rgba(11,11,11,0.04), 0 4px 16px rgba(11,11,11,0.04)",
    ),
    "dark": dict(
        card="#1a1a19", border="rgba(255,255,255,0.10)", border_solid="#2c2c2a",
        text1="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
        brand="#d95926", brand_text="#f58a62", brand_soft="rgba(240,106,58,0.14)", amber_soft="rgba(237,161,0,0.08)",
        s1="#3987e5", s2="#d95926", s3="#199e70", err="#898781",
        seq=["#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        shadow="0 1px 2px rgba(0,0,0,0.3)",
    ),
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tokens() -> dict:
    return TOKENS[mode()]


def inject_css() -> None:
    t = tokens()
    st.markdown(f"""<style>
    .block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
    h1, h2, h3 {{ letter-spacing: -0.02em; }}
    .hero {{ position: relative; overflow: hidden; border: 1px solid {t['border']}; border-radius: 20px;
             padding: 26px 28px 22px; margin-bottom: 18px;
             background: linear-gradient(135deg, {t['brand_soft']} 0%, {t['amber_soft']} 60%, transparent 100%), {t['card']};
             box-shadow: {t['shadow']}; }}
    .hero .eyebrow {{ color: {t['brand_text']}; font-weight: 700; font-size: .78rem; letter-spacing: .08em;
                      text-transform: uppercase; margin-bottom: 4px; }}
    .hero h1 {{ font-size: 2.1rem; line-height: 1.15; margin: 0 0 6px; padding: 0; color: {t['text1']}; }}
    .hero p {{ color: {t['text2']}; margin: 0 0 14px; font-size: 1rem; max-width: 640px; }}
    .hero .slice {{ position: absolute; right: 22px; top: 14px; font-size: 64px; opacity: .9;
                    transform: rotate(-12deg); filter: drop-shadow(0 6px 12px rgba(0,0,0,.12)); }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 999px;
             background: {t['card']}; border: 1px solid {t['border']}; color: {t['text2']}; font-size: .8rem; }}
    .chip b {{ color: {t['text1']}; font-weight: 600; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 18px; }}
    @media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .hero .slice {{ display: none; }} }}
    .kpi {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 16px; padding: 14px 16px 13px;
            box-shadow: {t['shadow']}; min-width: 0; }}
    .kpi.accent {{ border-color: {t['brand']}; box-shadow: 0 0 0 1px {t['brand']} inset, {t['shadow']}; }}
    .kpi .label {{ color: {t['muted']}; font-size: .72rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; }}
    .kpi .value {{ color: {t['text1']}; font-size: 1.9rem; font-weight: 750; line-height: 1.2; margin-top: 4px;
                   white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .kpi .value small {{ font-size: .95rem; font-weight: 600; color: {t['text2']}; margin-left: 2px; }}
    .kpi .sub {{ color: {t['text2']}; font-size: .82rem; margin-top: 4px; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .76rem; font-weight: 600;
             background: {t['brand_soft']}; color: {t['brand_text']}; }}
    .section {{ display: flex; align-items: baseline; gap: 8px; margin: 2px 0 6px; }}
    .section .t {{ font-weight: 700; font-size: 1.02rem; color: {t['text1']}; }}
    .section .d {{ color: {t['muted']}; font-size: .85rem; }}
    .policies {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 6px 0 14px; }}
    @media (max-width: 900px) {{ .policies {{ grid-template-columns: 1fr; }} }}
    .policy {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 16px; padding: 16px 18px; box-shadow: {t['shadow']}; }}
    .policy.best {{ border-color: {t['brand']}; box-shadow: 0 0 0 1px {t['brand']} inset, {t['shadow']}; }}
    .policy .name {{ font-weight: 700; color: {t['text1']}; margin-bottom: 2px; }}
    .policy .rule {{ color: {t['muted']}; font-size: .8rem; margin-bottom: 12px; min-height: 2.8em; line-height: 1.4; }}
    .policy .row {{ display: flex; justify-content: space-between; padding: 7px 0; border-top: 1px solid {t['border']};
                    font-size: .9rem; color: {t['text2']}; }}
    .policy .row b {{ color: {t['text1']}; font-variant-numeric: tabular-nums; }}
    .sidebar-brand {{ display: flex; align-items: center; gap: 10px; margin: -6px 0 6px; }}
    .sidebar-brand .logo {{ font-size: 26px; }}
    .sidebar-brand .n {{ font-weight: 800; font-size: 1.05rem; line-height: 1.1; color: {t['text1']}; }}
    .sidebar-brand .s {{ font-size: .75rem; color: {t['muted']}; }}
    div[data-testid="stTabs"] button[role="tab"] p {{ font-size: .93rem; font-weight: 600; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px; }}
    </style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- HTML pieces
def hero(eyebrow: str, title: str, text: str, chips: list[tuple[str, str]]) -> None:
    chip_html = "".join(f'<span class="chip">{escape(k)} <b>{escape(v)}</b></span>' for k, v in chips)
    st.markdown(f'<div class="hero"><div class="slice">🍕</div><div class="eyebrow">{escape(eyebrow)}</div>'
                f'<h1>{escape(title)}</h1><p>{escape(text)}</p><div class="chips">{chip_html}</div></div>',
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", unit: str = "", pill: str = "", accent: bool = False) -> str:
    unit_html = f"<small>{escape(unit)}</small>" if unit else ""
    pill_html = f'<span class="pill">{escape(pill)}</span> ' if pill else ""
    return (f'<div class="kpi{" accent" if accent else ""}"><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}{unit_html}</div><div class="sub">{pill_html}{escape(sub)}</div></div>')


def kpi_row(cards: list[str]) -> None:
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def section(title: str, desc: str = "") -> None:
    d = f'<span class="d">{escape(desc)}</span>' if desc else ""
    st.markdown(f'<div class="section"><span class="t">{escape(title)}</span>{d}</div>', unsafe_allow_html=True)


def policy_cards(rows: list[dict]) -> None:
    html = ""
    for r in rows:
        stats = "".join(f'<div class="row"><span>{escape(k)}</span><b>{escape(v)}</b></div>' for k, v in r["stats"])
        html += (f'<div class="policy{" best" if r.get("best") else ""}"><div class="name">{escape(r["name"])}</div>'
                 f'<div class="rule">{escape(r["rule"])}</div>{stats}</div>')
    st.markdown(f'<div class="policies">{html}</div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown('<div class="sidebar-brand"><span class="logo">🍕</span><div><div class="n">Slice Planner</div>'
                '<div class="s">Demand &amp; ingredient forecasting</div></div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- charts
def style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    t = tokens()
    fig.update_layout(
        height=height, margin=dict(l=4, r=12, t=28 if legend else 8, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=t["text2"], size=13), title=None,
        hoverlabel=dict(bgcolor=t["card"], bordercolor=t["border_solid"], font=dict(family=FONT, color=t["text1"])),
        showlegend=legend, legend=dict(orientation="h", x=0, y=1.1, font=dict(color=t["text2"]), title=None),
        barcornerradius=4, bargap=0.38, hovermode="closest",
    )
    axis = dict(automargin=True, gridcolor=t["grid"], linecolor=t["axis"], zeroline=False, tickfont=dict(color=t["muted"]),
                title=dict(font=dict(color=t["muted"])))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
