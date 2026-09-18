"""
app.py
======
Pakistan Cricket Analytics Dashboard
--------------------------------------
Main Streamlit entry-point.  Run with:
    streamlit run app.py

Architecture
------------
Data flow:
  CSV files  →  utils.data_loader  →  utils.data_processor  →  Plotly charts

Sections
--------
1. Page config + custom CSS
2. Cached data loaders
3. Sidebar filters
4. KPI metric cards
5. Tab 1 – Results Overview    (Donut chart + Win% trend line)
6. Tab 2 – Head-to-Head        (Grouped bar chart per opponent)
7. Tab 3 – Player Leaderboards (Horizontal bar: batters & bowlers)
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_loader    import load_matches, load_players
from utils.data_processor import (
    filter_matches,
    filter_players,
    compute_kpis,
    result_breakdown,
    performance_by_year,
    head_to_head,
    top_scorers,
    top_wicket_takers,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Pakistan Cricket Analytics Dashboard",
    page_icon  = "🏏",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background ───────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e8eaf0;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #00471b 0%, #006428 60%, #003d19 100%);
    border-right: 2px solid #00a651;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stMultiSelect > div { background-color: #005c22 !important; }

/* ── Section headings ───────────────────────────────────────────────────── */
h1 { color: #00c853 !important; letter-spacing: 1px; }
h2, h3 { color: #69f0ae !important; }

/* ── KPI cards ──────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a2a1a 0%, #0d1f0d 100%);
    border: 1px solid #00a651;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(0,166,81,0.2);
}
[data-testid="stMetricLabel"]  { color: #69f0ae !important; font-size: 0.85rem; font-weight: 600; }
[data-testid="stMetricValue"]  { color: #ffffff !important; font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricDelta"]  { color: #b9f6ca !important; }

/* ── Tab strip ──────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    color: #69f0ae !important;
    font-weight: 600;
    font-size: 0.95rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #00c853 !important;
    color: #00c853 !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: #00a651; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading match data …")
def get_matches() -> pd.DataFrame:
    """Load and cache the full match results dataset."""
    return load_matches()


@st.cache_data(show_spinner="Loading player data …")
def get_players() -> pd.DataFrame:
    """Load and cache the full player statistics dataset."""
    return load_players()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
df_matches_raw = get_matches()
df_players_raw = get_players()

all_opponents = sorted(df_matches_raw["opponent"].dropna().unique().tolist())
all_formats   = ["T20", "ODI", "Test"]
min_year, max_year = int(df_matches_raw["year"].min()), int(df_matches_raw["year"].max())


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / header
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <span style='font-size:3rem;'>🏏</span><br>
        <span style='font-size:1.1rem; font-weight:700; color:#ffffff;'>
            Pakistan Cricket<br>Analytics
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Filters")

    # Format filter
    selected_formats = st.multiselect(
        label   = "Match Format",
        options = all_formats,
        default = all_formats,
        help    = "Select one or more match formats to include.",
    )

    # Year range slider
    selected_years = st.slider(
        label = "Year Range",
        min_value = min_year,
        max_value = max_year,
        value     = (min_year, max_year),
        step      = 1,
        help      = "Drag to filter matches by year.",
    )

    # Opponent filter
    selected_opponents = st.multiselect(
        label   = "Opponent Team",
        options = all_opponents,
        default = all_opponents,
        help    = "Filter matches by opposition team.",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#b9f6ca;'>Data: 2015–2024 · 400 matches · 16 players</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────
# Fallback to all options when nothing is selected (avoid empty DataFrames)
active_formats   = selected_formats   if selected_formats   else all_formats
active_opponents = selected_opponents if selected_opponents else all_opponents

df_matches = filter_matches(
    df_matches_raw,
    formats    = active_formats,
    year_range = selected_years,
    opponents  = active_opponents,
)

df_players = filter_players(
    df_players_raw,
    formats    = active_formats,
    year_range = selected_years,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🏏 Pakistan Cricket Analytics Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#69f0ae; margin-top:-10px;'>"
    "Interactive performance analysis · 2015 – 2024</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# 7. KPI METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
kpis = compute_kpis(df_matches)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label = "🏟️ Total Matches",
        value = f"{kpis['total_matches']:,}",
    )
with col2:
    st.metric(
        label = "🏆 Win %",
        value = f"{kpis['win_pct']}%",
        delta = f"{kpis['wins']}W  {kpis['losses']}L  {kpis['draws']}D",
    )
with col3:
    st.metric(
        label = "📈 Avg Run Rate",
        value = f"{kpis['avg_run_rate']}",
        delta = "runs/over",
    )
with col4:
    st.metric(
        label = "🏏 Highest Score",
        value = f"{kpis['highest_score']}",
        delta = "all-time",
    )
with col5:
    st.metric(
        label = "📊 Avg Runs / Match",
        value = f"{kpis['avg_runs']}",
        delta = "batting",
    )

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Results Overview",
    "⚔️  Head-to-Head",
    "🏅 Player Leaderboards",
])


# ── Plotly shared theme ───────────────────────────────────────────────────────
PLOTLY_THEME   = "plotly_dark"
PAK_GREEN      = "#00c853"
PAK_RED        = "#f44336"
PAK_AMBER      = "#ffc107"
PAK_BLUE       = "#42a5f5"
CHART_BG       = "rgba(15,17,23,0)"   # transparent → inherits page bg
GRID_COLOR     = "#1e3a1e"
FONT_COLOR     = "#e8eaf0"

RESULT_COLORS  = {
    "Win"       : PAK_GREEN,
    "Loss"      : PAK_RED,
    "Draw"      : PAK_AMBER,
    "No Result" : PAK_BLUE,
}


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – RESULTS OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if df_matches.empty:
        st.warning("No matches found for the selected filters.")
    else:
        col_left, col_right = st.columns([1, 1.6])

        # ── Donut chart – Win/Loss/Draw breakdown ─────────────────────────
        with col_left:
            st.markdown("#### Match Result Breakdown")
            breakdown = result_breakdown(df_matches)

            fig_donut = go.Figure(
                go.Pie(
                    labels         = breakdown["result"],
                    values         = breakdown["count"],
                    hole           = 0.55,
                    marker_colors  = [
                        RESULT_COLORS.get(r, "#9e9e9e")
                        for r in breakdown["result"]
                    ],
                    textinfo       = "label+percent",
                    textfont_size  = 13,
                    hovertemplate  = (
                        "<b>%{label}</b><br>"
                        "Matches: %{value}<br>"
                        "Share: %{percent}<extra></extra>"
                    ),
                )
            )
            fig_donut.update_layout(
                template        = PLOTLY_THEME,
                paper_bgcolor   = CHART_BG,
                plot_bgcolor    = CHART_BG,
                font_color      = FONT_COLOR,
                margin          = dict(t=20, b=20, l=10, r=10),
                legend          = dict(
                    orientation = "h",
                    yanchor     = "bottom",
                    y           = -0.15,
                    xanchor     = "center",
                    x           = 0.5,
                ),
                # Centre annotation
                annotations = [{
                    "text"     : f"<b>{kpis['total_matches']}<br>Matches</b>",
                    "x"        : 0.5,
                    "y"        : 0.5,
                    "font"     : {"size": 16, "color": FONT_COLOR},
                    "showarrow": False,
                }],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Line chart – Win % trend over years ───────────────────────────
        with col_right:
            st.markdown("#### Win % & Runs Trend Over Time")
            year_df = performance_by_year(df_matches)

            fig_trend = make_subplots(
                specs        = [[{"secondary_y": True}]],
            )

            # Win % line (primary y)
            fig_trend.add_trace(
                go.Scatter(
                    x            = year_df["year"],
                    y            = year_df["win_pct"],
                    name         = "Win %",
                    mode         = "lines+markers",
                    line         = dict(color=PAK_GREEN, width=3),
                    marker       = dict(size=8, color=PAK_GREEN,
                                        line=dict(color="#ffffff", width=1.5)),
                    hovertemplate= "<b>%{x}</b><br>Win %%: %{y:.1f}<extra></extra>",
                ),
                secondary_y = False,
            )

            # Avg runs bars (secondary y)
            fig_trend.add_trace(
                go.Bar(
                    x            = year_df["year"],
                    y            = year_df["avg_runs"],
                    name         = "Avg Runs",
                    marker_color = "rgba(0,166,81,0.25)",
                    marker_line  = dict(color=PAK_GREEN, width=1),
                    hovertemplate= "<b>%{x}</b><br>Avg Runs: %{y:.1f}<extra></extra>",
                ),
                secondary_y = True,
            )

            fig_trend.update_layout(
                template      = PLOTLY_THEME,
                paper_bgcolor = CHART_BG,
                plot_bgcolor  = CHART_BG,
                font_color    = FONT_COLOR,
                hovermode     = "x unified",
                legend        = dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                ),
                margin        = dict(t=30, b=40, l=50, r=60),
                xaxis         = dict(
                    title      = "Year",
                    gridcolor  = GRID_COLOR,
                    tickmode   = "linear",
                    dtick      = 1,
                ),
            )
            fig_trend.update_yaxes(
                title_text = "Win %",
                gridcolor  = GRID_COLOR,
                range      = [0, 100],
                secondary_y= False,
            )
            fig_trend.update_yaxes(
                title_text  = "Avg Runs",
                showgrid    = False,
                secondary_y = True,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # ── Win/Loss per Format stacked bar ──────────────────────────────
        st.markdown("#### Win / Loss / Draw by Format")
        fmt_data = []
        for fmt, grp in df_matches.groupby("format"):
            fmt_data.append({
                "format": fmt,
                "Win"   : int((grp["result"] == "Win").sum()),
                "Loss"  : int((grp["result"] == "Loss").sum()),
                "Draw"  : int(((grp["result"] == "Draw") | (grp["result"] == "No Result")).sum()),
            })
        fmt_df = pd.DataFrame(fmt_data)

        if not fmt_df.empty:
            fig_fmt = go.Figure()
            for result_type, color in [("Win", PAK_GREEN), ("Loss", PAK_RED), ("Draw", PAK_AMBER)]:
                if result_type in fmt_df.columns:
                    fig_fmt.add_trace(go.Bar(
                        name         = result_type,
                        x            = fmt_df["format"],
                        y            = fmt_df[result_type],
                        marker_color = color,
                        hovertemplate= f"<b>%{{x}}</b><br>{result_type}: %{{y}}<extra></extra>",
                    ))
            fig_fmt.update_layout(
                barmode       = "group",
                template      = PLOTLY_THEME,
                paper_bgcolor = CHART_BG,
                plot_bgcolor  = CHART_BG,
                font_color    = FONT_COLOR,
                margin        = dict(t=10, b=40, l=40, r=20),
                xaxis         = dict(title="Format",     gridcolor=GRID_COLOR),
                yaxis         = dict(title="# Matches",  gridcolor=GRID_COLOR),
                legend        = dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_fmt, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – HEAD-TO-HEAD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if df_matches.empty:
        st.warning("No matches found for the selected filters.")
    else:
        h2h_df = head_to_head(df_matches)

        st.markdown("#### Pakistan Head-to-Head Record vs All Selected Opponents")

        # ── Grouped bar – W/L/D per opponent ────────────────────────────
        fig_h2h = go.Figure()
        for result_type, color in [("wins", PAK_GREEN), ("losses", PAK_RED), ("draws", PAK_AMBER)]:
            fig_h2h.add_trace(go.Bar(
                name         = result_type.capitalize(),
                x            = h2h_df["opponent"],
                y            = h2h_df[result_type],
                marker_color = color,
                text         = h2h_df[result_type],
                textposition = "outside",
                hovertemplate= (
                    f"<b>%{{x}}</b><br>"
                    f"{result_type.capitalize()}: %{{y}}<extra></extra>"
                ),
            ))

        fig_h2h.update_layout(
            barmode       = "group",
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            height        = 420,
            margin        = dict(t=20, b=60, l=40, r=20),
            xaxis         = dict(
                title     = "Opponent",
                tickangle = -30,
                gridcolor = GRID_COLOR,
            ),
            yaxis         = dict(title="# Matches", gridcolor=GRID_COLOR),
            legend        = dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_h2h, use_container_width=True)

        # ── Win % scatter – bubble size = matches played ──────────────────
        st.markdown("#### Win % vs Avg Runs Scored (bubble = matches played)")

        fig_bubble = px.scatter(
            h2h_df,
            x             = "avg_runs",
            y             = "win_pct",
            size          = "matches",
            color         = "opponent",
            text          = "opponent",
            size_max      = 55,
            template      = PLOTLY_THEME,
            labels        = {
                "avg_runs"  : "Avg Runs Scored",
                "win_pct"   : "Win %",
                "matches"   : "Matches Played",
                "opponent"  : "Opponent",
            },
            hover_data    = {
                "wins"    : True,
                "losses"  : True,
                "draws"   : True,
                "matches" : True,
            },
        )
        fig_bubble.update_traces(textposition="top center", textfont_size=11)
        fig_bubble.update_layout(
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            height        = 400,
            margin        = dict(t=20, b=40, l=50, r=20),
            xaxis         = dict(gridcolor=GRID_COLOR),
            yaxis         = dict(gridcolor=GRID_COLOR, range=[0, 100]),
            showlegend    = False,
        )
        # Reference line at 50% win rate
        fig_bubble.add_hline(
            y              = 50,
            line_dash      = "dash",
            line_color     = PAK_AMBER,
            annotation_text= "50% win rate",
            annotation_position = "bottom right",
            annotation_font_color = PAK_AMBER,
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        # ── Detailed H2H table ────────────────────────────────────────────
        st.markdown("#### Detailed Record")
        display_h2h = h2h_df[
            ["opponent", "matches", "wins", "losses", "draws", "win_pct", "avg_runs"]
        ].rename(columns={
            "opponent" : "Opponent",
            "matches"  : "Matches",
            "wins"     : "Wins",
            "losses"   : "Losses",
            "draws"    : "Draws",
            "win_pct"  : "Win %",
            "avg_runs" : "Avg Runs",
        })
        st.dataframe(
            display_h2h.style
                .background_gradient(subset=["Win %"],  cmap="Greens")
                .background_gradient(subset=["Avg Runs"], cmap="Blues")
                .format({"Win %": "{:.1f}%", "Avg Runs": "{:.1f}"}),
            use_container_width = True,
            hide_index          = True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – PLAYER LEADERBOARDS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if df_players.empty:
        st.warning("No player data found for the selected filters.")
    else:
        col_bat, col_bowl = st.columns(2)

        # ── Top Batters ───────────────────────────────────────────────────
        with col_bat:
            st.markdown("#### 🏏 Top Run-Scorers")
            batters_df = top_scorers(df_players, top_n=8)

            fig_bat = go.Figure(go.Bar(
                x            = batters_df["runs"],
                y            = batters_df["player"],
                orientation  = "h",
                marker       = dict(
                    color    = batters_df["runs"],
                    colorscale = [[0, "#005c22"], [1, "#00e676"]],
                    showscale  = False,
                    line       = dict(color="#00c853", width=0.8),
                ),
                text         = batters_df["runs"].apply(lambda v: f"{v:,}"),
                textposition = "outside",
                customdata   = batters_df[["batting_avg", "strike_rate", "hundreds", "fifties"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Runs: %{x:,}<br>"
                    "Avg: %{customdata[0]:.1f}<br>"
                    "SR: %{customdata[1]:.1f}<br>"
                    "100s: %{customdata[2]}  |  50s: %{customdata[3]}"
                    "<extra></extra>"
                ),
            ))
            fig_bat.update_layout(
                template      = PLOTLY_THEME,
                paper_bgcolor = CHART_BG,
                plot_bgcolor  = CHART_BG,
                font_color    = FONT_COLOR,
                height        = 400,
                margin        = dict(t=10, b=40, l=120, r=60),
                xaxis         = dict(title="Total Runs", gridcolor=GRID_COLOR),
                yaxis         = dict(
                    title    = "",
                    autorange= "reversed",
                    gridcolor= GRID_COLOR,
                ),
            )
            st.plotly_chart(fig_bat, use_container_width=True)

            # Batter detail table
            st.dataframe(
                batters_df[["player","runs","batting_avg","strike_rate","hundreds","fifties"]]
                .rename(columns={
                    "player"     : "Player",
                    "runs"       : "Runs",
                    "batting_avg": "Avg",
                    "strike_rate": "SR",
                    "hundreds"   : "100s",
                    "fifties"    : "50s",
                })
                .style
                .background_gradient(subset=["Runs"], cmap="Greens")
                .format({"Avg": "{:.1f}", "SR": "{:.1f}"}),
                use_container_width = True,
                hide_index          = True,
            )

        # ── Top Bowlers ───────────────────────────────────────────────────
        with col_bowl:
            st.markdown("#### 🎯 Top Wicket-Takers")
            bowlers_df = top_wicket_takers(df_players, top_n=8)

            fig_bowl = go.Figure(go.Bar(
                x            = bowlers_df["wickets"],
                y            = bowlers_df["player"],
                orientation  = "h",
                marker       = dict(
                    color      = bowlers_df["wickets"],
                    colorscale = [[0, "#4a0000"], [1, "#ef5350"]],
                    showscale  = False,
                    line       = dict(color="#f44336", width=0.8),
                ),
                text         = bowlers_df["wickets"].apply(lambda v: f"{v:,}"),
                textposition = "outside",
                customdata   = bowlers_df[["bowling_avg","economy","five_wickets"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Wickets: %{x:,}<br>"
                    "Avg: %{customdata[0]:.1f}<br>"
                    "Economy: %{customdata[1]:.2f}<br>"
                    "5-wkt hauls: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ))
            fig_bowl.update_layout(
                template      = PLOTLY_THEME,
                paper_bgcolor = CHART_BG,
                plot_bgcolor  = CHART_BG,
                font_color    = FONT_COLOR,
                height        = 400,
                margin        = dict(t=10, b=40, l=140, r=60),
                xaxis         = dict(title="Total Wickets", gridcolor=GRID_COLOR),
                yaxis         = dict(
                    title    = "",
                    autorange= "reversed",
                    gridcolor= GRID_COLOR,
                ),
            )
            st.plotly_chart(fig_bowl, use_container_width=True)

            # Bowler detail table
            st.dataframe(
                bowlers_df[["player","wickets","bowling_avg","economy","five_wickets"]]
                .rename(columns={
                    "player"      : "Player",
                    "wickets"     : "Wickets",
                    "bowling_avg" : "Avg",
                    "economy"     : "Economy",
                    "five_wickets": "5-WKTs",
                })
                .style
                .background_gradient(subset=["Wickets"], cmap="Reds")
                .format({"Avg": "{:.1f}", "Economy": "{:.2f}"}),
                use_container_width = True,
                hide_index          = True,
            )

        # ── Combined radar chart – balanced all-rounder view ──────────────
        st.markdown("---")
        st.markdown("#### 📡 Batter vs Bowler Performance Radar")

        top4_bat  = top_scorers(df_players, top_n=4)
        top4_bowl = top_wicket_takers(df_players, top_n=4)

        radar_categories = ["Runs (norm)", "Avg", "SR (norm)", "100s", "50s"]

        def normalize(series, lo=0, hi=None):
            hi = hi or series.max()
            return ((series - lo) / (hi - lo) * 100).round(1)

        fig_radar = go.Figure()

        colors_bat  = [PAK_GREEN, "#69f0ae", "#00e676", "#b9f6ca"]
        colors_bowl = [PAK_RED,   "#ef9a9a", "#e57373", "#ffcdd2"]

        for i, row in top4_bat.iterrows():
            r_norm  = normalize(pd.Series([row["runs"]]),   hi=top4_bat["runs"].max())[0]
            sr_norm = normalize(pd.Series([row["strike_rate"]]), hi=200)[0]
            vals    = [r_norm, min(row["batting_avg"], 100), sr_norm,
                       min(row["hundreds"]*10, 100), min(row["fifties"]*5, 100)]
            vals   += [vals[0]]  # close the polygon
            cats    = radar_categories + [radar_categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r         = vals,
                theta     = cats,
                fill      = "toself",
                name      = row["player"],
                line_color= colors_bat[i],
                fillcolor = colors_bat[i].replace(")", ",0.15)").replace("rgb", "rgba")
                             if "rgb" in colors_bat[i] else colors_bat[i] + "26",
                opacity   = 0.85,
            ))

        fig_radar.update_layout(
            polar = dict(
                radialaxis  = dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR,
                                   tickfont_color=FONT_COLOR),
                angularaxis = dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR),
                bgcolor     = "rgba(0,0,0,0)",
            ),
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            font_color    = FONT_COLOR,
            height        = 450,
            margin        = dict(t=40, b=40, l=60, r=60),
            legend        = dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
            showlegend    = True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4caf50; font-size:0.8rem;'>"
    "🏏 Pakistan Cricket Analytics Dashboard &nbsp;|&nbsp; "
    "Built with Streamlit &amp; Plotly &nbsp;|&nbsp; "
    "Data: 2015–2024 (Simulated)"
    "</p>",
    unsafe_allow_html=True,
)
