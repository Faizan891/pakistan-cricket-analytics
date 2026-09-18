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

from utils.data_loader import load_matches, load_players
from utils.data_processor import (
    filter_matches,
    filter_players,
    compute_kpis,
    toss_impact_analysis,
    venue_innings_splits,
    phase_overview_summary,
    top_phase_batters,
    top_phase_bowlers,
    middle_overs_spin_vs_pace,
    death_overs_finishing_metrics,
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

    # Innings Split Filter
    innings_options = ["All Innings", "1st Innings (Defending)", "2nd Innings (Chasing)"]
    selected_innings_label = st.selectbox(
        label   = "Innings Split",
        options = innings_options,
        index   = 0,
        help    = "Filter matches where Pakistan batted 1st vs chased (2nd).",
    )
    if "1st" in selected_innings_label:
        innings_filter = "1st"
    elif "2nd" in selected_innings_label:
        innings_filter = "2nd"
    else:
        innings_filter = "All"

    # Phase Selector (Focus for Phase module)
    phase_options = ["Powerplay (Overs 1–6)", "Middle Overs (Overs 7–15)", "Death Overs (Overs 16–20)"]
    selected_phase_label = st.selectbox(
        label   = "Phase Focus",
        options = phase_options,
        index   = 0,
        help    = "Active phase highlighted in deep-dive leaderboards.",
    )
    phase_focus_clean = "Powerplay" if "Powerplay" in selected_phase_label else ("Middle" if "Middle" in selected_phase_label else "Death")

    st.markdown("---")
    st.markdown(
        "<small style='color:#b9f6ca;'>✅ Toss Impact &amp; Phase Splits Enabled<br>Data: 2015–2024 · 400 matches · 16 players</small>",
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
    innings    = innings_filter,
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

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.metric(
        label = "🏟️ Total Matches",
        value = f"{kpis['total_matches']:,}",
        delta = f"{kpis['wins']}W - {kpis['losses']}L - {kpis['draws']}D",
    )
with k2:
    st.metric(
        label = "🏆 Win %",
        value = f"{kpis['win_pct']}%",
        delta = "overall record",
    )
with k3:
    st.metric(
        label = "📈 Avg Run Rate",
        value = f"{kpis['avg_run_rate']}",
        delta = f"Avg {kpis['avg_runs']} runs",
    )
with k4:
    st.metric(
        label = "🏏 Highest Score",
        value = f"{kpis['highest_score']}",
        delta = "innings peak",
    )
with k5:
    st.metric(
        label = "🪙 Toss Win Win-%",
        value = f"{kpis['toss_win_match_win_pct']}%",
        delta = f"Toss Won {kpis['toss_win_pct']}%",
    )
with k6:
    st.metric(
        label = "🛡️ Bat 1st vs Chase",
        value = f"{kpis['bat_1st_win_pct']}%",
        delta = f"Chase: {kpis['chase_win_pct']}%",
    )

st.markdown("---")


# ── Plotly Shared Palette ─────────────────────────────────────────────────────
PLOTLY_THEME = "plotly_dark"
PAK_GREEN    = "#00e676"
PAK_DARK_G   = "#008537"
PAK_RED      = "#ef5350"
PAK_AMBER    = "#ffca28"
PAK_BLUE     = "#42a5f5"
CHART_BG     = "rgba(13,17,23,0)"
GRID_COLOR   = "#1e3a1e"
FONT_COLOR   = "#e6edf3"

RESULT_COLORS = {
    "Win": PAK_GREEN,
    "Loss": PAK_RED,
    "Draw": PAK_AMBER,
    "No Result": PAK_BLUE,
}


# ─────────────────────────────────────────────────────────────────────────────
# 8. TABS ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
tab_results, tab_toss, tab_phases, tab_h2h, tab_players = st.tabs([
    "📊 Match Results & Trends",
    "🪙 Toss & Innings Splits",
    "⏱️ Phase Analysis (PP, Mid, Death)",
    "⚔️ Head-to-Head",
    "🏅 Leaderboards & Raw Data",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: MATCH RESULTS & TRENDS
# ═════════════════════════════════════════════════════════════════════════════
with tab_results:
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

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: TOSS IMPACT & INNINGS SPLITS
# ═════════════════════════════════════════════════════════════════════════════
with tab_toss:
    st.markdown("### 🪙 Toss Decision Impact & Innings Splits Analysis")
    st.markdown(
        "Analyze how toss luck, tactical decisions (`Bat` vs `Field`), and match conditions "
        "impact Pakistan's win rates when **defending totals (Batting 1st)** versus **chasing targets (Batting 2nd)**."
    )

    toss_data = toss_impact_analysis(df_matches)
    venue_splits = venue_innings_splits(df_matches)

    # Sub-metrics
    tm1, tm2, tm3, tm4 = st.columns(4)
    toss_out = toss_data["toss_outcome"]
    toss_won_row = toss_out[toss_out["Category"] == "Toss Won"]
    toss_lost_row = toss_out[toss_out["Category"] == "Toss Lost"]

    win_p_toss_won = toss_won_row["Win %"].values[0] if not toss_won_row.empty else 0.0
    win_p_toss_lost = toss_lost_row["Win %"].values[0] if not toss_lost_row.empty else 0.0

    dec_out = toss_data["decision_outcome"]
    chose_bat_row = dec_out[dec_out["Decision"] == "Chose to Bat"]
    chose_field_row = dec_out[dec_out["Decision"] == "Chose to Field"]

    win_p_bat = chose_bat_row["Win %"].values[0] if not chose_bat_row.empty else 0.0
    win_p_field = chose_field_row["Win %"].values[0] if not chose_field_row.empty else 0.0

    with tm1:
        st.metric(
            label = "🎯 Win % When Toss Won",
            value = f"{win_p_toss_won}%",
            delta = f"{(win_p_toss_won - win_p_toss_lost):+.1f}% vs Toss Lost",
        )
    with tm2:
        st.metric(
            label = "❌ Win % When Toss Lost",
            value = f"{win_p_toss_lost}%",
            delta = "disadvantaged",
            delta_color = "inverse",
        )
    with tm3:
        st.metric(
            label = "🏏 Chose to Bat First",
            value = f"{win_p_bat}%",
            delta = f"{chose_bat_row['Matches'].values[0] if not chose_bat_row.empty else 0} matches",
        )
    with tm4:
        st.metric(
            label = "🏃 Chose to Field / Chase",
            value = f"{win_p_field}%",
            delta = f"{chose_field_row['Matches'].values[0] if not chose_field_row.empty else 0} matches",
        )

    st.markdown("---")

    # Visualizations: Toss Decisions & Innings Impact
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("#### Toss Outcome & Decision Win Rates")
        fig_toss_bar = go.Figure()

        cats = ["Toss Won", "Toss Lost", "Chose to Bat", "Chose to Field"]
        win_rates = [win_p_toss_won, win_p_toss_lost, win_p_bat, win_p_field]
        bar_colors = [PAK_GREEN, PAK_RED, PAK_BLUE, PAK_AMBER]

        fig_toss_bar.add_trace(go.Bar(
            x            = cats,
            y            = win_rates,
            marker_color = bar_colors,
            text         = [f"{v:.1f}%" for v in win_rates],
            textposition = "outside",
            hovertemplate= "<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
        ))
        fig_toss_bar.add_hline(y=50, line_dash="dash", line_color="#81c784", annotation_text="50% baseline")
        fig_toss_bar.update_layout(
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            margin        = dict(t=20, b=30, l=40, r=20),
            yaxis         = dict(title="Win Percentage (%)", range=[0, 100], gridcolor=GRID_COLOR),
            xaxis         = dict(gridcolor=GRID_COLOR),
        )
        st.plotly_chart(fig_toss_bar, use_container_width=True)

    with t_col2:
        st.markdown("#### Innings Match Volume & Win Rates")
        inn_df = toss_data["innings_outcome"]
        fig_inn = go.Figure()
        for idx, row in inn_df.iterrows():
            color = PAK_GREEN if "1st" in row["Innings"] else PAK_BLUE
            fig_inn.add_trace(go.Bar(
                name         = row["Innings"],
                x            = [row["Innings"]],
                y            = [row["Win %"]],
                marker_color = color,
                text         = [f"{row['Win %']:.1f}% ({row['Wins']}/{row['Matches']} W)"],
                textposition = "outside",
                hovertemplate= f"<b>{row['Innings']}</b><br>Matches: {row['Matches']}<br>Win %: {row['Win %']}%<br>Avg Score: {row['Avg Runs']}<extra></extra>",
            ))
        fig_inn.add_hline(y=50, line_dash="dash", line_color="#81c784", annotation_text="50% Par")
        fig_inn.update_layout(
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            margin        = dict(t=20, b=30, l=40, r=20),
            yaxis         = dict(title="Win %", range=[0, 100], gridcolor=GRID_COLOR),
            showlegend    = False,
        )
        st.plotly_chart(fig_inn, use_container_width=True)

    # Venue Innings Split: Batting 1st vs Chasing Success Rate
    st.markdown("---")
    st.markdown("#### 🏟️ Batting 1st vs Chasing Success Rate by Venue")

    if not venue_splits.empty:
        fig_venue = go.Figure()

        fig_venue.add_trace(go.Bar(
            name         = "Batting 1st (Defending) Win %",
            x            = venue_splits["Venue"],
            y            = venue_splits["Bat 1st Win %"],
            marker_color = PAK_GREEN,
            text         = venue_splits["Bat 1st Win %"].apply(lambda v: f"{v:.0f}%"),
            textposition = "outside",
            hovertemplate= "<b>%{x}</b><br>Bat 1st Win %%: %{y:.1f}<br>1st Inn Avg Score: %{customdata:.1f}<extra></extra>",
            customdata   = venue_splits["Avg 1st Inn Score"],
        ))

        fig_venue.add_trace(go.Bar(
            name         = "Batting 2nd (Chasing) Win %",
            x            = venue_splits["Venue"],
            y            = venue_splits["Chase Win %"],
            marker_color = PAK_BLUE,
            text         = venue_splits["Chase Win %"].apply(lambda v: f"{v:.0f}%"),
            textposition = "outside",
            hovertemplate= "<b>%{x}</b><br>Chase Win %%: %{y:.1f}<br>2nd Inn Avg Score: %{customdata:.1f}<extra></extra>",
            customdata   = venue_splits["Avg 2nd Inn Score"],
        ))

        fig_venue.add_hline(y=50, line_dash="dot", line_color="#b0bec5", annotation_text="50% Equilibrium")

        fig_venue.update_layout(
            barmode       = "group",
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            height        = 420,
            margin        = dict(t=25, b=60, l=40, r=20),
            xaxis         = dict(title="Ground / Venue", tickangle=-30, gridcolor=GRID_COLOR),
            yaxis         = dict(title="Win % at Venue", range=[0, 110], gridcolor=GRID_COLOR),
            legend        = dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_venue, use_container_width=True)

        st.markdown("#### Venue Detailed Innings Splits Table")
        st.dataframe(
            venue_splits,
            use_container_width = True,
            hide_index          = True,
            column_config       = {
                "Bat 1st Win %": st.column_config.ProgressColumn("Bat 1st Win %", format="%.1f%%", min_value=0, max_value=100),
                "Chase Win %": st.column_config.ProgressColumn("Chase Win %", format="%.1f%%", min_value=0, max_value=100),
                "Avg 1st Inn Score": st.column_config.NumberColumn("Avg 1st Inn Score", format="%.1f"),
                "Avg 2nd Inn Score": st.column_config.NumberColumn("Avg 2nd Inn Score", format="%.1f"),
            }
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: PHASE ANALYSIS (POWERPLAY, MIDDLE, DEATH)
# ═════════════════════════════════════════════════════════════════════════════
with tab_phases:
    st.markdown("### ⏱️ Phase-Specific Performance Analysis")
    st.markdown(
        "Cricket matches are won or lost in distinct phases: **Powerplay (Overs 1–6)**, "
        "**Middle Overs (Overs 7–15)**, and **Death Overs (Overs 16–20)**. "
        "Examine strike rates, boundary frequencies, spin control vs pace, and death economy."
    )

    phase_sum = phase_overview_summary(df_players)

    if not phase_sum.empty:
        p_c1, p_c2, p_c3 = st.columns(3)

        pp_data  = phase_sum[phase_sum["Phase"].str.startswith("Powerplay")].iloc[0]
        mid_data = phase_sum[phase_sum["Phase"].str.startswith("Middle")].iloc[0]
        dth_data = phase_sum[phase_sum["Phase"].str.startswith("Death")].iloc[0]

        with p_c1:
            st.metric(
                label = "⚡ Powerplay (Overs 1–6)",
                value = f"{pp_data['Batting Strike Rate']:.1f} SR",
                delta = f"{pp_data['Total Runs']:,} Runs · {pp_data['Wickets Taken']} Wkts · Eco: {pp_data['Avg Economy']}",
            )
        with p_c2:
            st.metric(
                label = "🧭 Middle Overs (Overs 7–15)",
                value = f"{mid_data['Batting Strike Rate']:.1f} SR",
                delta = f"{mid_data['Total Runs']:,} Runs · {mid_data['Wickets Taken']} Wkts · Eco: {mid_data['Avg Economy']}",
            )
        with p_c3:
            st.metric(
                label = "🔥 Death Overs (Overs 16–20)",
                value = f"{dth_data['Batting Strike Rate']:.1f} SR",
                delta = f"{dth_data['Total Runs']:,} Runs · {dth_data['Wickets Taken']} Wkts · Eco: {dth_data['Avg Economy']}",
            )

    st.markdown("---")

    col_p_left, col_p_right = st.columns(2)

    with col_p_left:
        st.markdown("#### Phase Batting Strike Rate Progression")
        fig_phase_sr = go.Figure(go.Bar(
            x            = ["Powerplay (1–6)", "Middle (7–15)", "Death (16–20)"],
            y            = [pp_data['Batting Strike Rate'], mid_data['Batting Strike Rate'], dth_data['Batting Strike Rate']],
            marker_color = [PAK_GREEN, PAK_BLUE, PAK_AMBER],
            text         = [f"{v:.1f}" for v in [pp_data['Batting Strike Rate'], mid_data['Batting Strike Rate'], dth_data['Batting Strike Rate']]],
            textposition = "outside",
            hovertemplate= "<b>%{x}</b><br>Strike Rate: %{y:.1f}<extra></extra>",
        ))
        fig_phase_sr.update_layout(
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            margin        = dict(t=20, b=30, l=40, r=20),
            yaxis         = dict(title="Batting Strike Rate", gridcolor=GRID_COLOR),
        )
        st.plotly_chart(fig_phase_sr, use_container_width=True)

    with col_p_right:
        st.markdown("#### Phase Bowling Economy Progression")
        fig_phase_eco = go.Figure(go.Bar(
            x            = ["Powerplay (1–6)", "Middle (7–15)", "Death (16–20)"],
            y            = [pp_data['Avg Economy'], mid_data['Avg Economy'], dth_data['Avg Economy']],
            marker_color = [PAK_GREEN, PAK_BLUE, PAK_RED],
            text         = [f"{v:.2f}" for v in [pp_data['Avg Economy'], mid_data['Avg Economy'], dth_data['Avg Economy']]],
            textposition = "outside",
            hovertemplate= "<b>%{x}</b><br>Economy: %{y:.2f} rpo<extra></extra>",
        ))
        fig_phase_eco.update_layout(
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            margin        = dict(t=20, b=30, l=40, r=20),
            yaxis         = dict(title="Bowling Economy (Runs/Over)", gridcolor=GRID_COLOR),
        )
        st.plotly_chart(fig_phase_eco, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 🔍 Deep-Dive: {phase_focus_clean} Phase Leaders")

    col_ph_bat, col_ph_bowl = st.columns(2)
    top_batters_phase = top_phase_batters(df_players, phase=phase_focus_clean, top_n=8)
    top_bowlers_phase = top_phase_bowlers(df_players, phase=phase_focus_clean, top_n=8)

    with col_ph_bat:
        st.markdown(f"#### 🏏 Top {phase_focus_clean} Run Scorers &amp; Strike Rate")
        if not top_batters_phase.empty:
            fig_ph_bat = go.Figure(go.Bar(
                x            = top_batters_phase["Runs"],
                y            = top_batters_phase["player"],
                orientation  = "h",
                marker       = dict(
                    color      = top_batters_phase["Strike Rate"],
                    colorscale = [[0, "#004d1f"], [1, "#00e676"]],
                    colorbar   = dict(title="Strike Rate"),
                ),
                text         = top_batters_phase.apply(lambda r: f"{r['Runs']:,} r (SR {r['Strike Rate']:.0f})", axis=1),
                textposition = "outside",
                hovertemplate= "<b>%{y}</b><br>Phase Runs: %{x:,}<br>Strike Rate: %{marker.color:.1f}<extra></extra>",
            ))
            fig_ph_bat.update_layout(
                template      = PLOTLY_THEME,
                paper_bgcolor = CHART_BG,
                plot_bgcolor  = CHART_BG,
                font_color    = FONT_COLOR,
                height        = 380,
                margin        = dict(t=10, b=40, l=120, r=60),
                xaxis         = dict(title="Runs in Phase", gridcolor=GRID_COLOR),
                yaxis         = dict(autorange="reversed", gridcolor=GRID_COLOR),
            )
            st.plotly_chart(fig_ph_bat, use_container_width=True)

            st.dataframe(
                top_batters_phase[["player", "Runs", "Balls", "Strike Rate", "Phase Run Share %"]].rename(columns={"player": "Player"}),
                use_container_width = True,
                hide_index          = True,
                column_config       = {
                    "Strike Rate": st.column_config.NumberColumn("Strike Rate", format="%.1f"),
                    "Phase Run Share %": st.column_config.ProgressColumn("Phase Share %", format="%.1f%%", min_value=0, max_value=100),
                }
            )

    with col_ph_bowl:
        st.markdown(f"#### 🎯 Top {phase_focus_clean} Bowlers (Wickets &amp; Economy)")
        if not top_bowlers_phase.empty:
            fig_ph_bowl = go.Figure(go.Bar(
                x            = top_bowlers_phase["Wickets"],
                y            = top_bowlers_phase["player"],
                orientation  = "h",
                marker       = dict(
                    color      = top_bowlers_phase["Economy"],
                    colorscale = [[0, "#2e7d32"], [0.5, "#fbc02d"], [1, "#d32f2f"]],
                    colorbar   = dict(title="Economy"),
                ),
                text         = top_bowlers_phase.apply(lambda r: f"{r['Wickets']} wkts (Eco {r['Economy']:.1f})", axis=1),
                textposition = "outside",
                hovertemplate= "<b>%{y}</b><br>Wickets: %{x}<br>Economy: %{marker.color:.2f}<extra></extra>",
            ))
            fig_ph_bowl.update_layout(
                template      = PLOTLY_THEME,
                paper_bgcolor = CHART_BG,
                plot_bgcolor  = CHART_BG,
                font_color    = FONT_COLOR,
                height        = 380,
                margin        = dict(t=10, b=40, l=140, r=60),
                xaxis         = dict(title="Wickets in Phase", gridcolor=GRID_COLOR),
                yaxis         = dict(autorange="reversed", gridcolor=GRID_COLOR),
            )
            st.plotly_chart(fig_ph_bowl, use_container_width=True)

            st.dataframe(
                top_bowlers_phase[["player", "Wickets", "Economy", "Phase Wkt Share %"]].rename(columns={"player": "Player"}),
                use_container_width = True,
                hide_index          = True,
                column_config       = {
                    "Economy": st.column_config.NumberColumn("Economy", format="%.2f"),
                    "Phase Wkt Share %": st.column_config.ProgressColumn("Phase Wkt Share", format="%.1f%%", min_value=0, max_value=100),
                }
            )

    st.markdown("---")
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        st.markdown("#### 🌀 Middle Overs (7–15): Spin Control vs Pace Attack")
        spin_pace_df = middle_overs_spin_vs_pace(df_players)
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Bar(
            name         = "Middle Wickets",
            x            = spin_pace_df["Bowling Type"],
            y            = spin_pace_df["Middle Wickets"],
            marker_color = [PAK_GREEN, PAK_BLUE],
            text         = spin_pace_df["Middle Wickets"],
            textposition = "outside",
        ))
        fig_sp.update_layout(
            template      = PLOTLY_THEME,
            paper_bgcolor = CHART_BG,
            plot_bgcolor  = CHART_BG,
            font_color    = FONT_COLOR,
            margin        = dict(t=20, b=30, l=40, r=20),
            yaxis         = dict(title="Wickets Taken in Middle Overs", gridcolor=GRID_COLOR),
        )
        st.plotly_chart(fig_sp, use_container_width=True)
        st.dataframe(spin_pace_df, use_container_width=True, hide_index=True)

    with s_col2:
        st.markdown("#### 💣 Death Overs (16–20): Finishing Accelerators")
        death_metrics = death_overs_finishing_metrics(df_players)
        death_bat = death_metrics["batters"]
        if not death_bat.empty:
            fig_death = go.Figure(go.Scatter(
                x             = death_bat["Strike Rate"],
                y             = death_bat["Runs"],
                mode          = "markers+text",
                text          = death_bat["player"],
                textposition  = "top center",
                marker        = dict(
                    size       = death_bat["Est. Boundary Runs"] / 8 + 12,
                    color      = death_bat["Runs/Ball"],
                    colorscale = "Viridis",
                    showscale  = True,
                    colorbar   = dict(title="Runs/Ball"),
                ),
                hovertemplate = "<b>%{text}</b><br>Death Runs: %{y}<br>Death SR: %{x:.1f}<br>Boundary Runs: %{marker.size}<extra></extra>",
            ))
            fig_death.update_layout(
                template      = PLOTLY_THEME,
                paper_bgcolor = CHART_BG,
                plot_bgcolor  = CHART_BG,
                font_color    = FONT_COLOR,
                margin        = dict(t=20, b=30, l=40, r=20),
                xaxis         = dict(title="Death Overs Strike Rate", gridcolor=GRID_COLOR),
                yaxis         = dict(title="Death Overs Runs", gridcolor=GRID_COLOR),
            )
            st.plotly_chart(fig_death, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: HEAD-TO-HEAD
# ═════════════════════════════════════════════════════════════════════════════
with tab_h2h:
    if df_matches.empty:
        st.warning("No matches found for the selected filters.")
    else:
        st.markdown("#### Player Strike Rate and Economy by Format")
        rate_df = (
            df_players.groupby("format", as_index=False)
            .agg(strike_rate=("strike_rate", "mean"), economy=("economy", "mean"))
            .round({"strike_rate": 1, "economy": 2})
        )
        if not rate_df.empty:
            st.bar_chart(
                rate_df.set_index("format"),
                y=["strike_rate", "economy"],
                y_label="Average rate",
            )

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
            display_h2h,
            use_container_width = True,
            hide_index          = True,
            column_config       = {
                "Win %": st.column_config.ProgressColumn(
                    "Win %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Avg Runs": st.column_config.NumberColumn(
                    "Avg Runs",
                    format="%.1f",
                ),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: LEADERBOARDS & RAW DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab_players:
    st.markdown("#### Filtered Raw Data")
    explorer_view = st.radio(
        "Dataset",
        ["Match results", "Player statistics"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if explorer_view == "Match results":
        st.dataframe(df_matches, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_players, use_container_width=True, hide_index=True)

    st.markdown("---")
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
                }),
                use_container_width = True,
                hide_index          = True,
                column_config       = {
                    "Runs": st.column_config.ProgressColumn(
                        "Runs",
                        format="%d",
                        min_value=0,
                        max_value=int(batters_df["runs"].max()) if not batters_df.empty else 100,
                    ),
                    "Avg": st.column_config.NumberColumn("Avg", format="%.1f"),
                    "SR": st.column_config.NumberColumn("SR", format="%.1f"),
                },
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
                }),
                use_container_width = True,
                hide_index          = True,
                column_config       = {
                    "Wickets": st.column_config.ProgressColumn(
                        "Wickets",
                        format="%d",
                        min_value=0,
                        max_value=int(bowlers_df["wickets"].max()) if not bowlers_df.empty else 50,
                    ),
                    "Avg": st.column_config.NumberColumn("Avg", format="%.1f"),
                    "Economy": st.column_config.NumberColumn("Economy", format="%.2f"),
                },
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
