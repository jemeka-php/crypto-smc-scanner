"""
Crypto SMC Candidate Scanner - Streamlit Dashboard
Personal crypto perpetual market scanner for manual trading candidate discovery.
"""

from datetime import datetime
import json
import time
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st

from config import AppConfig, DEFAULT_CONFIG
from data.exchange import ExchangeClient
from scanner.candidates import Candidate
from scanner.pipeline import MarketScanner, ScanStats
from ai.gemini import explain_candidate_setup

# ----------------- Streamlit Page Configuration -----------------
st.set_page_config(
    page_title="Crypto SMC Candidate Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- Custom Styling -----------------
st.markdown(
    """
    <style>
    /* Dark Terminal Theme Styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .terminal-header {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-family: 'SF Mono', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    }
    
    .terminal-title {
        color: #58a6ff;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .terminal-subtitle {
        color: #8b949e;
        font-size: 13px;
        margin-top: 4px;
    }

    /* Stats Banner */
    .stats-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 14px;
        text-align: center;
    }
    .stats-val {
        font-size: 18px;
        font-weight: 700;
        color: #f0f6fc;
    }
    .stats-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Candidate Cards */
    .candidate-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .candidate-card:hover {
        border-color: #58a6ff;
    }
    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .card-sym {
        font-size: 18px;
        font-weight: 700;
        color: #58a6ff;
        font-family: monospace;
    }
    .card-score {
        background: #1f6feb;
        color: #ffffff;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 13px;
    }
    .card-metrics {
        font-size: 13px;
        color: #c9d1d9;
        margin-bottom: 6px;
        font-family: monospace;
    }
    .card-context {
        font-size: 12px;
        color: #8b949e;
    }

    /* Badges */
    .badge-bullish {
        background: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-bearish {
        background: rgba(248, 81, 73, 0.2);
        color: #f85149;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-stage-confirmed {
        background: rgba(46, 160, 67, 0.25);
        color: #3fb950;
        border: 1px solid #2ea043;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-stage-forming {
        background: rgba(210, 153, 34, 0.25);
        color: #d29922;
        border: 1px solid #bb8009;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-stage-watch {
        background: rgba(56, 139, 253, 0.2);
        color: #58a6ff;
        border: 1px solid #388bfd;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-ltf-confirmed {
        background: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        border: 1px solid #2ea043;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-ltf-partial {
        background: rgba(210, 153, 34, 0.2);
        color: #d29922;
        border: 1px solid #bb8009;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-ltf-watching {
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid #388bfd;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-ltf-notrigger {
        background: rgba(139, 148, 158, 0.15);
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
    }
    .badge-ltf-invalidated {
        background: rgba(248, 81, 73, 0.2);
        color: #f85149;
        border: 1px solid #f85149;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
    }

    /* Plan & Alert boxes */
    .info-callout {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .waiting-callout {
        background: #1c1813;
        border-left: 4px solid #d29922;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .trade-plan-box {
        background: #131d1b;
        border: 1px solid #238636;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 10px;
    }
    .atas-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Session State Initialization -----------------
if "scanner_config" not in st.session_state:
    st.session_state.scanner_config = AppConfig()

if "candidates" not in st.session_state:
    st.session_state.candidates = []

if "scan_stats" not in st.session_state:
    st.session_state.scan_stats = None

if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = None

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None

if "atas_checklist" not in st.session_state:
    st.session_state.atas_checklist = {}

if "ai_explanations" not in st.session_state:
    st.session_state.ai_explanations = {}


# ----------------- Sidebar Configuration -----------------
with st.sidebar:
    st.markdown("### ⚙ Scanner Controls")

    # Refresh
    col_ref1, col_ref2 = st.columns([1, 1])
    with col_ref1:
        refresh_clicked = st.button("🔄 Refresh Now", use_container_width=True)
    with col_ref2:
        demo_toggle = st.toggle("Demo Mode", value=st.session_state.scanner_config.demo_mode)
        if demo_toggle != st.session_state.scanner_config.demo_mode:
            st.session_state.scanner_config.demo_mode = demo_toggle
            refresh_clicked = True

    auto_refresh = st.checkbox("Auto-refresh timer", value=False)
    refresh_rate = st.slider("Interval (seconds)", 30, 300, 60, step=15)

    st.markdown("---")
    st.markdown("### 🎯 Quantitative Filters")
    min_vol_m = st.number_input(
        "Min 24H Quote Vol ($M)",
        min_value=1.0,
        max_value=500.0,
        value=float(st.session_state.scanner_config.filters.min_24h_quote_volume / 1_000_000.0),
        step=5.0,
    )
    st.session_state.scanner_config.filters.min_24h_quote_volume = min_vol_m * 1_000_000.0

    min_rvol = st.slider(
        "Min 1H RVOL",
        min_value=1.0,
        max_value=4.0,
        value=float(st.session_state.scanner_config.filters.min_rvol),
        step=0.1,
    )
    st.session_state.scanner_config.filters.min_rvol = min_rvol

    min_price_chg = st.slider(
        "Min |24H Price Change| (%)",
        min_value=1.0,
        max_value=15.0,
        value=float(st.session_state.scanner_config.filters.min_abs_price_change_pct),
        step=0.5,
    )
    st.session_state.scanner_config.filters.min_abs_price_change_pct = min_price_chg

    min_oi_chg = st.slider(
        "Min 1H OI Change (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(st.session_state.scanner_config.filters.min_1h_oi_change_pct),
        step=0.5,
    )
    st.session_state.scanner_config.filters.min_1h_oi_change_pct = min_oi_chg

    st.markdown("---")
    st.markdown("### ⚖ Scoring Weights (100 pts)")
    with st.expander("Customize Weights"):
        w_liq = st.slider("Liquidity", 0, 25, 10)
        w_rvol = st.slider("RVOL", 0, 25, 15)
        w_oi = st.slider("Open Interest", 0, 25, 15)
        w_vol = st.slider("Volatility", 0, 20, 10)
        w_htf = st.slider("HTF Location", 0, 25, 15)
        w_sweep = st.slider("Liquidity Sweep", 0, 20, 10)
        w_disp = st.slider("Displacement", 0, 20, 10)
        w_bos = st.slider("BOS / MSS", 0, 20, 10)
        w_conf = st.slider("OB / FVG Confluence", 0, 10, 5)

        st.session_state.scanner_config.weights.liquidity = float(w_liq)
        st.session_state.scanner_config.weights.rvol = float(w_rvol)
        st.session_state.scanner_config.weights.oi = float(w_oi)
        st.session_state.scanner_config.weights.volatility = float(w_vol)
        st.session_state.scanner_config.weights.htf_location = float(w_htf)
        st.session_state.scanner_config.weights.liquidity_sweep = float(w_sweep)
        st.session_state.scanner_config.weights.displacement = float(w_disp)
        st.session_state.scanner_config.weights.bos_mss = float(w_bos)
        st.session_state.scanner_config.weights.ob_fvg_confluence = float(w_conf)

        total_weight = int(w_liq + w_rvol + w_oi + w_vol + w_htf + w_sweep + w_disp + w_bos + w_conf)
        st.markdown(f"**Total scoring weight:** `{total_weight} / 100`")
        if total_weight != 100:
            st.warning(f"⚠ Total weight is {total_weight} / 100. Weights must sum to 100 to avoid misleading candidate ranking.")

    if not st.session_state.scanner_config.weights.is_valid():
        st.sidebar.warning(f"⚠ Scoring weights sum to {int(st.session_state.scanner_config.weights.total())} / 100.")

    st.markdown("---")
    st.markdown("### 🤖 Gemini API Key")
    gemini_key_input = st.text_input(
        "Optional API Key for 'Explain Setup'",
        value=st.session_state.scanner_config.gemini_api_key,
        type="password",
        help="Used only when '🤖 Explain Setup' button is pressed.",
    )
    if gemini_key_input != st.session_state.scanner_config.gemini_api_key:
        st.session_state.scanner_config.gemini_api_key = gemini_key_input


# ----------------- Scanning Logic Execution -----------------
def perform_scan():
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(pct: int, msg: str):
        progress_bar.progress(pct)
        status_text.text(msg)

    scanner = MarketScanner(config=st.session_state.scanner_config)
    candidates, stats = scanner.run_scan(progress_callback=update_progress)

    st.session_state.candidates = candidates
    st.session_state.scan_stats = stats
    st.session_state.last_refresh_time = datetime.now().strftime("%H:%M:%S")

    progress_bar.empty()
    status_text.empty()


# Initial run if candidates are empty or refresh requested
if refresh_clicked or not st.session_state.candidates:
    perform_scan()


# ----------------- Dashboard Header & Statistics -----------------
stats: Optional[ScanStats] = st.session_state.scan_stats
last_time_str = st.session_state.last_refresh_time or datetime.now().strftime("%H:%M:%S")

st.markdown(
    f"""
    <div class="terminal-header">
        <div class="terminal-title">⚡ CRYPTO SMC CANDIDATE SCANNER</div>
        <div class="terminal-subtitle">
            Personal Market Sorting Terminal &bull; Binance USDT-M Perpetuals &bull; Zero Lookahead SMC
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if stats and getattr(stats, "is_stale", False):
    err_msg = getattr(stats, "error_message", None)
    err_detail = f"\n\n**Details:** {err_msg}" if err_msg else ""
    st.error(
        f"⚠ DATA STALE: Unable to establish live connection to exchange. Displaying cached data.{err_detail}"
    )

# Statistics Metrics Row
col_s1, col_s2, col_s3, col_s4, col_s5, col_s6, col_s7 = st.columns(7)
with col_s1:
    st.markdown(
        f"""<div class="stats-card"><div class="stats-val">{stats.markets_scanned if stats else 0}</div><div class="stats-label">Scanned</div></div>""",
        unsafe_allow_html=True,
    )
with col_s2:
    st.markdown(
        f"""<div class="stats-card"><div class="stats-val">{stats.stage1_candidates if stats else 0}</div><div class="stats-label">Stage 1 Shortlist</div></div>""",
        unsafe_allow_html=True,
    )
with col_s3:
    st.markdown(
        f"""<div class="stats-card"><div class="stats-val">{stats.stage2_candidates if stats else 0}</div><div class="stats-label">Stage 2 (LTF)</div></div>""",
        unsafe_allow_html=True,
    )
with col_s4:
    st.markdown(
        f"""<div class="stats-card"><div class="stats-val">{stats.passed_rvol if stats else 0}</div><div class="stats-label">Passed RVOL</div></div>""",
        unsafe_allow_html=True,
    )
with col_s5:
    st.markdown(
        f"""<div class="stats-card"><div class="stats-val">{stats.passed_oi if stats else 0}</div><div class="stats-label">Passed OI</div></div>""",
        unsafe_allow_html=True,
    )
with col_s6:
    s1_t = f"{stats.stage1_duration_s:.1f}s" if stats and stats.stage1_duration_s else "0s"
    s2_t = f"{stats.stage2_duration_s:.1f}s" if stats and stats.stage2_duration_s else "0s"
    st.markdown(
        f"""<div class="stats-card"><div class="stats-val">{s1_t} / {s2_t}</div><div class="stats-label">S1 / S2 Time</div></div>""",
        unsafe_allow_html=True,
    )
with col_s7:
    st.markdown(
        f"""<div class="stats-card"><div class="stats-val">{last_time_str}</div><div class="stats-label">Updated</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ----------------- Section: Prominent TOP Candidates -----------------
candidates: List[Candidate] = st.session_state.candidates
top_candidates = candidates[:6]

if top_candidates:
    st.markdown("### 🔥 Top Ranked Candidates")
    cols = st.columns(len(top_candidates))
    for idx, (col, c) in enumerate(zip(cols, top_candidates)):
        with col:
            stage_class = (
                "badge-stage-confirmed"
                if c.stage == "CONFIRMED"
                else ("badge-stage-forming" if c.stage == "SETUP_FORMING" else "badge-stage-watch")
            )
            bias_class = "badge-bullish" if c.structure_1h.bias == "BULLISH" else ("badge-bearish" if c.structure_1h.bias == "BEARISH" else "")

            ltf_st = c.ltf_confirmation.status
            ltf_class = (
                "badge-ltf-confirmed" if ltf_st == "CONFIRMED"
                else ("badge-ltf-partial" if ltf_st == "PARTIAL_CONFIRMATION"
                else ("badge-ltf-watching" if ltf_st == "WATCHING"
                else ("badge-ltf-invalidated" if ltf_st == "INVALIDATED"
                else "badge-ltf-notrigger")))
            )

            card_html = f"""
            <div class="candidate-card">
                <div class="card-top">
                    <span class="card-sym">#{idx+1} {c.symbol}</span>
                    <span class="card-score">{c.total_score:.0f}/100</span>
                </div>
                <div class="card-metrics">
                    RVOL <b>{c.rvol:.1f}x</b> | OI <b>{c.oi_metrics.change_1h_pct:+.1f}%</b> | 24H <b>{c.price_change_24h_pct:+.1f}%</b>
                </div>
                <div class="card-context">
                    <span class="{bias_class}">1H {c.structure_1h.bias}</span> &bull;
                    <span>1H: {c.location_1h}</span><br/>
                    <span>15M: {c.location_15m}</span> &bull;
                    <span>4H: {c.structure_4h.bias}</span><br/>
                    <span class="{stage_class}">{c.stage}</span> &bull;
                    <span class="{ltf_class}">LTF: {ltf_st}</span>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button(f"Inspect {c.symbol}", key=f"btn_top_{c.symbol}", use_container_width=True):
                st.session_state.selected_symbol = c.symbol

st.write("")

# ----------------- Section: Main Candidate Table -----------------
st.markdown("### 📊 Market Candidate Universe")

if candidates:
    # Build dataframe for display
    table_rows = []
    for rank, c in enumerate(candidates, 1):
        table_rows.append({
            "Rank": rank,
            "Symbol": c.symbol,
            "Price": f"${c.last_price:,.4f}" if c.last_price < 1 else f"${c.last_price:,.2f}",
            "24H Vol ($M)": round(c.quote_volume_24h / 1_000_000.0, 1),
            "24H Change %": c.price_change_24h_pct,
            "1H RVOL": c.rvol,
            "1H OI %": c.oi_metrics.change_1h_pct,
            "Funding": (
                f"{c.funding.rate_pct:+.3f}% {c.funding.label}"
                if c.funding and c.funding.rate_pct is not None and c.funding.label is not None
                else "N/A"
            ),
            "1H SMC Location": c.location_1h,
            "15M SMC Location": c.location_15m,
            "4H Macro Context": c.structure_4h.bias,
            "LTF Entry Status": c.ltf_confirmation.status,
            "Score": c.total_score,
        })

    df_table = pd.DataFrame(table_rows)

    sort_col = st.selectbox(
        "Sort table by:",
        ["Score", "1H RVOL", "1H OI %", "24H Vol ($M)", "24H Change %"],
        index=0,
    )
    df_table_sorted = df_table.sort_values(
        by=sort_col,
        ascending=False,
    )

    st.dataframe(
        df_table_sorted,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                format="%d",
                min_value=0,
                max_value=100,
            ),
            "24H Change %": st.column_config.NumberColumn(
                "24H Change %",
                format="%.2f%%",
            ),
            "1H RVOL": st.column_config.NumberColumn(
                "1H RVOL",
                format="%.2fx",
            ),
            "1H OI %": st.column_config.NumberColumn(
                "1H OI %",
                format="%+.2f%%",
            ),
        },
    )
else:
    st.info("No candidates currently pass the active filters. Adjust your thresholds in the sidebar.")

st.write("")

# ----------------- Section: Candidate Detail Panel -----------------
st.markdown("---")
st.markdown("## 🔍 Candidate Deep-Dive Inspector")

if candidates:
    symbol_list = [c.symbol for c in candidates]
    default_idx = (
        symbol_list.index(st.session_state.selected_symbol)
        if st.session_state.selected_symbol in symbol_list
        else 0
    )

    selected_symbol = st.selectbox(
        "Select candidate to inspect:",
        symbol_list,
        index=default_idx,
    )
    st.session_state.selected_symbol = selected_symbol

    selected_candidate: Candidate = next(c for c in candidates if c.symbol == selected_symbol)

    # 4 Detail Tabs
    tab_market, tab_structure, tab_setup, tab_atas_ai = st.tabs([
        "📈 Market & OI",
        "📐 Structure & SMC",
        "🎯 Setup & Trade Plan",
        "📝 ATAS & AI Confirmation",
    ])

    # ---------------- TAB 1: Market & OI ----------------
    with tab_market:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${selected_candidate.last_price:,.4f}")
        c2.metric("24H Change", f"{selected_candidate.price_change_24h_pct:+.2f}%")
        c3.metric("24H Quote Volume", f"${selected_candidate.quote_volume_24h:,.0f}")
        c4.metric("1H RVOL", f"{selected_candidate.rvol:.2f}x")

        c_oi1, c_oi2, c_oi3 = st.columns(3)
        c_oi1.metric("1H OI Change", f"{selected_candidate.oi_metrics.change_1h_pct:+.2f}%")
        c_oi2.metric("4H OI Change", f"{selected_candidate.oi_metrics.change_4h_pct:+.2f}%")
        c_oi3.metric(
            "Current OI Value",
            f"${selected_candidate.oi_metrics.current_oi:,.0f}" if selected_candidate.oi_metrics.current_oi else "N/A",
        )

        st.markdown(
            f"""
            <div class="info-callout">
                <b>PRICE / OI CLASSIFICATION:</b> <span style="color:#58a6ff; font-weight:700;">{selected_candidate.oi_metrics.classification}</span><br/>
                <span style="font-size:13px; color:#8b949e;">{selected_candidate.oi_metrics.interpretation}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("##### 🪙 Funding Rate")
        st.caption("🔒 DISPLAY-ONLY CONTEXT — Observational market rate; does not affect SMC score, ranking, or trade signals.")
        f_info = selected_candidate.funding
        if f_info and f_info.rate is not None:
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.metric("Funding Rate", f"{f_info.rate_pct:+.4f}%")
            f_col2.metric("Status", f_info.label or "N/A")
            f_ts_str = (
                datetime.fromtimestamp(f_info.timestamp / 1000.0).strftime("%Y-%m-%d %H:%M UTC")
                if f_info.timestamp
                else "N/A"
            )
            f_next_str = (
                datetime.fromtimestamp(f_info.next_funding_time / 1000.0).strftime("%Y-%m-%d %H:%M UTC")
                if f_info.next_funding_time
                else "N/A"
            )
            f_col3.metric("Funding Timestamp", f_ts_str)
            f_col4.metric("Next Funding", f_next_str)
        else:
            st.info("Funding rate data unavailable for this asset (N/A).")

    # ---------------- TAB 2: Structure & SMC ----------------
    with tab_structure:
        st.markdown("#### Primary Setup & Macro Structure Analysis")
        st.caption("🔒 Automated SMC locations and zones represent objective algorithmic context based on price action rules; they are not discretionary manual chart markings.")

        col_st1, col_st2, col_st3 = st.columns(3)
        with col_st1:
            st.markdown("##### 1H Primary Setup Structure")
            st.info(selected_candidate.location_1h_desc)
            st.write(f"- **1H Bias:** `{selected_candidate.structure_1h.bias}`")
            latest_bos_1h = selected_candidate.structure_1h.latest_bos
            latest_mss_1h = selected_candidate.structure_1h.latest_mss
            st.write(f"- **Latest 1H BOS:** {f'{latest_bos_1h.direction} (${latest_bos_1h.broken_level:.2f})' if latest_bos_1h else 'None'}")
            st.write(f"- **Latest 1H MSS:** {f'{latest_mss_1h.direction} (${latest_mss_1h.broken_level:.2f})' if latest_mss_1h else 'None'}")
            st.write(f"- **Active 1H Order Blocks:** {len(selected_candidate.active_obs_1h)}")
            st.write(f"- **Active 1H FVGs:** {len(selected_candidate.active_fvgs_1h)}")
            st.write(f"- **Recent 1H Sweeps:** {len(selected_candidate.sweeps_1h)}")
            if selected_candidate.displacements_1h:
                st.caption(f"Last 1H displacement: {selected_candidate.displacements_1h[-1].direction} ({selected_candidate.displacements_1h[-1].displacement_ratio:.1f}x ATR)")

        with col_st2:
            st.markdown("##### 15M Primary Setup Refinement")
            st.info(selected_candidate.location_15m_desc)
            st.write(f"- **15M Bias:** `{selected_candidate.structure_15m.bias if selected_candidate.structure_15m else 'N/A'}`")
            latest_bos_15m = selected_candidate.structure_15m.latest_bos if selected_candidate.structure_15m else None
            latest_mss_15m = selected_candidate.structure_15m.latest_mss if selected_candidate.structure_15m else None
            st.write(f"- **Latest 15M BOS:** {f'{latest_bos_15m.direction} (${latest_bos_15m.broken_level:.2f})' if latest_bos_15m else 'None'}")
            st.write(f"- **Latest 15M MSS:** {f'{latest_mss_15m.direction} (${latest_mss_15m.broken_level:.2f})' if latest_mss_15m else 'None'}")
            st.write(f"- **Active 15M Order Blocks:** {len(selected_candidate.active_obs_15m)}")
            st.write(f"- **Active 15M FVGs:** {len(selected_candidate.active_fvgs_15m)}")
            st.write(f"- **Recent 15M Sweeps:** {len(selected_candidate.sweeps_15m)}")
            if selected_candidate.displacements_15m:
                st.caption(f"Last 15M displacement: {selected_candidate.displacements_15m[-1].direction} ({selected_candidate.displacements_15m[-1].displacement_ratio:.1f}x ATR)")

        with col_st3:
            st.markdown("##### 4H Macro Context (Optional)")
            st.write(f"- **4H Bias:** `{selected_candidate.structure_4h.bias}`")
            st.write(f"- **Macro Note:** {selected_candidate.macro_context_4h or 'Neutral macro structure'}")
            st.write(f"- **Nearby Liquidity Pools:** {len(selected_candidate.nearby_liquidity)}")
            if selected_candidate.nearby_liquidity:
                for lvl in selected_candidate.nearby_liquidity[:3]:
                    st.caption(f"{lvl.timeframe} {lvl.source} ({lvl.side}) at ${lvl.price:.2f}")

    # ---------------- TAB 3: Setup & Trade Plan ----------------
    with tab_setup:
        ltf = selected_candidate.ltf_confirmation
        ltf_badge_class = (
            "badge-ltf-confirmed" if ltf.status == "CONFIRMED"
            else ("badge-ltf-partial" if ltf.status == "PARTIAL_CONFIRMATION"
            else ("badge-ltf-watching" if ltf.status == "WATCHING"
            else ("badge-ltf-invalidated" if ltf.status == "INVALIDATED"
            else "badge-ltf-notrigger")))
        )

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <span style="font-size:20px; font-weight:700;">Stage: {selected_candidate.stage}</span>
                    &bull; <span style="font-size:16px;">Bias: {selected_candidate.direction}</span>
                    &bull; <span class="{ltf_badge_class}">LTF: {ltf.status}</span>
                </div>
                <div style="background:#1f6feb; color:#fff; font-size:18px; font-weight:700; padding:4px 12px; border-radius:6px;">
                    Score: {selected_candidate.total_score:.1f} / 100
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stage 2 LTF Entry Confirmation Box
        st.markdown(
            f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:12px 16px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-weight:700; font-size:14px; color:#58a6ff;">⚡ LOWER-TIMEFRAME ENTRY CONFIRMATION (5M / 3M / 1M)</span>
                    <span class="{ltf_badge_class}">{ltf.status}</span>
                </div>
                <div style="font-size:13px; color:#c9d1d9; margin-bottom:8px;">
                    <b>Status:</b> {ltf.summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_ltf1, col_ltf2, col_ltf3 = st.columns(3)
        with col_ltf1:
            st.markdown("<b>5M Entry Trigger</b>", unsafe_allow_html=True)
            st.write(f"- {ltf.m5_sweep_desc or '5M liquidity sweep pending'}")
        with col_ltf2:
            st.markdown("<b>3M Refinement</b>", unsafe_allow_html=True)
            st.write(f"- {ltf.m3_displacement_desc or '3M displacement pending'}")
        with col_ltf3:
            st.markdown("<b>1M Microstructure</b>", unsafe_allow_html=True)
            st.write(f"- {ltf.m1_mss_desc or '1M MSS pending'}")

        st.caption("🔒 1H/15M determine setup quality & ranking; 5M/3M/1M provide entry timing confirmation without altering the candidate's main score.")

        # Why Interesting
        st.markdown(
            f"""
            <div class="info-callout">
                <b>💡 WHY IT IS INTERESTING:</b><br/>
                {selected_candidate.why_interesting}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # What Am I Waiting For
        st.markdown(
            f"""
            <div class="waiting-callout">
                <b>⏳ WHAT I AM WAITING FOR BEFORE CONSIDERING A TRADE:</b><br/>
                {selected_candidate.waiting_for}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Score Breakdown Expansion
        with st.expander("Score Component Breakdown"):
            sb = selected_candidate.score_breakdown
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.write(f"- **Liquidity:** {sb.liquidity_score} pts ({sb.details.get('liquidity', '')})")
                st.write(f"- **RVOL:** {sb.rvol_score} pts ({sb.details.get('rvol', '')})")
                st.write(f"- **Open Interest:** {sb.oi_score} pts ({sb.details.get('oi', '')})")
                st.write(f"- **Volatility:** {sb.volatility_score} pts ({sb.details.get('volatility', '')})")
                st.write(f"- **HTF Location:** {sb.htf_location_score} pts ({sb.details.get('htf_location', '')})")
            with col_b2:
                st.write(f"- **Liquidity Sweep:** {sb.liquidity_sweep_score} pts ({sb.details.get('liquidity_sweep', '')})")
                st.write(f"- **Displacement:** {sb.displacement_score} pts ({sb.details.get('displacement', '')})")
                st.write(f"- **BOS / MSS:** {sb.bos_mss_score} pts ({sb.details.get('bos_mss', '')})")
                st.write(f"- **OB/FVG Confluence:** {sb.ob_fvg_confluence_score} pts ({sb.details.get('confluence', '')})")

        # Reference Trade Plan
        plan = selected_candidate.trade_plan
        if plan:
            st.markdown(
                f"""
                <div class="trade-plan-box">
                    <div style="color:#3fb950; font-weight:700; font-size:14px; margin-bottom:8px;">
                        📌 REFERENCE TRADE PLAN ({plan.direction})
                    </div>
                    <div style="font-family:monospace; font-size:13px; line-height:1.6;">
                        <b>Entry Zone:</b> ${plan.entry:,.4f}<br/>
                        <b>Stop Loss:</b> ${plan.stop:,.4f} (Risk: ${plan.risk_amount:,.4f})<br/>
                        <b>Target 1R:</b> ${plan.target_1r:,.4f}<br/>
                        <b>Target 2R:</b> ${plan.target_2r:,.4f}<br/>
                        <b>Target 3R:</b> ${plan.target_3r:,.4f}<br/>
                        <b>Risk / Reward:</b> {plan.risk_reward}
                    </div>
                    <div style="font-size:11px; color:#8b949e; margin-top:8px;">
                        * Reference model based on OB edge and 1.0x ATR buffer. Not an automated execution or financial advice.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("No trade plan generated: market has not formed an unmitigated order block in setup direction.")

    # ---------------- TAB 4: ATAS & AI Confirmation ----------------
    with tab_atas_ai:
        col_atas, col_ai = st.columns([1, 1])

        # ATAS Manual Confirmation Checklist
        with col_atas:
            st.markdown("### 📋 ATAS Orderflow Confirmation")
            st.caption("Manual inspection checklist for ATAS / footprint orderflow:")

            sym = selected_candidate.symbol
            checklist_state = st.session_state.atas_checklist.get(sym, {
                "footprint": False,
                "delta": False,
                "absorption": False,
                "poc": False,
                "imbalance": False,
                "mss_micro": False,
                "status": "NOT CHECKED",
                "notes": "",
            })

            fp = st.checkbox("Footprint reviewed", value=checklist_state["footprint"], key=f"fp_{sym}")
            dl = st.checkbox("Delta supports setup", value=checklist_state["delta"], key=f"dl_{sym}")
            ab = st.checkbox("Absorption observed", value=checklist_state["absorption"], key=f"ab_{sym}")
            poc = st.checkbox("POC behaviour supports setup", value=checklist_state["poc"], key=f"poc_{sym}")
            imb = st.checkbox("Imbalance supports setup", value=checklist_state["imbalance"], key=f"imb_{sym}")
            micro = st.checkbox("1M / 5M MSS confirmed", value=checklist_state["mss_micro"], key=f"micro_{sym}")

            status_opt = st.radio(
                "ATAS STATUS:",
                ["NOT CHECKED", "SUPPORTS", "DOES NOT SUPPORT"],
                index=["NOT CHECKED", "SUPPORTS", "DOES NOT SUPPORT"].index(checklist_state["status"]),
                key=f"status_{sym}",
                horizontal=True,
            )

            notes = st.text_area(
                "Manual Notes:",
                value=checklist_state["notes"],
                key=f"notes_{sym}",
                placeholder="e.g. Significant trapped buyers at swing low, delta flipped positive at POC...",
                height=80,
            )

            st.session_state.atas_checklist[sym] = {
                "footprint": fp,
                "delta": dl,
                "absorption": ab,
                "poc": poc,
                "imbalance": imb,
                "mss_micro": micro,
                "status": status_opt,
                "notes": notes,
            }

            st.caption("🔒 Automated scanner calculations and manual ATAS observations remain strictly separated.")

        # Gemini AI Setup Explanation
        with col_ai:
            st.markdown("### 🤖 Strategic Setup Explanation")
            st.caption("Synthesize deterministic scanner observations with Google Gemini.")

            if st.button("🤖 Explain Setup", key=f"ai_btn_{sym}", use_container_width=True):
                with st.spinner("Analyzing candidate evidence..."):
                    ai_res = explain_candidate_setup(
                        candidate=selected_candidate,
                        api_key=st.session_state.scanner_config.gemini_api_key,
                    )
                    st.session_state.ai_explanations[sym] = ai_res

            cached_ai = st.session_state.ai_explanations.get(sym)
            if cached_ai:
                st.markdown(
                    f"""
                    <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:12px; margin-top:10px;">
                        <div><b>Assessment:</b> <span style="color:#58a6ff;">{cached_ai.get('assessment')}</span> | <b>Bias:</b> {cached_ai.get('bias')}</div>
                        <p style="margin-top:8px; font-size:13px; line-height:1.5;">{cached_ai.get('reasoning')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Identified Strengths:**")
                for s in cached_ai.get("strengths", []):
                    st.write(f"- ✅ {s}")

                st.markdown("**Key Strategic Risks:**")
                for r in cached_ai.get("risks", []):
                    st.write(f"- ⚠️ {r}")

                st.markdown("**Missing Confirmations:**")
                for m in cached_ai.get("missing_confirmation", []):
                    st.write(f"- ⏳ {m}")

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
