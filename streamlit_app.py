import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="PetroQuant Alpha",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better typography and clean look
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stDataFrame { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent
    
    # Load CSVs
    try:
        panel_df = pd.read_csv(base_dir / "data" / "processed" / "panel.csv", parse_dates=["Date"], index_col="Date")
        benchmark_df = pd.read_csv(base_dir / "data" / "processed" / "benchmark.csv", parse_dates=["Date"], index_col="Date")
        equity_df = pd.read_csv(base_dir / "results" / "tables" / "equity_curve.csv", parse_dates=["date"], index_col="date")
        trade_log = pd.read_csv(base_dir / "results" / "tables" / "trade_log.csv", parse_dates=["entry_date", "exit_date"])
        param_sweep = pd.read_csv(base_dir / "results" / "tables" / "parameter_sweep_results.csv")
        yearly_summary = pd.read_csv(base_dir / "results" / "tables" / "yearly_crack_summary.csv", index_col="Year")
        
        # Compute Benchmark NAVs (starting at $1M like the strategy)
        benchmark_df["SPY_NAV"] = benchmark_df["SPY"] / benchmark_df["SPY"].iloc[0] * 1e6
        benchmark_df["WTI_NAV"] = benchmark_df["WTI_BH"] / benchmark_df["WTI_BH"].iloc[0] * 1e6
        
        with open(base_dir / "results" / "tables" / "performance_metrics.json", "r") as f:
            metrics = json.load(f)
            
        return panel_df, benchmark_df, equity_df, trade_log, param_sweep, yearly_summary, metrics
    except Exception as e:
        st.error(f"Error loading data. Ensure pipeline has been run. Details: {e}")
        st.stop()

panel_df, benchmark_df, equity_df, trade_log, param_sweep, yearly_summary, metrics = load_data()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🛢️ PetroQuant Alpha")
st.sidebar.markdown("**Systematic Energy Trading**")
st.sidebar.markdown("---")

st.sidebar.markdown("""
**Strategy Overview**
Exploiting structural mean-reversion in the 3:2:1 Crude Oil Crack Spread using a rolling z-score framework.
""")

st.sidebar.markdown("---")
st.sidebar.info(
    "Data Period:\n"
    f"{equity_df.index.min().strftime('%Y-%m-%d')} to "
    f"{equity_df.index.max().strftime('%Y-%m-%d')}"
)

# ==========================================
# MAIN APP HEADER
# ==========================================
st.title("Crude Oil Crack Spread Strategy (3:2:1)")
st.markdown("Quantitative backtest results and analytics dashboard.")

# Top-level metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
with col2:
    st.metric("CAGR", f"{metrics.get('cagr_pct', 0):.2f}%")
with col3:
    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
with col4:
    st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
with col5:
    st.metric("Win Rate", f"{metrics.get('hit_rate_pct', 0):.1f}%")

st.markdown("---")

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Market Overview", 
    "💸 Strategy Performance", 
    "⚙️ Parameter Analytics", 
    "📋 Trade Log"
])

# ------------------------------------------
# TAB 1: Market Overview
# ------------------------------------------
with tab1:
    st.subheader("Commodity Components & The 3:2:1 Spread")
    
    # 3:2:1 Spread Plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=equity_df.index, y=equity_df["crack"], mode="lines", name="3:2:1 Crack Spread ($/bbl)", line=dict(color="#2E86AB")))
    fig1.update_layout(
        title="Historical 3:2:1 Crack Spread",
        yaxis_title="Spread ($/bbl)",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("### Component Prices")
    st.markdown("WTI Crude vs RBOB Gasoline vs Heating Oil (converted to $/bbl).")
    
    # Component Prices
    fig2 = go.Figure()
    # Clip WTI to avoid -37 breaking the chart scale
    wti_clipped = panel_df["WTI"].clip(lower=-5)
    
    fig2.add_trace(go.Scatter(x=panel_df.index, y=wti_clipped, mode="lines", name="WTI Crude", line=dict(color="#1F2937", width=1)))
    fig2.add_trace(go.Scatter(x=panel_df.index, y=panel_df["RBOB"], mode="lines", name="RBOB Gasoline", line=dict(color="#A23B72", width=1)))
    fig2.add_trace(go.Scatter(x=panel_df.index, y=panel_df["HO"], mode="lines", name="Heating Oil", line=dict(color="#F18F01", width=1)))
    
    fig2.update_layout(
        yaxis_title="Price ($/bbl)",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified"
    )
    # Force reasonable y-axis limits to avoid COVID distortions ruining the view
    fig2.update_yaxes(range=[wti_clipped.min() * 1.1, max(panel_df["RBOB"].max(), panel_df["HO"].max()) * 1.1])
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------
# TAB 2: Strategy Performance
# ------------------------------------------
with tab2:
    st.subheader("Equity Curve vs Benchmarks")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=equity_df.index, y=equity_df["nav"] / 1e6, mode="lines", name="Crack Strategy", line=dict(color="#2E86AB", width=2)))
    fig3.add_trace(go.Scatter(x=benchmark_df.index, y=benchmark_df["SPY_NAV"] / 1e6, mode="lines", name="SPY (Buy & Hold)", line=dict(color="#A23B72", width=1.5, dash="dot")))
    fig3.add_trace(go.Scatter(x=benchmark_df.index, y=benchmark_df["WTI_NAV"] / 1e6, mode="lines", name="WTI (Buy & Hold)", line=dict(color="#F18F01", width=1.5, dash="dot")))
    
    fig3.update_layout(
        yaxis_title="NAV (Millions $)",
        yaxis_tickformat="$.2f",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Strategy positioning & z-score
    st.subheader("Z-Score & Positioning")
    
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Top panel: Z-score
    fig4.add_trace(go.Scatter(x=equity_df.index, y=equity_df["z_score"], name="Z-Score", line=dict(color="#2E86AB", width=1)), row=1, col=1)
    fig4.add_hline(y=1.5, line_dash="dash", line_color="red", row=1, col=1, annotation_text="+1.5σ (Short Entry)")
    fig4.add_hline(y=-1.5, line_dash="dash", line_color="green", row=1, col=1, annotation_text="-1.5σ (Long Entry)")
    
    # Bottom panel: Position
    fig4.add_trace(go.Scatter(x=equity_df.index, y=equity_df["position"], name="Position", line=dict(color="#1F2937", width=1, shape="hv")), row=2, col=1)
    
    fig4.update_layout(height=500, hovermode="x unified", showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------
# TAB 3: Parameter Analytics
# ------------------------------------------
with tab3:
    st.subheader("Walk-Forward Parameter Optimisation")
    st.markdown("Evaluating Out-Of-Sample (OOS) Sharpe Ratio across Rolling Windows and Entry Thresholds.")
    
    if len(param_sweep) > 0:
        pivot = param_sweep.pivot(index="window", columns="entry_thresh", values="oos_sharpe")
        
        fig5 = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(x) for x in pivot.columns],
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 2),
            texttemplate="%{text}",
            hoverongaps=False
        ))
        
        fig5.update_layout(
            xaxis_title="Entry Threshold (σ)",
            yaxis_title="Rolling Window (Days)",
            height=500
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Parameter sweep data not found or empty.")

# ------------------------------------------
# TAB 4: Trade Log
# ------------------------------------------
with tab4:
    st.subheader("Detailed Trade Log")
    
    # Formatting for display
    display_log = trade_log.copy()
    display_log["entry_date"] = display_log["entry_date"].dt.strftime("%Y-%m-%d")
    if "exit_date" in display_log.columns:
        display_log["exit_date"] = pd.to_datetime(display_log["exit_date"]).dt.strftime("%Y-%m-%d")
    
    numeric_cols = ["entry_spread", "exit_spread", "net_pnl", "return_pct"]
    for col in numeric_cols:
        if col in display_log.columns:
            display_log[col] = display_log[col].round(2)
            
    # Function to apply colors to PnL
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
        
    st.dataframe(
        display_log.style.map(color_pnl, subset=["net_pnl"]) if hasattr(display_log.style, "map") else display_log.style.applymap(color_pnl, subset=["net_pnl"]),
        use_container_width=True,
        height=600
    )
