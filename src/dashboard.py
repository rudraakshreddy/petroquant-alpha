"""
src/dashboard.py — Interactive Plotly HTML Dashboard.

Generates a fully self-contained single HTML file with 6 tabbed panels.
No server required — open directly in any modern browser.

Tabs
----
1. Market Overview    : Crack spread history + component prices
2. Statistical Tests  : ADF/KPSS/Hurst/OU results table + R/S plot
3. Signal & Trades    : Z-score + entry/exit markers (interactive zoom)
4. Performance        : Equity curve + drawdown + annual bar chart
5. Risk Analytics     : Metrics table + P&L distribution + rolling Sharpe
6. Optimisation       : Parameter heatmap + walk-forward OOS Sharpe grid
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)

# Colour palette (consistent with matplotlib figures)
BLUE   = "#2E86AB"
RED    = "#E84855"
GREEN  = "#3BB273"
ORANGE = "#F18F01"
GREY   = "#6B7280"
DARK   = "#1F2937"
BG     = "#F9FAFB"
PANEL  = "#FFFFFF"


# ---------------------------------------------------------------------------
# Helper: Common Layout
# ---------------------------------------------------------------------------

def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=DARK, family="Inter, sans-serif")),
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", size=11, color=DARK),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor=GREY, borderwidth=1),
        margin=dict(l=55, r=20, t=60, b=45),
        xaxis=dict(gridcolor="#E5E7EB", showgrid=True, linecolor="#D1D5DB"),
        yaxis=dict(gridcolor="#E5E7EB", showgrid=True, linecolor="#D1D5DB"),
        hovermode="x unified",
    )


# ---------------------------------------------------------------------------
# Tab 1: Market Overview
# ---------------------------------------------------------------------------

def tab_market_overview(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.55], vertical_spacing=0.08,
        subplot_titles=("Component Prices ($/bbl)", "3:2:1 Crack Spread ($/bbl)"),
    )

    # Component prices — clip to sensible range (exclude COVID -$37 WTI spike)
    wti_clipped  = df["WTI"].clip(lower=-5)   # -$5 floor; show the crash, not the -37 artifact
    rbob_clipped = df["RBOB"].clip(lower=0)
    ho_clipped   = df["HO"].clip(lower=0)

    for series, color, name in [
        (wti_clipped,  RED,    "WTI Crude ($/bbl)"),
        (rbob_clipped, BLUE,   "RBOB Gasoline ($/bbl)"),
        (ho_clipped,   ORANGE, "Heating Oil ($/bbl)"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=series,
            name=name, line=dict(color=color, width=1.4), opacity=0.9,
            hovertemplate=f"{name}: $%{{y:.2f}}<br>%{{x}}",
        ), row=1, col=1)

    # Fix y-axis range for component panel to show all three lines clearly
    comp_max = max(rbob_clipped.max(), ho_clipped.max(), wti_clipped.max()) * 1.05
    comp_min = min(wti_clipped.min(), 0) - 5
    fig.update_yaxes(range=[comp_min, comp_max], row=1, col=1)

    # Crack spread + bands
    crack = df["crack"]
    fig.add_trace(go.Scatter(
        x=df.index.tolist() + df.index.tolist()[::-1],
        y=df["roll_upper_252"].tolist() + df["roll_lower_252"].tolist()[::-1],
        fill="toself", fillcolor="rgba(46,134,171,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Rolling mean ± 2σ band", showlegend=True,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["roll_mean_252"], name="252d Rolling Mean",
        line=dict(color=BLUE, width=1.2, dash="dash"), opacity=0.8,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=crack, name="Crack Spread",
        line=dict(color=DARK, width=1.1), opacity=0.9,
        hovertemplate="Crack: $%{y:.2f}/bbl<br>%{x}",
    ), row=2, col=1)

    fig.update_layout(**_base_layout("Market Overview — 3:2:1 Crack Spread (2019–2024)"))
    fig.update_yaxes(tickprefix="$", ticksuffix="/bbl")
    return fig


# ---------------------------------------------------------------------------
# Tab 2: Statistical Tests
# ---------------------------------------------------------------------------

def tab_statistical_tests(stat_results: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        subplot_titles=("R/S Analysis — Hurst Exponent Estimation", "Statistical Test Results"),
    )

    # Left: Hurst R/S scatter + fit
    h       = stat_results["hurst"]
    log_n   = np.array(h["log_n"])
    log_rs  = np.array(h["log_rs"])
    H       = h["H"]
    c_int   = h["intercept"]
    x_fit   = np.linspace(log_n.min(), log_n.max(), 200)

    fig.add_trace(go.Scatter(x=log_n.tolist(), y=log_rs.tolist(), mode="markers",
                              marker=dict(color=BLUE, size=8), name="Observed R/S",
                              hovertemplate="log(n): %{x:.3f}<br>log(R/S): %{y:.3f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_fit.tolist(), y=(H * x_fit + c_int).tolist(), mode="lines",
                              line=dict(color=RED, width=2),
                              name=f"OLS fit: H = {H:.4f} (R²={h['r_squared']:.3f})"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_fit.tolist(),
        y=(0.5 * x_fit + np.mean(log_rs) - 0.5 * np.mean(log_n)).tolist(),
        mode="lines", line=dict(color=GREY, width=1.2, dash="dot"),
        name="H = 0.5 (Random Walk)", opacity=0.7,
    ), row=1, col=1)

    fig.update_xaxes(title_text="log(Sub-period size n)", row=1, col=1)
    fig.update_yaxes(title_text="log(Mean R/S)", row=1, col=1)

    # Right: Summary table
    adf  = stat_results["adf"]["primary"]
    kpss = stat_results["kpss"]["level"]
    ou   = stat_results["ou"]

    headers  = ["Test", "Statistic", "p-value", "Result"]
    rows_data = [
        ["ADF (AIC)", f"{adf['adf_stat']:.4f}", f"{adf['p_value']:.5f}",
         "✓ Stationary" if adf["reject_null_5pct"] else "✗ Unit Root"],
        ["KPSS (level)", f"{kpss['kpss_stat']:.4f}", f"{kpss['p_value']:.5f}",
         "✓ Stationary" if kpss["fail_to_reject_5pct"] else "✗ Non-stationary"],
        [f"Hurst H", f"{H:.4f}", f"R²={h['r_squared']:.3f}",
         "✓ Mean-Rev." if H < 0.5 else "✗ Random Walk"],
        ["OU Half-Life", f"β={ou['beta']:.5f}", f"τ={ou['half_life_days']:.1f}d",
         "✓ Mean-Rev." if ou["beta"] < 0 else "✗ No Rev."],
    ]
    cell_colors = []
    for row in rows_data:
        result = row[-1]
        cell_colors.append(["white"] * 3 + [("#D1FAE5" if "✓" in result else "#FEE2E2")])

    fig.add_trace(go.Table(
        header=dict(values=headers, fill_color=DARK, font=dict(color="white", size=11),
                    align="left", height=32),
        cells=dict(values=list(zip(*rows_data)), fill_color=list(zip(*cell_colors)),
                   font=dict(size=10), align="left", height=28),
    ), row=1, col=2)

    fig.update_layout(**_base_layout("Statistical Evidence for Mean Reversion"))
    return fig


# ---------------------------------------------------------------------------
# Tab 3: Signal and Trades
# ---------------------------------------------------------------------------

def tab_signal_trades(df: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5], vertical_spacing=0.06,
        subplot_titles=("Crack Spread with Trade Markers", "Rolling Z-Score"),
    )

    crack = df["crack"]
    z     = df.get("z_score", pd.Series(np.nan, index=df.index))

    # Crack spread
    fig.add_trace(go.Scatter(x=df.index, y=crack, name="Crack Spread",
                              line=dict(color=DARK, width=1.0), opacity=0.85,
                              hovertemplate="$%{y:.2f}<br>%{x}"), row=1, col=1)

    # Trade entries/exits
    if len(trades) > 0:
        long_entries  = trades[trades["direction"] ==  1]
        short_entries = trades[trades["direction"] == -1]
        for trade_set, color, sym, name in [
            (long_entries,  GREEN,  "triangle-up",   "Long Entry"),
            (short_entries, RED,    "triangle-down",  "Short Entry"),
        ]:
            if len(trade_set) > 0:
                fig.add_trace(go.Scatter(
                    x=trade_set["entry_date"], mode="markers",
                    y=df["crack"].reindex(trade_set["entry_date"]).values,
                    name=name, marker=dict(color=color, size=10, symbol=sym),
                    hovertemplate=f"{name}: $%{{y:.2f}}<br>%{{x}}",
                ), row=1, col=1)
        # Exits
        exit_dates = trades["exit_date"].dropna()
        fig.add_trace(go.Scatter(
            x=exit_dates, mode="markers",
            y=df["crack"].reindex(exit_dates).values,
            name="Exit", marker=dict(color=GREY, size=8, symbol="x"),
            hovertemplate="Exit: $%{y:.2f}<br>%{x}",
        ), row=1, col=1)

    # Z-score with threshold bands
    fig.add_trace(go.Scatter(x=df.index, y=z, name="Z-Score",
                              line=dict(color=BLUE, width=1.0), opacity=0.9,
                              hovertemplate="z=%{y:.2f}<br>%{x}"), row=2, col=1)
    for val, color, dash, lbl in [
        (2.0, RED, "dash", "Entry ±2σ"), (-2.0, RED, "dash", None),
        (0.5, GREY, "dot", "Exit ±0.5σ"), (-0.5, GREY, "dot", None),
        (4.0, DARK, "dashdot", "Stop ±4σ"), (-4.0, DARK, "dashdot", None),
    ]:
        fig.add_hline(y=val, line=dict(color=color, dash=dash, width=0.9),
                      row=2, col=1, annotation_text=lbl if lbl else "",
                      annotation_position="right")

    fig.update_layout(**_base_layout("Signal Visualisation & Trade History"))
    fig.update_yaxes(title_text="Crack Spread ($/bbl)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score (σ)", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Tab 4: Performance
# ---------------------------------------------------------------------------

def tab_performance(equity: pd.DataFrame, config: Config) -> go.Figure:
    from src.risk_metrics import compute_max_drawdown

    dd_info = compute_max_drawdown(equity["nav"])
    dd_ser  = dd_info["drawdown_series"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
        subplot_titles=("Net Asset Value", "Drawdown (%)", "Daily P&L (USD)"),
    )

    nav = equity["nav"]

    # NAV
    fig.add_trace(go.Scatter(x=nav.index, y=nav / 1e6, name="NAV",
                              line=dict(color=BLUE, width=1.5),
                              hovertemplate="$%{y:.3f}M<br>%{x}",
                              fill="tozeroy", fillcolor="rgba(46,134,171,0.08)"), row=1, col=1)
    fig.add_hline(y=config.INITIAL_NAV / 1e6, line=dict(color=GREY, dash="dot", width=0.8),
                  row=1, col=1, annotation_text="Initial NAV")

    # Drawdown
    fig.add_trace(go.Scatter(x=dd_ser.index, y=dd_ser * 100, name="Drawdown",
                              fill="tozeroy", fillcolor="rgba(232,72,85,0.3)",
                              line=dict(color=RED, width=0.8),
                              hovertemplate="%{y:.2f}%<br>%{x}"), row=2, col=1)

    # Daily P&L
    dpnl = equity["daily_pnl"]
    colors_pnl = [GREEN if v >= 0 else RED for v in dpnl.values]
    fig.add_trace(go.Bar(x=dpnl.index, y=dpnl, name="Daily P&L",
                          marker_color=colors_pnl, opacity=0.7,
                          hovertemplate="$%{y:,.0f}<br>%{x}"), row=3, col=1)

    fig.update_layout(**_base_layout("Strategy Performance — Equity, Drawdown & Daily P&L"))
    fig.update_yaxes(title_text="NAV ($M)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", ticksuffix="%", autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", tickprefix="$", row=3, col=1)
    return fig


# ---------------------------------------------------------------------------
# Tab 5: Risk Analytics
# ---------------------------------------------------------------------------

def tab_risk_analytics(equity: pd.DataFrame, trades: pd.DataFrame,
                        metrics: dict, config: Config) -> go.Figure:
    from src.risk_metrics import compute_rolling_sharpe
    from scipy.stats import gaussian_kde

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Rolling 252-Day Sharpe Ratio",
            "Trade Net P&L Distribution",
            "Key Performance Metrics",
            "Exit Reason Attribution",
        ),
        specs=[
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "table"}, {"type": "pie"}],
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    nav  = equity["nav"]
    rets = nav.pct_change().dropna()

    # Rolling Sharpe
    rs = compute_rolling_sharpe(rets, 252, config.RISK_FREE_RATE).dropna()
    fig.add_trace(go.Scatter(x=rs.index, y=rs, name="Rolling Sharpe",
                              line=dict(color=BLUE, width=1.2),
                              fill="tozeroy",
                              fillcolor="rgba(46,134,171,0.1)",
                              hovertemplate="Sharpe: %{y:.2f}<br>%{x}"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=GREY, dash="dash", width=0.7), row=1, col=1)
    fig.add_hline(y=1, line=dict(color=GREEN, dash="dot", width=0.9), row=1, col=1,
                  annotation_text="Sharpe = 1")

    # P&L Histogram
    if len(trades) > 0:
        pnl   = trades["net_pnl"].dropna()
        wins  = pnl[pnl > 0]
        losses= pnl[pnl <= 0]
        bins  = np.linspace(pnl.min() * 1.1, pnl.max() * 1.1, 35).tolist()
        fig.add_trace(go.Histogram(x=wins.tolist(),   xbins=dict(start=bins[0], end=bins[-1], size=(bins[-1]-bins[0])/34),
                                    name="Wins",   marker_color=GREEN, opacity=0.65), row=1, col=2)
        fig.add_trace(go.Histogram(x=losses.tolist(), xbins=dict(start=bins[0], end=bins[-1], size=(bins[-1]-bins[0])/34),
                                    name="Losses", marker_color=RED, opacity=0.65), row=1, col=2)

    # Metrics table
    display_metrics = {
        "Total Return": f"{metrics.get('total_return_pct', 0):.2f}%",
        "CAGR":         f"{metrics.get('cagr_pct', 0):.2f}%",
        "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.4f}",
        "Sortino Ratio":f"{metrics.get('sortino_ratio', 0):.4f}",
        "Calmar Ratio": f"{metrics.get('calmar_ratio', 0):.4f}",
        "Max Drawdown": f"{metrics.get('max_drawdown_pct', 0):.2f}%",
        "Ann. Vol":     f"{metrics.get('ann_volatility_pct', 0):.2f}%",
        "VaR 95% (1d)": f"{metrics.get('var_95_daily_pct', 0):.3f}%",
        "CVaR 95% (1d)":f"{metrics.get('cvar_95_daily_pct', 0):.3f}%",
        "Hit Rate":     f"{metrics.get('hit_rate_pct', 0):.1f}%",
        "Profit Factor":f"{metrics.get('profit_factor', 0):.3f}",
        "# Trades":     str(metrics.get('n_trades', 0)),
        "Avg Hold":     f"{metrics.get('avg_hold_days', 0):.1f} days",
        "Total Costs":  f"${metrics.get('total_costs_usd', 0):,.0f}",
    }
    k_list = list(display_metrics.keys())
    v_list = list(display_metrics.values())
    row_colors = ["#EFF6FF" if i % 2 == 0 else "white" for i in range(len(k_list))]

    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], fill_color=DARK,
                    font=dict(color="white", size=11), align="left", height=28),
        cells=dict(values=[k_list, v_list],
                   fill_color=[row_colors, row_colors],
                   font=dict(size=10), align=["left", "right"], height=24),
    ), row=2, col=1)

    # Exit reason pie
    if len(trades) > 0 and "exit_reason" in trades.columns:
        reason_counts = trades["exit_reason"].value_counts()
        fig.add_trace(go.Pie(
            labels=reason_counts.index.tolist(), values=reason_counts.values.tolist(),
            name="Exit Reasons",
            marker=dict(colors=[GREEN, BLUE, RED, ORANGE]),
            textinfo="label+percent", hole=0.35,
        ), row=2, col=2)

    fig.update_layout(**_base_layout("Risk Analytics Dashboard"))
    return fig


# ---------------------------------------------------------------------------
# Tab 6: Optimisation
# ---------------------------------------------------------------------------

def tab_optimisation(sweep_results: pd.DataFrame, best_params: dict) -> go.Figure:
    pivot = sweep_results.pivot_table(
        values="oos_sharpe", index="window", columns="entry_thresh"
    ).round(3)

    z_vals   = pivot.values.tolist()
    x_labels = [str(c) + "σ" for c in pivot.columns.tolist()]
    y_labels = [str(r) + "d" for r in pivot.index.tolist()]

    fig = go.Figure(go.Heatmap(
        z=z_vals, x=x_labels, y=y_labels,
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z_vals],
        texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="OOS Sharpe"),
        hovertemplate="Window: %{y}<br>Entry: %{x}<br>OOS Sharpe: %{z:.3f}<extra></extra>",
    ))

    # Mark best
    if best_params:
        bw = str(best_params["window"]) + "d"
        be = str(best_params["entry_thresh"]) + "σ"
        if bw in y_labels and be in x_labels:
            fig.add_trace(go.Scatter(
                x=[be], y=[bw], mode="markers+text",
                marker=dict(symbol="star", size=18, color=BLUE,
                            line=dict(color="white", width=1.5)),
                text=["Best"], textposition="top center",
                name="Selected Parameters", showlegend=True,
            ))

    fig.update_layout(
        **_base_layout("Walk-Forward Parameter Optimisation — OOS Sharpe Heatmap"),
        xaxis_title="Entry Threshold (σ)",
        yaxis_title="Rolling Window (days)",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML Assembly
# ---------------------------------------------------------------------------

def build_dashboard(df: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame,
                    stat_results: dict, sweep_results: pd.DataFrame,
                    best_params: dict, metrics: dict, config: Config) -> str:
    """
    Assemble all six tabs into a single self-contained HTML file.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 8: BUILDING INTERACTIVE DASHBOARD")
    logger.info("=" * 60)

    from src.risk_metrics import compute_benchmark_metrics

    figures = {
        "Market Overview":    tab_market_overview(df),
        "Statistical Tests":  tab_statistical_tests(stat_results),
        "Signal & Trades":    tab_signal_trades(df, trades),
        "Performance":        tab_performance(equity, config),
        "Risk Analytics":     tab_risk_analytics(equity, trades, metrics, config),
        "Optimisation":       tab_optimisation(sweep_results, best_params),
    }

    # Convert each figure to an HTML div (no Plotly.js inline — use CDN)
    divs = {}
    for name, fig in figures.items():
        divs[name] = pio.to_html(fig, include_plotlyjs=False, full_html=False,
                                  config={"responsive": True, "displayModeBar": True,
                                          "toImageButtonOptions": {"format": "png", "scale": 2}})

    # Build Bootstrap 5 HTML with tab navigation
    tab_ids   = [f"tab-{i}" for i in range(len(divs))]
    tab_names = list(divs.keys())

    nav_items = "\n".join(
        f'<li class="nav-item" role="presentation">'
        f'<button class="nav-link {"active" if i == 0 else ""}" '
        f'id="{tab_ids[i]}-btn" data-bs-toggle="tab" '
        f'data-bs-target="#{tab_ids[i]}" type="button" role="tab">'
        f'{name}</button></li>'
        for i, name in enumerate(tab_names)
    )

    tab_panes = "\n".join(
        f'<div class="tab-pane fade {"show active" if i == 0 else ""}" '
        f'id="{tab_ids[i]}" role="tabpanel">'
        f'<div class="chart-container">{divs[name]}</div></div>'
        for i, name in enumerate(tab_names)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Crude Oil Crack Spread — Strategy Dashboard</title>
  <!-- Bootstrap 5 -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
        rel="stylesheet">
  <!-- Plotly -->
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
  <style>
    :root {{
      --blue:   #2E86AB;
      --dark:   #1F2937;
      --bg:     #F9FAFB;
    }}
    body {{ background: var(--bg); font-family: 'Inter', 'Segoe UI', sans-serif; }}
    .header {{
      background: linear-gradient(135deg, var(--dark), #374151);
      color: white; padding: 1.2rem 2rem; margin-bottom: 0;
    }}
    .header h1 {{ font-size: 1.5rem; font-weight: 700; margin: 0; }}
    .header p  {{ font-size: 0.85rem; opacity: 0.75; margin: 0.2rem 0 0 0; }}
    .nav-tabs .nav-link {{ color: var(--dark); font-size: 0.88rem; font-weight: 500;
                           border-radius: 0; padding: 0.55rem 1.1rem; }}
    .nav-tabs .nav-link.active {{ color: var(--blue); border-bottom: 2.5px solid var(--blue);
                                   background: white; font-weight: 700; }}
    .nav-tabs {{ background: white; border-bottom: 1px solid #E5E7EB;
                 padding: 0 1rem; position: sticky; top: 0; z-index: 100;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.07); }}
    .chart-container {{ padding: 1rem 1.5rem; min-height: 600px; }}
    .footer {{ background: var(--dark); color: rgba(255,255,255,0.55);
               font-size: 0.78rem; text-align: center; padding: 0.8rem 1rem; }}
    .metrics-badge {{
      display: inline-block; background: var(--blue); color: white;
      border-radius: 4px; padding: 0.2rem 0.6rem; font-size: 0.8rem; margin: 0.15rem;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🛢️ Crude Oil 3:2:1 Crack Spread — Mean-Reversion Strategy</h1>
    <p>Rolling Z-Score Signal · Event-Loop Backtester · Walk-Forward Optimisation · 2019–2024</p>
  </div>

  <!-- Tab navigation -->
  <ul class="nav nav-tabs" id="dashboardTabs" role="tablist">
    {nav_items}
  </ul>

  <!-- Tab content -->
  <div class="tab-content" id="dashboardTabContent">
    {tab_panes}
  </div>

  <div class="footer">
    Generated by Crack Spread Strategy Pipeline &nbsp;|&nbsp;
    3:2:1 Crack = (2×RBOB + HO − 3×WTI) / 3 &nbsp;|&nbsp;
    Data: Yahoo Finance CME front-month settlements &nbsp;|&nbsp;
    All prices in $/bbl
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    output_path = Path(config.DASHBOARD_DIR) / "crack_spread_dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"  Dashboard saved → {output_path}")
    return str(output_path)
