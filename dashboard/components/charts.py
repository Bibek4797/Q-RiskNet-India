"""
Q-RiskNet India — Executive Plotly Chart Components
Copyright (c) 2026 Bibek Rout
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import src.visualization.plotly_plots as vis

# Plotly configuration — Modebar enabled with zoom/pan/autoscale/reset tools
CHART_CONFIG = {
    'scrollZoom': False,  # Keep page scroll smooth, while allowing box zoom, pan, and modebar zoom/pan
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d'],  # Keep zoom2d, pan2d, select2d, zoomIn2d, zoomOut2d, autoScale2d, resetScale2d
    'doubleClick': 'reset+autosize'
}

# Premium colour palette for chart traces
_PALETTE = [
    '#818cf8', '#a78bfa', '#c084fc', '#fb7185', '#34d399',
    '#38bdf8', '#fbbf24', '#f97316', '#e879f9', '#2dd4bf'
]


def _premium_layout(fig, height=None):
    """Applies clean professional dark layout to any Plotly figure with interactive axes."""
    h = height or 420
    fig.update_layout(
        height=h,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(family='Inter, sans-serif', size=12, color='#94a3b8'),
        title=dict(
            font=dict(family='Inter, sans-serif', size=13, color='#94a3b8'),
            x=0.0, xanchor='left', pad=dict(l=2, b=8)
        ),
        legend=dict(
            bgcolor='rgba(11,17,32,0.8)',
            bordercolor='rgba(99,102,241,0.2)',
            borderwidth=1,
            font=dict(size=11, color='#64748b'),
            itemsizing='constant'
        ),
        margin=dict(l=4, r=4, t=44, b=8),
        colorway=_PALETTE,
        uirevision='dataset_view',  # Preserves user zoom state during minor reruns, but autoranges on reset
    )
    fig.update_xaxes(
        fixedrange=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False,
        tickfont=dict(size=10, color='#475569'),
        linecolor='rgba(255,255,255,0.08)',
        showline=True,
    )
    fig.update_yaxes(
        fixedrange=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False,
        tickfont=dict(size=10, color='#475569'),
        linecolor='rgba(255,255,255,0.08)',
        showline=True,
    )
    return fig


def _render_plotly(fig, height=None, key=None):
    """Applies clean layout and renders Plotly chart with modebar interaction controls."""
    fig = _premium_layout(fig, height)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key=key)


def render_prices_chart(prices_df, key=None):
    """Renders base-100 normalized price chart."""
    norm_prices = (prices_df / prices_df.iloc[0]) * 100
    fig = px.line(
        norm_prices,
        x=norm_prices.index,
        y=norm_prices.columns,
        title="Base-100 Normalized Price Trends",
        labels={"value": "Normalized Level (Base=100)", "variable": "Sector"}
    )
    _render_plotly(fig, height=450, key=key)


def render_drawdowns_chart(drawdowns_df, key=None):
    """Renders peak-to-trough percentage drawdown chart."""
    fig = px.line(
        drawdowns_df,
        x=drawdowns_df.index,
        y=drawdowns_df.columns,
        title="Sectoral Historical Drawdowns (%)",
        labels={"value": "Drawdown (%)", "variable": "Sector"}
    )
    _render_plotly(fig, height=400, key=key)


def render_rolling_volatility_chart(vol_df, window_label="20-Day", key=None):
    """Renders rolling annualized volatility chart."""
    fig = px.line(
        vol_df,
        x=vol_df.index,
        y=vol_df.columns,
        title=f"Annualized Rolling Volatility ({window_label} Window %)",
        labels={"value": "Annualized Volatility (%)", "variable": "Sector"}
    )
    _render_plotly(fig, height=400, key=key)


def render_conditional_volatility_chart(returns_series, cond_vol_series, model_name, key=None):
    """Overlays return series with ±2σ conditional volatility bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns_series.index, y=returns_series.values,
        mode='lines', name='Daily Return (%)',
        line=dict(color='#64748b', width=1), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=cond_vol_series.index, y=2.0 * cond_vol_series.values,
        mode='lines', name='+2σ Upper Volatility Band',
        line=dict(color='#ef4444', width=1.8, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=cond_vol_series.index, y=-2.0 * cond_vol_series.values,
        mode='lines', name='-2σ Lower Volatility Band',
        line=dict(color='#ef4444', width=1.8, dash='dash')
    ))
    fig.update_layout(
        title=f"Conditional Volatility Envelopes (±2σ) - {model_name} ({returns_series.name})",
        xaxis_title="Date", yaxis_title="Return (%) / Volatility"
    )
    _render_plotly(fig, height=450, key=key)


def render_forecast_benchmark_chart(preds_df, target_sector, key=None):
    """Renders Out-of-Sample actual returns vs model predictions line plot."""
    fig = px.line(
        preds_df,
        x=preds_df.index,
        y=preds_df.columns,
        title=f"Out-of-Sample Forecast Predictions Benchmark ({target_sector})",
        labels={"value": "Daily Log Return (%)", "variable": "Model / Actual"}
    )
    _render_plotly(fig, height=450, key=key)


def render_feature_importance_chart(feat_series, title_str, key=None):
    """Renders Feature Importance bar chart."""
    top_feats = feat_series.sort_values(ascending=True).tail(10)
    fig = px.bar(
        x=top_feats.values,
        y=top_feats.index,
        orientation='h',
        title=title_str,
        labels={"x": "Gini Importance", "y": "Lagged Feature"}
    )
    _render_plotly(fig, height=380, key=key)


def render_qvar_heatmap(coeff_matrix, quantile_val, key=None):
    """Renders QVAR Autoregressive Coefficient Matrix Heatmap."""
    fig = px.imshow(
        coeff_matrix,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"QVAR Coefficient Matrix Φ₁(τ={quantile_val:.2f}) [Row=Target, Col=Source]"
    )
    _render_plotly(fig, height=420, key=key)


def render_qvar_girf_chart(girf_df, shocked_sector, quantile_val, key=None):
    """Renders Generalized Impulse Response Function (GIRF) curves."""
    fig = px.line(
        girf_df,
        x=girf_df.index,
        y=girf_df.columns,
        title=f"QVAR Generalized Impulse Responses (GIRF, τ={quantile_val:.2f}) to +2σ Shock in {shocked_sector}",
        labels={"x": "Horizon (Days)", "value": "Response (%)", "variable": "Sector"}
    )
    _render_plotly(fig, height=420, key=key)


def render_acf_pacf_chart(lags, acf_vals, pacf_vals, sector_name, key=None):
    """Renders ACF and PACF bar charts."""
    col_a, col_p = st.columns(2)
    key_acf = f"{key}_acf" if key else None
    key_pacf = f"{key}_pacf" if key else None
    with col_a:
        fig_acf = px.bar(
            x=lags, y=acf_vals,
            title=f"Autocorrelation (ACF) - {sector_name}",
            labels={"x": "Lag", "y": "ACF"}
        )
        _render_plotly(fig_acf, height=350, key=key_acf)
    with col_p:
        fig_pacf = px.bar(
            x=lags, y=pacf_vals,
            title=f"Partial Autocorrelation (PACF) - {sector_name}",
            labels={"x": "Lag", "y": "PACF"}
        )
        _render_plotly(fig_pacf, height=350, key=key_pacf)


def render_kde_comparison_chart(series, x_grid, norm_pdf, key=None):
    """Renders Empirical Return Histogram with Gaussian PDF Overlay."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=series, histnorm='probability density',
        name='Empirical Returns', marker_color='#3b82f6', opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x_grid, y=norm_pdf, mode='lines',
        name='Gaussian Normal Fit', line=dict(color='#ef4444', width=2.5, dash='dash')
    ))
    fig.update_layout(
        title=f"Empirical Returns Distribution vs Gaussian Fit ({series.name})",
        xaxis_title="Return (%)", yaxis_title="Density"
    )
    _render_plotly(fig, height=400, key=key)


def render_rolling_variance_chart(roll_var, sector_name, key=None):
    """Renders 20-Day Rolling Variance plot for volatility clustering."""
    fig = px.line(
        roll_var, x=roll_var.index, y=roll_var.values,
        title=f"Rolling Sample Variance (20-Day Window) - {sector_name}",
        labels={"value": "Variance", "index": "Date"}
    )
    _render_plotly(fig, height=380, key=key)


def render_correlation_chart(corr_df, key=None):
    """Renders Pearson correlation matrix heatmap."""
    fig = vis.render_correlation_heatmap(corr_df)
    _render_plotly(fig, height=500, key=key)


def render_spillover_charts(metrics, key_net=None, key_tf=None):
    """Renders Net Risk Flow bar chart and Gross Risk Flow (Transmitted vs Received) bar chart."""
    col1, col2 = st.columns(2)
    with col1:
        net_series = metrics['NET'].sort_values()
        fig_net = px.bar(
            x=net_series.values, y=net_series.index,
            orientation='h', color=net_series.values,
            color_continuous_scale="RdYlGn_r",
            title="Net Risk Flow (Risk Transmitted - Received)",
            labels={"x": "Net Risk Flow (%)", "y": "Sector"}
        )
        _render_plotly(fig_net, height=400, key=key_net)

    with col2:
        to_from_df = pd.DataFrame({"Risk Transmitted": metrics['TO'], "Risk Received": metrics['FROM']})
        fig_tf = px.bar(
            to_from_df, barmode='group',
            title="Gross Risk Flow (Transmitted vs Received)"
        )
        _render_plotly(fig_tf, height=400, key=key_tf)


def render_network_graph(spill_df, comms, min_edge, layout_style, key=None):
    """Renders Plotly directed risk spillover network graph."""
    fig = vis.render_spillover_network(
        spill_df, communities=comms,
        min_threshold_pct=min_edge, layout_type=layout_style
    )
    _render_plotly(fig, height=600, key=key)


def render_mst_graph(mst_graph, dist_matrix, key=None):
    """Renders Minimum Spanning Tree (MST) graph."""
    fig = vis.render_mst_network(mst_graph, dist_matrix)
    _render_plotly(fig, height=550, key=key)


def render_rolling_tci_chart(rolling_tci_df, window_size, step_size, key=None):
    """Renders rolling window TCI line chart with 50% reference line."""
    fig = px.line(
        rolling_tci_df, y="Rolling TCI (%)",
        title=f"Rolling Systemic Connectedness — Window {window_size}d, Step {step_size}d",
        labels={"value": "TCI (%)", "Date": "Date"}
    )
    # Reference line at 50% — moderate/high connectedness boundary
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="rgba(245,158,11,0.4)",
        annotation_text="50% threshold",
        annotation_position="bottom right",
        annotation_font_color="#64748b",
        annotation_font_size=10,
    )
    _render_plotly(fig, height=400, key=key)
