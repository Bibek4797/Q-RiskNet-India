"""
Q-RiskNet India — Network Topology Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st

import src.network.mst as mst
import src.network.spectral as spectral
import src.network.network_runner as net_runner
import src.diagnostics.logger as diag
from dashboard.components.charts import render_network_graph, render_mst_graph
from dashboard.components.exports import download_csv
from dashboard.components.status import render_empty_state


def render_page(returns_df):
    """Renders the Financial Network Topology page."""

    st.header("🕸️ Financial Network Science & Systemic Topology")
    st.caption("Directed risk-spillover networks, centrality rankings, spectral community "
               "detection, and minimum spanning tree (MST) risk backbone.")

    if st.session_state.get("spillover_df") is None:
        render_empty_state("Calculate connectedness metrics in the "
                           "**🌊 Connectedness & Spillover** page first.")
        return

    spill_df = st.session_state["spillover_df"]

    # ── Network Controls ──────────────────────────────────────────────
    net_col, ctrl_col = st.columns([3, 1])
    with ctrl_col:
        st.markdown("### Controls")
        min_edge = st.slider("Min Edge (%)", 0.0, 15.0, 2.0, 0.5)
        layout = st.selectbox("Layout", ["circular", "spring"])
        comm_mode = st.radio("Communities", ["Auto", "Manual"], horizontal=True)

        if comm_mode == "Manual":
            max_c = max(2, len(spill_df.columns) - 1)
            n_comm = st.slider("# Communities", 2, max_c, min(3, max_c)) if max_c > 2 else 2
        else:
            n_comm = "auto"

    with net_col:
        try:
            comms = spectral.detect_communities(spill_df, n_communities=n_comm)
            render_network_graph(spill_df, comms, min_edge, layout)
        except Exception as e:
            diag.log_error("Network rendering failure", e)
            st.error(f"Error: {str(e)}")

    st.markdown("---")

    # ── Centrality Rankings ───────────────────────────────────────────
    st.subheader("📊 Network Centrality Rankings")
    try:
        net_res = net_runner.run_all_network_analysis(
            spill_df, returns_df, threshold_pct=min_edge, save_reports=True)
        cent_df = net_res["centrality_df"]
        gs = net_res["global_stats"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Directed Edges", gs["Edge_Count"])
        c2.metric("Density", f"{gs['Network_Density']:.3f}")
        c3.metric("Top Hub", cent_df.iloc[0]["Sector"] if not cent_df.empty else "N/A")
        c4.metric("Top Bridge",
                  cent_df.sort_values("Betweenness_Centrality", ascending=False).iloc[0]["Sector"]
                  if not cent_df.empty else "N/A")

        st.dataframe(cent_df, use_container_width=True)
        download_csv(cent_df, "network_centrality.csv", key="dl_cent")
    except Exception as e:
        st.error(f"Centrality error: {str(e)}")

    st.markdown("---")

    # ── MST ───────────────────────────────────────────────────────────
    st.subheader("🌲 Minimum Spanning Tree (MST)")
    try:
        dist = mst.compute_correlation_distance(returns_df)
        mst_g = mst.construct_mst(dist)
        render_mst_graph(mst_g, dist)
    except Exception as e:
        st.error(f"MST error: {str(e)}")
