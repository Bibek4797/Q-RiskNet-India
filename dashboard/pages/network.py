"""
Q-RiskNet India — Financial Network Science Section
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.network.mst as mst
import src.network.spectral as spectral
import src.network.network_runner as net_runner
import src.diagnostics.logger as diag
from dashboard.components.charts import render_network_graph, render_mst_graph
from dashboard.components.exports import download_csv


def render_page(returns_df):
    """Renders the Financial Network Science page."""

    st.markdown("### Risk Network & Topology (Diebold–Yılmaz Network Mapping)")
    st.caption("Visualises the Diebold–Yılmaz (2014) directed spillover network topology to identify systemic hubs, shock transmitters, and vulnerable receivers.")

    spill_df = st.session_state.get("spillover_df")

    if spill_df is None:
        st.info(
            "**No spillover data available.** Navigate to **Connectedness** and click **Run Analysis** first, "
            "or return to **Overview** to auto-compute baseline spillovers."
        )
        return

    # ── Network Graph ──────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl2:
        st.markdown("**Graph controls**")
        if st.button("↺ Reset view", key="btn_reset_net_graph"):
            st.session_state["key_net_graph"] = st.session_state.get("key_net_graph", 0) + 1
        min_edge = st.slider(
            "Min edge strength (%)",
            0.0, 15.0, 2.0, 0.5,
            help="Hide edges below this spillover threshold to reduce clutter"
        )

    with ctrl1:
        try:
            net_key = f"net_graph_{st.session_state.get('key_net_graph', 0)}"
            render_network_graph(spill_df, None, min_edge, layout_style="circular", key=net_key)
        except Exception as e:
            diag.log_error("Network rendering failure", e)
            st.error(f"Network rendering error: {str(e)}")

    st.markdown("---")

    # ── Key Systemic Sectors ───────────────────────────────────────────
    st.markdown("### Key systemic sectors")

    try:
        net_res = net_runner.run_all_network_analysis(
            spill_df, returns_df, threshold_pct=min_edge, save_reports=True
        )
        cent_df = net_res["centrality_df"]
        gs = net_res["global_stats"]

        # 4 summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        top_transmitter = (
            cent_df.sort_values("Out_Degree_Export", ascending=False).iloc[0]["Sector"]
            if not cent_df.empty else "N/A"
        )
        top_receiver = (
            cent_df.sort_values("In_Degree_Import", ascending=False).iloc[0]["Sector"]
            if not cent_df.empty else "N/A"
        )
        top_hub = (
            cent_df.sort_values("PageRank_Centrality", ascending=False).iloc[0]["Sector"]
            if "PageRank_Centrality" in cent_df.columns and not cent_df.empty else "N/A"
        )
        top_bridge = (
            cent_df.sort_values("Betweenness_Centrality", ascending=False).iloc[0]["Sector"]
            if "Betweenness_Centrality" in cent_df.columns and not cent_df.empty else "N/A"
        )

        k1.metric("Top Risk Transmitter", top_transmitter,
                  help="Sector with the highest total outbound risk spillover")
        k2.metric("Top Risk Receiver", top_receiver,
                  help="Sector with the highest total inbound risk spillover")
        k3.metric("Most Systemically Important", top_hub,
                  help="Sector with the highest network influence (PageRank)")
        k4.metric("Key Bridge Sector", top_bridge,
                  help="Sector that most frequently lies on the shortest risk-transmission paths (betweenness)")

        # Simplified ranking table — friendly columns only
        friendly_cols = {
            "Out_Degree_Export": "Risk Transmitted (%)",
            "In_Degree_Import": "Risk Received (%)",
            "Net_Degree_Export": "Net Risk Flow (%)"
        }
        show_cols = ["Sector"] + [v for k, v in friendly_cols.items() if k in cent_df.columns]
        friendly_df = cent_df.rename(columns=friendly_cols)[show_cols]
        friendly_df = friendly_df.sort_values("Risk Transmitted (%)", ascending=False) if "Risk Transmitted (%)" in friendly_df.columns else friendly_df

        st.dataframe(friendly_df, use_container_width=True, hide_index=True)

        with st.expander("Advanced network metrics"):
            st.caption("PageRank = systemic influence. Betweenness = bridge/contagion role. Eigenvector = connection to other important sectors.")
            full_rename = {
                "Out_Degree_Export": "Risk Transmitted (%)",
                "In_Degree_Import": "Risk Received (%)",
                "Net_Degree_Export": "Net Risk Flow (%)",
                "PageRank_Centrality": "Systemic Influence (PageRank)",
                "Eigenvector_Centrality": "Systemic Importance (Eigenvector)",
                "Betweenness_Centrality": "Bridge Role (Betweenness)",
                "Closeness_Centrality": "Closeness"
            }
            st.dataframe(cent_df.rename(columns=full_rename), use_container_width=True, hide_index=True)

            st.markdown(f"""
            **Network summary:**
            - Risk connections above threshold: **{gs.get('Edge_Count', '—')}**
            - Network density: **{gs.get('Network_Density', 0):.3f}**
            """)

            with st.expander("Download network metrics"):
                download_csv(cent_df.rename(columns=full_rename), "network_systemic_rankings.csv", key="dl_cent")

    except Exception as e:
        st.error(f"Centrality calculation error: {str(e)}")

    st.markdown("---")

    # ── Minimum Spanning Tree ─────────────────────────────────────────
    st.markdown("### Risk Backbone (Minimum Spanning Tree — Prim's Algorithm)")
    st.caption(
        "Extracts the acyclic topological backbone connecting all 10 sectoral indices via the strongest risk channels "
        "using Prim's algorithm on the total bilateral spillover weights ($W_{ij} = S_{ij} + S_{ji}$) from the directed network above."
    )

    try:
        dist_matrix, weights_df = mst.compute_spillover_distance(spill_df, method="sum")
        mst_g = mst.construct_mst(dist_matrix, algorithm="prim", weights_df=weights_df)
        plot_title = "Systemic Risk Backbone — Prim's MST (Bilateral Weight Sum: S_ij + S_ji)"

        render_mst_graph(mst_g, dist_matrix, title=plot_title)

        # MST Edges Detail Table
        mst_edges = []
        for u, v, d in mst_g.edges(data=True):
            mst_edges.append({
                "Sector 1": u,
                "Sector 2": v,
                "Bilateral Spillover Weight (%)": round(float(d.get("bilateral_weight", 0.0)), 2),
                "Prim MST Distance": round(float(d.get("weight", 0.0)), 4)
            })

        mst_edges_df = pd.DataFrame(mst_edges).sort_values("Bilateral Spillover Weight (%)", ascending=False)

        with st.expander("View MST Backbone Edges (Prim's Algorithm)", expanded=False):
            st.dataframe(mst_edges_df, use_container_width=True, hide_index=True)
            download_csv(mst_edges_df, "mst_backbone_edges.csv", key="dl_mst_edges")

    except Exception as e:
        st.error(f"MST error: {str(e)}")
