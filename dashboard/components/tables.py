import pandas as pd
import streamlit as st

def render_descriptive_table(desc_stats_df):
    """
    Renders econometric descriptive statistics table.
    """
    st.dataframe(desc_stats_df, use_container_width=True, height=450)
    st.caption("JB: Jarque-Bera Normality Test (p < 0.05 indicates fat tails). ADF: Augmented Dickey-Fuller Unit Root Test (p < 0.05 indicates stationarity).")

def render_spillover_matrix_table(spill_df, metrics):
    """
    Renders Diebold-Yilmaz spillover matrix table with styling and fallback.
    """
    display_spill = spill_df.copy()
    display_spill["TO OTHERS"] = metrics["TO"]
    from_row = pd.Series(metrics["FROM"], name="FROM OTHERS")
    display_spill = pd.concat([display_spill, pd.DataFrame([from_row])])
    display_spill.loc["FROM OTHERS", "TO OTHERS"] = metrics["TCI"]

    try:
        st.dataframe(
            display_spill.style.format("{:.2f}%").background_gradient(cmap="Reds", subset=(spill_df.index, spill_df.columns)), 
            use_container_width=True
        )
    except Exception:
        st.dataframe(display_spill.style.format("{:.2f}%"), use_container_width=True)

    st.caption("Rows represent receiving sectors (FROM); columns represent transmitting sectors (TO). Diagonal represents self-spillover.")
