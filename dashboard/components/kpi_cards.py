"""
Q-RiskNet India — Premium KPI Cards Component
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def render_kpi_cards(metrics):
    """
    Renders premium glassmorphism KPI cards for TCI, Top Systemic Transmitter,
    and Top Risk Receiver.
    """
    tci_val = metrics['TCI']
    max_transmitter = metrics['NET'].idxmax()
    max_receiver = metrics['NET'].idxmin()
    net_out = metrics['NET'][max_transmitter]
    net_in = metrics['NET'][max_receiver]

    # Colour-code TCI: green < 40, amber 40-65, red > 65
    if tci_val < 40:
        tci_colour = "#22c55e"
        tci_label = "Low Connectedness"
    elif tci_val < 65:
        tci_colour = "#f59e0b"
        tci_label = "Moderate Connectedness"
    else:
        tci_colour = "#ef4444"
        tci_label = "High Connectedness"

    col_m1, col_m2, col_m3 = st.columns(3, gap="medium")

    with col_m1:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02));
                    border:1px solid rgba(99,102,241,0.28);border-radius:14px;padding:20px 22px;
                    position:relative;overflow:hidden;'>
            <div style='position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,#6366f1,#a78bfa);'></div>
            <div style='font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.09em;margin-bottom:8px;'>Total Connectedness Index</div>
            <div style='font-family:"Space Grotesk",sans-serif;font-size:2.2rem;font-weight:700;
                        color:{tci_colour};line-height:1;'>{tci_val:.2f}%</div>
            <div style='font-size:0.78rem;color:{tci_colour};opacity:0.75;margin-top:6px;
                        font-weight:500;'>{tci_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(255,255,255,0.02));
                    border:1px solid rgba(239,68,68,0.25);border-radius:14px;padding:20px 22px;
                    position:relative;overflow:hidden;'>
            <div style='position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,#ef4444,#f97316);'></div>
            <div style='font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.09em;margin-bottom:8px;'>Top Risk Transmitter</div>
            <div style='font-family:"Space Grotesk",sans-serif;font-size:1.6rem;font-weight:700;
                        color:#fca5a5;line-height:1.15;'>{max_transmitter}</div>
            <div style='font-size:0.78rem;color:#f87171;margin-top:6px;font-weight:500;'>
                ↑ +{net_out:.2f}% Net Outflow</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(34,197,94,0.08),rgba(255,255,255,0.02));
                    border:1px solid rgba(34,197,94,0.25);border-radius:14px;padding:20px 22px;
                    position:relative;overflow:hidden;'>
            <div style='position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,#22c55e,#10b981);'></div>
            <div style='font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.09em;margin-bottom:8px;'>Top Risk Receiver</div>
            <div style='font-family:"Space Grotesk",sans-serif;font-size:1.6rem;font-weight:700;
                        color:#86efac;line-height:1.15;'>{max_receiver}</div>
            <div style='font-size:0.78rem;color:#4ade80;margin-top:6px;font-weight:500;'>
                ↓ {net_in:.2f}% Net Inflow</div>
        </div>
        """, unsafe_allow_html=True)
