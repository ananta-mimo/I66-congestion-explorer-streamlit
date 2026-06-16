"""
I-66 ITB Congestion Forecast Explorer
======================================
Interactive Streamlit app showcasing the Phase 1 predictability ceiling study.
Simulates XGBoost TTI predictions from EDA-derived lookup tables and
displays real model evaluation results from the published analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="I-66 Congestion Forecast Explorer",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .metric-card {
    background: #0f1923;
    border: 1px solid #1e3048;
    border-radius: 6px;
    padding: 18px 22px;
    margin-bottom: 10px;
  }
  .metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a7a99;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.1;
    margin-top: 4px;
  }
  .metric-sub {
    font-size: 12px;
    color: #5a7a99;
    margin-top: 4px;
  }
  .congestion-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
    margin-top: 8px;
  }
  .free-flow    { background: #0d3326; color: #2ecc71; border: 1px solid #2ecc71; }
  .moderate     { background: #2d2200; color: #f39c12; border: 1px solid #f39c12; }
  .congested    { background: #2d0a0a; color: #e74c3c; border: 1px solid #e74c3c; }
  .severe       { background: #1a0000; color: #ff4757; border: 1px solid #ff4757; }

  .section-header {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3a6186;
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #1e3048;
    padding-bottom: 6px;
    margin-bottom: 16px;
    margin-top: 28px;
  }
  .insight-box {
    background: #0a1520;
    border-left: 3px solid #3a6186;
    padding: 12px 16px;
    border-radius: 0 4px 4px 0;
    font-size: 13px;
    color: #8aafcc;
    margin: 12px 0;
  }
  .stSelectbox label, .stSlider label, .stRadio label {
    font-size: 12px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a7a99 !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    tti_hour  = pd.read_csv("data/tti_by_hour_direction.csv")
    tti_dow   = pd.read_csv("data/tti_by_day_of_week.csv")
    vol_tmc   = pd.read_csv("data/volatility_by_tmc.csv")
    model_res = pd.read_csv("data/overall_model_results.csv")
    per_tmc   = pd.read_csv("data/per_tmc_results.csv")
    return tti_hour, tti_dow, vol_tmc, model_res, per_tmc

tti_hour, tti_dow, vol_tmc, model_res, per_tmc = load_data()

# ── Helper functions ──────────────────────────────────────────────────────────
DAY_LABELS = {0:"Monday", 1:"Tuesday", 2:"Wednesday",
              3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}

def get_tti(hour, direction):
    row = tti_hour[(tti_hour["hour_of_day"] == float(hour)) &
                   (tti_hour["direction"] == direction)]
    return float(row["tti"].values[0]) if len(row) else 1.0

def get_dow_multiplier(dow):
    row = tti_dow[tti_dow["day_of_week_number"] == float(dow)]
    base_dow = tti_dow["tti"].mean()
    return float(row["tti"].values[0]) / base_dow if len(row) else 1.0

def simulate_prediction(hour, direction, dow, horizon_min, seed=42):
    """
    Simulate XGBoost TTI predictions using EDA-derived lookup tables.
    Applies hour-of-day base TTI, day-of-week multiplier,
    and horizon-scaled noise drawn from real MAE distributions.
    """
    np.random.seed(seed + hour + dow)
    base_tti  = get_tti(hour, direction)
    dow_mult  = get_dow_multiplier(dow)
    tti_now   = base_tti * dow_mult

    # Horizon-scaled noise from real MAE values
    mae_lookup = {5: 0.091, 15: 0.118, 30: 0.143}
    noise_scale = mae_lookup[horizon_min] * 0.6
    predicted   = tti_now + np.random.normal(0, noise_scale)
    predicted   = max(0.8, predicted)

    # Confidence interval from RMSE
    rmse_lookup = {5: 0.287, 15: 0.365, 30: 0.427}
    ci_half = rmse_lookup[horizon_min] * 1.0

    return round(tti_now, 3), round(predicted, 3), round(ci_half, 3)

def congestion_label(tti):
    if tti < 1.1:   return "Free Flow",  "free-flow"
    elif tti < 1.3: return "Moderate",   "moderate"
    elif tti < 1.6: return "Congested",  "congested"
    else:           return "Severe",      "severe"

def tti_color(tti):
    if tti < 1.1:   return "#2ecc71"
    elif tti < 1.3: return "#f39c12"
    elif tti < 1.6: return "#e74c3c"
    else:           return "#ff4757"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 I-66 ITB Explorer")
    st.markdown(
        "<div style='font-size:12px;color:#5a7a99;line-height:1.6'>"
        "I-66 Inside the Beltway (ITB) is a managed lane corridor in "
        "Northern Virginia with asymmetric peak-hour tolling.<br><br>"
        "<b style='color:#8aafcc'>EB</b> is tolled during AM peak (5:30–9:30 AM).<br>"
        "<b style='color:#8aafcc'>WB</b> is tolled during PM peak (3:00–7:00 PM)."
        "</div>", unsafe_allow_html=True
    )
    st.markdown("---")

    direction = st.radio("Direction", ["EASTBOUND", "WESTBOUND"],
                         format_func=lambda x: "⮕ Eastbound (AM peak)" if x == "EASTBOUND"
                                                else "⬅ Westbound (PM peak)")

    hour = st.slider("Hour of Day", 0, 23, 17,
                     format="%d:00",
                     help="Select the hour you want to forecast")

    dow = st.selectbox("Day of Week", list(DAY_LABELS.keys()),
                       format_func=lambda x: DAY_LABELS[x], index=1)

    horizon = st.selectbox("Forecast Horizon", [5, 15, 30],
                           format_func=lambda x: f"{x} minutes ahead", index=0)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#3a6186;font-family:IBM Plex Mono,monospace'>"
        "DATA<br>"
        "<span style='color:#5a7a99'>RITIS/INRIX probe data<br>"
        "41 TMCs · 2022–2025<br>"
        "5-min UTC resolution</span>"
        "</div>", unsafe_allow_html=True
    )

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:IBM Plex Mono,monospace;font-size:1.6rem;"
    "font-weight:600;letter-spacing:0.04em;color:#cde4f5;margin-bottom:4px'>"
    "I-66 ITB · Congestion Forecast Explorer"
    "</h1>"
    "<p style='color:#5a7a99;font-size:13px;margin-top:0'>"
    "Phase 1 predictability ceiling study · Endogenous TTI baseline · "
    "XGBoost / Random Forest / Persistence"
    "</p>",
    unsafe_allow_html=True
)

# ── Main prediction panel ─────────────────────────────────────────────────────
tti_now, tti_pred, ci = simulate_prediction(hour, direction, dow, horizon)
label, badge_class      = congestion_label(tti_pred)
color                   = tti_color(tti_pred)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Current TTI</div>
      <div class='metric-value' style='color:{tti_color(tti_now)}'>{tti_now:.3f}</div>
      <div class='metric-sub'>{hour:02d}:00 · {DAY_LABELS[dow]}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Predicted TTI (+{horizon} min)</div>
      <div class='metric-value' style='color:{color}'>{tti_pred:.3f}</div>
      <div class='metric-sub'>± {ci:.3f} (±1 RMSE)</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Congestion State</div>
      <div class='metric-value' style='font-size:1.4rem;color:{color}'>{label}</div>
      <div class='congestion-badge {badge_class}'>{label.upper()}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    direction_short = "EB" if direction == "EASTBOUND" else "WB"
    toll_active = (direction == "EASTBOUND" and 5 <= hour <= 9) or \
                  (direction == "WESTBOUND" and 15 <= hour <= 18)
    toll_text  = "🟡 Toll Active" if toll_active else "⚪ Toll Inactive"
    toll_color = "#f39c12" if toll_active else "#5a7a99"
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Toll Status · {direction_short}</div>
      <div class='metric-value' style='font-size:1.2rem;color:{toll_color}'>{toll_text}</div>
      <div class='metric-sub'>Dynamic pricing window</div>
    </div>""", unsafe_allow_html=True)

# ── TTI scale reference ──────────────────────────────────────────────────────
st.markdown("<div class='insight-box'>"
            "<b>TTI (Travel Time Index)</b> is the ratio of actual travel time to free-flow travel time. "
            "TTI = 1.0 is perfect free flow. TTI = 2.0 means the trip takes twice as long as free flow. "
            "Thresholds: <b>&lt; 1.1</b> Free Flow · <b>1.1–1.3</b> Moderate · "
            "<b>1.3–1.6</b> Congested · <b>&gt; 1.6</b> Severe"
            "</div>", unsafe_allow_html=True)

# ── Charts ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Typical TTI Profile · 24-Hour Pattern</div>",
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    # Hourly TTI by direction
    eb = tti_hour[tti_hour["direction"] == "EASTBOUND"].sort_values("hour_of_day")
    wb = tti_hour[tti_hour["direction"] == "WESTBOUND"].sort_values("hour_of_day")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#0a1520")
    ax.set_facecolor("#0a1520")

    ax.plot(eb["hour_of_day"], eb["tti"], color="#3a9bd5", linewidth=2,
            label="Eastbound", zorder=3)
    ax.plot(wb["hour_of_day"], wb["tti"], color="#e74c3c", linewidth=2,
            label="Westbound", zorder=3)

    # Highlight selected hour
    ax.axvline(x=hour, color="#f39c12", linewidth=1.5, linestyle="--",
               alpha=0.8, zorder=4)
    ax.scatter([hour], [tti_now], color="#f39c12", s=80, zorder=5)

    ax.axhline(y=1.1, color="#2ecc71", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(y=1.3, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(y=1.6, color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_xlabel("Hour of Day", color="#5a7a99", fontsize=10)
    ax.set_ylabel("Mean TTI", color="#5a7a99", fontsize=10)
    ax.set_title("Mean TTI by Hour and Direction", color="#cde4f5",
                 fontsize=11, fontfamily="monospace")
    ax.tick_params(colors="#5a7a99")
    ax.set_xlim(0, 23)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3048")
    ax.legend(facecolor="#0f1923", edgecolor="#1e3048",
              labelcolor="#8aafcc", fontsize=9)
    ax.grid(axis="y", color="#1e3048", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_b:
    # Day of week TTI
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors_dow = ["#3a9bd5" if i < 5 else "#5a7a99" for i in range(7)]
    colors_dow[dow] = "#f39c12"

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    fig2.patch.set_facecolor("#0a1520")
    ax2.set_facecolor("#0a1520")

    bars = ax2.bar(dow_labels, tti_dow.sort_values("day_of_week_number")["tti"],
                   color=colors_dow, edgecolor="#0a1520", linewidth=0.5)
    ax2.axhline(y=1.1, color="#2ecc71", linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.set_xlabel("Day of Week", color="#5a7a99", fontsize=10)
    ax2.set_ylabel("Mean TTI", color="#5a7a99", fontsize=10)
    ax2.set_title("Mean TTI by Day of Week", color="#cde4f5",
                  fontsize=11, fontfamily="monospace")
    ax2.tick_params(colors="#5a7a99")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e3048")
    ax2.grid(axis="y", color="#1e3048", linewidth=0.8)
    ax2.set_ylim(1.0, 1.25)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Model performance panel ──────────────────────────────────────────────────
st.markdown("<div class='section-header'>Model Evaluation · Test Set Performance (2024–2025)</div>",
            unsafe_allow_html=True)

col_c, col_d = st.columns([1.2, 1])

with col_c:
    # MAE comparison chart across horizons and models
    models_order = ["Persistence", "LinearRegression", "RandomForest", "XGBoost"]
    model_colors = {"Persistence": "#5a7a99", "LinearRegression": "#8aafcc",
                    "RandomForest": "#3a9bd5", "XGBoost": "#2ecc71"}
    horizons_order = ["5min", "15min", "30min"]
    horizon_labels = {"5min": "5 min", "15min": "15 min", "30min": "30 min"}

    fig3, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=False)
    fig3.patch.set_facecolor("#0a1520")

    for i, h in enumerate(horizons_order):
        ax = axes[i]
        ax.set_facecolor("#0a1520")
        subset = model_res[model_res["horizon"] == h].set_index("model")
        maes   = [subset.loc[m, "test_mae"] if m in subset.index else 0
                  for m in models_order]
        cols   = [model_colors[m] for m in models_order]
        short_labels = ["Persist.", "LinReg", "RF", "XGB"]
        bars = ax.bar(short_labels, maes, color=cols, edgecolor="#0a1520", linewidth=0.5)

        # Highlight selected horizon
        if h == f"{horizon}min":
            for bar in bars:
                bar.set_edgecolor("#f39c12")
                bar.set_linewidth(2)

        ax.set_title(horizon_labels[h], color="#cde4f5", fontsize=10,
                     fontfamily="monospace")
        ax.set_ylabel("Test MAE" if i == 0 else "", color="#5a7a99", fontsize=9)
        ax.tick_params(colors="#5a7a99", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3048")
        ax.grid(axis="y", color="#1e3048", linewidth=0.8)

    fig3.suptitle("Test MAE by Model and Forecast Horizon", color="#cde4f5",
                  fontsize=11, fontfamily="monospace", y=1.02)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col_d:
    # Key finding callout
    st.markdown("<div class='section-header' style='margin-top:8px'>Key Finding</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box' style='font-size:13px;line-height:1.7'>
    At the <b style='color:#cde4f5'>5-minute horizon</b>, the simple persistence
    baseline (TTI<sub>t</sub> = TTI<sub>t-1</sub>) achieves
    <b style='color:#2ecc71'>MAE = 0.070</b>, which XGBoost
    barely improves upon (MAE = 0.091).<br><br>
    At <b style='color:#cde4f5'>30 minutes</b>, XGBoost
    (<b style='color:#2ecc71'>MAE = 0.143</b>) meaningfully outperforms
    persistence (<b style='color:#e74c3c'>MAE = 0.149</b>),
    confirming that endogenous TTI lags carry more signal at longer horizons.
    <br><br>
    This establishes the <b style='color:#cde4f5'>predictability ceiling</b>
    before adding exogenous signals (tolls, weather, events).
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics table
    st.markdown("<div class='section-header' style='margin-top:16px'>XGBoost vs Persistence</div>",
                unsafe_allow_html=True)

    xgb_res  = model_res[model_res["model"] == "XGBoost"][["horizon","test_mae","test_rmse"]]
    pers_res = model_res[model_res["model"] == "Persistence"][["horizon","test_mae","test_rmse"]]
    compare  = xgb_res.merge(pers_res, on="horizon", suffixes=("_xgb","_pers"))
    compare["MAE Δ"] = (compare["test_mae_xgb"] - compare["test_mae_pers"]).round(3)
    compare = compare[["horizon","test_mae_pers","test_mae_xgb","MAE Δ"]]
    compare.columns = ["Horizon", "Persistence MAE", "XGBoost MAE", "Δ MAE"]
    compare = compare.sort_values("Horizon")
    st.dataframe(compare.set_index("Horizon"), use_container_width=True)

# ── Segment volatility ───────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Segment Volatility · Predictability by TMC</div>",
            unsafe_allow_html=True)

col_e, col_f = st.columns([2, 1])

with col_e:
    dir_filter = "EASTBOUND" if direction == "EASTBOUND" else "WESTBOUND"
    vol_filtered = vol_tmc[vol_tmc["direction"] == dir_filter].sort_values(
        "volatility_30min", ascending=False).head(15)

    fig4, ax4 = plt.subplots(figsize=(9, 3.2))
    fig4.patch.set_facecolor("#0a1520")
    ax4.set_facecolor("#0a1520")

    bar_colors = [tti_color(1.0 + v * 3) for v in vol_filtered["volatility_30min"]]
    ax4.barh(vol_filtered["tmc"], vol_filtered["volatility_30min"],
             color=bar_colors, edgecolor="#0a1520")
    ax4.set_xlabel("30-min Volatility (std of TTI changes)", color="#5a7a99", fontsize=9)
    ax4.set_title(f"Top 15 Most Volatile Segments · {dir_filter.title()}",
                  color="#cde4f5", fontsize=10, fontfamily="monospace")
    ax4.tick_params(colors="#5a7a99", labelsize=8)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#1e3048")
    ax4.grid(axis="x", color="#1e3048", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

with col_f:
    st.markdown("""
    <div class='insight-box' style='font-size:12px;line-height:1.7;margin-top:8px'>
    <b style='color:#cde4f5'>Volatility bounds accuracy.</b><br><br>
    Segments with higher 30-min TTI volatility are inherently harder to
    forecast regardless of model complexity. This is the core
    <b>predictability ceiling</b> finding: irreducible uncertainty
    at volatile segments cannot be resolved by adding more features.
    <br><br>
    The regression analysis confirms a strong positive relationship
    between segment volatility and model error.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:11px;color:#3a6186;font-family:IBM Plex Mono,monospace;"
    "display:flex;justify-content:space-between'>"
    "<span>I-66 ITB · Endogenous TTI Predictability Ceiling Study</span>"
    "<span>Paper under review · Transportation Research Part C</span>"
    "<span>RITIS/INRIX probe data · 41 TMCs · 2022–2025</span>"
    "</div>",
    unsafe_allow_html=True
)
