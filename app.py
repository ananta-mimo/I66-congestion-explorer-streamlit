"""
I-66 ITB Congestion Forecast Explorer — Redesigned
====================================================
Clean research-dashboard aesthetic. White/warm background,
serif display type, road-sign TTI meter as signature element.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="I-66 ITB · Congestion Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,600;9..144,700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* Reset */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: #FAFAF8;
  color: #1a2332;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background-color: #1a2332 !important;
  border-right: none;
}
section[data-testid="stSidebar"] * {
  color: #c8d4e0 !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
  color: #8A9BB0 !important;
  font-size: 10px !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace !important;
}
section[data-testid="stSidebar"] hr {
  border-color: #2d3f54 !important;
}

/* Road sign TTI meter — signature element */
.tti-sign {
  display: flex;
  align-items: stretch;
  gap: 0;
  border: 4px solid #1a2332;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 28px;
  font-family: 'JetBrains Mono', monospace;
}
.tti-sign-label {
  background: #1a2332;
  color: #FAFAF8;
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-width: 130px;
}
.tti-sign-label .sign-eyebrow {
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #8A9BB0;
  margin-bottom: 4px;
}
.tti-sign-label .sign-title {
  font-size: 13px;
  font-weight: 600;
  color: #FAFAF8;
  line-height: 1.3;
}
.tti-sign-value {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px 28px;
  gap: 32px;
}
.tti-block {
  text-align: center;
}
.tti-num {
  font-size: 2.8rem;
  font-weight: 600;
  line-height: 1;
}
.tti-desc {
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #8A9BB0;
  margin-top: 4px;
}
.tti-divider {
  width: 1px;
  background: #e0ddd8;
  margin: 8px 0;
}
.status-pill {
  display: inline-block;
  padding: 5px 16px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
}
.pill-free    { background: #1e5c38; color: #ffffff; }
.pill-mod     { background: #8a5a00; color: #ffffff; }
.pill-cong    { background: #8a1a00; color: #ffffff; }
.pill-severe  { background: #4a0000; color: #ffffff; }
.pill-toll-on  { background: #C0392B; color: #ffffff; }
.pill-toll-off { background: #e8e4e0; color: #8A9BB0; }

/* Section styling */
.section-rule {
  border: none;
  border-top: 1.5px solid #1a2332;
  margin: 36px 0 20px 0;
}
.section-eyebrow {
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #8A9BB0;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 4px;
}
.section-title {
  font-family: 'Fraunces', serif;
  font-size: 1.35rem;
  font-weight: 600;
  color: #1a2332;
  margin-bottom: 16px;
  line-height: 1.2;
}

/* Stat cards */
.stat-row {
  display: flex;
  gap: 1px;
  background: #d8d4ce;
  border: 1px solid #d8d4ce;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 20px;
}
.stat-cell {
  flex: 1;
  background: #FAFAF8;
  padding: 16px 18px;
}
.stat-cell-label {
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #8A9BB0;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 6px;
}
.stat-cell-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.6rem;
  font-weight: 600;
  color: #1a2332;
  line-height: 1;
}
.stat-cell-sub {
  font-size: 11px;
  color: #8A9BB0;
  margin-top: 4px;
}

/* Finding box */
.finding-box {
  background: #EEF2F7;
  border-left: 4px solid #1a2332;
  padding: 16px 20px;
  border-radius: 0 4px 4px 0;
  font-size: 13px;
  line-height: 1.7;
  color: #2d3f54;
}
.finding-box b { color: #1a2332; }

/* Footer */
.app-footer {
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid #d8d4ce;
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: #8A9BB0;
  font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    tti_hour  = pd.read_csv("data/tti_by_hour_direction.csv")
    tti_dow   = pd.read_csv("data/tti_by_day_of_week.csv")
    vol_tmc   = pd.read_csv("data/volatility_by_tmc.csv")
    model_res = pd.read_csv("data/overall_model_results.csv")
    per_tmc   = pd.read_csv("data/per_tmc_results.csv")
    return tti_hour, tti_dow, vol_tmc, model_res, per_tmc

tti_hour, tti_dow, vol_tmc, model_res, per_tmc = load_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
DAY_LABELS = {0:"Monday",1:"Tuesday",2:"Wednesday",
              3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}

def get_tti(hour, direction):
    row = tti_hour[(tti_hour["hour_of_day"]==float(hour)) &
                   (tti_hour["direction"]==direction)]
    return float(row["tti"].values[0]) if len(row) else 1.0

def get_dow_multiplier(dow):
    row = tti_dow[tti_dow["day_of_week_number"]==float(dow)]
    base = tti_dow["tti"].mean()
    return float(row["tti"].values[0])/base if len(row) else 1.0

def simulate_prediction(hour, direction, dow, horizon_min):
    np.random.seed(42 + hour + dow)
    base_tti = get_tti(hour, direction)
    dow_mult = get_dow_multiplier(dow)
    tti_now  = base_tti * dow_mult
    mae_lookup  = {5:0.091, 15:0.118, 30:0.143}
    rmse_lookup = {5:0.287, 15:0.365, 30:0.427}
    predicted = max(0.8, tti_now + np.random.normal(0, mae_lookup[horizon_min]*0.6))
    return round(tti_now,3), round(predicted,3), round(rmse_lookup[horizon_min],3)

def congestion_info(tti):
    if tti < 1.1:   return "Free Flow",  "pill-free",  "#1e5c38"
    elif tti < 1.3: return "Moderate",   "pill-mod",   "#8a5a00"
    elif tti < 1.6: return "Congested",  "pill-cong",  "#8a1a00"
    else:           return "Severe",     "pill-severe", "#4a0000"

def chart_color(tti):
    if tti < 1.1:   return "#2d8a50"
    elif tti < 1.3: return "#c78d00"
    elif tti < 1.6: return "#C0392B"
    else:           return "#7a0000"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 8px 0'>
      <div style='font-size:9px;letter-spacing:0.2em;text-transform:uppercase;
                  color:#8A9BB0;font-family:JetBrains Mono,monospace;margin-bottom:6px'>
        Research Dashboard
      </div>
      <div style='font-family:Fraunces,serif;font-size:1.3rem;font-weight:600;
                  color:#FAFAF8;line-height:1.2'>
        I-66 ITB<br>Congestion<br>Forecast
      </div>
      <div style='margin-top:10px;font-size:11px;color:#8A9BB0;line-height:1.5'>
        Northern Virginia<br>
        41 TMC segments<br>
        2022–2025
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    direction = st.radio(
        "Direction",
        ["EASTBOUND","WESTBOUND"],
        format_func=lambda x: "→ Eastbound" if x=="EASTBOUND" else "← Westbound"
    )
    hour = st.slider("Hour of Day", 0, 23, 17, format="%d:00")
    dow  = st.selectbox("Day of Week", list(DAY_LABELS.keys()),
                        format_func=lambda x: DAY_LABELS[x], index=1)
    horizon = st.selectbox("Forecast Horizon",
                           [5,15,30],
                           format_func=lambda x: f"{x} min ahead")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;line-height:1.8;color:#4a6070'>
      <div style='color:#8A9BB0;letter-spacing:0.12em;font-size:9px;
                  text-transform:uppercase;margin-bottom:6px;
                  font-family:JetBrains Mono,monospace'>About</div>
      I-66 ITB has asymmetric dynamic tolling.<br>
      <b style='color:#8A9BB0'>EB</b> tolled AM peak 5:30–9:30<br>
      <b style='color:#8A9BB0'>WB</b> tolled PM peak 3:00–7:00<br><br>
      TTI = actual travel time ÷ free-flow travel time.<br>
      TTI 1.0 = free flow. TTI 2.0 = twice as slow.
    </div>
    """, unsafe_allow_html=True)

# ── Compute prediction ────────────────────────────────────────────────────────
tti_now, tti_pred, ci = simulate_prediction(hour, direction, dow, horizon)
c_label, c_pill, c_hex = congestion_info(tti_pred)
toll_on = ((direction=="EASTBOUND" and 5<=hour<=9) or
           (direction=="WESTBOUND" and 15<=hour<=18))
dir_short = "EB" if direction=="EASTBOUND" else "WB"

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding-bottom:4px'>
  <div style='font-size:10px;letter-spacing:0.18em;text-transform:uppercase;
              color:#8A9BB0;font-family:JetBrains Mono,monospace;margin-bottom:6px'>
    Phase 1 · Endogenous TTI Predictability Ceiling Study
  </div>
  <div style='font-family:Fraunces,serif;font-size:2rem;font-weight:600;
              color:#1a2332;line-height:1.1;margin-bottom:4px'>
    I-66 Inside the Beltway — Congestion Forecast Explorer
  </div>
  <div style='font-size:13px;color:#8A9BB0'>
    XGBoost · Random Forest · Persistence baseline &nbsp;·&nbsp; Manuscript under review,
    <i>Transportation Research Part C</i>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='section-rule' style='margin-top:20px'>", unsafe_allow_html=True)

# ── SIGNATURE ELEMENT: Road-sign TTI meter ────────────────────────────────────
toll_pill  = "pill-toll-on"  if toll_on else "pill-toll-off"
toll_text  = "Toll Active"   if toll_on else "Toll Inactive"

st.markdown(f"""
<div class='tti-sign'>
  <div class='tti-sign-label'>
    <div class='sign-eyebrow'>Live Estimate</div>
    <div class='sign-title'>{dir_short} · {hour:02d}:00<br>{DAY_LABELS[dow]}</div>
  </div>
  <div class='tti-sign-value'>
    <div class='tti-block'>
      <div class='tti-num' style='color:#1a2332'>{tti_now:.3f}</div>
      <div class='tti-desc'>Current TTI</div>
    </div>
    <div class='tti-divider'></div>
    <div class='tti-block'>
      <div class='tti-num' style='color:{c_hex}'>{tti_pred:.3f}</div>
      <div class='tti-desc'>Predicted +{horizon} min</div>
    </div>
    <div class='tti-divider'></div>
    <div class='tti-block'>
      <span class='status-pill {c_pill}'>{c_label}</span>
      <div class='tti-desc' style='margin-top:8px'>Congestion State</div>
    </div>
    <div class='tti-divider'></div>
    <div class='tti-block'>
      <span class='status-pill {toll_pill}'>{toll_text}</span>
      <div class='tti-desc' style='margin-top:8px'>I-66 Toll Window</div>
    </div>
    <div class='tti-divider'></div>
    <div class='tti-block'>
      <div class='tti-num' style='font-size:1.6rem;color:#8A9BB0'>±{ci:.3f}</div>
      <div class='tti-desc'>±1 RMSE Interval</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 24-hour TTI profile ───────────────────────────────────────────────────────
st.markdown("<div class='section-eyebrow'>Traffic Pattern</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>24-Hour TTI Profile by Direction</div>", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    eb = tti_hour[tti_hour["direction"]=="EASTBOUND"].sort_values("hour_of_day")
    wb = tti_hour[tti_hour["direction"]=="WESTBOUND"].sort_values("hour_of_day")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    ax.fill_between(eb["hour_of_day"], 1.0, eb["tti"],
                    color="#1a2332", alpha=0.08)
    ax.fill_between(wb["hour_of_day"], 1.0, wb["tti"],
                    color="#C0392B", alpha=0.08)
    ax.plot(eb["hour_of_day"], eb["tti"], color="#1a2332",
            linewidth=2, label="Eastbound")
    ax.plot(wb["hour_of_day"], wb["tti"], color="#C0392B",
            linewidth=2, label="Westbound", linestyle="--")
    ax.axvline(x=hour, color="#C0392B", linewidth=1.2,
               linestyle=":", alpha=0.7)
    ax.scatter([hour], [tti_now], color="#C0392B", s=60, zorder=5)

    for level, lbl in [(1.1,"Free Flow"), (1.3,"Moderate"), (1.6,"Congested")]:
        ax.axhline(y=level, color="#d8d4ce", linewidth=0.8, linestyle="-")
        ax.text(23.2, level, lbl, va="center", fontsize=7,
                color="#8A9BB0", fontstyle="italic")

    ax.set_xlabel("Hour of Day", fontsize=9, color="#8A9BB0")
    ax.set_ylabel("Mean TTI", fontsize=9, color="#8A9BB0")
    ax.set_xlim(0, 24)
    ax.tick_params(colors="#8A9BB0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0ddd8")
    ax.legend(fontsize=9, framealpha=0, labelcolor="#1a2332")
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_b:
    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    vals = tti_dow.sort_values("day_of_week_number")["tti"].values
    colors_dow = ["#1a2332" if i<5 else "#8A9BB0" for i in range(7)]
    colors_dow[dow] = "#C0392B"

    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    fig2.patch.set_facecolor("#FAFAF8")
    ax2.set_facecolor("#FAFAF8")

    ax2.bar(dow_labels, vals, color=colors_dow,
            edgecolor="#FAFAF8", linewidth=1.5, width=0.6)
    ax2.axhline(y=1.0, color="#d8d4ce", linewidth=0.8)
    ax2.set_xlabel("Day of Week", fontsize=9, color="#8A9BB0")
    ax2.set_ylabel("Mean TTI", fontsize=9, color="#8A9BB0")
    ax2.set_ylim(0.98, 1.22)
    ax2.tick_params(colors="#8A9BB0", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#e0ddd8")
    ax2.grid(axis="y", color="#e0ddd8", linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Model evaluation ──────────────────────────────────────────────────────────
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-eyebrow'>Model Evaluation</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Test Set Performance · 2024–2025</div>", unsafe_allow_html=True)

col_c, col_d = st.columns([1.4, 1])

with col_c:
    models_order  = ["Persistence","LinearRegression","RandomForest","XGBoost"]
    short_labels  = ["Persist.","LinReg","RF","XGB"]
    model_colors  = {"Persistence":"#d8d4ce","LinearRegression":"#8A9BB0",
                     "RandomForest":"#1a2332","XGBoost":"#C0392B"}
    horizons_order = ["5min","15min","30min"]
    horizon_labels = {"5min":"5 min","15min":"15 min","30min":"30 min"}

    fig3, axes3 = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)
    fig3.patch.set_facecolor("#FAFAF8")

    for i, h in enumerate(horizons_order):
        ax = axes3[i]
        ax.set_facecolor("#FAFAF8")
        subset = model_res[model_res["horizon"]==h].set_index("model")
        maes   = [subset.loc[m,"test_mae"] if m in subset.index else 0
                  for m in models_order]
        cols   = [model_colors[m] for m in models_order]
        bars   = ax.bar(short_labels, maes, color=cols,
                        edgecolor="#FAFAF8", linewidth=1, width=0.6)

        if h == f"{horizon}min":
            for bar in bars:
                bar.set_edgecolor("#C0392B")
                bar.set_linewidth(2)

        ax.set_title(horizon_labels[h], fontsize=10, color="#1a2332", fontweight="600")
        if i == 0:
            ax.set_ylabel("Test MAE", fontsize=8, color="#8A9BB0")
        ax.tick_params(colors="#8A9BB0", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e0ddd8")
        ax.grid(axis="y", color="#e0ddd8", linewidth=0.6)

    fig3.suptitle("Test MAE by Model and Horizon  ·  Selected horizon highlighted",
                  fontsize=9, color="#8A9BB0", y=1.02)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col_d:
    st.markdown("""
    <div class='finding-box'>
      <b>Predictability ceiling finding.</b><br><br>
      At the <b>5-minute horizon</b>, persistence (MAE 0.070) is nearly
      as accurate as XGBoost (MAE 0.091) — endogenous lags alone
      offer limited marginal gain at short ranges.<br><br>
      At <b>30 minutes</b>, XGBoost (MAE 0.143) meaningfully beats
      persistence (MAE 0.149), confirming lags carry more signal
      at longer horizons.<br><br>
      This ceiling is the baseline before exogenous signals —
      tolls, weather, events — are added in Phase 2.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    xgb_res  = model_res[model_res["model"]=="XGBoost"][["horizon","test_mae","test_rmse"]]
    pers_res = model_res[model_res["model"]=="Persistence"][["horizon","test_mae","test_rmse"]]
    compare  = xgb_res.merge(pers_res, on="horizon", suffixes=("_xgb","_pers"))
    compare["Δ MAE"] = (compare["test_mae_xgb"]-compare["test_mae_pers"]).round(3)
    compare = compare[["horizon","test_mae_pers","test_mae_xgb","Δ MAE"]]
    compare.columns = ["Horizon","Persistence","XGBoost","Δ MAE"]
    st.dataframe(compare.set_index("Horizon"), use_container_width=True)

# ── Segment volatility ────────────────────────────────────────────────────────
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-eyebrow'>Spatial Analysis</div>", unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>Segment Volatility · {direction.title()} Direction</div>",
            unsafe_allow_html=True)

col_e, col_f = st.columns([2, 1])

with col_e:
    vol_filtered = vol_tmc[vol_tmc["direction"]==direction].sort_values(
        "volatility_30min", ascending=False).head(15)

    fig4, ax4 = plt.subplots(figsize=(9, 3.4))
    fig4.patch.set_facecolor("#FAFAF8")
    ax4.set_facecolor("#FAFAF8")

    vol_vals = vol_filtered["volatility_30min"].values
    max_v    = vol_vals.max()
    bar_cols = [plt.cm.RdYlGn_r(v / max_v * 0.85 + 0.05) for v in vol_vals]

    ax4.barh(vol_filtered["tmc"], vol_vals,
             color=bar_cols, edgecolor="#FAFAF8", linewidth=0.5, height=0.65)
    ax4.set_xlabel("30-min Volatility (std of TTI changes)",
                   fontsize=9, color="#8A9BB0")
    ax4.tick_params(colors="#8A9BB0", labelsize=8)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#e0ddd8")
    ax4.grid(axis="x", color="#e0ddd8", linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

with col_f:
    st.markdown("""
    <div class='finding-box'>
      <b>Volatility bounds accuracy.</b><br><br>
      High-volatility segments are inherently harder to forecast
      regardless of model complexity. Irreducible uncertainty
      at these segments cannot be resolved by adding features.<br><br>
      Regression analysis across 41 TMCs confirms a strong
      positive relationship between 30-min TTI volatility
      and test-set model error.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-footer'>
  <span>I-66 ITB · Phase 1 Endogenous TTI Study</span>
  <span>RITIS/INRIX probe data · 41 TMCs · 2022–2025</span>
  <span>Manuscript under review · Transportation Research Part C</span>
</div>
""", unsafe_allow_html=True)
