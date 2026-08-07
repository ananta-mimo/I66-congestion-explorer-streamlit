"""
I-66 ITB Congestion Forecast Explorer — v2
============================================
Changes from v1:
  - Slider defaults to current local hour (US/Eastern)
  - Day-of-week defaults to today
  - Fonts switched to DM Sans + DM Mono (cleaner, less AI-default)
  - Volatility chart shows intersection name alongside TMC code
  - Font sizes bumped up across the board
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import pytz

st.set_page_config(
    page_title="I-66 ITB · Congestion Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Current time in US/Eastern ────────────────────────────────────────────────
eastern   = pytz.timezone("America/New_York")
now_et    = datetime.now(eastern)
now_hour  = now_et.hour
now_dow   = now_et.weekday()   # 0 = Monday

# ── Design tokens ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: #F7F5F2;
  color: #1C2B3A;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background-color: #EEECEA !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
  color: #1C2B3A !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
  font-size: 11px !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #5A7A90 !important;
  font-family: 'DM Mono', monospace !important;
}
section[data-testid="stSidebar"] hr {
  border-color: #C8C4BE !important;
  margin: 18px 0 !important;
}

/* ── Road sign TTI meter ── */
.tti-sign {
  display: flex;
  align-items: stretch;
  gap: 0;
  border: 3px solid #1C2B3A;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 32px;
  font-family: 'DM Mono', monospace;
  box-shadow: 4px 4px 0px #1C2B3A;
}
.tti-sign-label {
  background: #1C2B3A;
  color: #F7F5F2;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-width: 140px;
}
.sign-eyebrow {
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #6A8AA0;
  margin-bottom: 6px;
  font-family: 'DM Mono', monospace;
}
.sign-title {
  font-size: 15px;
  font-weight: 600;
  color: #F7F5F2;
  line-height: 1.4;
  font-family: 'DM Sans', sans-serif;
}
.tti-sign-value {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: space-around;
  padding: 20px 24px;
  background: #FFFFFF;
  gap: 8px;
}
.tti-block { text-align: center; }
.tti-num {
  font-size: 3rem;
  font-weight: 700;
  line-height: 1;
  font-family: 'DM Mono', monospace;
}
.tti-desc {
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #6A8AA0;
  margin-top: 6px;
  font-family: 'DM Mono', monospace;
}
.tti-divider {
  width: 1px;
  background: #E8E4DE;
  margin: 4px 0;
  align-self: stretch;
}
.status-pill {
  display: inline-block;
  padding: 6px 18px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-family: 'DM Mono', monospace;
}
.pill-free   { background:#1B5E35; color:#fff; }
.pill-mod    { background:#7A4F00; color:#fff; }
.pill-cong   { background:#8B1A00; color:#fff; }
.pill-severe { background:#3D0000; color:#fff; }
.pill-toll-on  { background:#B03020; color:#fff; }
.pill-toll-off { background:#E8E4DE; color:#8A9BB0; }

/* ── Section styling ── */
.section-rule {
  border: none;
  border-top: 2px solid #1C2B3A;
  margin: 40px 0 22px 0;
}
.section-eyebrow {
  font-size: 11px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #6A8AA0;
  font-family: 'DM Mono', monospace;
  margin-bottom: 6px;
}
.section-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.55rem;
  font-weight: 700;
  color: #1C2B3A;
  margin-bottom: 20px;
  line-height: 1.2;
}

/* ── Finding box ── */
.finding-box {
  background: #EDF2F7;
  border-left: 4px solid #1C2B3A;
  padding: 18px 22px;
  border-radius: 0 6px 6px 0;
  font-size: 14px;
  line-height: 1.75;
  color: #2D3F52;
}
.finding-box b { color: #1C2B3A; }

/* ── Footer ── */
.app-footer {
  margin-top: 52px;
  padding-top: 18px;
  border-top: 1px solid #D8D4CE;
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #8A9BB0;
  font-family: 'DM Mono', monospace;
}

/* ── Page header ── */
.page-eyebrow {
  font-size: 11px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #6A8AA0;
  font-family: 'DM Mono', monospace;
  margin-bottom: 8px;
}
.page-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.2rem;
  font-weight: 700;
  color: #1C2B3A;
  line-height: 1.15;
  margin-bottom: 6px;
}
.page-sub {
  font-size: 14px;
  color: #6A8AA0;
  font-family: 'DM Sans', sans-serif;
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
    tmc_id    = pd.read_csv("data/TMC_Identification.csv")
    # Build clean TMC → intersection label (keep one row per TMC)
    tmc_labels = (tmc_id[['tmc','intersection']]
                  .drop_duplicates(subset='tmc')
                  .set_index('tmc')['intersection']
                  .to_dict())
    return tti_hour, tti_dow, vol_tmc, model_res, per_tmc, tmc_labels

tti_hour, tti_dow, vol_tmc, model_res, per_tmc, tmc_labels = load_data()

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
    if tti < 1.1:   return "Free Flow",  "pill-free",  "#1B5E35"
    elif tti < 1.3: return "Moderate",   "pill-mod",   "#7A4F00"
    elif tti < 1.6: return "Congested",  "pill-cong",  "#8B1A00"
    else:           return "Severe",     "pill-severe", "#3D0000"

def chart_color(tti):
    if tti < 1.1:   return "#2A9D5C"
    elif tti < 1.3: return "#D4900A"
    elif tti < 1.6: return "#B03020"
    else:           return "#7A0000"

def tmc_display_label(tmc):
    """Return 'INTERSECTION (TMC)' or just TMC if not found."""
    intersection = tmc_labels.get(tmc, "")
    if intersection:
        # Clean up the intersection string a bit
        clean = intersection.replace("/EXIT", " Exit").replace("/", " / ")
        return f"{clean}  ({tmc})"
    return tmc

# ── Sidebar — defaults to current time ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 10px 0'>
      <div class='sign-eyebrow'>Research Dashboard</div>
      <div style='font-family:Playfair Display,serif;font-size:1.4rem;
                  font-weight:700;color:#1C2B3A;line-height:1.25;margin-top:4px'>
        I-66 ITB<br>Congestion<br>Forecast
      </div>
      <div style='margin-top:12px;font-size:13px;color:#6A8AA0;line-height:1.6;
                  font-family:DM Sans,sans-serif'>
        Northern Virginia<br>41 TMC segments · 2022–2025
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    direction = st.radio(
        "Direction",
        ["EASTBOUND","WESTBOUND"],
        format_func=lambda x: "→ Eastbound" if x=="EASTBOUND" else "← Westbound"
    )

    # Default to current Eastern hour
    hour = st.slider("Hour of Day", 0, 23, now_hour, format="%d:00",
                     help="Defaults to current time in US/Eastern")

    # Default to today's day of week
    dow = st.selectbox("Day of Week", list(DAY_LABELS.keys()),
                       format_func=lambda x: DAY_LABELS[x],
                       index=now_dow)

    horizon = st.selectbox("Forecast Horizon",
                           [5, 15, 30],
                           format_func=lambda x: f"{x} min ahead")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px;line-height:1.8;color:#4a6070;font-family:DM Sans,sans-serif'>
      <div class='sign-eyebrow' style='margin-bottom:8px'>About</div>
      I-66 ITB uses asymmetric dynamic tolling.<br>
      <b style='color:#8A9BB0'>EB</b> tolled AM peak 5:30–9:30<br>
      <b style='color:#8A9BB0'>WB</b> tolled PM peak 3:00–7:00<br><br>
      TTI = actual ÷ free-flow travel time.<br>
      TTI 1.0 = free flow · TTI 2.0 = 2× slower.<br><br>
      <span style='color:#4a6070'>Current time (ET): {now_et.strftime('%H:%M')}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Compute prediction ────────────────────────────────────────────────────────
tti_now, tti_pred, ci = simulate_prediction(hour, direction, dow, horizon)
c_label, c_pill, c_hex = congestion_info(tti_pred)
toll_on = ((direction=="EASTBOUND" and 5<=hour<=9) or
           (direction=="WESTBOUND" and 15<=hour<=18))
dir_short = "EB" if direction=="EASTBOUND" else "WB"

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding-bottom:6px'>
  <div class='page-eyebrow'>Endogenous Travel Time Index (TTI) Predictability Ceiling Study</div>
  <div class='page-title'>I-66 Inside the Beltway<br>Congestion Forecast Explorer</div>
  <div class='page-sub'>
    XGBoost · Random Forest · Linear Regression · Persistence baseline &nbsp;·&nbsp;
    
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='section-rule' style='margin-top:22px'>", unsafe_allow_html=True)

# ── SIGNATURE ELEMENT: Road-sign TTI meter ────────────────────────────────────
toll_pill = "pill-toll-on"  if toll_on else "pill-toll-off"
toll_text = "Toll Active"   if toll_on else "Toll Inactive"

st.markdown(f"""
<div class='tti-sign'>
  <div class='tti-sign-label'>
    <div class='sign-eyebrow'>Live Estimate </div>
    <div class='sign-eyebrow'>(from historic data)</div>
    <div class='sign-title'>
      {dir_short} · {hour:02d}:00<br>{DAY_LABELS[dow]}
    </div>
  </div>
  <div class='tti-sign-value'>
    <div class='tti-block'>
      <div class='tti-num' style='color:#1C2B3A'>{tti_now:.3f}</div>
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
      <div class='tti-desc' style='margin-top:10px'>Congestion State</div>
    </div>
    <div class='tti-divider'></div>
    <div class='tti-block'>
      <span class='status-pill {toll_pill}'>{toll_text}</span>
      <div class='tti-desc' style='margin-top:10px'>I-66 Toll Window</div>
    </div>
    <div class='tti-divider'></div>
    <div class='tti-block'>
      <div class='tti-num' style='font-size:2rem;color:#8A9BB0'>±{ci:.3f}</div>
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

    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor("#F0DBB2")
    ax.set_facecolor("#FFFFFF")

    ax.fill_between(eb["hour_of_day"], 1.0, eb["tti"], color="#1C2B3A", alpha=0.07)
    ax.fill_between(wb["hour_of_day"], 1.0, wb["tti"], color="#B03020", alpha=0.07)
    ax.plot(eb["hour_of_day"], eb["tti"], color="#1C2B3A", linewidth=2.2,
            label="Eastbound")
    ax.plot(wb["hour_of_day"], wb["tti"], color="#B03020", linewidth=2.2,
            label="Westbound", linestyle="--")
    ax.axvline(x=hour, color="#B03020", linewidth=1.4, linestyle=":", alpha=0.8)
    ax.scatter([hour], [tti_now], color="#B03020", s=70, zorder=5)

    for level, lbl in [(1.1,"Free Flow"), (1.3,"Moderate"), (1.6,"Congested")]:
        ax.axhline(y=level, color="#C9300A", linewidth=0.9)
        ax.text(23.3, level, lbl, va="center", fontsize=8,
                color="#0F151D", fontstyle="italic")

    ax.set_xlabel("Hour of Day", fontsize=10, color="#0E171D",
                  fontfamily="DM Sans")
    ax.set_ylabel("Mean TTI", fontsize=10, color="#0E171D",
                  fontfamily="DM Sans")
    ax.set_xlim(0, 24)
    ax.tick_params(colors="#0B1520", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0F0C08")
    ax.legend(fontsize=10, framealpha=0, labelcolor="#1C2B3A")
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_b:
    dow_labels_short = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    vals = tti_dow.sort_values("day_of_week_number")["tti"].values
    colors_dow = ["#1C2B3A" if i < 5 else "#8A9BB0" for i in range(7)]
    colors_dow[dow] = "#B03020"

    fig2, ax2 = plt.subplots(figsize=(7, 3.4))
    fig2.patch.set_facecolor("#F5E2C5")
    ax2.set_facecolor("#FFFFFF")

    ax2.bar(dow_labels_short, vals, color=colors_dow,
            edgecolor="#FFFFFF", linewidth=1.5, width=0.6)
    ax2.axhline(y=1.0, color="#E0DDD8", linewidth=0.9)
    ax2.set_xlabel("Day of Week", fontsize=10, color="#1A689C")
    ax2.set_ylabel("Mean TTI", fontsize=10, color="#206492")
    ax2.set_ylim(0.98, 1.22)
    ax2.tick_params(colors="#0E141B", labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#E8E4DE")
    ax2.grid(axis="y", color="#E8E4DE", linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Model evaluation ──────────────────────────────────────────────────────────
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-eyebrow'>Model Evaluation</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Test Set Performance · 2024–2025</div>",
            unsafe_allow_html=True)

col_c, col_d = st.columns([1.4, 1])

with col_c:
    models_order  = ["Persistence","LinearRegression","RandomForest","XGBoost"]
    short_labels  = ["Persist.","LinReg","RF","XGB"]
    model_colors  = {"Persistence":"#D8D4CE","LinearRegression":"#8A9BB0",
                     "RandomForest":"#1C2B3A","XGBoost":"#B03020"}
    horizons_order = ["5min","15min","30min"]
    horizon_labels = {"5min":"5 min","15min":"15 min","30min":"30 min"}

    fig3, axes3 = plt.subplots(1, 3, figsize=(9, 3.4), sharey=False)
    fig3.patch.set_facecolor("#F7F5F2")

    for i, h in enumerate(horizons_order):
        ax = axes3[i]
        ax.set_facecolor("#FFFFFF")
        subset = model_res[model_res["horizon"]==h].set_index("model")
        maes   = [subset.loc[m,"test_mae"] if m in subset.index else 0
                  for m in models_order]
        cols   = [model_colors[m] for m in models_order]
        bars   = ax.bar(short_labels, maes, color=cols,
                        edgecolor="#FFFFFF", linewidth=1.2, width=0.6)

        if h == f"{horizon}min":
            for bar in bars:
                bar.set_edgecolor("#B03020")
                bar.set_linewidth(2.2)

        ax.set_title(horizon_labels[h], fontsize=11, color="#1C2B3A",
                     fontweight="bold", pad=8)
        if i == 0:
            ax.set_ylabel("Test MAE", fontsize=9, color="#6A8AA0")
        ax.tick_params(colors="#08448D", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#E8E4DE")
        ax.grid(axis="y", color="#E8E4DE", linewidth=0.6)

    fig3.suptitle("Test MAE by Model and Horizon  ·  Selected horizon outlined in red",
                  fontsize=9, color="#124D96", y=1.02)
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

# ── Segment volatility with intersection names ────────────────────────────────
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-eyebrow'>Spatial Analysis</div>", unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>Segment Volatility/Variability · {direction.title()}</div>",
            unsafe_allow_html=True)

col_e, col_f = st.columns([2, 1])

with col_e:
# Merge road_order from TMC identification file
    tmc_order = (pd.read_csv("data/TMC_Identification.csv")
                  [['tmc','road_order']]
                  .drop_duplicates(subset='tmc'))

    vol_filtered = (vol_tmc[vol_tmc["direction"]==direction]
                    .merge(tmc_order, on='tmc', how='left')
                    .sort_values("road_order")
                    .copy())

    # Build display labels: "INTERSECTION (TMC)"
    vol_filtered["label"] = vol_filtered["tmc"].apply(tmc_display_label)

    fig4, ax4 = plt.subplots(figsize=(9, 4.2))
    fig4.patch.set_facecolor("#F7F5F2")
    ax4.set_facecolor("#FFFFFF")

    vol_vals = vol_filtered["volatility_30min"].values
    max_v    = vol_vals.max() if vol_vals.max() > 0 else 1
    bar_cols = [plt.cm.RdYlGn_r(v / max_v * 0.80 + 0.08) for v in vol_vals]

    ax4.barh(vol_filtered["label"], vol_vals,
             color=bar_cols, edgecolor="#FFFFFF", linewidth=0.5, height=0.65)
    ax4.set_xlabel("30-min Volatility (std of TTI changes)",
                   fontsize=10, color="#01090E")
    ax4.tick_params(colors="#020C13", labelsize=9)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#E8E4DE")
    ax4.grid(axis="x", color="#E8E4DE", linewidth=0.6)
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
      and test-set model error.<br><br>
      Labels show the nearest intersection so you can locate
      each segment on the corridor.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-footer'>
  <span>I-66 ITB · Phase 1 Endogenous TTI Study</span>
  <span>NPMRDS probe data · 41 TMCs · 2022–2025</span>
  <span>Manuscript under review</span>
</div>
""", unsafe_allow_html=True)
