# I-66 ITB Congestion Forecast Explorer

An interactive Streamlit app showcasing the Phase 1 predictability ceiling study on the I-66 Inside the Beltway (ITB) corridor in Northern Virginia.

**[Live App →]** (https://i66-congestion-explorer.streamlit.app/) 
---

## What This App Shows

Select a **direction**, **hour of day**, **day of week**, and **forecast horizon** to explore:

- **Live TTI prediction** using XGBoost-derived estimates from the endogenous baseline model
- **Congestion state classification** (Free Flow / Moderate / Congested / Severe)
- **Toll window indicator** based on I-66 ITB's asymmetric pricing structure
- **24-hour TTI profiles** by direction from 4 years of probe data
- **Model evaluation results** — XGBoost vs persistence baseline across 5, 15, and 30-minute horizons
- **Segment volatility ranking** showing which TMCs are inherently hardest to forecast

---

## Key Research Finding

At the **5-minute horizon**, a simple persistence baseline achieves MAE = 0.070, which XGBoost barely improves upon (MAE = 0.091). At **30 minutes**, XGBoost (MAE = 0.143) meaningfully outperforms persistence (MAE = 0.149). This establishes the **predictability ceiling** for endogenous TTI signals before adding exogenous inputs like tolls, weather, or events.

---

## Run Locally

```bash
git clone https://github.com/ananta-mimo/i66-congestion-explorer.git
cd i66-congestion-explorer
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
i66-congestion-explorer/
├── app.py                  # Main Streamlit application
├── data/
│   ├── tti_by_hour_direction.csv     # Mean TTI per hour and direction
│   ├── tti_by_day_of_week.csv        # Mean TTI per day of week
│   ├── volatility_by_tmc.csv         # 30-min volatility per TMC segment
│   ├── overall_model_results.csv     # Aggregate model performance
│   └── per_tmc_results.csv           # Per-segment model performance
├── requirements.txt
└── README.md
```

---

## Data

All data is derived from the research analysis and contains no raw probe records.

| File | Description |
|---|---|
| `tti_by_hour_direction.csv` | Mean TTI aggregated by hour and direction across 2022–2025 |
| `tti_by_day_of_week.csv` | Mean TTI by day of week |
| `volatility_by_tmc.csv` | 30-min TTI volatility per TMC segment |
| `overall_model_results.csv` | Aggregate test MAE and RMSE for all models and horizons |
| `per_tmc_results.csv` | Per-segment validation and test metrics |

Raw RITIS/INRIX probe data is not included (institutional license required).

---

## Related

- [Phase 1 Research Repo](https://github.com/ananta-mimo/I66-congestion-predictability) — Full pipeline, notebooks, and model outputs
- Paper under review at *Transportation Research Part C: Emerging Technologies*

---

## Tech Stack

`Streamlit` · `Pandas` · `NumPy` · `Matplotlib`
