# Architecture

## High-level
- **Ingestion** pulls:
  - EirGrid "System Data Qtr Hourly" (demand + wind + solar) -> hourly features
  - CSO quarterly data-centre electricity consumption -> per-quarter DC share
  - (Optional) Ember hourly day-ahead prices -> price spikes, negative price frequency
- **Feature builder** creates a single hourly feature table in Postgres (`dcgrid.features_hourly`)
- **Training** fits:
  - baseline **XGBoost** (default)
  - optional deep learning **Temporal Fusion Transformer (TFT)** (PyTorch Forecasting)
  - optional daily **Prophet**
- **Serving** exposes a FastAPI endpoint `/forecast` + `/stress/alerts`
- **Dashboard** Streamlit charts for demand, DC load proxy, and stress score
- **Storage** MinIO (S3 API) stores trained models (`models/<REGION>/<MODEL>.pkl`) and MLflow artifacts

## Grid stress score (transparent composite)
`stress_score` is a weighted sum of rolling z-scores:
- price level (proxy scarcity)
- negative price frequency (proxy oversupply / curtailment stress)
- renewable variability (proxy balancing difficulty)
- ramp rate magnitude (proxy operational stress)

This is *not* a physics-based grid security metric; it’s an early-warning **market + variability + ramp** indicator that tends to rise before sustained planning signals appear.
