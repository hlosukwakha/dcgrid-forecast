# Data sources

This project is deliberately built on public datasets.

## Ireland data-centre consumption
- CSO release: *Data Centres Metered Electricity Consumption 2023* (quarterly GWh series 2015–2023), used to compute a quarterly **data-centre share** and to scale system demand into a DC load proxy.

## Ireland system time series
- EirGrid *System Data Qtr Hourly* spreadsheet (quarter-hourly system demand + renewable generation). Converted to hourly features.

## European day-ahead prices (optional)
- Ember *European Wholesale Electricity Price Data* (hourly day-ahead prices by country). If supplied, the pipeline computes:
  - price spikes (rolling 30d 99th percentile)
  - negative price frequency (rolling 7d)

If you want to generalise beyond Ireland, add ENTSO-E Transparency Platform ingestion (requires an API token).
