# FleetIQ Freight Pricing

**Version:** 1
**Run:** 792c772cfb53478fab3cd443fcd5e50e
**Trained:** 2026-10-31T03:21:42.871600+00:00
**Training rows:** 1371
**Reference year:** 2025

## Intended use

Produce a booking-time freight quote and a calibrated 90% cost interval for a lane.

## Metrics

| Metric | Value |
| --- | --- |
| baseline_mae | 30.0212 |
| baseline_mape | 16.6360 |
| coverage_90 | 0.9014 |
| mae | 13.1478 |
| mape | 7.4173 |
| medae | 10.6168 |
| r2 | 0.9616 |
| rmse | 18.0121 |

## Inputs

weight_kg, distance_miles, month, day_of_week, day_of_year, is_weekend, weight_per_mile, origin_warehouse, destination, carrier, lane, distance_band, weight_band

## Excluded columns

shipment_id, delivery_date, status, transit_days, cost

## Limitations

Freight cost is heavy-tailed. The point model optimises absolute error, so R2 is low while MAE and MAPE are the useful operational metrics. The 90% interval is a split-conformal estimate and assumes future shipments resemble the 2023 US lane mix.
