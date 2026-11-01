# FleetIQ Freight Pricing

**Version:** None
**Run:** local-20261031T031914Z-8af9a7
**Trained:** 2026-10-31T03:19:14.513882+00:00
**Training rows:** 170
**Reference year:** 2025

## Intended use

Produce a booking-time freight quote and a calibrated 90% cost interval for a lane.

## Metrics

| Metric | Value |
| --- | --- |
| baseline_mae | 38.0747 |
| baseline_mape | 20.0521 |
| coverage_90 | 0.7838 |
| mae | 25.6940 |
| mape | 12.9935 |
| medae | 18.1346 |
| r2 | 0.8702 |
| rmse | 37.5529 |

## Inputs

weight_kg, distance_miles, month, day_of_week, day_of_year, is_weekend, weight_per_mile, origin_warehouse, destination, carrier, lane, distance_band, weight_band

## Excluded columns

shipment_id, delivery_date, status, transit_days, cost

## Limitations

Freight cost is heavy-tailed. The point model optimises absolute error, so R2 is low while MAE and MAPE are the useful operational metrics. The 90% interval is a split-conformal estimate and assumes future shipments resemble the 2023 US lane mix.
