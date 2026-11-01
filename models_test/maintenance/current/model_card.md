# FleetIQ Maintenance Risk

**Version:** None
**Run:** local-20261031T031911Z-f724af
**Trained:** 2026-10-31T03:19:11.560260+00:00
**Training rows:** 3517
**Reference year:** 2025

## Intended use

Rank vehicles for scheduled maintenance using telemetry and the last known service state.

## Metrics

| Metric | Value |
| --- | --- |
| accuracy | 1.0000 |
| brier | 0.0000 |
| f1 | 1.0000 |
| log_loss | 0.0001 |
| pr_auc | 1.0000 |
| precision | 1.0000 |
| recall | 1.0000 |
| roc_auc | 1.0000 |

## Inputs

usage_hours, load_capacity, actual_load, engine_temperature, tire_pressure, fuel_consumption, battery_status, vibration_levels, oil_quality, failure_history, anomalies_detected, diagnostic_trouble_code_count, can_message_rate_hz, sensor_packet_loss_rate, historical_maintenance_cost, vehicle_age, usage_per_year, load_ratio, temp_delta, vibration_per_1k_hours, dtc_per_10k_hours, packet_loss_pct, maintenance_cost_per_hour, brake_wear_flag, failure_flag, anomaly_flag, make_and_model, vehicle_type, route_info, weather_conditions, road_conditions, communication_interface, telematics_gateway, edge_device_class, maintenance_type, brake_condition

## Excluded columns

ads, cmes, data_split, days_since_last_maintenance, downtime_maintenance, impact_on_efficiency, last_maintenance_date, maintenance_cost, maintenance_severity, maintenance_severity_id, mbf, ohi, pcr, pre_event_record, predictive_score, tpi, uer, uir

## Limitations

The public dataset is synthetic and encodes maintenance need as a deterministic function of service age and telemetry. Discrimination is therefore near-perfect. Days since last maintenance, severity, cost and post-service fields are excluded to avoid direct leakage. Re-validate on real fleet data before operational use.
