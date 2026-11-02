CREATE INDEX IF NOT EXISTS ix_pod_monitoring_api_reported_date
    ON pod_monitoring_api (reported_date, pod_id_seq);

CREATE INDEX IF NOT EXISTS ix_pod_forecast_api_reported_date
    ON pod_forecast_api (reported_date, pod_id_seq, year_frcast);
