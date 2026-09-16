-- Run against the Clio database. Read-only snapshot. Full row scans, run off-peak on large databases.
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '60s';
SELECT now() AS measured_at, pg_database_size(current_database()) AS database_bytes;
SELECT relname, pg_table_size(relid) AS table_bytes,
       pg_indexes_size(relid) AS index_bytes, pg_total_relation_size(relid) AS total_bytes,
       n_live_tup, n_dead_tup, last_autovacuum
FROM pg_stat_user_tables
WHERE schemaname = 'clio' AND relname IN ('solar_wind_observation', 'geomagnetic_observation', 'observation_source_status');
SELECT kind, spacecraft, count(*) AS rows, count(*) FILTER (WHERE active) AS active_rows,
       min(observed_at) AS first_measurement, max(observed_at) AS last_measurement,
       round(avg(pg_column_size(s))) AS mean_row_bytes,
       round(avg(pg_column_size(raw))) AS mean_raw_bytes,
       round(avg(pg_column_size(s."values"))) AS mean_values_bytes
FROM clio.solar_wind_observation s GROUP BY kind, spacecraft ORDER BY kind, spacecraft;
SELECT metric, count(*) AS rows, min(interval_start) AS first_measurement,
       max(interval_start) AS last_measurement, round(avg(pg_column_size(g))) AS mean_row_bytes
FROM clio.geomagnetic_observation g GROUP BY metric ORDER BY metric;
SELECT date_trunc('day', observed_at AT TIME ZONE 'UTC') AS day_utc, kind,
       count(*) AS rows, count(DISTINCT spacecraft) AS spacecraft,
       count(DISTINCT observed_at) AS minute_timestamps,
       count(DISTINCT observed_at) FILTER (WHERE active) AS active_minute_timestamps
FROM clio.solar_wind_observation
GROUP BY 1, 2 ORDER BY 1, 2;
COMMIT;
