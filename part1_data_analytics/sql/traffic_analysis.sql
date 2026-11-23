-- ===========================================================================
-- Part 1 Task 1: SQL-based traffic analysis (SQLite)
--
-- All analysis queries read traffic_clean (one row per hour, D1/D2).
-- Each query starts with a "-- name:" line so run_sql_analysis.py can run it
-- and save its result to part1_data_analytics/results/<name>.csv.
-- The file can also be pasted into any SQLite tool (e.g. DB Browser for SQLite).
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- Task 1.1  Verify the load
-- ---------------------------------------------------------------------------

-- name: 1_1_load_verification
SELECT 'traffic_raw'    AS table_name, COUNT(*) AS n_rows, COUNT(DISTINCT date_time) AS n_distinct_hours,
       MIN(date_time) AS first_hour, MAX(date_time) AS last_hour, SUM(traffic_volume) AS total_volume
FROM traffic_raw
UNION ALL
SELECT 'traffic_hourly', COUNT(*), COUNT(DISTINCT date_time), MIN(date_time), MAX(date_time), SUM(traffic_volume)
FROM traffic_hourly
UNION ALL
SELECT 'traffic_clean',  COUNT(*), COUNT(DISTINCT date_time), MIN(date_time), MAX(date_time), SUM(traffic_volume)
FROM traffic_clean;

-- name: 1_1_rows_per_month
-- Coverage check: months with no rows at all are the gap (Sep 2014 - May 2015).
SELECT strftime('%Y-%m', date_time) AS year_month,
       COUNT(*)                     AS n_hours
FROM traffic_clean
GROUP BY year_month
ORDER BY year_month;


-- ---------------------------------------------------------------------------
-- Task 1.2  Annual traffic trends, 2012-2017
-- Totals depend on how many hours were recorded, so hours and the hourly
-- average are reported next to the total.
-- ---------------------------------------------------------------------------

-- name: 1_2_annual_traffic
WITH yearly AS (
    SELECT CAST(strftime('%Y', date_time) AS INTEGER) AS year,
           SUM(traffic_volume)                        AS total_volume,
           COUNT(*)                                   AS hours_recorded,
           ROUND(AVG(traffic_volume), 1)              AS avg_per_hour
    FROM traffic_clean
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
)
SELECT year,
       total_volume,
       total_volume - LAG(total_volume) OVER (ORDER BY year)                         AS change_total,
       ROUND(100.0 * (total_volume - LAG(total_volume) OVER (ORDER BY year))
             / LAG(total_volume) OVER (ORDER BY year), 1)                            AS pct_change_total,
       CASE WHEN total_volume > LAG(total_volume) OVER (ORDER BY year) THEN 'Increase'
            WHEN total_volume < LAG(total_volume) OVER (ORDER BY year) THEN 'Decrease'
       END                                                                           AS direction,
       hours_recorded,
       ROUND(100.0 * hours_recorded / (CASE WHEN year % 4 = 0 THEN 8784 ELSE 8760 END), 1) AS pct_hours_covered,
       avg_per_hour,
       ROUND(avg_per_hour - LAG(avg_per_hour) OVER (ORDER BY year), 1)               AS change_avg_per_hour,
       ROUND(100.0 * (avg_per_hour - LAG(avg_per_hour) OVER (ORDER BY year))
             / LAG(avg_per_hour) OVER (ORDER BY year), 1)                            AS pct_change_avg
FROM yearly
ORDER BY year;

-- name: 1_2_raw_vs_hourly_totals
-- Why the hourly table matters: summing the raw table double-counts repeated hours.
SELECT r.year,
       r.raw_total,
       c.clean_total,
       r.raw_total - c.clean_total                                  AS overstated_by,
       ROUND(100.0 * (r.raw_total - c.clean_total) / c.clean_total, 1) AS pct_overstated
FROM (SELECT strftime('%Y', date_time) AS year, SUM(traffic_volume) AS raw_total
      FROM traffic_raw GROUP BY year) r
JOIN (SELECT strftime('%Y', date_time) AS year, SUM(traffic_volume) AS clean_total
      FROM traffic_clean GROUP BY year) c USING (year)
WHERE r.year BETWEEN '2012' AND '2017'
ORDER BY r.year;


-- ---------------------------------------------------------------------------
-- Task 1.3  Temperature around holidays (New Year's Day, Labor Day)
-- 2015 New Year's Day falls in the data gap, so 2014 and 2018 are shown as
-- context. Holiday labels mark the OBSERVED federal holiday (2017-01-02).
-- ---------------------------------------------------------------------------

-- name: 1_3_holiday_temperature
SELECT holiday,
       DATE(date_time)                              AS holiday_date,
       CAST(strftime('%Y', date_time) AS INTEGER)   AS year,
       COUNT(*)                                     AS hours_recorded,
       ROUND(AVG(temp) - 273.15, 1)                 AS avg_temp_c,
       ROUND(MIN(temp) - 273.15, 1)                 AS min_temp_c,
       ROUND(MAX(temp) - 273.15, 1)                 AS max_temp_c,
       ROUND(AVG(traffic_volume), 0)                AS avg_traffic,
       GROUP_CONCAT(DISTINCT weather_main)          AS weather_seen
FROM traffic_clean
WHERE holiday IN ('New Years Day', 'Labor Day')
  AND strftime('%Y', date_time) BETWEEN '2014' AND '2018'
GROUP BY holiday, holiday_date
ORDER BY holiday DESC, holiday_date;

-- name: 1_3_holiday_year_on_year
-- Year-on-year change in average holiday temperature and traffic (2015-2017 focus).
WITH h AS (
    SELECT holiday,
           CAST(strftime('%Y', date_time) AS INTEGER) AS year,
           AVG(temp) - 273.15                         AS avg_temp_c,
           AVG(traffic_volume)                        AS avg_traffic
    FROM traffic_clean
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND strftime('%Y', date_time) BETWEEN '2014' AND '2018'
    GROUP BY holiday, year
)
SELECT holiday, year,
       ROUND(avg_temp_c, 1)                                                          AS avg_temp_c,
       ROUND(avg_temp_c - LAG(avg_temp_c) OVER (PARTITION BY holiday ORDER BY year), 1) AS temp_change_c,
       ROUND(avg_traffic, 0)                                                         AS avg_traffic,
       ROUND(avg_traffic - LAG(avg_traffic) OVER (PARTITION BY holiday ORDER BY year), 0) AS traffic_change
FROM h
ORDER BY holiday DESC, year;

-- name: 1_3_holiday_vs_normal_days
-- Context: are holidays quieter than ordinary days in the same month?
SELECT CASE WHEN is_holiday = 1 THEN holiday ELSE 'Ordinary day (same month)' END AS day_type,
       strftime('%m', date_time)       AS month,
       COUNT(*)                        AS hours,
       ROUND(AVG(traffic_volume), 0)   AS avg_traffic,
       ROUND(AVG(temp) - 273.15, 1)    AS avg_temp_c
FROM traffic_clean
WHERE strftime('%m', date_time) IN ('01', '09')
  AND (holiday IN ('New Years Day', 'Labor Day') OR is_holiday = 0)
GROUP BY day_type, month
ORDER BY month, day_type;
