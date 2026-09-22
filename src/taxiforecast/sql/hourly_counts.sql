-- Hourly pickups per borough. Only trips with a real duration (0-180 minutes)
-- count; payment_type is not filtered, because every record here is a trip
-- that happened and this is a demand count, not a fare analysis.
SELECT
    date_trunc('hour', r.tpep_pickup_datetime) AS hour,
    z.Borough                                  AS borough,
    count(*)                                   AS trips
FROM raw_trips AS r
JOIN raw_zones AS z ON z.LocationID = r.PULocationID
WHERE r.tpep_pickup_datetime >= getvariable('window_start')
  AND r.tpep_pickup_datetime <  getvariable('window_end')
  AND r.tpep_dropoff_datetime >  r.tpep_pickup_datetime
  AND date_diff('minute', r.tpep_pickup_datetime, r.tpep_dropoff_datetime) <= 180
  AND z.Borough IN ('Manhattan', 'Queens', 'Brooklyn', 'Bronx')
GROUP BY ALL
ORDER BY hour, borough;
