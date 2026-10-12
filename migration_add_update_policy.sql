-- ====================================================================
-- MIGRATION: Column-Restricted Update Policy for sensor_data
-- Project: Air Quality AI Monitor
-- Security Warning: A broad `UPDATE FOR TO anon USING (true)` allows
-- any client with the publishable key to overwrite physical sensor telemetry
-- (aqi_ppm, temp, humidity).
-- Below is the column-restricted approach to grant update privileges ONLY
-- on the prediction columns (is_anomaly, anomaly_score).
-- ====================================================================

-- 1. Ensure RLS is enabled
ALTER TABLE public.sensor_data ENABLE ROW LEVEL SECURITY;

-- 2. Restrict column-level UPDATE permissions to only ML fields for anon role
REVOKE UPDATE ON public.sensor_data FROM anon;
GRANT UPDATE (is_anomaly, anomaly_score) ON public.sensor_data TO anon;

-- 3. Create RLS policy for updating prediction columns
DROP POLICY IF EXISTS "Allow anon update predictions on sensor_data" ON public.sensor_data;
CREATE POLICY "Allow anon update predictions on sensor_data"
    ON public.sensor_data
    FOR UPDATE
    TO anon
    USING (true)
    WITH CHECK (true);
