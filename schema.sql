-- ====================================================================
-- STAGE 1: Supabase Database Schema Setup
-- Project: Cloud-Based Air Quality Monitoring System with Automated Alert
-- ====================================================================

-- 1. Create table for sensor readings
CREATE TABLE IF NOT EXISTS public.sensor_data (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    aqi_ppm DOUBLE PRECISION NOT NULL,
    temperature DOUBLE PRECISION NOT NULL,
    humidity DOUBLE PRECISION NOT NULL,
    buzzer_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_score DOUBLE PRECISION DEFAULT NULL
);

-- 2. Add performance indexes for rapid querying by timestamp and anomaly status
CREATE INDEX IF NOT EXISTS idx_sensor_data_created_at ON public.sensor_data (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_data_is_anomaly ON public.sensor_data (is_anomaly);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE public.sensor_data ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies
-- Policy 1: Allow anonymous INSERT (used by ESP32 microcontrollers)
DROP POLICY IF EXISTS "Allow anon insert" ON public.sensor_data;
CREATE POLICY "Allow anon insert"
    ON public.sensor_data
    FOR INSERT
    TO anon
    WITH CHECK (true);

-- Policy 2: Allow anonymous SELECT (used by Web Dashboard for live feeds)
DROP POLICY IF EXISTS "Allow anon select" ON public.sensor_data;
CREATE POLICY "Allow anon select"
    ON public.sensor_data
    FOR SELECT
    TO anon
    USING (true);

DROP POLICY IF EXISTS "Allow anon update" ON public.sensor_data;

-- 5. Enable Realtime subscriptions for live dashboard streaming
ALTER PUBLICATION supabase_realtime ADD TABLE public.sensor_data;
