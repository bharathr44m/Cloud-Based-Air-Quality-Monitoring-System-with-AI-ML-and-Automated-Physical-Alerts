-- ====================================================================
-- STAGE 2 MIGRATION: Least-Privilege ML Predictions Table (Option A)
-- Project: Air Quality AI Monitor
-- Description: Creates an isolated predictions table linked to sensor_data
--              by foreign key. Sensor data remains read-only.
-- Security:
--   - SELECT: Granted to `anon` (Web Dashboard can read predictions).
--   - INSERT/UPDATE: Restricted to `service_role` (Server backend only).
--     Anonymous web clients cannot insert or modify prediction scores.
-- ====================================================================

-- 1. Create dedicated ml_predictions table
CREATE TABLE IF NOT EXISTS public.ml_predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sensor_data_id BIGINT NOT NULL REFERENCES public.sensor_data(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_anomaly BOOLEAN NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'isolation_forest_v1',
    CONSTRAINT unique_sensor_prediction UNIQUE (sensor_data_id)
);

-- 2. Performance indexes
CREATE INDEX IF NOT EXISTS idx_ml_predictions_sensor_id ON public.ml_predictions (sensor_data_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_created_at ON public.ml_predictions (created_at DESC);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE public.ml_predictions ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies
-- Policy 1: Allow anonymous SELECT (Web Dashboard can read & render predictions)
DROP POLICY IF EXISTS "Allow anon select predictions" ON public.ml_predictions;
CREATE POLICY "Allow anon select predictions"
    ON public.ml_predictions
    FOR SELECT
    TO anon
    USING (true);

-- Policy 2: Allow service_role INSERT (Server-side Python backend write only)
DROP POLICY IF EXISTS "Allow service_role insert predictions" ON public.ml_predictions;
CREATE POLICY "Allow service_role insert predictions"
    ON public.ml_predictions
    FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Policy 3: Allow service_role UPDATE (Server-side Python backend update only)
DROP POLICY IF EXISTS "Allow service_role update predictions" ON public.ml_predictions;
CREATE POLICY "Allow service_role update predictions"
    ON public.ml_predictions
    FOR UPDATE
    TO service_role
    USING (true)
    WITH CHECK (true);

-- 5. Safe idempotent Realtime publication addition
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
          AND schemaname = 'public'
          AND tablename = 'ml_predictions'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE public.ml_predictions;
    END IF;
END $$;
