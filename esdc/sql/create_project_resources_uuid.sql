-- add new column to store project_uuid
ALTER TABLE project_resources
ADD COLUMN IF NOT EXISTS project_uuid TEXT;

-- generate UUID v4 in project_resources table. UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
UPDATE project_resources
SET project_uuid = gen_random_uuid()::TEXT;