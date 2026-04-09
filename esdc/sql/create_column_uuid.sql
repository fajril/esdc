-- add new column to store project_uuid
ALTER TABLE {table_name}
ADD COLUMN IF NOT EXISTS uuid TEXT;

-- generate UUID v4 in {table_name} table. UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
UPDATE {table_name}
SET uuid = gen_random_uuid()::TEXT;