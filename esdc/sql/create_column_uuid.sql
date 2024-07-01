-- add new column to store project_uuid
ALTER TABLE {table_name}
ADD COLUMN uuid TEXT;

-- generate UUID v4 in {table_name} table. UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
UPDATE {table_name}
SET uuid = (
    lower(hex(randomblob(4))) || '-' ||
    lower(hex(randomblob(2))) || '-' ||
    '4' || substr(hex(randomblob(2)), 2) || '-' ||
    substr('89ab', 1 + (abs(random()) % 4), 1) || substr(hex(randomblob(2)), 2) || '-' ||
    lower(hex(randomblob(6)))
);