-- add new column to store project_uuid
ALTER TABLE project_resources
ADD COLUMN project_uuid TEXT;

-- generate UUID v4 in project_resources table. UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
UPDATE project_resources
SET project_uuid = (
  lower(hex(randomblob(4))) || '-' ||
  lower(hex(randomblob(2))) || '-' ||
  '4' || substr(hex(randomblob(2)), 2) || '-' ||
  substr('89ab', 1 + (abs(random()) % 4), 1) || substr(hex(randomblob(2)), 2) || '-' ||
  lower(hex(randomblob(6)))
);