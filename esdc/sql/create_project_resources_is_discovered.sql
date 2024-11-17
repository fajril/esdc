ALTER TABLE project_resources
ADD COLUMN is_discovered INT;

UPDATE project_resources
SET is_discovered = 
    CASE
        WHEN SUBSTRING(project_class, 1, 1) = '1' THEN 1
        WHEN SUBSTRING(project_class, 1, 1) = '2' THEN 1
        WHEN SUBSTRING(project_class, 1, 1) = '3' THEN 0
    END;