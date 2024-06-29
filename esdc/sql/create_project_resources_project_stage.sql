ALTER TABLE project_resources
ADD COLUMN project_stage VARCHAR(15);

UPDATE project_resources
SET project_stage = 
    CASE
        WHEN SUBSTRING(project_level, 1, 1) = 'E' THEN '1. Exploitation'
        WHEN SUBSTRING(project_level, 1, 1) = 'X' THEN '2. Exploration'
        WHEN SUBSTRING(project_level, 1, 1) = 'A' THEN '3. Abandoned'
    END;