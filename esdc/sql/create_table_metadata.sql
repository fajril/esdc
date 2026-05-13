CREATE TABLE IF NOT EXISTS _metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR
);

INSERT OR REPLACE INTO _metadata (key, value)
VALUES ('last_updated', CURRENT_TIMESTAMP::VARCHAR);