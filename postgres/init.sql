CREATE TABLE IF NOT EXISTS detection_results (
    id SERIAL PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    detection_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_image_name ON detection_results (image_name);