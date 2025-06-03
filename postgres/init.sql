-- Créer la base de données et la table
CREATE DATABASE yolo_db;

-- Se connecter à yolo_db puis exécuter :
CREATE TABLE IF NOT EXISTS detection_results (
    id SERIAL PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    detection_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour améliorer les performances
CREATE INDEX idx_detection_results_created_at ON detection_results(created_at DESC);
CREATE INDEX idx_detection_results_image_name ON detection_results(image_name);