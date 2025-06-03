import os
import cv2
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import psycopg2
from datetime import datetime
import mlflow
import mlflow.pyfunc
from ultralytics import YOLO
import torch

app = FastAPI()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "yolo_db"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "secret")
    )

# Load YOLO model
def load_yolo_model():
    try:
        # Utiliser ultralytics YOLO v8
        model = YOLO('yolov8n.pt')  # Télécharge automatiquement si pas présent
        print("YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise

model = load_yolo_model()

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Vérifier le type de fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Lire le fichier image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Faire la détection avec YOLO v8
        results = model(img)
        
        # Extraire les détections - format YOLO v8
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Coordonnées de la boîte
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Confiance
                    confidence = float(box.conf[0].cpu().numpy())
                    # Classe
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detection = {
                        'xmin': float(x1),
                        'ymin': float(y1),
                        'xmax': float(x2),
                        'ymax': float(y2),
                        'confidence': confidence,
                        'class': class_id,
                        'name': class_name
                    }
                    detections.append(detection)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("model", "yolov8n")
            mlflow.log_param("image_name", file.filename)
            mlflow.log_metric("num_detections", len(detections))
            mlflow.log_metric("image_width", img.shape[1])
            mlflow.log_metric("image_height", img.shape[0])
            
            # Log des métriques de détection
            if detections:
                avg_confidence = sum(det['confidence'] for det in detections) / len(detections)
                mlflow.log_metric("avg_confidence", avg_confidence)
                max_confidence = max(det['confidence'] for det in detections)
                mlflow.log_metric("max_confidence", max_confidence)
        
        # Sauvegarder en base de données
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Convertir les détections en JSON pour PostgreSQL
            detection_json = json.dumps({'detections': detections})
            
            cur.execute(
                "INSERT INTO detection_results (image_name, detection_data, created_at) VALUES (%s, %s, %s)",
                (file.filename, detection_json, datetime.now())
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Continue même si la DB échoue
        
        return {
            "filename": file.filename,
            "detections": detections,
            "num_detections": len(detections),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Error during detection: {str(e)}")

@app.get("/results/")
async def get_results():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT image_name, detection_data, created_at FROM detection_results ORDER BY created_at DESC LIMIT 10"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        results = []
        for row in rows:
            try:
                # Parser le JSON des détections
                detection_data = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                results.append({
                    "image_name": row[0],
                    "detections": detection_data,
                    "created_at": row[2].isoformat() if row[2] else None
                })
            except json.JSONDecodeError:
                # Si le JSON est mal formé, inclure quand même le résultat
                results.append({
                    "image_name": row[0],
                    "detections": {"error": "Invalid JSON data"},
                    "created_at": row[2].isoformat() if row[2] else None
                })
        
        return results
        
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "yolo_version": "ultralytics-yolov8"
    }

@app.get("/")
async def root():
    return {
        "message": "YOLO Object Detection API",
        "version": "1.0",
        "endpoints": [
            "/detect/ - POST: Upload image for detection",
            "/results/ - GET: Get recent detection results",
            "/health/ - GET: Health check"
        ]
    }