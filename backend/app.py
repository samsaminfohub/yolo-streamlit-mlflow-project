import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
import psycopg2
from datetime import datetime
import mlflow
import mlflow.pyfunc
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
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_yolo_model()

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run detection
    results = model(img)
    detections = results.pandas().xyxy[0].to_dict(orient='records')
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model", "yolov5s")
        mlflow.log_metric("num_detections", len(detections))
        for i, det in enumerate(detections):
            mlflow.log_metric(f"detection_{i}_confidence", det['confidence'])
    
    # Save to database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detection_results (image_name, detection_data) VALUES (%s, %s)",
        (file.filename, {'detections': detections})
    )
    conn.commit()
    cur.close()
    conn.close()
    
    return {"filename": file.filename, "detections": detections}

@app.get("/results/")
async def get_results():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT image_name, detection_data, created_at FROM detection_results ORDER BY created_at DESC LIMIT 10")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    return [{"image_name": row[0], "detections": row[1], "created_at": row[2]} for row in rows]