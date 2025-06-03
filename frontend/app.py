import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Send to backend for detection
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{BACKEND_URL}/detect/", files=files)
    
    if response.status_code == 200:
        result = response.json()
        detections = result["detections"]
        
        # Display detections
        st.subheader("Detection Results")
        
        # Create figure and axes
        fig, ax = plt.subplots()
        ax.imshow(image)
        
        # Draw bounding boxes
        for det in detections:
            if det['confidence'] > 0.5:  # Only show confident detections
                box = patches.Rectangle(
                    (det['xmin'], det['ymin']),
                    det['xmax'] - det['xmin'],
                    det['ymax'] - det['ymin'],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(box)
                plt.text(
                    det['xmin'],
                    det['ymin'],
                    f"{det['name']} {det['confidence']:.2f}",
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5)
                )
        
        st.pyplot(fig)
        
        # Display detection data
        st.table([{
            "Class": det["name"],
            "Confidence": f"{det['confidence']:.2f}",
            "Location": f"({det['xmin']:.0f}, {det['ymin']:.0f}) to ({det['xmax']:.0f}, {det['ymax']:.0f})"
        } for det in detections if det['confidence'] > 0.5])
    else:
        st.error("Error in object detection")

# Show recent results
st.sidebar.header("Recent Detections")
recent_results = requests.get(f"{BACKEND_URL}/results/").json()
for result in recent_results:
    st.sidebar.text(f"{result['image_name']}\n{result['created_at']}")