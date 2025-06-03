FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    pip install --upgrade pip
      
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Install YOLOv5 dependencies
RUN apt-get update && apt-get install -y libgl1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]