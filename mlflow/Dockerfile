FROM python:3.8-slim

WORKDIR /app

RUN pip install mlflow psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres/${POSTGRES_DB}", \
     "--default-artifact-root", "s3://mlflow/", \
     "--host", "0.0.0.0"]