FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files and model artefacts
COPY FloodPrediction.py .
COPY routerecom.py .
COPY api.py .
COPY kerala_flood_labels_lon_lat.csv .
COPY flood_rf_model.pkl .

EXPOSE 8003

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8003"]
