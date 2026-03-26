FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and API
COPY xgboost_emission_model.pkl .
COPY api.py .

EXPOSE 8002

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]
