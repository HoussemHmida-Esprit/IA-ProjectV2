# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only essential files
COPY backend/ ./backend/
COPY src/ ./src/
COPY utils/ ./utils/
COPY pages/ ./pages/

# Copy only essential model files (not Random Forest)
COPY models/*.py ./models/
COPY models/__init__.py ./models/
COPY models/xgb_nopca_multitarget.pkl ./models/
COPY models/tab_transformer_best.pth ./models/
COPY models/lstm_forecaster.pth ./models/
COPY models/stacking_ensemble.pkl ./models/ 2>/dev/null || true
COPY models/ensemble_info.pkl ./models/ 2>/dev/null || true
COPY models/collision_labels.pkl ./models/ 2>/dev/null || true

# Copy only model_ready.csv for SHAP (skip other large CSVs)
RUN mkdir -p ./data
COPY data/model_ready.csv ./data/ 2>/dev/null || true
COPY data/.gitkeep ./data/ 2>/dev/null || true

# Expose port (Railway will set PORT env variable)
EXPOSE 8000

# Start command
CMD cd backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
