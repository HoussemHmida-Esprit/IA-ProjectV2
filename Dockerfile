# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy supporting Python modules
COPY src/ ./src/
COPY utils/ ./utils/
COPY pages/ ./pages/

# Copy models directory structure
COPY models/*.py ./models/
COPY models/__init__.py ./models/

# Copy only small essential model files that exist in Git
COPY models/xgb_nopca_multitarget.pkl ./models/
COPY models/tab_transformer_best.pth ./models/
COPY models/lstm_forecaster.pth ./models/
COPY models/collision_labels.pkl ./models/

# Copy data directory with only model_ready.csv
RUN mkdir -p ./data
COPY data/model_ready.csv ./data/

# Expose port
EXPOSE 8000

# Start command
CMD cd backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
