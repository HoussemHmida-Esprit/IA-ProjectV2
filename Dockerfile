# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU-only version first (smaller than GPU version)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy supporting Python modules  
COPY src/ ./src/
COPY utils/ ./utils/
COPY pages/ ./pages/

# Copy models directory
COPY models/*.py ./models/
COPY models/__init__.py ./models/

# Copy model files
COPY models/xgb_nopca_multitarget.pkl ./models/
COPY models/tab_transformer_best.pth ./models/
COPY models/lstm_forecaster.pth ./models/
COPY models/collision_labels.pkl ./models/

# Copy data
RUN mkdir -p ./data
COPY data/model_ready.csv ./data/

# Change working directory to backend
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Start command - use exec form with sh to handle PORT variable
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
