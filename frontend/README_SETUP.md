# Setup Instructions

## Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
# Windows
start.bat

# Linux/Mac
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Verify Setup

1. Check API health: http://localhost:8000/api/health
2. Check frontend: http://localhost:3000

## Features

### Live Prediction
- Select accident parameters (lighting, location, intersection, etc.)
- Get ensemble prediction from multiple models
- View individual model predictions
- See confidence and model agreement metrics

### Model Explainability (XAI)
- View global feature importance
- Calculate SHAP values for specific predictions
- Understand which features drive predictions
- Interactive SHAP force plots

### Forecasting
- Predict future accident counts (1-30 days)
- LSTM-based time-series forecasting
- View daily predictions and trends
- Analyze peak days and totals

### Overview Dashboard
- System metrics and status
- Model performance trends
- Quick actions
- Real-time health monitoring

## Troubleshooting

### Backend Issues
- Make sure all models are trained and in the `models/` directory
- Check that `data/model_ready.csv` and `data/cleaned_accidents.csv` exist
- Verify Python dependencies are installed

### Frontend Issues
- Clear browser cache
- Check console for errors
- Verify backend is running on port 8000

### CORS Issues
- Backend is configured to allow localhost:3000 and localhost:5173
- If using different ports, update CORS settings in `backend/main.py`
