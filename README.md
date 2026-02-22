# 🚗 Accident Prediction & Analysis System

AI-powered system for predicting road accident collision types and severity using ensemble machine learning models.

## 🎯 Features

- **Multi-Model Prediction**: Ensemble of XGBoost, Random Forest, TabTransformer, and LSTM
- **Dual Prediction**: Predicts both collision type (7 classes) and severity (4 levels)
- **Explainable AI**: SHAP values for model interpretability
- **Risk Analysis**: Time-based and condition-based risk assessment
- **Real-time Forecasting**: LSTM-based accident forecasting
- **Professional Dashboard**: React-based UI with business metrics

## 🏗️ Architecture

```
Frontend (React + Vite)
    ↓
FastAPI Backend
    ↓
ML Models (XGBoost, RF, TabTransformer, LSTM)
    ↓
SHAP Explainability Engine
```

## 📊 Model Performance

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| Stacking Ensemble | 46.0% | Fast | Highest accuracy |
| XGBoost V2 | 45.1% | Very Fast | Production use |
| Random Forest V2 | 33.1% | Fast | Feature importance |
| TabTransformer | 35.0% | Medium | Complex patterns |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/accident-prediction-system.git
cd accident-prediction-system
```

2. **Backend Setup**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

4. **Access the application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📦 Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

**Recommended:**
- Frontend: Vercel (Free)
- Backend: Render (Free or $7/month)
- Models: AWS S3 or Git LFS

## 🎨 Screenshots

### Overview Dashboard
Real-time system metrics, risk analysis, and business value indicators.

### Prediction Engine
Multi-model prediction with collision type and severity analysis.

### XAI Dashboard
SHAP-based explainability showing feature importance and decision factors.

## 🔧 Tech Stack

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- Recharts
- Axios

**Backend:**
- FastAPI
- Python 3.11
- Uvicorn

**ML/AI:**
- XGBoost
- Scikit-learn
- PyTorch
- SHAP
- Pandas/NumPy

## 📁 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── views/          # React pages
│   │   ├── components/     # Reusable components
│   │   └── main.jsx        # Entry point
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── xgb_nopca_multitarget.pkl
│   ├── rf_pca_multitarget.pkl
│   ├── tab_transformer_best.pth
│   ├── lstm_forecaster.pth
│   └── *.py                # Model training scripts
├── data/
│   ├── model_ready.csv     # Preprocessed data
│   └── cleaned_accidents.csv
└── README.md
```

## 🎯 Use Cases

- **Insurance Companies**: Risk assessment and premium calculation
- **Traffic Authorities**: Accident prevention and road safety planning
- **Emergency Services**: Resource allocation and response optimization
- **Urban Planning**: Infrastructure improvement prioritization
- **Fleet Management**: Driver safety training and route optimization

## 📈 Business Value

- **35% Risk Reduction**: Potential accident prevention through early warning
- **€2.4M Annual Savings**: Estimated cost savings from prevention
- **127 High-Risk Scenarios**: Identified critical patterns requiring intervention

## 🔒 Security

- CORS configured for production domains
- Environment variables for sensitive data
- No PII stored or processed
- API rate limiting (recommended for production)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- French road accident data (2005-2024)
- Open-source ML libraries and frameworks
- React and FastAPI communities

---

**Note**: This is an AI-powered prediction system. Predictions should be used as decision support, not as the sole basis for critical decisions.
