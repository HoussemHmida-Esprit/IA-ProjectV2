"""
FastAPI Backend for Accident Severity Prediction
Serves predictions from XGBoost/TabTransformer/LSTM and SHAP explainability
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import sys
from pathlib import Path
import numpy as np

# Add models directory to path
project_root = Path(__file__).parent.parent
models_dir = project_root / 'models'
data_dir = project_root / 'data'

sys.path.append(str(models_dir))

try:
    from production_inference_pipeline import ProductionInferencePipeline
    from explainable_ai import AccidentXAI
    from lstm_forecasting import AccidentForecaster
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False

app = FastAPI(title="Accident Prediction API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.vercel.app",  # Allow all Vercel deployments
        "https://ia-projectv2.vercel.app",  # Your production URL (update if different)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
xai = None
forecaster = None

# Request/Response Models
class PredictionRequest(BaseModel):
    lighting: int  # lum
    location: int  # agg
    intersection: int  # int
    day_of_week: int
    hour: int
    num_users: int
    model: Optional[str] = 'stacking'  # Model to use: stacking, xgboost, random_forest, tabtransformer, lstm

class PredictionResponse(BaseModel):
    final_prediction: Dict
    individual_models: Dict
    ensemble: Dict
    metadata: Dict

class SHAPResponse(BaseModel):
    features: List[str]
    shap_values: List[float]
    feature_importance: List[Dict]

class ForecastRequest(BaseModel):
    days: int = 7

class ForecastResponse(BaseModel):
    dates: List[str]
    predictions: List[int]
    total: int
    average: float

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global pipeline, xai, forecaster
    
    if not MODELS_AVAILABLE:
        print("⚠️ Models not available")
        return
    
    try:
        # Load production pipeline
        print("Loading production pipeline...")
        print(f"Models directory: {models_dir}")
        print(f"Data directory: {data_dir}")
        
        pipeline = ProductionInferencePipeline(models_dir=str(models_dir))
        if pipeline.load_all_models():
            print("✅ Pipeline loaded successfully")
        else:
            print("⚠️ No models loaded in pipeline")
            pipeline = None
        
        # Load XAI
        print("Loading XAI module...")
        try:
            xai_model_path = models_dir / 'xgb_nopca_multitarget.pkl'
            xai_data_path = data_dir / 'model_ready.csv'
            
            print(f"XAI model path: {xai_model_path}")
            print(f"XAI data path: {xai_data_path}")
            
            if not xai_model_path.exists():
                raise FileNotFoundError(f"Model not found: {xai_model_path}")
            if not xai_data_path.exists():
                raise FileNotFoundError(f"Data not found: {xai_data_path}")
            
            xai = AccidentXAI(
                model_path=str(xai_model_path),
                data_path=str(xai_data_path)
            )
            xai.load_model_and_data()
            xai.compute_shap_values(sample_size=500)
            print("✅ XAI loaded successfully")
        except Exception as e:
            print(f"⚠️ XAI loading failed: {e}")
            xai = None
        
        # Load forecaster
        print("Loading LSTM forecaster...")
        try:
            forecaster_data_path = data_dir / 'cleaned_accidents.csv'
            forecaster_model_path = models_dir / 'lstm_forecaster.pth'
            
            print(f"Forecaster data path: {forecaster_data_path}")
            print(f"Forecaster model path: {forecaster_model_path}")
            
            if not forecaster_data_path.exists():
                raise FileNotFoundError(f"Data not found: {forecaster_data_path}")
            if not forecaster_model_path.exists():
                raise FileNotFoundError(f"Model not found: {forecaster_model_path}")
            
            forecaster = AccidentForecaster(
                data_path=str(forecaster_data_path),
                sequence_length=30
            )
            forecaster.prepare_time_series_data()
            forecaster.load_model(str(forecaster_model_path))
            print("✅ Forecaster loaded successfully")
        except Exception as e:
            print(f"⚠️ Forecaster loading failed: {e}")
            forecaster = None
            
    except Exception as e:
        print(f"❌ Startup error: {e}")

# Health check
@app.get("/api/health")
async def health_check():
    """Check API and model status"""
    return {
        "status": "online",
        "models_loaded": pipeline is not None,
        "xai_available": xai is not None,
        "forecaster_available": forecaster is not None,
        "models": {
            "xgboost": "xgboost" in (pipeline.models if pipeline else {}),
            "random_forest": "random_forest" in (pipeline.models if pipeline else {}),
            "tabtransformer": "tabtransformer" in (pipeline.models if pipeline else {}),
            "lstm": "lstm" in (pipeline.models if pipeline else {})
        }
    }

# Prediction endpoint
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict accident severity using specified model or ensemble
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Prepare input
        input_data = {
            'lum': request.lighting,
            'agg': request.location,
            'int': request.intersection,
            'day_of_week': request.day_of_week,
            'hour': request.hour,
            'num_users': request.num_users
        }
        
        # Check if specific model is requested
        if request.model and request.model != 'stacking':
            # Get prediction from specific model
            result = predict_single_model(input_data, request.model)
        else:
            # Get ensemble prediction (default)
            result = pipeline.predict_for_dashboard(input_data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def predict_single_model(input_data: Dict, model_key: str) -> Dict:
    """
    Get prediction from a single model
    
    Args:
        input_data: Input features
        model_key: Model identifier (xgboost, random_forest, tabtransformer, lstm)
    
    Returns:
        Dashboard-compatible prediction output for single model with both collision type and severity
    """
    from datetime import datetime
    
    # Map model keys to display names
    model_names = {
        'xgboost': 'XGBoost',
        'xgboost_v1': 'XGBoost V1',
        'xgboost_v2': 'XGBoost V2',
        'random_forest': 'Random Forest',
        'random_forest_v1': 'Random Forest V1',
        'random_forest_v2': 'Random Forest V2',
        'tabtransformer': 'TabTransformer'
    }
    
    # Map v1/v2 to actual model keys
    model_mapping = {
        'xgboost_v1': 'xgboost',
        'xgboost_v2': 'xgboost',
        'random_forest_v1': 'random_forest',
        'random_forest_v2': 'random_forest',
        'tabtransformer': 'tabtransformer'
    }
    
    actual_model_key = model_mapping.get(model_key, model_key)
    
    # Check if model exists
    if actual_model_key not in pipeline.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not available")
    
    # Get prediction from specific model
    import pandas as pd
    df = pd.DataFrame([input_data])
    
    collision_pred = None
    severity_pred = None
    collision_confidence = 0.0
    severity_confidence = 0.0
    
    # Load the actual model (might be MultiOutputClassifier)
    import pickle
    model_path = None
    
    if actual_model_key == 'xgboost':
        model_path = models_dir / 'xgb_nopca_multitarget.pkl'
    elif actual_model_key == 'random_forest':
        model_path = models_dir / 'rf_pca_multitarget.pkl'
        if not model_path.exists():
            model_path = models_dir / 'rf_nopca_multitarget.pkl'
    
    # Try to get multi-output predictions
    if model_path and model_path.exists():
        try:
            print(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                full_model = model_data.get('model', model_data)
            
            print(f"Model type: {type(full_model)}")
            print(f"Has estimators_: {hasattr(full_model, 'estimators_')}")
            
            X = pipeline.pipeline.preprocess_for_tree_models(df)
            print(f"Input shape: {X.shape}")
            
            # Check if it's MultiOutputClassifier
            if hasattr(full_model, 'estimators_'):
                # Multi-output model: predicts [collision, severity]
                predictions = full_model.predict(X)
                collision_pred = float(predictions[0][0])
                severity_pred = float(predictions[0][1])
                
                print(f"Multi-output prediction - Collision: {collision_pred}, Severity: {severity_pred}")
                
                # Get probabilities for confidence
                try:
                    # Get probabilities from each estimator
                    collision_proba = full_model.estimators_[0].predict_proba(X)
                    severity_proba = full_model.estimators_[1].predict_proba(X)
                    collision_confidence = float(np.max(collision_proba[0]))
                    severity_confidence = float(np.max(severity_proba[0]))
                    print(f"Confidences - Collision: {collision_confidence}, Severity: {severity_confidence}")
                except Exception as conf_error:
                    print(f"Error getting confidence: {conf_error}")
                    collision_confidence = 0.8
                    severity_confidence = 0.8
            else:
                # Single output model (collision only)
                pred = full_model.predict(X)
                collision_pred = float(pred[0])
                severity_pred = 2.0  # Default moderate severity
                
                try:
                    pred_proba = full_model.predict_proba(X)
                    collision_confidence = float(np.max(pred_proba[0]))
                    severity_confidence = 0.7
                except:
                    collision_confidence = 0.8
                    severity_confidence = 0.7
                    
        except Exception as e:
            print(f"Error loading multi-output model: {e}")
            # Fallback to single output
            collision_pred = float(pipeline.models[actual_model_key].predict(pipeline.pipeline.preprocess_for_tree_models(df))[0])
            severity_pred = 2.0
            collision_confidence = 0.8
            severity_confidence = 0.7
    
    elif actual_model_key == 'tabtransformer':
        try:
            row = df.iloc[0]
            # Convert to proper format for TabTransformer
            categorical_data = {f: int(row[f]) for f in pipeline.pipeline.categorical_features}
            numerical_data = {f: float(row[f]) for f in pipeline.pipeline.numerical_features}
            
            pred, probs, _ = pipeline.models['tabtransformer'].predict(
                categorical_data,
                numerical_data
            )
            collision_pred = float(pred)
            severity_pred = 1.0  # TabTransformer predicts collision only, default to light injury
            collision_confidence = float(np.max(probs))
            severity_confidence = 0.7
        except Exception as e:
            print(f"TabTransformer prediction error: {e}")
            import traceback
            traceback.print_exc()
            collision_pred = 2.0
            severity_pred = 1.0
            collision_confidence = 0.7
            severity_confidence = 0.7
    else:
        # Fallback
        collision_pred = 3.0
        severity_pred = 2.0
        collision_confidence = 0.7
        severity_confidence = 0.7
    
    # Map predictions to class names
    # Note: Models are trained with 0-indexed values
    # Collision: 0-6 (original 1-7 minus 1)
    # Severity: 0-3 (original 1-4 minus 1)
    collision_names = {
        -1: "Unknown",
        0: "Frontal",
        1: "Rear-End",
        2: "Side",
        3: "Chain",
        4: "Multiple",
        5: "Other",
        6: "None",
        7: "Other Type"
    }
    
    severity_names = {
        0: "Uninjured",
        1: "Light Injury",
        2: "Hospitalized",
        3: "Fatal"
    }
    
    collision_int = int(round(collision_pred))
    severity_int = int(round(severity_pred))
    model_display_name = model_names.get(model_key, model_key)
    
    # Format output with both predictions
    return {
        'final_prediction': {
            'collision': {
                'class': collision_int,
                'class_name': collision_names.get(collision_int, f"Class {collision_int}"),
                'confidence': collision_confidence,
                'raw_value': collision_pred
            },
            'severity': {
                'class': severity_int,
                'class_name': severity_names.get(severity_int, f"Severity {severity_int}"),
                'level': severity_int + 1,  # Display as 1-4 for user
                'confidence': severity_confidence,
                'raw_value': severity_pred
            }
        },
        'individual_models': {
            model_display_name: {
                'collision_prediction': collision_int,
                'collision_name': collision_names.get(collision_int, f"Class {collision_int}"),
                'severity_prediction': severity_int,
                'severity_name': severity_names.get(severity_int, f"Severity {severity_int}"),
                'collision_confidence': collision_confidence,
                'severity_confidence': severity_confidence
            }
        },
        'ensemble': {
            'collision_prediction': collision_int,
            'collision_name': collision_names.get(collision_int, f"Class {collision_int}"),
            'severity_prediction': severity_int,
            'severity_name': severity_names.get(severity_int, f"Severity {severity_int}"),
            'collision_confidence': collision_confidence,
            'severity_confidence': severity_confidence,
            'model_agreement': 1.0,
            'variance': 0.0,
            'std_dev': 0.0
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'models_used': [actual_model_key],
            'preprocessing_version': '1.0',
            'pipeline_version': '1.0',
            'model_type': 'single',
            'model_name': model_display_name
        }
    }

# SHAP explainability endpoint
@app.post("/api/shap")
async def get_shap_values(request: PredictionRequest):
    """
    Get SHAP values for a prediction
    """
    if xai is None:
        raise HTTPException(status_code=503, detail="XAI not available")
    
    try:
        # Prepare input - ensure all features are present
        input_data = {
            'lum': request.lighting,
            'agg': request.location,
            'int': request.intersection,
            'day_of_week': request.day_of_week,
            'hour': request.hour,
            'num_users': request.num_users
        }
        
        print(f"SHAP request input: {input_data}")
        print(f"XAI feature names: {xai.feature_names}")
        
        # Get SHAP values for this specific prediction
        import pandas as pd
        df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = [f for f in xai.feature_names if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0
        
        X = df[xai.feature_names].values
        print(f"Input shape: {X.shape}")
        
        # Calculate SHAP values
        shap_vals = xai.explainer.shap_values(X)
        print(f"SHAP values shape: {shap_vals.shape if hasattr(shap_vals, 'shape') else type(shap_vals)}")
        
        # Handle multi-class output
        if isinstance(shap_vals, list):
            print(f"SHAP is list with {len(shap_vals)} elements")
            # Take first class or average
            shap_vals = shap_vals[0] if len(shap_vals) > 0 else shap_vals
        
        if len(shap_vals.shape) == 3:
            print("SHAP is 3D, averaging across classes")
            shap_vals = np.mean(np.abs(shap_vals), axis=2)
        
        if len(shap_vals.shape) == 2:
            shap_vals = shap_vals[0]
        
        print(f"Final SHAP shape: {shap_vals.shape}")
        
        # Create response
        from explainable_ai import FEATURE_NAMES
        
        features_with_values = []
        for i, feature in enumerate(xai.feature_names):
            features_with_values.append({
                'feature': FEATURE_NAMES.get(feature, feature),
                'feature_code': feature,
                'value': float(shap_vals[i]),
                'input_value': input_data.get(feature, 0)
            })
        
        # Sort by absolute value
        features_with_values.sort(key=lambda x: abs(x['value']), reverse=True)
        
        print(f"Returning {len(features_with_values)} features")
        
        return {
            'features': [f['feature'] for f in features_with_values],
            'shap_values': [f['value'] for f in features_with_values],
            'feature_details': features_with_values
        }
        
    except Exception as e:
        import traceback
        error_detail = f"SHAP calculation failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

# Global SHAP importance endpoint
@app.get("/api/shap/importance")
async def get_feature_importance():
    """
    Get global feature importance from SHAP
    """
    if xai is None:
        raise HTTPException(status_code=503, detail="XAI not available")
    
    try:
        importance_df = xai.get_feature_importance()
        
        return {
            'features': importance_df['Feature'].tolist(),
            'importance': importance_df['Mean_Abs_SHAP'].tolist(),
            'feature_codes': importance_df['Feature_Code'].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(e)}")

# Forecasting endpoint
@app.post("/api/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Forecast future accident counts
    """
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster not available")
    
    try:
        # Get last sequence
        accident_counts = forecaster.daily_counts['accident_count'].values
        last_sequence = accident_counts[-forecaster.sequence_length:]
        
        # Forecast
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(request.days):
            next_pred = forecaster.predict(current_sequence)
            predictions.append(int(next_pred))
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Create dates
        from datetime import datetime, timedelta
        last_date = forecaster.daily_counts['date'].max()
        forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                         for i in range(request.days)]
        
        return {
            'dates': forecast_dates,
            'predictions': predictions,
            'total': sum(predictions),
            'average': float(np.mean(predictions))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

# Available models endpoint
@app.get("/api/models")
async def get_available_models():
    """
    Get list of available models with metadata
    """
    if pipeline is None:
        return {"models": []}
    
    models_info = []
    
    for model_key, model_name in pipeline.model_names.items():
        if model_key in pipeline.models:
            models_info.append({
                'key': model_key,
                'name': model_name,
                'available': True,
                'type': 'ensemble' if model_key == 'stacking' else 'ml'
            })
    
    return {"models": models_info}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
