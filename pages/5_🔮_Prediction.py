"""
Collision Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'models'))

MODELS_DIR = Path("models")

LIGHTING_OPTIONS = {
    1: "Daylight", 2: "Dusk/Dawn", 3: "Night with street lights",
    4: "Night without street lights", 5: "Not specified"
}

LOCATION_OPTIONS = {1: "Urban area", 2: "Rural area"}

INTERSECTION_OPTIONS = {
    1: "None", 2: "X intersection", 3: "T intersection", 4: "Y intersection",
    5: "5+ branches", 6: "Roundabout", 7: "Square", 8: "Level crossing", 9: "Other"
}

COLLISION_LABELS = {
    0: "Frontal Collision", 1: "Rear-End Collision", 2: "Side Collision",
    3: "Chain Collision", 4: "Multiple Collisions", 5: "Other Collision", 6: "No Collision"
}

SEVERITY_LABELS = {
    0: "Unharmed", 1: "Killed", 2: "Hospitalized", 3: "Light Injury"
}

DAY_OPTIONS = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Collision Prediction")


@st.cache_resource
def discover_models():
    """Discover all available models with their accuracies."""
    models = {}
    
    if not MODELS_DIR.exists():
        return models
    
    # XGBoost V2 (Optimized)
    if (MODELS_DIR / "xgboost_optimized_latest.pkl").exists():
        models["XGBoost V2"] = {
            'path': MODELS_DIR / "xgboost_optimized_latest.pkl",
            'type': 'sklearn',
            'accuracy': 45.14
        }
    
    # XGBoost V1 (Baseline)
    if (MODELS_DIR / "xgb_nopca_multitarget.pkl").exists():
        models["XGBoost V1"] = {
            'path': MODELS_DIR / "xgb_nopca_multitarget.pkl",
            'type': 'sklearn',
            'accuracy': 45.0
        }
    
    # Random Forest V2 (Optimized)
    if (MODELS_DIR / "random_forest_optimized_latest.pkl").exists():
        models["Random Forest V2"] = {
            'path': MODELS_DIR / "random_forest_optimized_latest.pkl",
            'type': 'sklearn',
            'accuracy': 33.11
        }
    
    # Random Forest V1 (Baseline)
    if (MODELS_DIR / "rf_pca_multitarget.pkl").exists():
        models["Random Forest V1"] = {
            'path': MODELS_DIR / "rf_pca_multitarget.pkl",
            'type': 'sklearn',
            'accuracy': 32.0
        }
    
    # TabTransformer V1 (only if PyTorch available)
    try:
        import torch
        if (MODELS_DIR / "tab_transformer_best.pth").exists():
            models["TabTransformer V1"] = {
                'path': MODELS_DIR / "tab_transformer_best.pth",
                'type': 'pytorch',
                'accuracy': 35.0
            }
    except ImportError:
        pass  # PyTorch not available, skip TabTransformer
    
    # Stacking Ensemble
    try:
        from production_inference_pipeline import ProductionInferencePipeline
        pipeline = ProductionInferencePipeline(models_dir='models')
        if pipeline.load_all_models():
            models["Stacking"] = {
                'path': None,
                'type': 'ensemble',
                'pipeline': pipeline,
                'accuracy': 46.0
            }
    except:
        pass
    
    return models


@st.cache_resource
def load_sklearn_model(model_path):
    """Load sklearn model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        return model_data.get('model', model_data)
    return model_data


@st.cache_resource
def load_pytorch_model(model_path):
    """Load PyTorch model."""
    import torch
    from tab_transformer import AccidentTabTransformer
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    categorical_encoders = checkpoint['categorical_encoders']
    categorical_dims = [len(enc.classes_) for enc in categorical_encoders.values()]
    num_classes = len(checkpoint['target_encoder'].classes_)
    
    tab_transformer = AccidentTabTransformer('data/model_ready.csv')
    tab_transformer.load_model(str(model_path), categorical_dims=categorical_dims, num_classes=num_classes)
    
    return {
        'transformer': tab_transformer,
        'categorical_features': checkpoint['categorical_features'],
        'numerical_features': checkpoint['numerical_features']
    }


def predict_sklearn(model, features):
    """Predict collision and severity with sklearn model."""
    feature_names = ['lum', 'agg', 'int', 'day_of_week', 'hour', 'num_users']
    X = np.array([[features.get(f, 0) for f in feature_names]])
    
    predictions = model.predict(X)
    
    # Handle multi-output
    if len(predictions.shape) == 2 and predictions.shape[1] == 2:
        col_pred = int(predictions[0, 0])
        sev_pred = int(predictions[0, 1])
    else:
        col_pred = int(predictions[0])
        sev_pred = None
    
    return col_pred, sev_pred


def predict_pytorch(model_data, features):
    """Predict with PyTorch model."""
    transformer = model_data['transformer']
    cat_features = model_data['categorical_features']
    num_features = model_data['numerical_features']
    
    categorical_data = {f: features.get(f, 0) for f in cat_features}
    numerical_data = {f: features.get(f, 0) for f in num_features}
    
    predicted_label, probs_dict, _ = transformer.predict(categorical_data, numerical_data)
    
    return predicted_label, None


def predict_ensemble(pipeline, features):
    """Predict with ensemble."""
    input_data = {
        'lum': features['lum'],
        'agg': features['agg'],
        'int': features['int'],
        'day_of_week': features['day_of_week'],
        'hour': features['hour'],
        'num_users': features['num_users']
    }
    
    result = pipeline.predict_for_dashboard(input_data)
    
    pred = result['final_prediction']['class']
    
    return pred, None, result


# Discover models
available_models = discover_models()

if not available_models:
    st.error("⚠️ No models found.")
    st.stop()

# Model selection with accuracy
col1, col2 = st.columns([3, 1])

with col1:
    model_name = st.selectbox(
        "Model",
        list(available_models.keys()),
        format_func=lambda x: f"{x} ({available_models[x]['accuracy']:.1f}%)"
    )

with col2:
    model_info = available_models[model_name]
    st.metric("Accuracy", f"{model_info['accuracy']:.1f}%")

st.divider()

# Input form
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    lighting_label = st.selectbox("Lighting", list(LIGHTING_OPTIONS.values()))
    lighting = [k for k, v in LIGHTING_OPTIONS.items() if v == lighting_label][0]
    
    location_label = st.selectbox("Location", list(LOCATION_OPTIONS.values()))
    location = [k for k, v in LOCATION_OPTIONS.items() if v == location_label][0]

with col2:
    intersection_label = st.selectbox("Intersection", list(INTERSECTION_OPTIONS.values()))
    intersection = [k for k, v in INTERSECTION_OPTIONS.items() if v == intersection_label][0]
    
    day_label = st.selectbox("Day", list(DAY_OPTIONS.values()))
    day = [k for k, v in DAY_OPTIONS.items() if v == day_label][0]

with col3:
    hour = st.slider("Hour", 0, 23, 12)
    num_users = st.number_input("People Involved", 1, 10, 2)

features = {
    'lum': lighting,
    'agg': location,
    'int': intersection,
    'hour': hour,
    'day_of_week': day,
    'num_users': num_users
}

st.divider()

# Predict
if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Predicting..."):
        
        ensemble_result = None
        
        if model_info['type'] == 'sklearn':
            model = load_sklearn_model(model_info['path'])
            col_pred, sev_pred = predict_sklearn(model, features)
            
        elif model_info['type'] == 'pytorch':
            model_data = load_pytorch_model(model_info['path'])
            col_pred, sev_pred = predict_pytorch(model_data, features)
            
        elif model_info['type'] == 'ensemble':
            pipeline = model_info['pipeline']
            col_pred, sev_pred, ensemble_result = predict_ensemble(pipeline, features)
        
        # Display results
        st.subheader("Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            col_label = COLLISION_LABELS.get(col_pred, f"Unknown ({col_pred})")
            st.success("**Collision Type**")
            st.markdown(f"## {col_label}")
        
        with result_col2:
            if sev_pred is not None:
                sev_label = SEVERITY_LABELS.get(sev_pred, f"Unknown ({sev_pred})")
                st.warning("**Severity**")
                st.markdown(f"## {sev_label}")
            else:
                st.info("**Severity**")
                st.markdown("## Not Available")
        
        # Ensemble details
        if ensemble_result:
            st.divider()
            st.subheader("Ensemble Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Agreement", f"{ensemble_result['ensemble']['model_agreement']*100:.1f}%")
            
            with col2:
                st.metric("Variance", f"{ensemble_result['ensemble']['variance']:.4f}")
            
            # Individual models
            st.markdown("**Individual Models:**")
            model_data = []
            for model_name, data in ensemble_result['individual_models'].items():
                model_data.append({
                    "Model": model_name,
                    "Prediction": data['prediction_name']
                })
            st.dataframe(pd.DataFrame(model_data), hide_index=True, use_container_width=True)
