"""
Collision Prediction - Production Tool
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

MODELS_DIR = Path("models")

# Label mappings
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

DAY_OPTIONS = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Collision Prediction")


@st.cache_resource
def discover_models():
    """Discover available models with V1/V2 naming."""
    models = {}
    
    if not MODELS_DIR.exists():
        return models
    
    # V2 Models (Optimized)
    if (MODELS_DIR / "xgboost_optimized_latest.pkl").exists():
        models["XGBoost V2"] = {
            'path': MODELS_DIR / "xgboost_optimized_latest.pkl",
            'version': 'V2',
            'accuracy': 45.14
        }
    
    if (MODELS_DIR / "random_forest_optimized_latest.pkl").exists():
        models["Random Forest V2"] = {
            'path': MODELS_DIR / "random_forest_optimized_latest.pkl",
            'version': 'V2',
            'accuracy': 33.11
        }
    
    # V1 Models (Baseline)
    if (MODELS_DIR / "xgb_nopca_multitarget.pkl").exists():
        models["XGBoost V1"] = {
            'path': MODELS_DIR / "xgb_nopca_multitarget.pkl",
            'version': 'V1',
            'accuracy': 45.0
        }
    
    if (MODELS_DIR / "rf_nopca_multitarget.pkl").exists():
        models["Random Forest V1"] = {
            'path': MODELS_DIR / "rf_nopca_multitarget.pkl",
            'version': 'V1',
            'accuracy': 32.0
        }
    
    return models


@st.cache_resource
def load_model(model_path):
    """Load model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        return model_data.get('model', model_data)
    return model_data


def predict(model, features: dict):
    """Make prediction."""
    feature_names = ['lum', 'agg', 'int', 'day_of_week', 'hour', 'num_users']
    X = np.array([[features.get(f, 0) for f in feature_names]])
    
    pred = model.predict(X)[0]
    
    # Get probabilities if available
    proba = {}
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X)
            if isinstance(probs, list):
                proba = {i: float(p) for i, p in enumerate(probs[0][0])}
            else:
                proba = {i: float(p) for i, p in enumerate(probs[0])}
        except:
            pass
    
    return int(pred), proba


# Discover models
available_models = discover_models()

if not available_models:
    st.error("⚠️ No models found. Please train models first.")
    st.stop()

# Model selection
col1, col2 = st.columns([3, 1])

with col1:
    model_name = st.selectbox(
        "Model",
        options=list(available_models.keys()),
        format_func=lambda x: f"{x} ({available_models[x]['accuracy']:.1f}%)"
    )

with col2:
    model_info = available_models[model_name]
    if model_info['version'] == 'V2':
        st.success("✓ Optimized")
    else:
        st.info("Baseline")

# Load model
model = load_model(model_info['path'])

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

# Collect features
features = {
    'lum': lighting,
    'agg': location,
    'int': intersection,
    'hour': hour,
    'day_of_week': day,
    'num_users': num_users
}

st.divider()

# Predict button
if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Predicting..."):
        pred, proba = predict(model, features)
        
        # Display result
        col_label = COLLISION_LABELS.get(pred, f"Unknown ({pred})")
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            st.markdown(f"### Prediction: **{col_label}**")
        
        with result_col2:
            if proba:
                confidence = proba.get(pred, 0) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
        
        # Probabilities table
        if proba:
            st.divider()
            st.subheader("All Probabilities")
            
            prob_data = []
            for code, prob in sorted(proba.items(), key=lambda x: x[1], reverse=True):
                label = COLLISION_LABELS.get(code, f"Code {code}")
                prob_data.append({
                    "Collision Type": label,
                    "Probability": f"{prob * 100:.1f}%"
                })
            
            st.dataframe(
                pd.DataFrame(prob_data),
                hide_index=True,
                use_container_width=True
            )
