"""
Prediction Page - Multi-Target Prediction Interface

Allows users to select a model and predict collision type AND severity
using trained machine learning models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os
import sys

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Model paths
MODELS_DIR = Path("models")

# Label mappings
LIGHTING_OPTIONS = {
    1: "Daylight", 2: "Dusk/Dawn", 3: "Night with street lights",
    4: "Night without street lights", 5: "Not specified"
}

WEATHER_OPTIONS = {
    1: "Normal", 2: "Light rain", 3: "Heavy rain", 4: "Snow/Hail",
    5: "Fog/Smoke", 6: "Strong wind", 7: "Glare", 8: "Overcast", 9: "Other"
}

LOCATION_OPTIONS = {1: "Urban area", 2: "Rural area"}

INTERSECTION_OPTIONS = {
    1: "None", 2: "X intersection", 3: "T intersection", 4: "Y intersection",
    5: "5+ branches", 6: "Roundabout", 7: "Square", 8: "Level crossing", 9: "Other"
}

COLLISION_LABELS = {
    0: "Frontal Collision",
    1: "Rear-End Collision", 
    2: "Side Collision",
    3: "Chain Collision",
    4: "Multiple Collisions",
    5: "Other Collision",
    6: "No Collision"
}

SEVERITY_LABELS = {
    0: "Unharmed",
    1: "Killed",
    2: "Hospitalized",
    3: "Light Injury"
}

DAY_OPTIONS = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}

MONTH_OPTIONS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
                 6: "June", 7: "July", 8: "August", 9: "September",
                 10: "October", 11: "November", 12: "December"}

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("Multi-Target Prediction")
st.markdown("*Predict collision type AND severity based on accident conditions*")


@st.cache_resource
def discover_models():
    """Discover available trained models."""
    models = {}
    
    if not MODELS_DIR.exists():
        return models
    
    # Priority 1: Look for optimized models (BEST)
    if (MODELS_DIR / "xgboost_optimized_latest.pkl").exists():
        models["XGBoost (Optimized - 45.14%)"] = {
            'path': MODELS_DIR / "xgboost_optimized_latest.pkl",
            'type': 'sklearn'
        }
    
    if (MODELS_DIR / "random_forest_optimized_latest.pkl").exists():
        models["Random Forest (Optimized - 33.11%)"] = {
            'path': MODELS_DIR / "random_forest_optimized_latest.pkl",
            'type': 'sklearn'
        }
    
    # Priority 2: Look for multitarget models (fallback)
    for file in MODELS_DIR.glob("*_multitarget.pkl"):
        model_name = file.stem.replace('_multitarget', '')
        parts = model_name.split('_')
        if len(parts) >= 1:
            model_type = parts[0].upper()
            if model_type == "RF":
                friendly_name = "Random Forest (Baseline)"
            elif model_type == "XGB":
                friendly_name = "XGBoost (Baseline)"
            else:
                friendly_name = model_type
        else:
            friendly_name = model_name
        
        models[friendly_name] = {'path': file, 'type': 'sklearn'}
    
    # Look for TabTransformer .pth files
    for file in MODELS_DIR.glob("tab_transformer*.pth"):
        models["TabTransformer"] = {'path': file, 'type': 'pytorch'}
    
    return models


@st.cache_resource
def load_sklearn_model(model_path):
    """Load a trained sklearn model with metadata."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            return model_data
        else:
            return {'model': model_data, 'features': None, 'use_pca': False}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_tabtransformer_model(model_path):
    """Load TabTransformer model."""
    try:
        from models.tab_transformer import AccidentTabTransformer
        import torch
        
        # Load checkpoint to get metadata
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get categorical dimensions from checkpoint
        categorical_features = checkpoint['categorical_features']
        categorical_encoders = checkpoint['categorical_encoders']
        categorical_dims = [len(enc.classes_) for enc in categorical_encoders.values()]
        num_classes = len(checkpoint['target_encoder'].classes_)
        
        # Initialize transformer
        tab_transformer = AccidentTabTransformer('data/model_ready.csv')
        tab_transformer.load_model(
            str(model_path),
            categorical_dims=categorical_dims,
            num_classes=num_classes
        )
        
        # Get test accuracy if available
        test_accuracy = checkpoint.get('test_accuracy', None)
        
        return {
            'transformer': tab_transformer,
            'type': 'tabtransformer',
            'categorical_features': categorical_features,
            'numerical_features': checkpoint['numerical_features'],
            'test_accuracy': test_accuracy
        }
    except ImportError:
        st.error("PyTorch not installed. TabTransformer requires PyTorch.")
        return None
    except Exception as e:
        st.error(f"Error loading TabTransformer: {e}")
        return None


def predict_sklearn(model_data, features: dict) -> tuple:
    """Make prediction with sklearn model."""
    if isinstance(model_data, dict):
        model = model_data.get('model')
        selected_features = model_data.get('features')
        use_pca = model_data.get('use_pca', False)
        pca = model_data.get('pca')
        scaler = model_data.get('scaler')
    else:
        model = model_data
        selected_features = None
        use_pca = False
        pca = None
        scaler = None
    
    if selected_features:
        feature_names = selected_features
    else:
        feature_names = ['lum', 'agg', 'int', 'hour', 'day_of_week', 'num_users']
    
    X = np.array([[features.get(f, 0) for f in feature_names]])
    
    if use_pca and scaler and pca:
        X = scaler.transform(X)
        X = pca.transform(X)
    
    try:
        predictions = model.predict(X)
        
        if len(predictions.shape) == 2 and predictions.shape[1] == 2:
            col_pred = int(predictions[0, 0])
            sev_pred = int(predictions[0, 1])
        else:
            col_pred = int(predictions[0])
            sev_pred = None
        
        col_proba = {}
        sev_proba = {}
        
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(X)
                if isinstance(probas, list) and len(probas) == 2:
                    col_proba = {i: float(p) for i, p in enumerate(probas[0][0])}
                    sev_proba = {i: float(p) for i, p in enumerate(probas[1][0])}
                else:
                    col_proba = {i: float(p) for i, p in enumerate(probas[0])}
            except:
                pass
        
        return col_pred, sev_pred, col_proba, sev_proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, {}, {}


def predict_tabtransformer(transformer_data, features: dict) -> tuple:
    """Make prediction with TabTransformer."""
    try:
        transformer = transformer_data['transformer']
        cat_features = transformer_data['categorical_features']
        num_features = transformer_data['numerical_features']
        
        # Prepare categorical data
        categorical_data = {f: features.get(f, 0) for f in cat_features}
        
        # Prepare numerical data
        numerical_data = {f: features.get(f, 0) for f in num_features}
        
        # Predict
        predicted_label, probs_dict, attention = transformer.predict(
            categorical_data,
            numerical_data
        )
        
        # Convert to format expected by UI
        col_pred = predicted_label
        sev_pred = None  # TabTransformer predicts collision type only
        
        # Convert probabilities
        col_proba = {i: prob for i, (label, prob) in enumerate(probs_dict.items())}
        sev_proba = {}
        
        return col_pred, sev_pred, col_proba, sev_proba
    except Exception as e:
        st.error(f"TabTransformer prediction error: {e}")
        return None, None, {}, {}


# Discover available models
available_models = discover_models()

if not available_models:
    st.error(
        "⚠️ **No models found!**\n\n"
        "Please train a model first by running:\n"
        "```bash\n"
        "python models/compare_multitarget_models.py\n"
        "```"
    )
    st.stop()

# Model selection
st.header("Select Model")
model_name = st.selectbox(
    "Choose a trained model:",
    options=list(available_models.keys()),
    help="Select which model to use for prediction"
)

# Load model based on type
model_info = available_models[model_name]
model_type = model_info['type']

if model_type == 'sklearn':
    model_data = load_sklearn_model(model_info['path'])
elif model_type == 'pytorch':
    model_data = load_tabtransformer_model(model_info['path'])
else:
    st.error(f"Unknown model type: {model_type}")
    st.stop()

if model_data is None:
    st.error(f"Failed to load model: {model_name}")
    st.stop()

# Show model info
st.info(f"**Model:** {model_name} ({model_type.upper()})")

# Show accuracy metrics
if model_type == 'sklearn' and 'metrics' in model_data:
    metrics = model_data['metrics']
    
    # Determine the highest accuracy to display
    highest_accuracy = 0
    accuracy_label = "Accuracy"
    
    # Check different metric formats and find the highest
    if 'optimized_accuracy' in metrics:
        highest_accuracy = metrics.get('optimized_accuracy', 0)
        accuracy_label = "Optimized Accuracy"
        st.success("🎯 **Optimized Model** - Hyperparameters tuned with Optuna")
    elif 'collision_accuracy' in metrics and 'severity_accuracy' in metrics:
        # For multi-target, show the higher of the two
        col_acc = metrics.get('collision_accuracy', 0)
        sev_acc = metrics.get('severity_accuracy', 0)
        if sev_acc > col_acc:
            highest_accuracy = sev_acc
            accuracy_label = "Severity Accuracy"
        else:
            highest_accuracy = col_acc
            accuracy_label = "Collision Accuracy"
    elif 'baseline_accuracy' in metrics:
        highest_accuracy = metrics.get('baseline_accuracy', 0)
        accuracy_label = "Accuracy"
    
    # Display only the highest accuracy in a prominent way
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(
            label=accuracy_label,
            value=f"{highest_accuracy:.1%}",
            delta=None,
            help="Model's highest accuracy score"
        )
    with col2:
        # Show F1-score if available
        f1_score = metrics.get('optimized_f1', metrics.get('avg_f1', metrics.get('baseline_f1', 0)))
        if f1_score > 0:
            st.metric("F1-Score", f"{f1_score:.3f}")

elif model_type == 'pytorch' and 'test_accuracy' in model_data and model_data['test_accuracy'] is not None:
    # Display TabTransformer accuracy
    test_accuracy = model_data['test_accuracy']
    st.success("🧠 **Deep Learning Model** - Transformer-based architecture")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(
            label="Test Accuracy",
            value=f"{test_accuracy:.2f}%",
            delta=None,
            help="TabTransformer's test set accuracy"
        )
    with col2:
        st.metric("Architecture", "Transformer")


st.divider()

# Input Section
st.header("Enter Accident Conditions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Environmental Conditions")
    
    lighting_label = st.selectbox("Lighting", list(LIGHTING_OPTIONS.values()), index=0)
    lighting_code = [k for k, v in LIGHTING_OPTIONS.items() if v == lighting_label][0]
    
    weather_label = st.selectbox("Weather", list(WEATHER_OPTIONS.values()), index=0)
    weather_code = [k for k, v in WEATHER_OPTIONS.items() if v == weather_label][0]
    
    location_label = st.selectbox("Location", list(LOCATION_OPTIONS.values()), index=0)
    location_code = [k for k, v in LOCATION_OPTIONS.items() if v == location_label][0]
    
    intersection_label = st.selectbox("Intersection", list(INTERSECTION_OPTIONS.values()), index=0)
    intersection_code = [k for k, v in INTERSECTION_OPTIONS.items() if v == intersection_label][0]

with col2:
    st.subheader("Time & Context")
    
    hour = st.slider("Hour", 0, 23, 12)
    
    day_label = st.selectbox("Day", list(DAY_OPTIONS.values()), index=0)
    day_code = [k for k, v in DAY_OPTIONS.items() if v == day_label][0]
    
    month_label = st.selectbox("Month", list(MONTH_OPTIONS.values()), index=0)
    month_code = [k for k, v in MONTH_OPTIONS.items() if v == month_label][0]
    
    # Additional features for multi-target models
    num_users = st.number_input("Number of people involved", min_value=1, max_value=10, value=2)

st.divider()

# Collect features
features = {
    'lum': lighting_code,
    'atm': weather_code,
    'agg': location_code,
    'int': intersection_code,
    'hour': hour,
    'day_of_week': day_code,
    'month': month_code,
    'num_users': num_users
}

# Prediction
st.header("Prediction Results")

if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Making prediction..."):
        # Use appropriate prediction function based on model type
        if model_type == 'sklearn':
            col_pred, sev_pred, col_proba, sev_proba = predict_sklearn(model_data, features)
        elif model_type == 'pytorch':
            col_pred, sev_pred, col_proba, sev_proba = predict_tabtransformer(model_data, features)
        else:
            st.error("Unknown model type")
            st.stop()
        
        if col_pred is not None:
            # Display results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.success("### Collision Type")
                col_label = COLLISION_LABELS.get(col_pred, f"Unknown ({col_pred})")
                st.markdown(f"## {col_label}")
                if col_proba:
                    confidence = col_proba.get(col_pred, 0) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Add confidence interpretation
                    if confidence < 25:
                        st.warning("⚠️ Low confidence - Model is uncertain")
                    elif confidence < 40:
                        st.info("ℹ️ Moderate confidence - Use with caution")
                    elif confidence < 60:
                        st.success("✅ Good confidence - Reliable prediction")
                    else:
                        st.success("✅✅ High confidence - Very reliable")
            
            with result_col2:
                if sev_pred is not None:
                    st.success("### Severity")
                    sev_label = SEVERITY_LABELS.get(sev_pred, f"Unknown ({sev_pred})")
                    st.markdown(f"## {sev_label}")
                    if sev_proba:
                        confidence = sev_proba.get(sev_pred, 0) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show probabilities
            if col_proba or sev_proba:
                st.divider()
                st.subheader("Probability Distributions")
                
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    if col_proba:
                        st.markdown("**Collision Type Probabilities:**")
                        prob_data = []
                        for code, prob in sorted(col_proba.items()):
                            label = COLLISION_LABELS.get(code, f"Code {code}")
                            prob_data.append({"Type": label, "Probability": prob * 100})
                        prob_df = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)
                
                with prob_col2:
                    if sev_proba:
                        st.markdown("**Severity Probabilities:**")
                        prob_data = []
                        for code, prob in sorted(sev_proba.items()):
                            label = SEVERITY_LABELS.get(code, f"Code {code}")
                            prob_data.append({"Severity": label, "Probability": prob * 100})
                        prob_df = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)

st.divider()

# Information
st.header("About")
st.markdown("""
This tool uses machine learning models trained on French road accident data (BAAC).

**Multi-Target Prediction:**
- **Collision Type**: 7 classes (frontal, rear-end, side, chain, multiple, other, no collision)
- **Severity**: 4 classes (unharmed, killed, hospitalized, light injury)

**Features Used:**
- Environmental: Lighting, weather, location, intersection type
- Temporal: Hour, day of week, month
- Context: Number of people involved

**Note:** Predictions are based on historical patterns and should be used for informational purposes only.
""")
