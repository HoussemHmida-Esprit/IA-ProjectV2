"""
Ensemble Prediction Page - Production Inference Pipeline
Uses stacking ensemble with all 4 models
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

try:
    from production_inference_pipeline import ProductionInferencePipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    st.error(f"⚠️ Production pipeline not available: {e}")

# Page config
st.set_page_config(page_title="Ensemble Prediction", page_icon="🎯", layout="wide")

st.title("🎯 Ensemble Prediction")
st.markdown("*Multi-model stacking ensemble with confidence scoring*")

# Feature options
LIGHTING_OPTIONS = {
    1: "Daylight", 2: "Dusk/Dawn", 3: "Night with street lights",
    4: "Night without street lights", 5: "Not specified"
}

LOCATION_OPTIONS = {1: "Urban area", 2: "Rural area"}

INTERSECTION_OPTIONS = {
    1: "None", 2: "X intersection", 3: "T intersection", 4: "Y intersection",
    5: "5+ branches", 6: "Roundabout", 7: "Square", 8: "Level crossing", 9: "Other"
}

DAY_OPTIONS = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    """Load production inference pipeline"""
    if not PIPELINE_AVAILABLE:
        return None
    
    try:
        pipeline = ProductionInferencePipeline(models_dir='models')
        if pipeline.load_all_models():
            return pipeline
        return None
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

pipeline = load_pipeline()

if pipeline is None:
    st.error("""
    ⚠️ **Ensemble Pipeline Not Available**
    
    The production inference pipeline could not be loaded. This could be because:
    - Models are not trained yet
    - Required dependencies are missing
    - Model files are not in the correct location
    
    Please ensure all models are trained and available.
    """)
    st.stop()

# Show pipeline info
st.success(f"✅ **Pipeline Loaded** - {len(pipeline.models)} models ready")

with st.expander("ℹ️ About Ensemble Prediction"):
    st.markdown("""
    ### How It Works
    
    This page uses a **Stacking Ensemble** that combines predictions from 4 models:
    
    1. **XGBoost** - Gradient boosting (optimized)
    2. **Random Forest** - Tree ensemble (optimized)
    3. **TabTransformer** - Deep learning with attention
    4. **LSTM** - Temporal risk forecasting
    
    ### Meta-Learning
    
    A **Ridge Regression** meta-model learns the optimal way to combine these predictions,
    resulting in better accuracy than any single model.
    
    ### Confidence Scoring
    
    - **Confidence**: How certain the ensemble is (based on prediction variance)
    - **Model Agreement**: How much the models agree with each other
    - Higher values = more reliable prediction
    """)

st.divider()

# Input Section
st.header("Enter Accident Conditions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Environmental Conditions")
    
    lighting_label = st.selectbox("Lighting", list(LIGHTING_OPTIONS.values()), index=0)
    lighting_code = [k for k, v in LIGHTING_OPTIONS.items() if v == lighting_label][0]
    
    location_label = st.selectbox("Location", list(LOCATION_OPTIONS.values()), index=0)
    location_code = [k for k, v in LOCATION_OPTIONS.items() if v == location_label][0]
    
    intersection_label = st.selectbox("Intersection", list(INTERSECTION_OPTIONS.values()), index=0)
    intersection_code = [k for k, v in INTERSECTION_OPTIONS.items() if v == intersection_label][0]

with col2:
    st.subheader("Time & Context")
    
    hour = st.slider("Hour", 0, 23, 12)
    
    day_label = st.selectbox("Day", list(DAY_OPTIONS.values()), index=0)
    day_code = [k for k, v in DAY_OPTIONS.items() if v == day_label][0]
    
    num_users = st.number_input("Number of people involved", min_value=1, max_value=10, value=2)

st.divider()

# Prediction
if st.button("🎯 Predict with Ensemble", type="primary", use_container_width=True):
    with st.spinner("Running multi-model inference..."):
        # Prepare input
        input_data = {
            'lum': lighting_code,
            'agg': location_code,
            'int': intersection_code,
            'day_of_week': day_code,
            'hour': hour,
            'num_users': num_users
        }
        
        # Get prediction
        try:
            result = pipeline.predict_for_dashboard(input_data)
            
            # Display results
            st.header("Prediction Results")
            
            # Final Prediction (Large Display)
            st.markdown("### 🎯 Ensemble Prediction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Collision Type",
                    result['final_prediction']['class_name'],
                    help="Final prediction from stacking ensemble"
                )
            
            with col2:
                confidence_pct = result['ensemble']['confidence'] * 100
                st.metric(
                    "Confidence",
                    f"{confidence_pct:.1f}%",
                    help="How certain the ensemble is about this prediction"
                )
            
            with col3:
                agreement_pct = result['ensemble']['model_agreement'] * 100
                st.metric(
                    "Model Agreement",
                    f"{agreement_pct:.1f}%",
                    help="How much the models agree with each other"
                )
            
            st.divider()
            
            # Individual Model Predictions
            st.markdown("### 📊 Individual Model Predictions")
            
            # Create comparison table
            model_data = []
            for model_name, data in result['individual_models'].items():
                model_data.append({
                    'Model': model_name,
                    'Prediction': data['prediction_name'],
                    'Raw Value': f"{data['raw_value']:.2f}",
                    'Confidence': f"{data['confidence']*100:.1f}%"
                })
            
            df_models = pd.DataFrame(model_data)
            st.dataframe(df_models, hide_index=True, use_container_width=True)
            
            # Visualize predictions
            st.markdown("### 📈 Prediction Distribution")
            
            import plotly.graph_objects as go
            
            models = list(result['individual_models'].keys())
            predictions = [data['raw_value'] for data in result['individual_models'].values()]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=models,
                    y=predictions,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f"{p:.2f}" for p in predictions],
                    textposition='auto'
                )
            ])
            
            # Add ensemble prediction line
            fig.add_hline(
                y=result['final_prediction']['raw_value'],
                line_dash="dash",
                line_color="red",
                annotation_text="Ensemble Prediction",
                annotation_position="right"
            )
            
            fig.update_layout(
                title="Model Predictions Comparison",
                xaxis_title="Model",
                yaxis_title="Predicted Class",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Confidence Analysis
            st.markdown("### 🔍 Confidence Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Variance Analysis**")
                st.metric("Variance", f"{result['ensemble']['variance']:.4f}")
                st.metric("Standard Deviation", f"{result['ensemble']['std_dev']:.4f}")
                
                if result['ensemble']['variance'] < 0.5:
                    st.success("✅ Low variance - Models agree strongly")
                elif result['ensemble']['variance'] < 1.0:
                    st.info("ℹ️ Moderate variance - Some disagreement")
                else:
                    st.warning("⚠️ High variance - Models disagree significantly")
            
            with col2:
                st.markdown("**Reliability Score**")
                
                # Calculate overall reliability
                reliability = (result['ensemble']['confidence'] + result['ensemble']['model_agreement']) / 2
                reliability_pct = reliability * 100
                
                st.metric("Overall Reliability", f"{reliability_pct:.1f}%")
                
                if reliability > 0.8:
                    st.success("✅ Highly reliable prediction")
                elif reliability > 0.6:
                    st.info("ℹ️ Moderately reliable prediction")
                else:
                    st.warning("⚠️ Low reliability - Use with caution")
            
            # Metadata
            with st.expander("📋 Prediction Metadata"):
                st.json(result['metadata'])
        
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            import traceback
            st.code(traceback.format_exc())

st.divider()

# Information
st.header("About")
st.markdown("""
### Stacking Ensemble

This prediction system uses **meta-learning** to combine multiple models:

**Level 0 (Base Models):**
- XGBoost, Random Forest, TabTransformer, LSTM

**Level 1 (Meta-Model):**
- Ridge Regression learns optimal combination weights

**Advantages:**
- ✅ Better accuracy than any single model
- ✅ More robust predictions
- ✅ Confidence scoring
- ✅ Model agreement metrics

**Use Cases:**
- Critical decisions requiring high confidence
- Comparing different model perspectives
- Understanding prediction reliability
""")
