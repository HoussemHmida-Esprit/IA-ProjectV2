"""
Model Explainability Page - SHAP Analysis

Provides interpretability for machine learning models using SHAP values.
Shows which features are most important for predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.explainable_ai import AccidentXAI
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Model Explainability",
    page_icon="🔍",
    layout="wide"
)

st.title("Model Explainability")
st.markdown("*Understand how the model makes predictions using SHAP values*")

if not SHAP_AVAILABLE:
    st.error("""
    ⚠️ **SHAP library not installed**
    
    To use this feature, install SHAP:
    ```bash
    pip install shap
    ```
    """)
    st.stop()

# Check if models exist
MODELS_DIR = Path("models")
DATA_PATH = Path("data/model_ready.csv")

available_models = {}
for model_file in MODELS_DIR.glob("*_multitarget.pkl"):
    model_name = model_file.stem.replace('_multitarget', '')
    if model_name in ['rf_pca', 'xgb_nopca']:
        friendly_name = "Random Forest" if model_name == 'rf_pca' else "XGBoost"
        available_models[friendly_name] = model_file

if not available_models:
    st.error("⚠️ No models found. Please train a model first.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"⚠️ Data file not found: {DATA_PATH}")
    st.stop()

# Sidebar: Model selection
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select Model to Explain",
    options=list(available_models.keys())
)

sample_size = st.sidebar.slider(
    "Sample Size for SHAP",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
    help="Larger samples give more accurate results but take longer"
)

# Main content
st.header("What is SHAP?")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** is a method to explain individual predictions by computing 
the contribution of each feature to the prediction.

- **Positive SHAP value**: Feature pushes prediction higher
- **Negative SHAP value**: Feature pushes prediction lower
- **Magnitude**: How much the feature matters
""")

st.divider()

# Initialize XAI
@st.cache_resource
def load_xai(model_path, data_path):
    """Load XAI module"""
    xai = AccidentXAI(str(model_path), str(data_path))
    xai.load_model_and_data()
    return xai

def compute_shap_if_needed(xai, sample_size):
    """Compute SHAP values if not already computed"""
    if xai.shap_values is None:
        xai.compute_shap_values(sample_size=sample_size)
    return xai

# Load model
with st.spinner(f"Loading {selected_model} model..."):
    try:
        xai = load_xai(available_models[selected_model], DATA_PATH)
        st.success(f"✅ Loaded {selected_model} model")
        
        # Show model info
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Features used by this model:** {len(xai.feature_names)}")
            feature_list = ", ".join([f"`{f}`" for f in xai.feature_names])
            st.markdown(f"Features: {feature_list}")
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Compute SHAP values
with st.spinner("Computing SHAP values... This may take a minute."):
    try:
        xai = compute_shap_if_needed(xai, sample_size)
        st.success(f"✅ SHAP values computed for {sample_size} samples")
        
        # Get actual features from the model (needed for all tabs)
        actual_features = xai.feature_names
        
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        st.stop()

st.divider()

# Tab layout for different visualizations
tab1, tab2, tab3 = st.tabs(["Global Importance", "Feature Dependence", "Feature Importance Table"])

with tab1:
    st.header("Global Feature Importance")
    st.markdown("""
    **How to read this chart:**
    - Features are ranked from **most important** (top) to **least important** (bottom)
    - Each **dot** represents one prediction from your data
    - **Color coding:**
      - 🔴 **Red/Pink dots** = High feature value (e.g., late hour, many people)
      - 🔵 **Blue dots** = Low feature value (e.g., early hour, few people)
    - **Horizontal position (x-axis):**
      - **Right side (positive)** = Feature increases accident severity/collision risk
      - **Left side (negative)** = Feature decreases accident severity/collision risk
    - **Spread** = How much the feature's impact varies across different situations
    
    **Example:** If "Hour" has many red dots on the right, it means late hours increase accident risk.
    """)
    
    try:
        fig = xai.plot_global_summary()
        st.pyplot(fig)
        plt.close()
        
        # Add interpretation help
        st.info("""
        💡 **Quick Interpretation Tips:**
        - Look at the **top 3 features** - these have the biggest impact on predictions
        - Features with **wide spreads** have complex, non-linear effects
        - Features with **narrow spreads** have consistent, predictable effects
        """)
        
    except Exception as e:
        st.error(f"Error creating summary plot: {e}")

with tab2:
    st.header("Feature Dependence Analysis")
    st.markdown("""
    **How to read this chart:**
    - **X-axis** = The actual value of the feature (e.g., hour 0-23, number of people)
    - **Y-axis** = SHAP value (impact on prediction)
      - **Positive (above 0)** = Increases accident severity/collision risk
      - **Negative (below 0)** = Decreases accident severity/collision risk
    - **Color** = Shows interaction with another feature
    - **Pattern** = Shows the relationship:
      - **Upward slope** = Higher values increase risk
      - **Downward slope** = Higher values decrease risk
      - **Curved/scattered** = Complex non-linear relationship
    
    **Example:** If Hour shows an upward curve from 18-23, it means evening hours increase risk.
    """)
    
    # Feature selection
    feature_display_names = {
        'hour': 'Hour of Day',
        'lum': 'Lighting Conditions',
        'atm': 'Weather Conditions',
        'agg': 'Location Type',
        'int': 'Intersection Type',
        'day_of_week': 'Day of Week',
        'month': 'Month',
        'num_users': 'Number of People',
        'num_light_injury': 'Number of Light Injuries'
    }
    
    # Build feature options from actual features
    feature_options = {
        feat: feature_display_names.get(feat, feat.replace('_', ' ').title())
        for feat in actual_features
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox(
            "Select Feature to Analyze",
            options=list(feature_options.keys()),
            format_func=lambda x: feature_options[x],
            help="Choose which feature's effect you want to understand"
        )
    
    with col2:
        interaction_feature = st.selectbox(
            "Color by Feature (Interaction)",
            options=['auto'] + list(feature_options.keys()),
            format_func=lambda x: 'Auto' if x == 'auto' else feature_options[x],
            help="Color points by another feature to see interactions"
        )
    
    if st.button("Generate Dependence Plot", type="primary"):
        with st.spinner("Creating dependence plot..."):
            try:
                interaction = None if interaction_feature == 'auto' else interaction_feature
                fig = xai.plot_dependence(selected_feature, interaction_feature=interaction)
                st.pyplot(fig)
                plt.close()
                
                # Add interpretation based on selected feature
                st.success("✅ Plot generated!")
                
                with st.expander("💡 How to interpret this specific plot"):
                    if selected_feature == 'hour':
                        st.write("""
                        **Hour of Day interpretation:**
                        - Look for peaks during rush hours (7-9, 17-19)
                        - Night hours (22-6) often show different patterns
                        - Compare weekday vs weekend patterns if colored by day_of_week
                        """)
                    elif selected_feature == 'num_users':
                        st.write("""
                        **Number of People interpretation:**
                        - More people usually means more severe accidents
                        - Look for threshold effects (e.g., 3+ people)
                        - Multi-vehicle accidents have different patterns
                        """)
                    elif selected_feature == 'lum':
                        st.write("""
                        **Lighting Conditions interpretation:**
                        - 1 = Daylight, 2 = Twilight, 3 = Night (lit), 4 = Night (unlit), 5 = Unknown
                        - Night conditions (3-4) typically increase risk
                        - Interaction with hour shows day/night cycle effects
                        """)
                    else:
                        st.write(f"""
                        **{feature_options[selected_feature]} interpretation:**
                        - Look for upward/downward trends
                        - Check for threshold effects (sudden changes)
                        - Color patterns show how other features modify the effect
                        """)
                
            except Exception as e:
                st.error(f"Error creating dependence plot: {e}")

with tab3:
    st.header("Feature Importance Rankings")
    st.markdown("""
    **How to read this table:**
    - Features are ranked from **most important** to **least important**
    - **Mean Abs SHAP** = Average impact on predictions (higher = more important)
    - This is a simplified view - see the Global Importance chart for full details
    """)
    
    try:
        importance_df = xai.get_feature_importance()
        
        # Display as table with better formatting
        st.dataframe(
            importance_df[['Feature', 'Mean_Abs_SHAP']].style.format({
                'Mean_Abs_SHAP': '{:.4f}'
            }).background_gradient(subset=['Mean_Abs_SHAP'], cmap='Blues'),
            hide_index=True,
            use_container_width=True
        )
        
        # Bar chart with better styling
        import plotly.express as px
        fig = px.bar(
            importance_df,
            x='Mean_Abs_SHAP',
            y='Feature',
            orientation='h',
            title='Feature Importance (Mean Absolute SHAP)',
            color='Mean_Abs_SHAP',
            color_continuous_scale='RdYlGn_r',
            labels={'Mean_Abs_SHAP': 'Importance Score', 'Feature': ''}
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            height=400,
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        top_feature = importance_df.iloc[0]['Feature']
        top_score = importance_df.iloc[0]['Mean_Abs_SHAP']
        
        st.info(f"""
        💡 **Key Insight:** 
        - **{top_feature}** is the most important feature (score: {top_score:.4f})
        - The top 3 features account for most of the model's decision-making
        - Focus on these features for interventions and policy decisions
        """)
        
    except Exception as e:
        st.error(f"Error creating importance table: {e}")

st.divider()

# Information section with visual examples
st.header("📚 Understanding SHAP Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 What is SHAP?")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** explains predictions by showing how much each feature contributes.
    
    Think of it like this:
    - Your model predicts an accident will be severe
    - SHAP shows: "Hour (+0.5), Location (+0.3), Lighting (+0.2)"
    - This means the hour contributed most to the high severity prediction
    """)
    
    st.subheader("📊 Reading the Charts")
    st.markdown("""
    **Global Summary (Tab 1):**
    - Top features = Most important
    - Red dots = High values
    - Blue dots = Low values
    - Right side = Increases risk
    - Left side = Decreases risk
    
    **Dependence Plot (Tab 2):**
    - Shows exact relationship
    - X-axis = Feature value
    - Y-axis = Impact on prediction
    - Upward trend = Higher values → Higher risk
    """)

with col2:
    st.subheader("💡 Practical Examples")
    st.markdown("""
    **Example 1: Hour of Day**
    - If you see red dots (late hours) on the right
    - → Late hours increase accident severity
    - → Policy: Increase patrols at night
    
    **Example 2: Number of People**
    - If the line goes up as people increase
    - → More people = More severe accidents
    - → Policy: Focus on multi-vehicle accidents
    
    **Example 3: Lighting**
    - If poor lighting (high values) pushes right
    - → Dark conditions increase risk
    - → Policy: Improve street lighting
    """)
    
    st.subheader("🎓 Key Concepts")
    st.markdown("""
    - **Positive SHAP** = Feature increases prediction
    - **Negative SHAP** = Feature decreases prediction
    - **Large magnitude** = Feature has big impact
    - **Small magnitude** = Feature has little impact
    - **Spread** = Effect varies by situation
    """)

st.divider()

# Quick reference guide
with st.expander("🔍 Quick Reference: Feature Value Meanings"):
    st.markdown("""
    ### Lighting Conditions (lum)
    - 1 = Full daylight
    - 2 = Twilight/Dawn
    - 3 = Night with street lights
    - 4 = Night without street lights
    - 5 = Unknown
    
    ### Location Type (agg)
    - 1 = Outside urban area
    - 2 = In urban area
    
    ### Intersection Type (int)
    - 1 = Outside intersection
    - 2 = X intersection
    - 3 = T intersection
    - 4 = Y intersection
    - 5 = Multiple intersections
    - 6 = Roundabout
    - 7 = Railroad crossing
    - 8 = Other
    - 9 = Unknown
    
    ### Hour
    - 0-23 (24-hour format)
    - Peak hours: 7-9 (morning), 17-19 (evening)
    
    ### Number of People (num_users)
    - Total people involved in the accident
    - Higher numbers typically mean more vehicles
    """)

with st.expander("❓ Common Questions"):
    st.markdown("""
    **Q: Why are some features more important than others?**
    A: The model learned from data that some features (like hour, lighting) have stronger relationships with accident outcomes than others.
    
    **Q: Can I trust these explanations?**
    A: SHAP is mathematically rigorous and widely used. However, it shows correlation, not causation.
    
    **Q: What should I do with this information?**
    A: Use it to:
    - Understand what drives accident severity
    - Identify high-risk conditions
    - Guide policy and intervention decisions
    - Validate that the model makes sense
    
    **Q: Why do some plots look scattered?**
    A: Scatter indicates complex interactions - the feature's effect depends on other factors.
    
    **Q: What's a "good" SHAP value?**
    A: There's no universal threshold. Compare features relative to each other within your model.
    """)
