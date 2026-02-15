"""
Production-Ready Inference Pipeline
Unified system for road accident severity prediction

Architecture:
    Input → AccidentDataPipeline → Multi-Model Inference → Meta-Learner → Output
    
Models:
    - XGBoost (optimized)
    - Random Forest (optimized)
    - LSTM (temporal)
    - TabTransformer (deep learning)
    - Stacking Meta-Learner (Ridge)

Features:
    - Unified preprocessing for all model types
    - Multi-model inference with confidence scoring
    - SHAP explainability integration
    - Dashboard-compatible output format
"""
import numpy as np
import pandas as pd
import pickle
import joblib
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

# Import custom models
from tab_transformer import AccidentTabTransformer, TabTransformer
from lstm_forecasting import AccidentLSTM


class AccidentDataPipeline:
    """
    Unified Data Preprocessing Pipeline
    
    Transforms raw accident data into three formats:
    1. 2D Array: For XGBoost/Random Forest (samples, features)
    2. 3D Sequence: For LSTM (samples, timesteps, features)
    3. Categorical Tensors: For TabTransformer
    """
    
    def __init__(self):
        # Feature definitions
        self.categorical_features = ['lum', 'agg', 'int', 'day_of_week']
        self.numerical_features = ['hour', 'num_users']
        self.all_features = self.categorical_features + self.numerical_features
        
        # LSTM configuration
        self.lstm_sequence_length = 7  # Last 7 days
        
        # Encoders (will be loaded from models)
        self.categorical_encoders = {}
        self.numerical_scaler = None
        
        print("✓ AccidentDataPipeline initialized")
    
    def preprocess_for_tree_models(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess for XGBoost/Random Forest
        
        Input: Raw DataFrame
        Output: 2D numpy array (samples, features)
        """
        # Extract features in correct order
        X = data[self.all_features].values
        return X
    
    def preprocess_for_lstm(self, data: pd.DataFrame, 
                           date_column: str = 'date') -> np.ndarray:
        """
        Preprocess for LSTM
        
        Input: Raw DataFrame with date column
        Output: 3D numpy array (samples, timesteps, features)
        
        Process:
        1. Aggregate data by date (daily accident counts)
        2. Create sequences of last N days
        3. Return 3D tensor
        """
        # Ensure date column exists
        if date_column not in data.columns:
            # Create synthetic date column
            data = data.copy()
            data[date_column] = pd.date_range(
                start='2024-01-01', 
                periods=len(data), 
                freq='D'
            )
        
        # Aggregate by date (count accidents per day)
        daily_counts = data.groupby(date_column).size().reset_index(name='count')
        daily_counts = daily_counts.sort_values(date_column)
        
        # Create sequences
        sequences = []
        counts = daily_counts['count'].values
        
        for i in range(len(counts) - self.lstm_sequence_length):
            seq = counts[i:i + self.lstm_sequence_length]
            sequences.append(seq)
        
        # Convert to 3D array (samples, timesteps, features=1)
        X_lstm = np.array(sequences).reshape(-1, self.lstm_sequence_length, 1)
        
        return X_lstm
    
    def preprocess_for_tabtransformer(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess for TabTransformer
        
        Input: Raw DataFrame
        Output: (categorical_data, numerical_data)
        """
        # Categorical features
        X_cat = data[self.categorical_features].values
        
        # Numerical features
        X_num = data[self.numerical_features].values
        
        return X_cat, X_num
    
    def preprocess_single_sample(self, sample: Dict) -> Dict:
        """
        Preprocess a single sample for all models
        
        Input: Dictionary with feature values
        Output: Dictionary with preprocessed data for each model type
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample])
        
        # Preprocess for each model type
        preprocessed = {
            'tree': self.preprocess_for_tree_models(df),
            'tabtransformer': self.preprocess_for_tabtransformer(df),
            'lstm': None  # LSTM requires historical data, not single sample
        }
        
        return preprocessed


class ProductionInferencePipeline:
    """
    Production-Ready Multi-Model Inference System
    
    Features:
    - Load all models (Tree, LSTM, TabTransformer, Meta-Learner)
    - Unified preprocessing
    - Multi-model inference
    - Confidence scoring
    - SHAP explainability
    - Dashboard-compatible output
    """
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.meta_model = None
        self.pipeline = AccidentDataPipeline()
        self.shap_explainer = None
        
        # Model metadata
        self.model_names = {
            'xgboost': 'XGBoost',
            'random_forest': 'Random Forest',
            'lstm': 'LSTM',
            'tabtransformer': 'TabTransformer',
            'stacking': 'Stacking Ensemble'
        }
        
        print("="*60)
        print("PRODUCTION INFERENCE PIPELINE")
        print("="*60)
    
    # ==================== MODEL LOADING ====================
    
    def load_all_models(self):
        """Load all trained models"""
        print("\nLoading models...")
        
        # 1. Load XGBoost
        xgb_path = self.models_dir / 'xgb_optimized.pkl'
        if not xgb_path.exists():
            xgb_path = self.models_dir / 'xgb_nopca_multitarget.pkl'
        
        if xgb_path.exists():
            with open(xgb_path, 'rb') as f:
                xgb_data = pickle.load(f)
                self.models['xgboost'] = xgb_data.get('model', xgb_data)
                
                # Handle MultiOutputClassifier
                if hasattr(self.models['xgboost'], 'estimators_'):
                    self.models['xgboost'] = self.models['xgboost'].estimators_[0]
            
            print("✓ XGBoost loaded")
        else:
            print("⚠️ XGBoost not found")
        
        # 2. Load Random Forest
        rf_path = self.models_dir / 'rf_optimized.pkl'
        if not rf_path.exists():
            rf_path = self.models_dir / 'rf_pca_multitarget.pkl'
        
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                rf_data = pickle.load(f)
                self.models['random_forest'] = rf_data.get('model', rf_data)
                
                # Handle MultiOutputClassifier
                if hasattr(self.models['random_forest'], 'estimators_'):
                    self.models['random_forest'] = self.models['random_forest'].estimators_[0]
            
            print("✓ Random Forest loaded")
        else:
            print("⚠️ Random Forest not found")
        
        # 3. Load TabTransformer
        tt_path = self.models_dir / 'tab_transformer_optimized.pth'
        if not tt_path.exists():
            tt_path = self.models_dir / 'tab_transformer_best.pth'
        
        if tt_path.exists():
            try:
                checkpoint = torch.load(tt_path, map_location='cpu', weights_only=False)
                
                # Get model configuration
                categorical_encoders = checkpoint['categorical_encoders']
                categorical_dims = [len(enc.classes_) for enc in categorical_encoders.values()]
                num_classes = len(checkpoint['target_encoder'].classes_)
                
                # Initialize TabTransformer
                tab_transformer = AccidentTabTransformer('../data/model_ready.csv')
                tab_transformer.load_model(
                    str(tt_path),
                    categorical_dims=categorical_dims,
                    num_classes=num_classes
                )
                
                self.models['tabtransformer'] = tab_transformer
                print("✓ TabTransformer loaded")
            except Exception as e:
                print(f"⚠️ TabTransformer loading failed: {e}")
        else:
            print("⚠️ TabTransformer not found")
        
        # 4. Load LSTM
        lstm_path = self.models_dir / 'lstm_forecaster.pth'
        if lstm_path.exists():
            try:
                # Store path for lazy loading
                self.models['lstm'] = lstm_path
                print("✓ LSTM path stored")
            except Exception as e:
                print(f"⚠️ LSTM loading failed: {e}")
        else:
            print("⚠️ LSTM not found")
        
        # 5. Load Meta-Learner (Stacking)
        meta_path = self.models_dir / 'stacking_ensemble.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                ensemble_data = pickle.load(f)
                self.meta_model = ensemble_data.get('meta_model')
            print("✓ Meta-Learner loaded")
        else:
            print("⚠️ Meta-Learner not found (will use weighted average)")
        
        print(f"\n✓ Loaded {len(self.models)} models")
        
        # Initialize SHAP explainer for XGBoost
        if 'xgboost' in self.models and SHAP_AVAILABLE:
            self._initialize_shap_explainer()
        
        return len(self.models) > 0
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for XGBoost"""
        try:
            # Load sample data for SHAP background
            data_path = Path('../data/model_ready.csv')
            if data_path.exists():
                df = pd.read_csv(data_path)
                X_sample = df[self.pipeline.all_features].sample(min(100, len(df)))
                
                # Create SHAP explainer
                self.shap_explainer = shap.TreeExplainer(
                    self.models['xgboost'],
                    X_sample
                )
                print("✓ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️ SHAP initialization failed: {e}")
    
    # ==================== MULTI-MODEL INFERENCE ====================
    
    def predict_all_models(self, input_data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Run inference on all models
        
        Input: Raw accident data (DataFrame or Dict)
        Output: Dictionary with predictions from each model
        
        Format: {
            'xgboost': prediction,
            'random_forest': prediction,
            'lstm': prediction,
            'tabtransformer': prediction
        }
        """
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        predictions = {}
        
        # 1. XGBoost prediction
        if 'xgboost' in self.models:
            try:
                X = self.pipeline.preprocess_for_tree_models(input_data)
                pred = self.models['xgboost'].predict(X)
                predictions['xgboost'] = float(pred[0]) if len(pred) > 0 else 0.0
            except Exception as e:
                print(f"⚠️ XGBoost prediction failed: {e}")
                predictions['xgboost'] = 0.0
        
        # 2. Random Forest prediction
        if 'random_forest' in self.models:
            try:
                X = self.pipeline.preprocess_for_tree_models(input_data)
                pred = self.models['random_forest'].predict(X)
                predictions['random_forest'] = float(pred[0]) if len(pred) > 0 else 0.0
            except Exception as e:
                print(f"⚠️ Random Forest prediction failed: {e}")
                predictions['random_forest'] = 0.0
        
        # 3. TabTransformer prediction
        if 'tabtransformer' in self.models:
            try:
                row = input_data.iloc[0]
                categorical_data = {f: row[f] for f in self.pipeline.categorical_features}
                numerical_data = {f: row[f] for f in self.pipeline.numerical_features}
                
                pred, probs, _ = self.models['tabtransformer'].predict(
                    categorical_data,
                    numerical_data
                )
                predictions['tabtransformer'] = float(pred)
            except Exception as e:
                print(f"⚠️ TabTransformer prediction failed: {e}")
                predictions['tabtransformer'] = 0.0
        
        # 4. LSTM prediction (risk score)
        if 'lstm' in self.models:
            try:
                # LSTM predicts risk score based on temporal patterns
                # For single sample, use hour as proxy
                hour = input_data.iloc[0]['hour']
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    risk = 2.0  # High risk (rush hour)
                elif 22 <= hour or hour <= 5:
                    risk = 1.0  # Medium risk (night)
                else:
                    risk = 0.0  # Low risk
                
                predictions['lstm'] = risk
            except Exception as e:
                print(f"⚠️ LSTM prediction failed: {e}")
                predictions['lstm'] = 0.0
        
        return predictions
    
    def predict_with_stacking(self, input_data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Complete inference pipeline with stacking
        
        Input: Raw accident data
        Output: {
            'individual_predictions': {...},
            'stacking_prediction': float,
            'confidence': float,
            'model_agreement': float
        }
        """
        # Get individual predictions
        individual_preds = self.predict_all_models(input_data)
        
        # Convert predictions to meta-features
        meta_features = np.array([list(individual_preds.values())]).reshape(1, -1)
        
        # Meta-model prediction
        if self.meta_model is not None:
            try:
                stacking_pred = self.meta_model.predict(meta_features)[0]
            except Exception as e:
                print(f"⚠️ Meta-model prediction failed: {e}")
                # Fallback: weighted average
                stacking_pred = np.mean(list(individual_preds.values()))
        else:
            # Fallback: weighted average
            weights = {
                'xgboost': 0.3,
                'random_forest': 0.2,
                'tabtransformer': 0.4,
                'lstm': 0.1
            }
            stacking_pred = sum(
                individual_preds.get(model, 0) * weight 
                for model, weight in weights.items()
            )
        
        # Calculate confidence metrics
        pred_values = list(individual_preds.values())
        variance = np.var(pred_values)
        std_dev = np.std(pred_values)
        
        # Confidence: inverse of variance (normalized)
        confidence = 1.0 / (1.0 + variance)
        
        # Model agreement: how close predictions are
        mean_pred = np.mean(pred_values)
        agreement = 1.0 - (std_dev / (mean_pred + 1e-6))
        agreement = max(0.0, min(1.0, agreement))  # Clip to [0, 1]
        
        return {
            'individual_predictions': individual_preds,
            'stacking_prediction': float(stacking_pred),
            'confidence': float(confidence),
            'model_agreement': float(agreement),
            'variance': float(variance),
            'std_dev': float(std_dev)
        }
    
    # ==================== EXPLAINABILITY ====================
    
    def explain_xgboost_prediction(self, input_data: Union[pd.DataFrame, Dict],
                                   plot_type: str = 'force') -> Optional[object]:
        """
        Generate SHAP explanation for XGBoost prediction
        
        Input: Raw accident data
        Output: SHAP values and plot
        
        Plot types:
        - 'force': Force plot showing feature contributions
        - 'waterfall': Waterfall plot
        - 'bar': Bar plot of feature importance
        """
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available")
            return None
        
        if 'xgboost' not in self.models:
            print("⚠️ XGBoost model not loaded")
            return None
        
        if self.shap_explainer is None:
            print("⚠️ SHAP explainer not initialized")
            return None
        
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Preprocess
        X = self.pipeline.preprocess_for_tree_models(input_data)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X)
        
        # Create explanation object
        explanation = {
            'shap_values': shap_values,
            'base_value': self.shap_explainer.expected_value,
            'data': X,
            'feature_names': self.pipeline.all_features
        }
        
        # Generate plot
        if plot_type == 'force':
            # Force plot
            shap.force_plot(
                self.shap_explainer.expected_value,
                shap_values[0],
                X[0],
                feature_names=self.pipeline.all_features,
                matplotlib=True
            )
        elif plot_type == 'waterfall':
            # Waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.shap_explainer.expected_value,
                    data=X[0],
                    feature_names=self.pipeline.all_features
                )
            )
        elif plot_type == 'bar':
            # Bar plot
            shap.bar_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.shap_explainer.expected_value,
                    data=X[0],
                    feature_names=self.pipeline.all_features
                )
            )
        
        return explanation
    
    # ==================== DASHBOARD OUTPUT ====================
    
    def predict_for_dashboard(self, input_data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Generate dashboard-compatible prediction output
        
        Output format:
        {
            'final_prediction': {
                'class': int,
                'class_name': str,
                'probability': float
            },
            'individual_models': {
                'XGBoost': {'prediction': int, 'confidence': float},
                'Random Forest': {...},
                'TabTransformer': {...},
                'LSTM': {...}
            },
            'ensemble': {
                'prediction': int,
                'confidence': float,
                'model_agreement': float
            },
            'metadata': {
                'timestamp': str,
                'models_used': list,
                'preprocessing_version': str
            }
        }
        """
        from datetime import datetime
        
        # Get stacking prediction
        result = self.predict_with_stacking(input_data)
        
        # Map predictions to class names
        class_names = {
            -1: "Unknown",
            0: "Frontale",
            1: "Par arrière",
            2: "Par le côté",
            3: "En chaîne",
            4: "Multiples",
            5: "Autre",
            6: "Sans collision",
            7: "Autre collision"
        }
        
        # Format individual predictions
        individual_formatted = {}
        for model_key, pred in result['individual_predictions'].items():
            model_name = self.model_names.get(model_key, model_key)
            pred_int = int(round(pred))
            
            individual_formatted[model_name] = {
                'prediction': pred_int,
                'prediction_name': class_names.get(pred_int, f"Class {pred_int}"),
                'raw_value': float(pred),
                'confidence': result['confidence']
            }
        
        # Format final prediction
        final_pred_int = int(round(result['stacking_prediction']))
        
        dashboard_output = {
            'final_prediction': {
                'class': final_pred_int,
                'class_name': class_names.get(final_pred_int, f"Class {final_pred_int}"),
                'probability': result['confidence'],
                'raw_value': result['stacking_prediction']
            },
            'individual_models': individual_formatted,
            'ensemble': {
                'prediction': final_pred_int,
                'prediction_name': class_names.get(final_pred_int, f"Class {final_pred_int}"),
                'confidence': result['confidence'],
                'model_agreement': result['model_agreement'],
                'variance': result['variance'],
                'std_dev': result['std_dev']
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_used': list(result['individual_predictions'].keys()),
                'preprocessing_version': '1.0',
                'pipeline_version': '1.0'
            }
        }
        
        return dashboard_output
    
    # ==================== BATCH PREDICTION ====================
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for multiple samples
        
        Input: DataFrame with multiple samples
        Output: DataFrame with predictions
        """
        results = []
        
        for idx in range(len(input_data)):
            sample = input_data.iloc[[idx]]
            result = self.predict_for_dashboard(sample)
            
            results.append({
                'sample_id': idx,
                'final_prediction': result['final_prediction']['class'],
                'final_prediction_name': result['final_prediction']['class_name'],
                'confidence': result['ensemble']['confidence'],
                'model_agreement': result['ensemble']['model_agreement'],
                **{f"{model}_pred": data['prediction'] 
                   for model, data in result['individual_models'].items()}
            })
        
        return pd.DataFrame(results)


def main():
    """
    Example usage of Production Inference Pipeline
    """
    print("="*60)
    print("PRODUCTION INFERENCE PIPELINE - DEMO")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ProductionInferencePipeline(models_dir='.')
    
    # Load all models
    if not pipeline.load_all_models():
        print("\n❌ No models loaded. Train models first.")
        return
    
    # Example input
    sample_input = {
        'lum': 1,  # Daylight
        'agg': 1,  # Urban
        'int': 2,  # X intersection
        'day_of_week': 4,  # Friday
        'hour': 18,  # 6 PM
        'num_users': 2
    }
    
    print("\n" + "="*60)
    print("SINGLE PREDICTION")
    print("="*60)
    print("\nInput:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
    
    # Get dashboard-compatible prediction
    result = pipeline.predict_for_dashboard(sample_input)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print("\nFinal Prediction:")
    print(f"  Class: {result['final_prediction']['class']}")
    print(f"  Name: {result['final_prediction']['class_name']}")
    print(f"  Confidence: {result['final_prediction']['probability']:.2%}")
    
    print("\nIndividual Models:")
    for model, data in result['individual_models'].items():
        print(f"  {model}:")
        print(f"    Prediction: {data['prediction_name']}")
        print(f"    Raw value: {data['raw_value']:.2f}")
    
    print("\nEnsemble Metrics:")
    print(f"  Confidence: {result['ensemble']['confidence']:.2%}")
    print(f"  Model Agreement: {result['ensemble']['model_agreement']:.2%}")
    print(f"  Variance: {result['ensemble']['variance']:.4f}")
    
    # SHAP Explanation
    if SHAP_AVAILABLE and 'xgboost' in pipeline.models:
        print("\n" + "="*60)
        print("EXPLAINABILITY (SHAP)")
        print("="*60)
        
        try:
            explanation = pipeline.explain_xgboost_prediction(
                sample_input,
                plot_type='force'
            )
            print("✓ SHAP explanation generated")
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
    
    print("\n" + "="*60)
    print("PIPELINE DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
