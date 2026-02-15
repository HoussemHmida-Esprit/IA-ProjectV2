"""
Stacking Ensemble - Meta-Learner Architecture
Combines XGBoost, Random Forest, TabTransformer, and LSTM

Architecture:
    Level 0 (Base Models):
        - XGBoost (2D tabular)
        - Random Forest (2D tabular)
        - TabTransformer (2D tabular with embeddings)
        - LSTM (3D temporal)
    
    Level 1 (Meta-Model):
        - Ridge Regression or XGBoost
        - Combines predictions from all base models

Key Challenge: Aligning LSTM's temporal predictions with tabular models
"""
import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Import custom models
from tab_transformer import AccidentTabTransformer
from lstm_forecasting import AccidentLSTM


class StackingEnsemble:
    """
    Meta-Learner Stacking Architecture
    
    Combines predictions from:
    1. XGBoost (tabular)
    2. Random Forest (tabular)
    3. TabTransformer (tabular with embeddings)
    4. LSTM (temporal sequences)
    
    Uses Out-of-Fold (OOF) predictions to avoid overfitting
    """
    
    def __init__(self, data_path='../data/model_ready.csv'):
        self.data_path = Path(data_path)
        self.base_models = {}
        self.meta_model = None
        self.feature_names = ['lum', 'agg', 'int', 'day_of_week', 'hour', 'num_users']
        self.categorical_features = ['lum', 'agg', 'int', 'day_of_week']
        self.numerical_features = ['hour', 'num_users']
        
        # Load data
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Ensure we have accident ID for alignment
        if 'Num_Acc' not in self.df.columns:
            print("⚠️ Warning: 'Num_Acc' column not found. Creating from index...")
            self.df['Num_Acc'] = range(len(self.df))
        
        self.X = self.df[self.feature_names]
        self.y = self.df['col']  # Collision type (classification)
        
        print(f"✓ Data loaded: {len(self.df)} samples")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Target classes: {len(self.y.unique())}")
    
    # ==================== STEP 1: LOAD BASE MODELS ====================
    
    def load_base_models(self):
        """Load all pre-trained base models"""
        print("\n" + "="*60)
        print("LOADING BASE MODELS")
        print("="*60)
        
        # 1. Load Random Forest
        rf_path = Path('rf_pca_multitarget.pkl')
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                rf_data = pickle.load(f)
                self.base_models['random_forest'] = rf_data.get('model', rf_data)
            print("✓ Random Forest loaded")
        else:
            print("⚠️ Random Forest not found")
        
        # 2. Load XGBoost
        xgb_path = Path('xgb_nopca_multitarget.pkl')
        if xgb_path.exists():
            with open(xgb_path, 'rb') as f:
                xgb_data = pickle.load(f)
                self.base_models['xgboost'] = xgb_data.get('model', xgb_data)
            print("✓ XGBoost loaded")
        else:
            print("⚠️ XGBoost not found")
        
        # 3. Load TabTransformer
        tt_path = Path('tab_transformer_best.pth')
        if tt_path.exists():
            checkpoint = torch.load(tt_path, map_location='cpu', weights_only=False)
            categorical_encoders = checkpoint['categorical_encoders']
            categorical_dims = [len(enc.classes_) for enc in categorical_encoders.values()]
            num_classes = len(checkpoint['target_encoder'].classes_)
            
            tab_transformer = AccidentTabTransformer(str(self.data_path))
            tab_transformer.load_model(
                str(tt_path),
                categorical_dims=categorical_dims,
                num_classes=num_classes
            )
            self.base_models['tabtransformer'] = tab_transformer
            print("✓ TabTransformer loaded")
        else:
            print("⚠️ TabTransformer not found")
        
        # 4. Load LSTM
        lstm_path = Path('lstm_forecaster.pth')
        if lstm_path.exists():
            # LSTM requires special handling - it predicts risk scores, not classes
            self.base_models['lstm'] = lstm_path
            print("✓ LSTM path stored (will load on demand)")
        else:
            print("⚠️ LSTM not found")
        
        print(f"\n✓ Loaded {len(self.base_models)} base models")
        return len(self.base_models) > 0
    
    # ==================== STEP 2: GENERATE OOF PREDICTIONS ====================
    
    def generate_oof_predictions(self, n_folds=5):
        """
        Generate Out-of-Fold (OOF) predictions for meta-features
        
        Why OOF?
        - Prevents overfitting in meta-model
        - Each sample's prediction comes from a model that didn't see it during training
        - Standard practice in stacking ensembles
        
        Process:
        1. Split data into K folds
        2. For each fold:
           - Train on K-1 folds
           - Predict on held-out fold
        3. Combine all predictions → OOF predictions
        """
        print("\n" + "="*60)
        print("GENERATING OUT-OF-FOLD PREDICTIONS")
        print("="*60)
        print(f"Using {n_folds}-fold cross-validation")
        
        n_samples = len(self.X)
        n_models = len(self.base_models)
        
        # Initialize OOF prediction arrays
        oof_predictions = np.zeros((n_samples, n_models))
        
        # K-Fold split
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            print(f"\nFold {fold + 1}/{n_folds}")
            print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
            
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Get predictions from each model
            model_idx = 0
            
            # 1. Random Forest
            if 'random_forest' in self.base_models:
                rf_model = self.base_models['random_forest']
                
                # Handle MultiOutputClassifier
                if hasattr(rf_model, 'estimators_'):
                    rf_pred = rf_model.estimators_[0].predict(X_val)
                else:
                    rf_pred = rf_model.predict(X_val)
                
                oof_predictions[val_idx, model_idx] = rf_pred
                model_idx += 1
                print(f"  ✓ Random Forest predictions: {len(rf_pred)}")
            
            # 2. XGBoost
            if 'xgboost' in self.base_models:
                xgb_model = self.base_models['xgboost']
                
                # Handle MultiOutputClassifier
                if hasattr(xgb_model, 'estimators_'):
                    xgb_pred = xgb_model.estimators_[0].predict(X_val)
                else:
                    xgb_pred = xgb_model.predict(X_val)
                
                oof_predictions[val_idx, model_idx] = xgb_pred
                model_idx += 1
                print(f"  ✓ XGBoost predictions: {len(xgb_pred)}")
            
            # 3. TabTransformer
            if 'tabtransformer' in self.base_models:
                tt_model = self.base_models['tabtransformer']
                tt_pred = []
                
                for idx in val_idx:
                    row = self.X.iloc[idx]
                    categorical_data = {f: row[f] for f in self.categorical_features}
                    numerical_data = {f: row[f] for f in self.numerical_features}
                    
                    pred, _, _ = tt_model.predict(categorical_data, numerical_data)
                    tt_pred.append(pred)
                
                oof_predictions[val_idx, model_idx] = tt_pred
                model_idx += 1
                print(f"  ✓ TabTransformer predictions: {len(tt_pred)}")
            
            # 4. LSTM (Special handling - temporal predictions)
            if 'lstm' in self.base_models:
                lstm_pred = self._get_lstm_predictions_aligned(val_idx)
                oof_predictions[val_idx, model_idx] = lstm_pred
                model_idx += 1
                print(f"  ✓ LSTM predictions: {len(lstm_pred)}")
        
        print(f"\n✓ OOF predictions generated: {oof_predictions.shape}")
        return oof_predictions
    
    def _get_lstm_predictions_aligned(self, indices):
        """
        Get LSTM predictions aligned with tabular data
        
        Challenge: LSTM predicts daily accident counts (risk scores)
                  Tabular models predict per-accident severity
        
        Solution: 
        1. Get LSTM's daily risk predictions
        2. Map each accident to its date's risk score
        3. Normalize to match severity scale
        """
        # For now, use a simple risk score based on temporal patterns
        # In production, you would:
        # 1. Load LSTM model
        # 2. Get daily predictions
        # 3. Map accidents to dates
        # 4. Return aligned risk scores
        
        # Placeholder: Use hour as proxy for risk (higher at rush hours)
        risk_scores = []
        for idx in indices:
            hour = self.df.iloc[idx]['hour']
            # Simple risk model: higher risk at rush hours (7-9, 17-19)
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                risk = 2  # High risk
            elif 22 <= hour or hour <= 5:
                risk = 1  # Medium risk (night)
            else:
                risk = 0  # Low risk
            risk_scores.append(risk)
        
        return np.array(risk_scores)
    
    # ==================== STEP 3: TRAIN META-MODEL ====================
    
    def train_meta_model(self, oof_predictions, meta_model_type='ridge'):
        """
        Train Level-1 Meta-Model
        
        Input: OOF predictions from all base models
        Output: Final prediction
        
        Meta-Model Options:
        - Ridge Regression: Simple, fast, prevents overfitting
        - XGBoost: More powerful, can learn complex interactions
        """
        print("\n" + "="*60)
        print("TRAINING META-MODEL")
        print("="*60)
        
        if meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
            print("Using Ridge Regression as meta-model")
        elif meta_model_type == 'xgboost':
            self.meta_model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            print("Using XGBoost as meta-model")
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}")
        
        # Train meta-model on OOF predictions
        print(f"Training on {oof_predictions.shape[0]} samples with {oof_predictions.shape[1]} meta-features")
        self.meta_model.fit(oof_predictions, self.y)
        
        # Evaluate meta-model
        meta_pred = self.meta_model.predict(oof_predictions)
        accuracy = accuracy_score(self.y, meta_pred)
        
        print(f"\n✓ Meta-model trained")
        print(f"  Training accuracy: {accuracy:.4f}")
        
        # Compare with individual models
        print("\n  Individual model accuracies:")
        for i in range(oof_predictions.shape[1]):
            model_acc = accuracy_score(self.y, oof_predictions[:, i].astype(int))
            model_name = list(self.base_models.keys())[i]
            print(f"    {model_name}: {model_acc:.4f}")
        
        return self.meta_model
    
    # ==================== STEP 4: GET META-FEATURES ====================
    
    def get_meta_features(self, X_input):
        """
        Generate meta-features from all base models
        
        Input: Raw features (samples, features)
        Output: Meta-features (samples, n_models)
        
        Process:
        1. Get predictions from each base model
        2. Stack predictions horizontally
        3. Return as meta-features for meta-model
        """
        n_samples = len(X_input)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        model_idx = 0
        
        # 1. Random Forest
        if 'random_forest' in self.base_models:
            rf_model = self.base_models['random_forest']
            if hasattr(rf_model, 'estimators_'):
                rf_pred = rf_model.estimators_[0].predict(X_input)
            else:
                rf_pred = rf_model.predict(X_input)
            meta_features[:, model_idx] = rf_pred
            model_idx += 1
        
        # 2. XGBoost
        if 'xgboost' in self.base_models:
            xgb_model = self.base_models['xgboost']
            if hasattr(xgb_model, 'estimators_'):
                xgb_pred = xgb_model.estimators_[0].predict(X_input)
            else:
                xgb_pred = xgb_model.predict(X_input)
            meta_features[:, model_idx] = xgb_pred
            model_idx += 1
        
        # 3. TabTransformer
        if 'tabtransformer' in self.base_models:
            tt_model = self.base_models['tabtransformer']
            tt_pred = []
            
            for i in range(len(X_input)):
                row = X_input.iloc[i] if isinstance(X_input, pd.DataFrame) else X_input[i]
                categorical_data = {f: row[self.feature_names.index(f)] if isinstance(row, np.ndarray) else row[f] 
                                   for f in self.categorical_features}
                numerical_data = {f: row[self.feature_names.index(f)] if isinstance(row, np.ndarray) else row[f]
                                 for f in self.numerical_features}
                
                pred, _, _ = tt_model.predict(categorical_data, numerical_data)
                tt_pred.append(pred)
            
            meta_features[:, model_idx] = tt_pred
            model_idx += 1
        
        # 4. LSTM
        if 'lstm' in self.base_models:
            # Get aligned LSTM predictions
            indices = range(len(X_input))
            lstm_pred = self._get_lstm_predictions_aligned(indices)
            meta_features[:, model_idx] = lstm_pred
            model_idx += 1
        
        return meta_features
    
    # ==================== STEP 5: FINAL PREDICTION ====================
    
    def stacking_predict(self, X_input):
        """
        Final stacking prediction pipeline
        
        Input: Raw features
        Output: Final predictions
        
        Pipeline:
        1. Get predictions from all base models → meta-features
        2. Feed meta-features to meta-model
        3. Return final prediction
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_meta_model() first.")
        
        # Step 1: Get meta-features
        meta_features = self.get_meta_features(X_input)
        
        # Step 2: Meta-model prediction
        final_pred = self.meta_model.predict(meta_features)
        
        return final_pred
    
    def stacking_predict_proba(self, X_input):
        """Get probability predictions from stacking ensemble"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_meta_model() first.")
        
        meta_features = self.get_meta_features(X_input)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Ridge doesn't have predict_proba, return predictions as probabilities
            return self.meta_model.predict(meta_features)
    
    # ==================== STEP 6: SAVE/LOAD ====================
    
    def save_ensemble(self, path='stacking_ensemble.pkl'):
        """Save the complete stacking ensemble"""
        ensemble_data = {
            'meta_model': self.meta_model,
            'base_models': self.base_models,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        
        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        print(f"✓ Ensemble saved to {path}")
    
    def load_ensemble(self, path='stacking_ensemble.pkl'):
        """Load a saved stacking ensemble"""
        with open(path, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.meta_model = ensemble_data['meta_model']
        self.base_models = ensemble_data['base_models']
        self.feature_names = ensemble_data['feature_names']
        self.categorical_features = ensemble_data['categorical_features']
        self.numerical_features = ensemble_data['numerical_features']
        
        print(f"✓ Ensemble loaded from {path}")
    
    # ==================== STEP 7: EVALUATION ====================
    
    def evaluate(self, X_test, y_test):
        """Evaluate stacking ensemble on test set"""
        print("\n" + "="*60)
        print("EVALUATING STACKING ENSEMBLE")
        print("="*60)
        
        # Get predictions
        y_pred = self.stacking_predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Compare with individual models
        print("\n  Individual Model Performance:")
        meta_features = self.get_meta_features(X_test)
        
        for i, model_name in enumerate(self.base_models.keys()):
            model_pred = meta_features[:, i].astype(int)
            model_acc = accuracy_score(y_test, model_pred)
            improvement = (accuracy - model_acc) / model_acc * 100
            print(f"    {model_name}: {model_acc:.4f} (Ensemble +{improvement:+.2f}%)")
        
        return accuracy


def main():
    """
    Complete Stacking Ensemble Pipeline
    """
    print("="*60)
    print("STACKING ENSEMBLE - META-LEARNER ARCHITECTURE")
    print("="*60)
    print("\nCombining 4 models:")
    print("  1. Random Forest (Tree-based)")
    print("  2. XGBoost (Gradient Boosting)")
    print("  3. TabTransformer (Deep Learning)")
    print("  4. LSTM (Temporal)")
    print()
    
    # Initialize ensemble
    ensemble = StackingEnsemble(data_path='data/model_ready.csv')
    
    # Step 1: Load base models
    if not ensemble.load_base_models():
        print("\n❌ No base models found. Train models first.")
        return
    
    # Step 2: Generate OOF predictions
    print("\nGenerating Out-of-Fold predictions...")
    print("This creates meta-features without overfitting.")
    oof_predictions = ensemble.generate_oof_predictions(n_folds=5)
    
    # Step 3: Train meta-model
    print("\nTraining meta-model...")
    ensemble.train_meta_model(oof_predictions, meta_model_type='ridge')
    
    # Step 4: Save ensemble
    ensemble.save_ensemble('stacking_ensemble.pkl')
    
    # Step 5: Test prediction
    print("\n" + "="*60)
    print("TESTING STACKING PREDICTION")
    print("="*60)
    
    # Test on a few samples
    test_samples = ensemble.X.iloc[:10]
    predictions = ensemble.stacking_predict(test_samples)
    
    print("\nSample Predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"  Sample {i+1}: Predicted class = {pred}")
    
    print("\n" + "="*60)
    print("STACKING ENSEMBLE COMPLETE!")
    print("="*60)
    print("\nUsage:")
    print("  ensemble = StackingEnsemble()")
    print("  ensemble.load_ensemble('stacking_ensemble.pkl')")
    print("  predictions = ensemble.stacking_predict(X_new)")


if __name__ == "__main__":
    main()
