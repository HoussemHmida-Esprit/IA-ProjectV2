"""
Optimized Stacking Ensemble with Hyperparameter Tuning
Uses Optuna to find the best configuration for the stacking ensemble
"""
import numpy as np
import pandas as pd
import pickle
import optuna
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptimizedStackingEnsemble:
    """
    Optimized Stacking Ensemble for Accident Prediction
    
    Optimizes:
    1. Base model hyperparameters
    2. Meta-model selection and hyperparameters
    3. Number of base models to use
    """
    
    def __init__(self, data_path='../data/model_ready.csv'):
        self.data_path = Path(data_path)
        
        # Load data
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Features
        self.feature_names = ['lum', 'agg', 'int', 'hour', 'num_users', 'num_light_injury']
        
        # Prepare data
        self.X = df[self.feature_names]
        self.y = df['col']
        
        # Fix class labels (ensure 0-based consecutive)
        self.y = self.y.replace(-1, 0)
        
        print(f"✓ Data loaded: {len(df)} samples")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Classes: {len(self.y.unique())}")
        
        self.best_config = None
        self.best_models = None
    
    # ==================== BASE MODEL OPTIMIZATION ====================
    
    def create_base_models(self, trial):
        """
        Create base models with trial-suggested hyperparameters
        """
        base_models = []
        
        # 1. Random Forest
        if trial.suggest_categorical('use_rf', [True, False]):
            rf = RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n_estimators', 50, 300, step=50),
                max_depth=trial.suggest_int('rf_max_depth', 10, 40, step=5),
                min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            base_models.append(('rf', rf))
        
        # 2. XGBoost
        if trial.suggest_categorical('use_xgb', [True, False]):
            xgb = XGBClassifier(
                n_estimators=trial.suggest_int('xgb_n_estimators', 100, 500, step=50),
                learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                max_depth=trial.suggest_int('xgb_max_depth', 3, 12),
                subsample=trial.suggest_float('xgb_subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            base_models.append(('xgb', xgb))
        
        # 3. LightGBM
        if trial.suggest_categorical('use_lgbm', [True, False]):
            lgbm = LGBMClassifier(
                n_estimators=trial.suggest_int('lgbm_n_estimators', 100, 500, step=50),
                learning_rate=trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True),
                max_depth=trial.suggest_int('lgbm_max_depth', 3, 12),
                num_leaves=trial.suggest_int('lgbm_num_leaves', 20, 100),
                subsample=trial.suggest_float('lgbm_subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            base_models.append(('lgbm', lgbm))
        
        # 4. Gradient Boosting
        if trial.suggest_categorical('use_gb', [True, False]):
            gb = GradientBoostingClassifier(
                n_estimators=trial.suggest_int('gb_n_estimators', 50, 200, step=25),
                learning_rate=trial.suggest_float('gb_learning_rate', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('gb_max_depth', 3, 10),
                subsample=trial.suggest_float('gb_subsample', 0.6, 1.0),
                random_state=42
            )
            base_models.append(('gb', gb))
        
        # Ensure at least 2 base models
        if len(base_models) < 2:
            # Add default RF and XGB
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ]
        
        return base_models
    
    def create_meta_model(self, trial):
        """
        Create meta-model with trial-suggested hyperparameters
        """
        meta_type = trial.suggest_categorical('meta_type', ['ridge', 'logistic', 'xgb', 'lgbm'])
        
        if meta_type == 'ridge':
            return Ridge(
                alpha=trial.suggest_float('ridge_alpha', 0.1, 10.0, log=True)
            )
        
        elif meta_type == 'logistic':
            return LogisticRegression(
                C=trial.suggest_float('logistic_C', 0.01, 10.0, log=True),
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        
        elif meta_type == 'xgb':
            return XGBClassifier(
                n_estimators=trial.suggest_int('meta_xgb_n_estimators', 50, 200, step=25),
                learning_rate=trial.suggest_float('meta_xgb_lr', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('meta_xgb_depth', 2, 6),
                random_state=42,
                n_jobs=-1
            )
        
        else:  # lgbm
            return LGBMClassifier(
                n_estimators=trial.suggest_int('meta_lgbm_n_estimators', 50, 200, step=25),
                learning_rate=trial.suggest_float('meta_lgbm_lr', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('meta_lgbm_depth', 2, 6),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
    
    # ==================== STACKING WITH OOF ====================
    
    def evaluate_stacking(self, base_models, meta_model, n_folds=3):
        """
        Evaluate stacking ensemble using Out-of-Fold predictions
        """
        n_samples = len(self.X)
        n_base_models = len(base_models)
        
        # Initialize OOF predictions
        oof_predictions = np.zeros((n_samples, n_base_models))
        
        # K-Fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Train and predict with each base model
            for i, (name, model) in enumerate(base_models):
                try:
                    model.fit(X_train, y_train)
                    
                    # Get predictions (probabilities if available)
                    if hasattr(model, 'predict_proba'):
                        # Use max probability as prediction strength
                        pred_proba = model.predict_proba(X_val)
                        oof_predictions[val_idx, i] = pred_proba.max(axis=1)
                    else:
                        oof_predictions[val_idx, i] = model.predict(X_val)
                except Exception as e:
                    # If model fails, use zeros
                    oof_predictions[val_idx, i] = 0
        
        # Train meta-model on OOF predictions
        try:
            meta_model.fit(oof_predictions, self.y)
            
            # Get final predictions
            if hasattr(meta_model, 'predict'):
                final_pred = meta_model.predict(oof_predictions)
            else:
                final_pred = (meta_model.predict(oof_predictions) > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y, final_pred)
            
            return accuracy
        
        except Exception as e:
            return 0.0
    
    # ==================== OPTUNA OBJECTIVE ====================
    
    def objective(self, trial):
        """
        Optuna objective function
        """
        # Create base models
        base_models = self.create_base_models(trial)
        
        # Create meta-model
        meta_model = self.create_meta_model(trial)
        
        # Evaluate stacking
        accuracy = self.evaluate_stacking(base_models, meta_model, n_folds=3)
        
        return accuracy
    
    # ==================== OPTIMIZATION ====================
    
    def optimize(self, n_trials=50):
        """
        Run hyperparameter optimization
        """
        print("\n" + "="*60)
        print("OPTIMIZING STACKING ENSEMBLE")
        print("="*60)
        print(f"Running {n_trials} trials...")
        print("This will take a while...\n")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best configuration
        self.best_config = study.best_params
        best_accuracy = study.best_value
        
        print(f"\n✅ Optimization complete!")
        print(f"   Best accuracy: {best_accuracy:.4f}")
        print(f"\n   Best configuration:")
        
        # Print configuration in organized way
        print("\n   Base Models:")
        for key, value in self.best_config.items():
            if key.startswith('use_'):
                model_name = key.replace('use_', '').upper()
                print(f"     {model_name}: {'✓' if value else '✗'}")
        
        print("\n   Meta-Model:")
        print(f"     Type: {self.best_config.get('meta_type', 'N/A')}")
        
        print("\n   Hyperparameters:")
        for key, value in sorted(self.best_config.items()):
            if not key.startswith('use_') and key != 'meta_type':
                print(f"     {key}: {value}")
        
        return self.best_config, best_accuracy
    
    # ==================== TRAIN FINAL MODEL ====================
    
    def train_final_model(self, config=None):
        """
        Train final stacking ensemble with best configuration
        """
        if config is None:
            if self.best_config is None:
                raise ValueError("No configuration provided. Run optimize() first.")
            config = self.best_config
        
        print("\n" + "="*60)
        print("TRAINING FINAL STACKING ENSEMBLE")
        print("="*60)
        
        # Recreate models with best config
        # This is a simplified version - you'd need to parse config properly
        base_models = []
        
        if config.get('use_rf', False):
            rf = RandomForestClassifier(
                n_estimators=config.get('rf_n_estimators', 100),
                max_depth=config.get('rf_max_depth', 20),
                min_samples_split=config.get('rf_min_samples_split', 10),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            base_models.append(('rf', rf))
            print("✓ Random Forest added")
        
        if config.get('use_xgb', False):
            xgb = XGBClassifier(
                n_estimators=config.get('xgb_n_estimators', 100),
                learning_rate=config.get('xgb_learning_rate', 0.1),
                max_depth=config.get('xgb_max_depth', 6),
                subsample=config.get('xgb_subsample', 0.8),
                colsample_bytree=config.get('xgb_colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            base_models.append(('xgb', xgb))
            print("✓ XGBoost added")
        
        if config.get('use_lgbm', False):
            lgbm = LGBMClassifier(
                n_estimators=config.get('lgbm_n_estimators', 100),
                learning_rate=config.get('lgbm_learning_rate', 0.1),
                max_depth=config.get('lgbm_max_depth', 6),
                num_leaves=config.get('lgbm_num_leaves', 31),
                subsample=config.get('lgbm_subsample', 0.8),
                colsample_bytree=config.get('lgbm_colsample_bytree', 0.8),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            base_models.append(('lgbm', lgbm))
            print("✓ LightGBM added")
        
        if config.get('use_gb', False):
            gb = GradientBoostingClassifier(
                n_estimators=config.get('gb_n_estimators', 100),
                learning_rate=config.get('gb_learning_rate', 0.1),
                max_depth=config.get('gb_max_depth', 5),
                subsample=config.get('gb_subsample', 0.8),
                random_state=42
            )
            base_models.append(('gb', gb))
            print("✓ Gradient Boosting added")
        
        # Create meta-model
        meta_type = config.get('meta_type', 'ridge')
        if meta_type == 'ridge':
            meta_model = Ridge(alpha=config.get('ridge_alpha', 1.0))
        elif meta_type == 'logistic':
            meta_model = LogisticRegression(
                C=config.get('logistic_C', 1.0),
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif meta_type == 'xgb':
            meta_model = XGBClassifier(
                n_estimators=config.get('meta_xgb_n_estimators', 100),
                learning_rate=config.get('meta_xgb_lr', 0.1),
                max_depth=config.get('meta_xgb_depth', 3),
                random_state=42,
                n_jobs=-1
            )
        else:
            meta_model = LGBMClassifier(
                n_estimators=config.get('meta_lgbm_n_estimators', 100),
                learning_rate=config.get('meta_lgbm_lr', 0.1),
                max_depth=config.get('meta_lgbm_depth', 3),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        print(f"✓ Meta-model: {meta_type}")
        
        # Train all base models on full data
        print("\nTraining base models...")
        for name, model in base_models:
            model.fit(self.X, self.y)
            print(f"  ✓ {name} trained")
        
        # Generate meta-features
        print("\nGenerating meta-features...")
        meta_features = np.zeros((len(self.X), len(base_models)))
        for i, (name, model) in enumerate(base_models):
            if hasattr(model, 'predict_proba'):
                meta_features[:, i] = model.predict_proba(self.X).max(axis=1)
            else:
                meta_features[:, i] = model.predict(self.X)
        
        # Train meta-model
        print("Training meta-model...")
        meta_model.fit(meta_features, self.y)
        
        # Store models
        self.best_models = {
            'base_models': base_models,
            'meta_model': meta_model,
            'config': config,
            'feature_names': self.feature_names
        }
        
        # Evaluate
        final_pred = meta_model.predict(meta_features)
        accuracy = accuracy_score(self.y, final_pred)
        f1 = f1_score(self.y, final_pred, average='weighted')
        
        print(f"\n✅ Final model trained!")
        print(f"   Training accuracy: {accuracy:.4f}")
        print(f"   Training F1-score: {f1:.4f}")
        
        return self.best_models
    
    # ==================== SAVE/LOAD ====================
    
    def save_model(self, path='stacking_ensemble_optimized.pkl'):
        """Save optimized stacking ensemble"""
        if self.best_models is None:
            raise ValueError("No model trained. Run train_final_model() first.")
        
        with open(path, 'wb') as f:
            pickle.dump(self.best_models, f)
        
        print(f"\n✓ Model saved to: {path}")


def main():
    """
    Run complete optimization pipeline
    """
    print("="*60)
    print("STACKING ENSEMBLE HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Initialize
    optimizer = OptimizedStackingEnsemble(data_path='../data/model_ready.csv')
    
    # Optimize
    best_config, best_accuracy = optimizer.optimize(n_trials=30)
    
    # Train final model
    optimizer.train_final_model()
    
    # Save
    optimizer.save_model('stacking_ensemble_optimized.pkl')
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
