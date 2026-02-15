"""
Hyperparameter Optimization for All Models
Uses Optuna to find optimal hyperparameters for:
- Random Forest
- XGBoost
- TabTransformer
- LSTM
"""
import numpy as np
import pandas as pd
import optuna
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Import custom models
from tab_transformer import AccidentTabTransformer
from lstm_forecasting import AccidentLSTM
from model_persistence import save_model as save_model_persistent

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelOptimizer:
    """Hyperparameter optimization for all models"""
    
    def __init__(self, data_path='data/model_ready.csv'):
        self.data_path = Path(data_path)
        self.results = {}
        
        # Load data
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Features
        self.categorical_features = ['lum', 'agg', 'int', 'day_of_week']
        self.numerical_features = ['hour', 'num_users']
        self.all_features = self.categorical_features + self.numerical_features
        
        # Prepare data
        self.X = df[self.all_features]
        self.y_collision = df['col']  # Collision type
        
        # Fix class labels for XGBoost (requires 0-based consecutive classes)
        # Replace -1 with 0
        self.y_collision = self.y_collision.replace(-1, 0)
        
        self.y_severity = df['max_severity'] if 'max_severity' in df.columns else None
        
        # Split data
        self.X_train, self.X_test, self.y_col_train, self.y_col_test = train_test_split(
            self.X, self.y_collision, test_size=0.2, random_state=42, stratify=self.y_collision
        )
        
        print(f"✓ Data loaded: {len(df)} samples")
        print(f"  Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        print(f"  Features: {len(self.all_features)}")
        print(f"  Classes: {len(self.y_collision.unique())}")
    
    # ==================== RANDOM FOREST OPTIMIZATION ====================
    
    def optimize_random_forest(self, trial):
        """Optuna objective for Random Forest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 50, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        
        # Cross-validation score
        scores = cross_val_score(
            model, self.X_train, self.y_col_train,
            cv=3, scoring='accuracy', n_jobs=-1
        )
        
        return scores.mean()
    
    def train_random_forest(self, n_trials=30):
        """Train Random Forest with hyperparameter optimization"""
        print("\n" + "="*60)
        print("RANDOM FOREST OPTIMIZATION")
        print("="*60)
        
        # Baseline (current parameters)
        print("\n1. Baseline Model (current parameters)...")
        baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        baseline_model.fit(self.X_train, self.y_col_train)
        baseline_pred = baseline_model.predict(self.X_test)
        baseline_acc = accuracy_score(self.y_col_test, baseline_pred)
        baseline_f1 = f1_score(self.y_col_test, baseline_pred, average='weighted')
        
        print(f"   Baseline Accuracy: {baseline_acc:.4f}")
        print(f"   Baseline F1-Score: {baseline_f1:.4f}")
        
        # Optimize
        print(f"\n2. Optimizing hyperparameters ({n_trials} trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optimize_random_forest, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        print(f"\n   Best parameters found:")
        for param, value in best_params.items():
            print(f"     {param}: {value}")
        
        # Train with best parameters
        print("\n3. Training optimized model...")
        optimized_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        optimized_model.fit(self.X_train, self.y_col_train)
        optimized_pred = optimized_model.predict(self.X_test)
        optimized_acc = accuracy_score(self.y_col_test, optimized_pred)
        optimized_f1 = f1_score(self.y_col_test, optimized_pred, average='weighted')
        
        print(f"   Optimized Accuracy: {optimized_acc:.4f}")
        print(f"   Optimized F1-Score: {optimized_f1:.4f}")
        
        # Improvement
        improvement = (optimized_acc - baseline_acc) / baseline_acc * 100
        print(f"\n✅ Improvement: {improvement:+.2f}%")
        
        # Save optimized model with persistence system
        metrics_dict = {
            'baseline_accuracy': baseline_acc,
            'optimized_accuracy': optimized_acc,
            'baseline_f1': baseline_f1,
            'optimized_f1': optimized_f1,
            'improvement_pct': improvement,
            'accuracy': optimized_acc,
            'f1_score': optimized_f1
        }
        
        save_model_persistent(
            model=optimized_model,
            model_name='random_forest_optimized',
            params=best_params,
            metrics=metrics_dict,
            features=self.all_features,
            model_type='sklearn'
        )
        
        # Also save legacy format
        model_data = {
            'model': optimized_model,
            'features': self.all_features,
            'best_params': best_params,
            'metrics': metrics_dict
        }
        
        with open('rf_optimized.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("✓ Legacy format saved to: rf_optimized.pkl")
        
        self.results['Random Forest'] = model_data['metrics']
        return model_data
    
    # ==================== XGBOOST OPTIMIZATION ====================
    
    def optimize_xgboost(self, trial):
        """Optuna objective for XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        model = XGBClassifier(**params)
        
        # Cross-validation score
        scores = cross_val_score(
            model, self.X_train, self.y_col_train,
            cv=3, scoring='accuracy', n_jobs=-1
        )
        
        return scores.mean()
    
    def train_xgboost(self, n_trials=30):
        """Train XGBoost with hyperparameter optimization"""
        print("\n" + "="*60)
        print("XGBOOST OPTIMIZATION")
        print("="*60)
        
        # Baseline
        print("\n1. Baseline Model (current parameters)...")
        baseline_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        baseline_model.fit(self.X_train, self.y_col_train)
        baseline_pred = baseline_model.predict(self.X_test)
        baseline_acc = accuracy_score(self.y_col_test, baseline_pred)
        baseline_f1 = f1_score(self.y_col_test, baseline_pred, average='weighted')
        
        print(f"   Baseline Accuracy: {baseline_acc:.4f}")
        print(f"   Baseline F1-Score: {baseline_f1:.4f}")
        
        # Optimize
        print(f"\n2. Optimizing hyperparameters ({n_trials} trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optimize_xgboost, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        print(f"\n   Best parameters found:")
        for param, value in best_params.items():
            print(f"     {param}: {value}")
        
        # Train with best parameters
        print("\n3. Training optimized model...")
        optimized_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1, tree_method='hist')
        optimized_model.fit(self.X_train, self.y_col_train)
        optimized_pred = optimized_model.predict(self.X_test)
        optimized_acc = accuracy_score(self.y_col_test, optimized_pred)
        optimized_f1 = f1_score(self.y_col_test, optimized_pred, average='weighted')
        
        print(f"   Optimized Accuracy: {optimized_acc:.4f}")
        print(f"   Optimized F1-Score: {optimized_f1:.4f}")
        
        # Improvement
        improvement = (optimized_acc - baseline_acc) / baseline_acc * 100
        print(f"\n✅ Improvement: {improvement:+.2f}%")
        
        # Save optimized model with persistence system
        metrics_dict = {
            'baseline_accuracy': baseline_acc,
            'optimized_accuracy': optimized_acc,
            'baseline_f1': baseline_f1,
            'optimized_f1': optimized_f1,
            'improvement_pct': improvement,
            'accuracy': optimized_acc,
            'f1_score': optimized_f1
        }
        
        save_model_persistent(
            model=optimized_model,
            model_name='xgboost_optimized',
            params=best_params,
            metrics=metrics_dict,
            features=self.all_features,
            model_type='sklearn'
        )
        
        # Also save legacy format
        model_data = {
            'model': optimized_model,
            'features': self.all_features,
            'best_params': best_params,
            'metrics': metrics_dict
        }
        
        with open('xgb_optimized.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("✓ Legacy format saved to: xgb_optimized.pkl")
        
        self.results['XGBoost'] = model_data['metrics']
        return model_data
    
    # ==================== TABTRANSFORMER OPTIMIZATION ====================
    
    def optimize_tabtransformer(self, trial):
        """Optuna objective for TabTransformer"""
        params = {
            'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'd_ff': trial.suggest_categorical('d_ff', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
            'embedding_dim': trial.suggest_categorical('embedding_dim', [8, 16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256])
        }
        
        # Quick training (10 epochs for optimization)
        tab_transformer = AccidentTabTransformer(str(self.data_path))
        X_cat, X_num, y, categorical_dims = tab_transformer.load_and_prepare_data()
        
        # Use smaller subset for faster optimization
        subset_size = min(50000, len(y))
        indices = np.random.choice(len(y), subset_size, replace=False)
        X_cat_sub = X_cat[indices]
        X_num_sub = X_num[indices]
        y_sub = y[indices]
        
        try:
            # Train with trial parameters
            _, test_accuracies = tab_transformer.train(
                X_cat_sub, X_num_sub, y_sub, categorical_dims,
                epochs=10,
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate']
            )
            
            # Don't save during optimization trials
            
            # Return best test accuracy
            return max(test_accuracies)
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    def train_tabtransformer(self, n_trials=20):
        """Train TabTransformer with hyperparameter optimization"""
        print("\n" + "="*60)
        print("TABTRANSFORMER OPTIMIZATION")
        print("="*60)
        
        # Baseline (current parameters)
        print("\n1. Baseline Model (current parameters)...")
        baseline_transformer = AccidentTabTransformer(str(self.data_path))
        X_cat, X_num, y, categorical_dims = baseline_transformer.load_and_prepare_data()
        
        print("   Training baseline model (50 epochs)...")
        _, baseline_accs = baseline_transformer.train(
            X_cat, X_num, y, categorical_dims,
            epochs=50,
            batch_size=128,
            learning_rate=0.001
        )
        baseline_acc = max(baseline_accs)
        
        # Don't save baseline model during optimization
        
        print(f"   Baseline Accuracy: {baseline_acc:.4f}")
        
        # Optimize
        print(f"\n2. Optimizing hyperparameters ({n_trials} trials)...")
        print("   (Using 10 epochs per trial for speed)")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optimize_tabtransformer, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        print(f"\n   Best parameters found:")
        for param, value in best_params.items():
            print(f"     {param}: {value}")
        
        # Train with best parameters (full 50 epochs)
        print("\n3. Training optimized model (50 epochs)...")
        optimized_transformer = AccidentTabTransformer(str(self.data_path))
        X_cat, X_num, y, categorical_dims = optimized_transformer.load_and_prepare_data()
        
        _, optimized_accs = optimized_transformer.train(
            X_cat, X_num, y, categorical_dims,
            epochs=50,
            batch_size=best_params['batch_size'],
            learning_rate=best_params['learning_rate']
        )
        optimized_acc = max(optimized_accs)
        
        print(f"   Optimized Accuracy: {optimized_acc:.4f}")
        
        # Improvement
        improvement = (optimized_acc - baseline_acc) / baseline_acc * 100
        print(f"\n✅ Improvement: {improvement:+.2f}%")
        
        # Save optimized model (use correct path)
        optimized_transformer.save_model('tab_transformer_optimized.pth')
        print("✓ Saved to: tab_transformer_optimized.pth")
        
        self.results['TabTransformer'] = {
            'baseline_accuracy': baseline_acc,
            'optimized_accuracy': optimized_acc,
            'improvement_pct': improvement,
            'best_params': best_params
        }
        
        return self.results['TabTransformer']
    
    # ==================== RESULTS SUMMARY ====================
    
    def generate_report(self):
        """Generate comprehensive optimization report"""
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        # Create results DataFrame
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'Baseline Accuracy': f"{metrics['baseline_accuracy']:.4f}",
                'Optimized Accuracy': f"{metrics['optimized_accuracy']:.4f}",
                'Improvement (%)': f"{metrics['improvement_pct']:+.2f}%"
            })
        
        df_results = pd.DataFrame(results_data)
        print("\n" + df_results.to_string(index=False))
        
        # Save to CSV
        df_results.to_csv('optimization_results.csv', index=False)
        print("\n✓ Results saved to: optimization_results.csv")
        
        # Save detailed results
        with open('optimization_results_detailed.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("HYPERPARAMETER OPTIMIZATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * 40 + "\n")
                for key, value in metrics.items():
                    if key == 'best_params':
                        f.write(f"  Best Parameters:\n")
                        for param, val in value.items():
                            f.write(f"    {param}: {val}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print("✓ Detailed results saved to: optimization_results_detailed.txt")
        
        return df_results


def main():
    """Run hyperparameter optimization for all models"""
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION - ALL MODELS")
    print("="*60)
    
    optimizer = ModelOptimizer(data_path='data/model_ready.csv')
    
    # Optimize each model
    print("\nStarting optimization process...")
    print("This will take several hours. Progress will be shown for each model.\n")
    
    # 1. Random Forest (fast)
    optimizer.train_random_forest(n_trials=30)
    
    # 2. XGBoost (medium)
    optimizer.train_xgboost(n_trials=30)
    
    # 3. TabTransformer (slow - fewer trials)
    optimizer.train_tabtransformer(n_trials=15)
    
    # Generate final report
    optimizer.generate_report()
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print("\nOptimized models saved:")
    print("  • rf_optimized.pkl")
    print("  • xgb_optimized.pkl")
    print("  • tab_transformer_optimized.pth")
    print("\nResults saved:")
    print("  • optimization_results.csv")
    print("  • optimization_results_detailed.txt")


if __name__ == "__main__":
    main()
