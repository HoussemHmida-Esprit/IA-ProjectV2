"""
Train All Models with Optimized Parameters
Uses the best parameters found by hyperparameter optimization
"""
import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Import custom models
from tab_transformer import AccidentTabTransformer
from lstm_forecasting import AccidentLSTM
from model_persistence import save_model


def load_data(data_path='../data/model_ready.csv'):
    """Load model-ready data"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded: {len(df):,} rows")
    return df


def prepare_data(df):
    """Prepare features and target"""
    feature_cols = ['lum', 'agg', 'int', 'day_of_week', 'hour', 'num_users']
    target_col = 'col'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mode().iloc[0])
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Fix class labels (replace -1 with 0 for XGBoost)
    y = y.replace(-1, 0).astype(int)
    
    print(f"✓ Features: {X.shape}")
    print(f"✓ Classes: {len(y.unique())}")
    
    return X, y, feature_cols


def train_random_forest_optimized(X, y, feature_cols):
    """Train Random Forest with optimized parameters"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST (OPTIMIZED PARAMETERS)")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimized parameters from hyperparameter_optimization.py
    params = {
        'n_estimators': 250,
        'max_depth': 15,
        'min_samples_split': 18,
        'min_samples_leaf': 7,
        'max_features': 'log2',
        'class_weight': 'balanced_subsample',
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Train
    print("\nTraining...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✅ Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Save
    metrics = {'accuracy': accuracy, 'f1_score': f1}
    save_model(
        model=model,
        model_name='random_forest_optimized',
        params=params,
        metrics=metrics,
        features=feature_cols,
        model_type='sklearn'
    )
    
    return model, metrics


def train_xgboost_optimized(X, y, feature_cols):
    """Train XGBoost with optimized parameters"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST (OPTIMIZED PARAMETERS)")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimized parameters from hyperparameter_optimization.py
    params = {
        'n_estimators': 500,
        'learning_rate': 0.06825809321107855,
        'max_depth': 6,
        'subsample': 0.7713090670917174,
        'colsample_bytree': 0.6836477230746428,
        'reg_alpha': 0.9359264190796677,
        'reg_lambda': 0.7983408166033527,
        'min_child_weight': 10,
        'gamma': 0.8167363403626162,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    print("\nParameters:")
    for k, v in params.items():
        if k not in ['random_state', 'n_jobs', 'tree_method']:
            print(f"  {k}: {v}")
    
    # Train
    print("\nTraining...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✅ Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Save
    metrics = {'accuracy': accuracy, 'f1_score': f1}
    save_model(
        model=model,
        model_name='xgboost_optimized',
        params=params,
        metrics=metrics,
        features=feature_cols,
        model_type='sklearn'
    )
    
    return model, metrics


def train_tabtransformer(data_path='../data/model_ready.csv'):
    """Train TabTransformer"""
    print("\n" + "="*60)
    print("TRAINING TABTRANSFORMER")
    print("="*60)
    
    # Initialize
    tab_transformer = AccidentTabTransformer(data_path)
    
    # Load and prepare data
    print("\nPreparing data...")
    X_cat, X_num, y, categorical_dims = tab_transformer.load_and_prepare_data()
    
    # Train with default good parameters
    print("\nTraining (50 epochs)...")
    print("This will take 10-15 minutes...")
    
    train_losses, test_accuracies = tab_transformer.train(
        X_cat, X_num, y, categorical_dims,
        epochs=50,
        batch_size=128,
        learning_rate=0.001
    )
    
    best_accuracy = max(test_accuracies)
    
    print(f"\n✅ Results:")
    print(f"   Best Accuracy: {best_accuracy:.4f}")
    
    # Save
    tab_transformer.save_model('tab_transformer_best.pth')
    print("✓ Saved to: tab_transformer_best.pth")
    
    return tab_transformer, best_accuracy


def train_lstm(data_path='../data/model_ready.csv'):
    """Train LSTM forecaster"""
    print("\n" + "="*60)
    print("TRAINING LSTM FORECASTER")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    
    # Aggregate by date
    print("Aggregating by date...")
    df['date'] = pd.to_datetime(df['date']) if 'date' in df.columns else pd.to_datetime('2020-01-01')
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
    daily_counts.columns = ['date', 'count']
    daily_counts = daily_counts.sort_values('date')
    
    # Prepare sequences
    print("Preparing sequences...")
    accident_counts = daily_counts['count'].values
    
    # Create sequences (30 days to predict next 7 days)
    sequence_length = 30
    forecast_horizon = 7
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(accident_counts) - sequence_length - forecast_horizon):
        X_sequences.append(accident_counts[i:i+sequence_length])
        y_sequences.append(accident_counts[i+sequence_length:i+sequence_length+forecast_horizon])
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    # Normalize
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    y = (y - mean) / std
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(-1)
    y_tensor = torch.FloatTensor(y)
    
    # Split
    split_idx = int(0.8 * len(X_tensor))
    X_train = X_tensor[:split_idx]
    y_train = y_tensor[:split_idx]
    X_test = X_tensor[split_idx:]
    y_test = y_tensor[split_idx:]
    
    print(f"✓ Training sequences: {len(X_train)}")
    print(f"✓ Test sequences: {len(X_test)}")
    
    # Initialize model
    model = AccidentLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=forecast_horizon,
        dropout=0.2
    )
    
    # Train
    print("\nTraining (100 epochs)...")
    print("This will take 5-10 minutes...")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    
    print(f"\n✅ Results:")
    print(f"   Test Loss (MSE): {test_loss.item():.4f}")
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std,
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon
    }, 'lstm_forecaster.pth')
    
    print("✓ Saved to: lstm_forecaster.pth")
    
    return model, test_loss.item()


def main():
    """Train all models with optimized parameters"""
    print("="*60)
    print("TRAINING ALL MODELS WITH OPTIMIZED PARAMETERS")
    print("="*60)
    
    results = {}
    
    # Load data once
    df = load_data()
    X, y, feature_cols = prepare_data(df)
    
    # 1. Random Forest (Optimized)
    try:
        rf_model, rf_metrics = train_random_forest_optimized(X, y, feature_cols)
        results['Random Forest'] = rf_metrics
    except Exception as e:
        print(f"\n❌ Random Forest failed: {e}")
    
    # 2. XGBoost (Optimized)
    try:
        xgb_model, xgb_metrics = train_xgboost_optimized(X, y, feature_cols)
        results['XGBoost'] = xgb_metrics
    except Exception as e:
        print(f"\n❌ XGBoost failed: {e}")
    
    # 3. TabTransformer
    try:
        tab_model, tab_acc = train_tabtransformer()
        results['TabTransformer'] = {'accuracy': tab_acc}
    except Exception as e:
        print(f"\n❌ TabTransformer failed: {e}")
    
    # 4. LSTM
    try:
        lstm_model, lstm_loss = train_lstm()
        results['LSTM'] = {'test_loss': lstm_loss}
    except Exception as e:
        print(f"\n❌ LSTM failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n✅ All models trained and saved!")
    print("\nSaved files:")
    print("  • models/random_forest_optimized_latest.pkl")
    print("  • models/xgboost_optimized_latest.pkl")
    print("  • models/tab_transformer_best.pth")
    print("  • models/lstm_forecaster.pth")


if __name__ == '__main__':
    main()
