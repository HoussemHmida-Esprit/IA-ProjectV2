"""
Model Training Script for Road Accident Collision Type Prediction

This script trains a Random Forest classifier to predict collision types
based on accident conditions (lighting, weather, location, time features).

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Import persistence manager
from model_persistence import save_model


# Label mappings for collision types
COLLISION_TYPE_LABELS = {
    1: 'Two vehicles - frontal',
    2: 'Two vehicles - rear',
    3: 'Two vehicles - side',
    4: 'Three+ vehicles - chain',
    5: 'Three+ vehicles - multiple',
    6: 'Other collision',
    7: 'No collision'
}


def load_data(data_path: str = '../data/model_ready.csv') -> pd.DataFrame:
    """Load model-ready data from CSV file.
    
    Args:
        data_path: Path to the model_ready.csv file
        
    Returns:
        DataFrame with model-ready data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Please run the preprocessing notebook first."
        )
    
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features (X) and target (y) for modeling.
    
    Features selected based on Requirements 4.2:
    - lum: Lighting conditions (1-5)
    - agg: Urban/rural (1-2)
    - int: Intersection type (1-9)
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0-6)
    - num_users: Number of people involved
    
    Target variable (Requirements 4.1):
    - col: Collision type (1-7)
    
    Args:
        df: DataFrame with model-ready data
        
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    # Define feature columns (based on actual data)
    feature_cols = ['lum', 'agg', 'int', 'hour', 'day_of_week', 'num_users']
    target_col = 'col'
    
    # Verify all columns exist
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Select features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.mode().iloc[0])
    
    # Drop rows where target is missing
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Convert target to integer
    y = y.astype(int)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """Split data into training and test sets with stratification.
    
    Requirements 4.3: Split data 80/20 with stratification
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for test set (default 0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Stratified sampling to maintain class distribution
    )
    
    print(f"\nTrain-Test Split:")
    print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """Train Random Forest classifier with balanced class weights.
    
    Requirements 4.4: Train model with balanced class weights and basic hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        Trained RandomForestClassifier model
    """
    print("\nTraining Random Forest classifier...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  class_weight: balanced")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',  # Handle class imbalance (Requirements 4.4)
        max_depth=15,  # Basic hyperparameter tuning
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    print("  Training complete!")
    
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, 
                   y_test: pd.Series) -> dict:
    """Evaluate model performance and generate metrics.
    
    Requirements 4.5: Calculate accuracy, precision, recall, F1-score
    Requirements 4.6: Generate confusion matrix and classification report
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics (Requirements 4.5)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Generate confusion matrix (Requirements 4.6)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generate classification report (Requirements 4.6)
    class_report = classification_report(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    # Print results
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    
    print(f"\nClassification Report:")
    print(class_report)
    
    return metrics


def save_model_legacy(model: RandomForestClassifier, model_path: str = 'models/collision_model.pkl',
               labels_path: str = 'models/collision_labels.pkl') -> None:
    """Save trained model and label mappings to files (legacy method).
    
    Requirements 4.7: Save model to models/collision_model.pkl
    
    Args:
        model: Trained model to save
        model_path: Path to save the model
        labels_path: Path to save the label mappings
    """
    # Ensure directory exists
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save label mappings for collision types
    with open(labels_path, 'wb') as f:
        pickle.dump(COLLISION_TYPE_LABELS, f)
    print(f"Labels saved to: {labels_path}")


def load_model(model_path: str = 'models/collision_model.pkl') -> RandomForestClassifier:
    """Load trained model from file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded RandomForestClassifier model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please train the model first."
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def load_labels(labels_path: str = 'models/collision_labels.pkl') -> dict:
    """Load collision type label mappings from file.
    
    Args:
        labels_path: Path to the saved labels
        
    Returns:
        Dictionary mapping collision codes to descriptions
    """
    path = Path(labels_path)
    if not path.exists():
        # Return default labels if file doesn't exist
        return COLLISION_TYPE_LABELS
    
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    return labels


def main():
    """Main function to run the complete model training pipeline."""
    print("=" * 60)
    print("COLLISION TYPE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load data (Requirements 4.1, 4.2)
    print("\n[Step 1] Loading data...")
    df = load_data()
    
    # Step 2: Prepare features and target
    print("\n[Step 2] Preparing features...")
    X, y = prepare_features(df)
    
    # Step 3: Split data (Requirements 4.3)
    print("\n[Step 3] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train model (Requirements 4.4)
    print("\n[Step 4] Training model...")
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model (Requirements 4.5, 4.6)
    print("\n[Step 5] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 6: Save model (Requirements 4.7)
    print("\n[Step 6] Saving model...")
    
    # Save with new persistence system
    params = {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    }
    
    feature_cols = ['lum', 'agg', 'int', 'hour', 'day_of_week', 'num_users']
    
    # Convert numpy arrays to lists for JSON serialization
    metrics_for_save = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score']
    }
    
    save_model(
        model=model,
        model_name='random_forest_baseline',
        params=params,
        metrics=metrics_for_save,
        features=feature_cols,
        model_type='sklearn',
        additional_info={'label_mapping': COLLISION_TYPE_LABELS}
    )
    
    # Also save legacy format for backward compatibility
    save_model_legacy(model)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, metrics


if __name__ == '__main__':
    main()
