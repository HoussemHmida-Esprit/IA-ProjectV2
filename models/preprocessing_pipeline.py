"""
Preprocessing Pipeline with Proper Train/Test Isolation
Prevents data leakage by fitting only on training data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


class PreprocessingPipeline:
    """
    Preprocessing pipeline that prevents data leakage
    
    Key Principles:
    1. Fit ONLY on training data
    2. Transform both train and test using fitted transformers
    3. Never let test data influence training statistics
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, categorical_features=None, numerical_features=None):
        """
        Fit preprocessing transformers on TRAINING data only
        
        Args:
            X_train: Training features
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
        """
        print("Fitting preprocessing pipeline on TRAINING data only...")
        
        X_train = X_train.copy()
        self.feature_names = X_train.columns.tolist()
        
        # Auto-detect feature types if not provided
        if categorical_features is None:
            categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if numerical_features is None:
            numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        # 1. Fit imputers on TRAINING data
        for col in numerical_features:
            if col in X_train.columns:
                imputer = SimpleImputer(strategy='median')
                imputer.fit(X_train[[col]])
                self.imputers[col] = imputer
                print(f"  ✓ Fitted imputer for {col} (median: {imputer.statistics_[0]:.2f})")
        
        for col in categorical_features:
            if col in X_train.columns:
                imputer = SimpleImputer(strategy='most_frequent')
                imputer.fit(X_train[[col]])
                self.imputers[col] = imputer
                print(f"  ✓ Fitted imputer for {col} (mode: {imputer.statistics_[0]})")
        
        # 2. Fit scalers on TRAINING data (after imputation)
        X_train_imputed = self._apply_imputation(X_train)
        
        for col in numerical_features:
            if col in X_train_imputed.columns:
                scaler = StandardScaler()
                scaler.fit(X_train_imputed[[col]])
                self.scalers[col] = scaler
                print(f"  ✓ Fitted scaler for {col} (mean: {scaler.mean_[0]:.2f}, std: {scaler.scale_[0]:.2f})")
        
        # 3. Fit encoders on TRAINING data (for categorical features if needed)
        for col in categorical_features:
            if col in X_train_imputed.columns:
                encoder = LabelEncoder()
                encoder.fit(X_train_imputed[col].astype(str))
                self.encoders[col] = encoder
                print(f"  ✓ Fitted encoder for {col} ({len(encoder.classes_)} classes)")
        
        self.is_fitted = True
        print("✓ Preprocessing pipeline fitted on training data")
        
        return self
    
    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputers"""
        X = X.copy()
        
        for col, imputer in self.imputers.items():
            if col in X.columns:
                X[col] = imputer.transform(X[[col]])
        
        return X
    
    def transform(self, X: pd.DataFrame, scale=True, encode=False) -> pd.DataFrame:
        """
        Transform data using fitted transformers
        
        Args:
            X: Features to transform (train or test)
            scale: Whether to apply scaling
            encode: Whether to apply encoding
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        X = X.copy()
        
        # 1. Apply imputation
        X = self._apply_imputation(X)
        
        # 2. Apply scaling if requested
        if scale:
            for col, scaler in self.scalers.items():
                if col in X.columns:
                    X[col] = scaler.transform(X[[col]])
        
        # 3. Apply encoding if requested
        if encode:
            for col, encoder in self.encoders.items():
                if col in X.columns:
                    # Handle unseen categories
                    X[col] = X[col].astype(str)
                    X[col] = X[col].apply(
                        lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                    )
                    X[col] = encoder.transform(X[col])
        
        return X
    
    def fit_transform(self, X_train: pd.DataFrame, categorical_features=None, 
                     numerical_features=None, scale=True, encode=False) -> pd.DataFrame:
        """
        Fit on training data and transform it
        
        Args:
            X_train: Training features
            categorical_features: List of categorical features
            numerical_features: List of numerical features
            scale: Whether to apply scaling
            encode: Whether to apply encoding
            
        Returns:
            Transformed training features
        """
        self.fit(X_train, categorical_features, numerical_features)
        return self.transform(X_train, scale=scale, encode=encode)
    
    def save(self, path: str):
        """Save fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")
        
        pipeline_data = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"✓ Pipeline saved to: {path}")
    
    def load(self, path: str):
        """Load fitted pipeline"""
        with open(path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.scalers = pipeline_data['scalers']
        self.imputers = pipeline_data['imputers']
        self.encoders = pipeline_data['encoders']
        self.feature_names = pipeline_data['feature_names']
        self.categorical_features = pipeline_data['categorical_features']
        self.numerical_features = pipeline_data['numerical_features']
        self.is_fitted = pipeline_data['is_fitted']
        
        print(f"✓ Pipeline loaded from: {path}")
        
        return self


def demonstrate_proper_preprocessing():
    """
    Demonstrate proper train/test preprocessing without data leakage
    """
    print("="*60)
    print("PROPER PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Load data
    data_path = Path('../data/model_ready.csv')
    if not data_path.exists():
        print("⚠️ Data file not found")
        return
    
    df = pd.read_csv(data_path)
    
    # Define features
    feature_cols = ['lum', 'agg', 'int', 'hour', 'num_users', 'num_light_injury']
    target_col = 'col'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data FIRST (before any preprocessing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n1. Data Split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Create pipeline
    pipeline = PreprocessingPipeline()
    
    # Fit on TRAINING data only
    print(f"\n2. Fitting on TRAINING data:")
    categorical_features = ['lum', 'agg', 'int']
    numerical_features = ['hour', 'num_users', 'num_light_injury']
    
    X_train_processed = pipeline.fit_transform(
        X_train,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        scale=True,
        encode=False
    )
    
    # Transform TEST data using fitted pipeline
    print(f"\n3. Transforming TEST data:")
    X_test_processed = pipeline.transform(X_test, scale=True, encode=False)
    
    print(f"\n4. Results:")
    print(f"   Train processed: {X_train_processed.shape}")
    print(f"   Test processed: {X_test_processed.shape}")
    print(f"   ✓ No data leakage - test data never influenced training statistics")
    
    # Save pipeline
    pipeline.save('preprocessing_pipeline.pkl')
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Points:")
    print("  1. Split data BEFORE any preprocessing")
    print("  2. Fit transformers ONLY on training data")
    print("  3. Transform both train and test using fitted transformers")
    print("  4. Save pipeline for production use")


if __name__ == '__main__':
    demonstrate_proper_preprocessing()
