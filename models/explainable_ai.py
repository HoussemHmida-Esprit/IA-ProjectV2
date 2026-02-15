"""
Explainable AI Module using SHAP
Provides interpretability for XGBoost models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

# Feature name mappings for better visualization
FEATURE_NAMES = {
    'lum': 'Lighting',
    'atm': 'Weather',
    'agg': 'Location',
    'int': 'Intersection',
    'hour': 'Hour',
    'day_of_week': 'Day of Week',
    'month': 'Month',
    'num_users': 'People Involved',
    'num_light_injury': 'Light Injuries',
    'num_serious_injury': 'Serious Injuries',
    'num_killed': 'Fatalities'
}


class AccidentXAI:
    """Explainable AI for accident prediction models"""
    
    def __init__(self, model_path: str, data_path: str):
        """
        Initialize XAI module
        
        Args:
            model_path: Path to trained model (.pkl)
            data_path: Path to model-ready data (.csv)
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X = None
        self.y = None
        
    def load_model_and_data(self):
        """Load trained model and data"""
        # Load model
        with open(self.model_path, 'rb') as f:
            model_data = joblib.load(f)
        
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.feature_names = model_data.get('features', None)
        else:
            self.model = model_data
            self.feature_names = None
        
        # Handle MultiOutputClassifier - extract first estimator for collision prediction
        if hasattr(self.model, 'estimators_'):
            print("Detected MultiOutputClassifier - using first estimator (collision type)")
            self.model = self.model.estimators_[0]
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Define features - use model's features if available, otherwise default
        if self.feature_names is None:
            # Try common feature sets
            possible_features = ['lum', 'atm', 'agg', 'int', 'hour', 'day_of_week', 'month', 'num_users', 'num_light_injury']
            self.feature_names = [f for f in possible_features if f in df.columns]
            
            if not self.feature_names:
                raise ValueError("Could not determine feature names from model or data")
        
        # Verify all features exist in data
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        
        # Prepare features
        self.X = df[self.feature_names].copy()
        
        # Get target (collision type)
        if 'col' in df.columns:
            self.y = df['col']
        elif 'max_severity' in df.columns:
            self.y = df['max_severity']
        elif 'grav' in df.columns:
            self.y = df['grav']
        else:
            self.y = None
        
        print(f"✓ Loaded model from {self.model_path}")
        print(f"✓ Loaded data: {len(self.X)} samples, {len(self.feature_names)} features")
        print(f"✓ Model type: {type(self.model).__name__}")
        
    def compute_shap_values(self, sample_size: int = 1000):
        """
        Compute SHAP values for the model
        
        Args:
            sample_size: Number of samples to use (for speed)
        """
        print("Computing SHAP values...")
        
        # Sample data for faster computation
        if len(self.X) > sample_size:
            sample_idx = np.random.choice(len(self.X), sample_size, replace=False)
            X_sample = self.X.iloc[sample_idx]
        else:
            X_sample = self.X
        
        # Store sample for later use
        self.X_sample = X_sample
        
        # Handle MultiOutputClassifier
        model_to_explain = self.model
        if hasattr(self.model, 'estimators_'):
            # MultiOutputClassifier - use first estimator (collision type)
            print("Detected MultiOutputClassifier, using first estimator (collision type)")
            model_to_explain = self.model.estimators_[0]
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(model_to_explain)
        
        # Compute SHAP values
        self.shap_values = self.explainer(X_sample)
        
        # For multi-class models, SHAP returns values for each class
        # We need to handle this properly
        if isinstance(self.shap_values.values, list) or (len(self.shap_values.values.shape) == 3):
            print(f"Multi-class output detected: {self.shap_values.values.shape}")
            # For multi-class, use the mean absolute SHAP across all classes
            # Or select the predicted class for each sample
            if len(self.shap_values.values.shape) == 3:
                # Shape is (samples, features, classes)
                # Average across classes for visualization
                print("Averaging SHAP values across classes for visualization")
                mean_shap_values = np.mean(np.abs(self.shap_values.values), axis=2)
                
                # Create new Explanation object with averaged values
                self.shap_values = shap.Explanation(
                    values=mean_shap_values,
                    base_values=np.mean(self.shap_values.base_values, axis=1) if len(self.shap_values.base_values.shape) > 1 else self.shap_values.base_values,
                    data=self.shap_values.data,
                    feature_names=self.feature_names
                )
        
        print(f"✓ SHAP values computed for {len(X_sample)} samples")
        print(f"  SHAP values shape: {self.shap_values.values.shape}")
        
    def plot_global_summary(self, save_path: str = None):
        """
        Create global summary plot showing feature importance
        
        Args:
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        # Create larger, clearer figure
        plt.figure(figsize=(12, 8))
        
        # Rename features for better readability
        feature_names_display = [FEATURE_NAMES.get(f, f) for f in self.feature_names]
        
        # Create summary plot with better styling
        shap.summary_plot(
            self.shap_values,
            features=self.shap_values.data,
            feature_names=feature_names_display,
            show=False,
            plot_size=(12, 8),
            max_display=len(self.feature_names)  # Show all features
        )
        
        plt.title("Global Feature Importance (SHAP)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=13, fontweight='bold')
        plt.ylabel("Features", fontsize=13, fontweight='bold')
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved global summary plot to {save_path}")
        
        return plt.gcf()
    
    def plot_dependence(self, feature: str, interaction_feature: str = None, save_path: str = None):
        """
        Create dependence plot for a specific feature
        
        Args:
            feature: Feature to analyze (e.g., 'hour')
            interaction_feature: Feature to color by (e.g., 'lum')
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in model features")
        
        feature_idx = self.feature_names.index(feature)
        
        # Create larger, clearer figure
        plt.figure(figsize=(12, 7))
        
        # Get display names
        feature_display = FEATURE_NAMES.get(feature, feature)
        interaction_display = FEATURE_NAMES.get(interaction_feature, interaction_feature) if interaction_feature else None
        
        # Get SHAP values - ensure 2D
        shap_vals = self.shap_values.values
        if len(shap_vals.shape) == 3:
            # Multi-class: average across classes
            shap_vals = np.mean(np.abs(shap_vals), axis=2)
        
        # Get feature data
        feature_data = self.shap_values.data
        
        # Create dependence plot with better styling
        shap.dependence_plot(
            feature_idx,
            shap_vals,
            feature_data,
            feature_names=self.feature_names,
            interaction_index=interaction_feature if interaction_feature else "auto",
            show=False,
            dot_size=40,
            alpha=0.6
        )
        
        plt.title(f"SHAP Dependence: {feature_display}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(f"{feature_display} Value", fontsize=13, fontweight='bold')
        plt.ylabel(f"SHAP Value for {feature_display}", fontsize=13, fontweight='bold')
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Add grid for easier reading
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dependence plot to {save_path}")
        
        return plt.gcf()
    
    def plot_waterfall(self, sample_idx: int, save_path: str = None):
        """
        Create waterfall plot for a single prediction
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        plt.figure(figsize=(10, 8))
        
        # Rename features
        feature_names_display = [FEATURE_NAMES.get(f, f) for f in self.feature_names]
        
        # Create waterfall plot
        shap.plots.waterfall(
            self.shap_values[sample_idx],
            show=False
        )
        
        plt.title(f"SHAP Explanation for Sample {sample_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved waterfall plot to {save_path}")
        
        return plt.gcf()
    
    def get_feature_importance(self):
        """
        Get feature importance as DataFrame
        
        Returns:
            DataFrame with features and their mean absolute SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        # Calculate mean absolute SHAP values
        shap_vals = self.shap_values.values
        
        # Handle different shapes
        if len(shap_vals.shape) == 3:
            # Multi-class: (samples, features, classes)
            # Average across samples and classes
            mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))
        elif len(shap_vals.shape) == 2:
            # Binary or averaged: (samples, features)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        else:
            # 1D - single sample
            mean_abs_shap = np.abs(shap_vals)
        
        # Ensure 1D array
        mean_abs_shap = np.atleast_1d(mean_abs_shap).flatten()
        
        # Ensure we have the right number of values
        if len(mean_abs_shap) != len(self.feature_names):
            raise ValueError(f"Shape mismatch: {len(mean_abs_shap)} SHAP values but {len(self.feature_names)} features. SHAP shape: {shap_vals.shape}")
        
        # Create DataFrame with explicit 1D arrays
        importance_df = pd.DataFrame({
            'Feature': [str(FEATURE_NAMES.get(f, f)) for f in self.feature_names],
            'Feature_Code': [str(f) for f in self.feature_names],
            'Mean_Abs_SHAP': mean_abs_shap.tolist()  # Convert to list to ensure 1D
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df


def main():
    """Example usage"""
    # Initialize XAI
    xai = AccidentXAI(
        model_path='models/xgb_nopca_multitarget.pkl',
        data_path='data/model_ready.csv'
    )
    
    # Load model and data
    xai.load_model_and_data()
    
    # Compute SHAP values
    xai.compute_shap_values(sample_size=1000)
    
    # Create visualizations
    xai.plot_global_summary(save_path='models/shap_global_summary.png')
    xai.plot_dependence('hour', interaction_feature='lum', save_path='models/shap_hour_dependence.png')
    
    # Get feature importance
    importance = xai.get_feature_importance()
    print("\nFeature Importance:")
    print(importance)
    
    # Save importance
    importance.to_csv('models/feature_importance_shap.csv', index=False)
    print("\n✓ XAI analysis complete!")


if __name__ == "__main__":
    main()
