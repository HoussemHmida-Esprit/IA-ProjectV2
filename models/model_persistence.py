"""
Model Persistence Manager
Handles saving and loading of all models with their parameters and metadata
"""
import pickle
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


class ModelPersistence:
    """Centralized model persistence with metadata tracking"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / 'models_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_sklearn_model(self, model: Any, model_name: str, 
                          params: Dict, metrics: Dict, 
                          features: list, additional_info: Optional[Dict] = None):
        """
        Save scikit-learn compatible model with full metadata
        
        Args:
            model: Trained model object
            model_name: Name identifier (e.g., 'random_forest', 'xgboost')
            params: Hyperparameters used
            metrics: Performance metrics (accuracy, f1, etc.)
            features: List of feature names
            additional_info: Any additional information to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # Package everything together
        model_package = {
            'model': model,
            'model_name': model_name,
            'params': params,
            'metrics': metrics,
            'features': features,
            'timestamp': timestamp,
            'additional_info': additional_info or {}
        }
        
        # Save model package
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Update metadata
        if model_name not in self.metadata:
            self.metadata[model_name] = []
        
        self.metadata[model_name].append({
            'filename': model_filename,
            'timestamp': timestamp,
            'params': params,
            'metrics': metrics,
            'features': features,
            'path': str(model_path)
        })
        
        # Also save as "latest" version
        latest_path = self.models_dir / f"{model_name}_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        self._save_metadata()
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Latest version: {latest_path}")
        
        return str(model_path)
    
    def save_pytorch_model(self, model: torch.nn.Module, model_name: str,
                          params: Dict, metrics: Dict,
                          features: list, additional_info: Optional[Dict] = None):
        """
        Save PyTorch model with full metadata
        
        Args:
            model: Trained PyTorch model
            model_name: Name identifier (e.g., 'tab_transformer', 'lstm')
            params: Hyperparameters used
            metrics: Performance metrics
            features: List of feature names
            additional_info: Any additional information
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_name}_{timestamp}.pth"
        model_path = self.models_dir / model_filename
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'params': params,
            'metrics': metrics,
            'features': features,
            'timestamp': timestamp,
            'additional_info': additional_info or {}
        }, model_path)
        
        # Update metadata
        if model_name not in self.metadata:
            self.metadata[model_name] = []
        
        self.metadata[model_name].append({
            'filename': model_filename,
            'timestamp': timestamp,
            'params': params,
            'metrics': metrics,
            'features': features,
            'path': str(model_path)
        })
        
        # Also save as "latest" version
        latest_path = self.models_dir / f"{model_name}_latest.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'params': params,
            'metrics': metrics,
            'features': features,
            'timestamp': timestamp,
            'additional_info': additional_info or {}
        }, latest_path)
        
        self._save_metadata()
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Latest version: {latest_path}")
        
        return str(model_path)
    
    def load_sklearn_model(self, model_name: str, version: str = 'latest') -> Dict:
        """
        Load scikit-learn model with metadata
        
        Args:
            model_name: Name of the model
            version: 'latest' or specific timestamp
            
        Returns:
            Dictionary with model and all metadata
        """
        if version == 'latest':
            model_path = self.models_dir / f"{model_name}_latest.pkl"
        else:
            model_path = self.models_dir / f"{model_name}_{version}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        print(f"✓ Loaded model: {model_path}")
        print(f"  Trained: {model_package['timestamp']}")
        print(f"  Metrics: {model_package['metrics']}")
        
        return model_package
    
    def load_pytorch_model(self, model_class: torch.nn.Module, 
                          model_name: str, version: str = 'latest') -> Dict:
        """
        Load PyTorch model with metadata
        
        Args:
            model_class: Uninitialized model class
            model_name: Name of the model
            version: 'latest' or specific timestamp
            
        Returns:
            Dictionary with model and all metadata
        """
        if version == 'latest':
            model_path = self.models_dir / f"{model_name}_latest.pth"
        else:
            model_path = self.models_dir / f"{model_name}_{version}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path)
        
        # Initialize model with saved params
        model = model_class(**checkpoint['params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Loaded model: {model_path}")
        print(f"  Trained: {checkpoint['timestamp']}")
        print(f"  Metrics: {checkpoint['metrics']}")
        
        return {
            'model': model,
            'params': checkpoint['params'],
            'metrics': checkpoint['metrics'],
            'features': checkpoint['features'],
            'timestamp': checkpoint['timestamp'],
            'additional_info': checkpoint.get('additional_info', {})
        }
    
    def list_models(self, model_name: Optional[str] = None):
        """
        List all saved models or specific model versions
        
        Args:
            model_name: Optional - filter by model name
        """
        if model_name:
            if model_name not in self.metadata:
                print(f"No saved versions found for: {model_name}")
                return
            
            print(f"\n{model_name} versions:")
            print("-" * 80)
            for version in self.metadata[model_name]:
                print(f"  Timestamp: {version['timestamp']}")
                print(f"  Metrics: {version['metrics']}")
                print(f"  Path: {version['path']}")
                print()
        else:
            print("\nAll saved models:")
            print("=" * 80)
            for name, versions in self.metadata.items():
                print(f"\n{name}: {len(versions)} version(s)")
                latest = versions[-1]
                print(f"  Latest: {latest['timestamp']}")
                print(f"  Metrics: {latest['metrics']}")
    
    def get_best_model(self, model_name: str, metric: str = 'accuracy') -> Dict:
        """
        Get the best performing version of a model
        
        Args:
            model_name: Name of the model
            metric: Metric to compare (default: 'accuracy')
            
        Returns:
            Metadata of best model version
        """
        if model_name not in self.metadata:
            raise ValueError(f"No saved versions found for: {model_name}")
        
        versions = self.metadata[model_name]
        best_version = max(versions, key=lambda v: v['metrics'].get(metric, 0))
        
        print(f"\nBest {model_name} (by {metric}):")
        print(f"  Timestamp: {best_version['timestamp']}")
        print(f"  {metric}: {best_version['metrics'].get(metric)}")
        
        return best_version
    
    def compare_models(self, metric: str = 'accuracy'):
        """
        Compare latest versions of all models
        
        Args:
            metric: Metric to compare
        """
        print(f"\nModel Comparison (by {metric}):")
        print("=" * 80)
        
        comparisons = []
        for model_name, versions in self.metadata.items():
            if versions:
                latest = versions[-1]
                metric_value = latest['metrics'].get(metric, 0)
                comparisons.append({
                    'Model': model_name,
                    metric.capitalize(): f"{metric_value:.4f}",
                    'Timestamp': latest['timestamp']
                })
        
        # Sort by metric
        comparisons.sort(key=lambda x: float(x[metric.capitalize()]), reverse=True)
        
        for comp in comparisons:
            print(f"{comp['Model']:20s} | {comp[metric.capitalize()]:8s} | {comp['Timestamp']}")


# Convenience functions
def save_model(model, model_name: str, params: Dict, metrics: Dict, 
               features: list, model_type: str = 'sklearn', **kwargs):
    """
    Quick save function
    
    Args:
        model: Trained model
        model_name: Model identifier
        params: Hyperparameters
        metrics: Performance metrics
        features: Feature list
        model_type: 'sklearn' or 'pytorch'
    """
    persistence = ModelPersistence()
    
    if model_type == 'sklearn':
        return persistence.save_sklearn_model(
            model, model_name, params, metrics, features, kwargs.get('additional_info')
        )
    elif model_type == 'pytorch':
        return persistence.save_pytorch_model(
            model, model_name, params, metrics, features, kwargs.get('additional_info')
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_model(model_name: str, version: str = 'latest', 
               model_type: str = 'sklearn', model_class=None):
    """
    Quick load function
    
    Args:
        model_name: Model identifier
        version: 'latest' or specific timestamp
        model_type: 'sklearn' or 'pytorch'
        model_class: Required for PyTorch models
    """
    persistence = ModelPersistence()
    
    if model_type == 'sklearn':
        return persistence.load_sklearn_model(model_name, version)
    elif model_type == 'pytorch':
        if model_class is None:
            raise ValueError("model_class required for PyTorch models")
        return persistence.load_pytorch_model(model_class, model_name, version)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == '__main__':
    # Demo usage
    persistence = ModelPersistence()
    persistence.list_models()
    persistence.compare_models('accuracy')
