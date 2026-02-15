"""
Verify All Models Can Be Loaded
Quick test to ensure all models are working
"""
import pickle
import torch
import numpy as np
from pathlib import Path

def test_random_forest():
    """Test Random Forest loading"""
    print("\n1. Testing Random Forest...")
    try:
        with open('random_forest_optimized_latest.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
            features = model_data.get('features', model_data.get('feature_names', []))
            params = model_data.get('best_params', model_data.get('params', {}))
        else:
            model = model_data
            features = []
            params = {}
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Model type: {type(model).__name__}")
        if features:
            print(f"   ✓ Features: {features}")
        if params:
            print(f"   ✓ n_estimators: {params.get('n_estimators', 'N/A')}")
        
        # Test prediction
        X_test = np.array([[5, 2, 1, 3, 18, 2]])  # Sample input
        pred = model.predict(X_test)
        print(f"   ✓ Test prediction: {pred[0]}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_xgboost():
    """Test XGBoost loading"""
    print("\n2. Testing XGBoost...")
    try:
        with open('xgboost_optimized_latest.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
            features = model_data.get('features', model_data.get('feature_names', []))
            params = model_data.get('best_params', model_data.get('params', {}))
        else:
            model = model_data
            features = []
            params = {}
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Model type: {type(model).__name__}")
        if features:
            print(f"   ✓ Features: {features}")
        if params:
            print(f"   ✓ n_estimators: {params.get('n_estimators', 'N/A')}")
        
        # Test prediction
        X_test = np.array([[5, 2, 1, 3, 18, 2]])  # Sample input
        pred = model.predict(X_test)
        print(f"   ✓ Test prediction: {pred[0]}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_tabtransformer():
    """Test TabTransformer loading"""
    print("\n3. Testing TabTransformer...")
    try:
        checkpoint = torch.load('tab_transformer_best.pth', map_location='cpu', weights_only=False)
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Keys: {list(checkpoint.keys())[:5]}")  # Show first 5 keys
        
        if 'model_state_dict' in checkpoint:
            print(f"   ✓ Model state dict found")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_lstm():
    """Test LSTM loading"""
    print("\n4. Testing LSTM...")
    try:
        checkpoint = torch.load('lstm_forecaster.pth', map_location='cpu', weights_only=False)
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            print(f"   ✓ Model state dict found")
            print(f"   ✓ Sequence length: {checkpoint.get('sequence_length', 'N/A')}")
            print(f"   ✓ Forecast horizon: {checkpoint.get('forecast_horizon', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("MODEL VERIFICATION TEST")
    print("="*60)
    
    results = {
        'Random Forest': test_random_forest(),
        'XGBoost': test_xgboost(),
        'TabTransformer': test_tabtransformer(),
        'LSTM': test_lstm()
    }
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name:20s} {status}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n✅ All models verified successfully!")
        print("   Ready for deployment!")
    else:
        print("\n⚠️ Some models failed verification")
        print("   Check errors above")
    
    return all_pass


if __name__ == '__main__':
    main()
