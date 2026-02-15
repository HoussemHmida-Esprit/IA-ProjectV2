"""
Test Production Inference Pipeline
Quick verification that everything works
"""
from production_inference_pipeline import ProductionInferencePipeline
import pandas as pd

def test_pipeline():
    """Test the production pipeline"""
    print("="*60)
    print("TESTING PRODUCTION INFERENCE PIPELINE")
    print("="*60)
    
    # Initialize
    print("\n1. Initializing pipeline...")
    pipeline = ProductionInferencePipeline(models_dir='.')
    
    # Load models
    print("\n2. Loading models...")
    if not pipeline.load_all_models():
        print("❌ No models loaded")
        return False
    
    print(f"✓ Loaded {len(pipeline.models)} models")
    
    # Test single prediction
    print("\n3. Testing single prediction...")
    test_input = {
        'lum': 1,
        'agg': 1,
        'int': 2,
        'day_of_week': 4,
        'hour': 18,
        'num_users': 2
    }
    
    try:
        result = pipeline.predict_for_dashboard(test_input)
        print("✓ Single prediction successful")
        print(f"  Prediction: {result['final_prediction']['class_name']}")
        print(f"  Confidence: {result['ensemble']['confidence']:.2%}")
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
        return False
    
    # Test batch prediction
    print("\n4. Testing batch prediction...")
    test_df = pd.DataFrame([test_input] * 5)
    
    try:
        batch_results = pipeline.predict_batch(test_df)
        print("✓ Batch prediction successful")
        print(f"  Processed {len(batch_results)} samples")
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False
    
    # Test individual models
    print("\n5. Testing individual models...")
    try:
        individual = pipeline.predict_all_models(test_input)
        print("✓ Individual predictions successful")
        for model, pred in individual.items():
            print(f"  {model}: {pred:.2f}")
    except Exception as e:
        print(f"❌ Individual predictions failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)
    print("\nPipeline is ready for production!")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)
