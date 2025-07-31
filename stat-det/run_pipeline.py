#!/usr/bin/env python3
"""
Complete pipeline for Stationarity Detection ML Model Training

This script runs the entire pipeline:
1. Feature extraction from CSV files
2. Model training and evaluation
3. Model saving and validation

Usage: python run_pipeline.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'), 
        ('scipy', 'scipy'), 
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("Missing required packages:", missing_packages)
        print("Install them with: pip install", " ".join(missing_packages))
        return False
    return True

def run_feature_extraction():
    """Run feature extraction step"""
    print("="*60)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*60)
    
    start_time = time.time()
    
    # Import and run feature extraction
    try:
        from feature_extraction import main as extract_features
        extract_features()
        
        elapsed = time.time() - start_time
        print(f"\nFeature extraction completed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return False

def run_model_training():
    """Run model training step"""
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    start_time = time.time()
    
    # Import and run model training
    try:
        from train_models import main as train_models
        train_models()
        
        elapsed = time.time() - start_time
        print(f"\nModel training completed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error in model training: {e}")
        return False

def validate_models():
    """Validate that models were created successfully"""
    print("\n" + "="*60)
    print("STEP 3: MODEL VALIDATION")
    print("="*60)
    
    models_dir = "models"
    required_files = [
        "best_model.pkl",
        "scaler.pkl",
        "training_results.json",
        "feature_info.json",
        "best_model_info.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(models_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:", missing_files)
        return False
    
    print("✅ All required model files created successfully")
    
    # Test the predictor
    try:
        from predict_new_data import StationarityPredictor
        predictor = StationarityPredictor(models_dir)
        
        # Test with synthetic data
        import numpy as np
        test_series = np.random.normal(0, 1, 50)
        result = predictor.predict_from_series(test_series)
        
        if result is not None:
            print("✅ Model prediction test successful")
            print(f"   Test prediction: {result['prediction_label']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            return True
        else:
            print("❌ Model prediction test failed")
            return False
            
    except Exception as e:
        print(f"❌ Model validation error: {e}")
        return False

def print_final_summary():
    """Print final summary and usage instructions"""
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nFiles created in 'models/' directory:")
    print("├── best_model.pkl          (Best performing model)")
    print("├── scaler.pkl              (Feature scaler)")
    print("├── training_results.json   (Training metrics)")
    print("├── feature_info.json       (Feature metadata)")
    print("├── best_model_info.json    (Best model info)")
    print("├── extracted_features.csv  (All extracted features)")
    print("└── *_model.pkl            (Individual models)")
    
    print("\nHow to use your trained model:")
    print("1. For single predictions:")
    print("   from predict_new_data import StationarityPredictor")
    print("   predictor = StationarityPredictor('models')")
    print("   result = predictor.predict_from_csv('your_file.csv')")
    
    print("\n2. For batch predictions:")
    print("   files = ['file1.csv', 'file2.csv', ...]")
    print("   results = predictor.predict_batch_csv_files(files)")
    
    print("\n3. Direct from time series:")
    print("   import numpy as np")
    print("   series = np.array([1, 2, 3, ...])  # Your time series")
    print("   result = predictor.predict_from_series(series)")
    
    print("\nResult format:")
    print("- prediction: 0 (Non-stationary) or 1 (Stationary)")
    print("- prediction_label: 'Stationary' or 'Non-stationary'")
    print("- confidence: Model confidence (0-1)")
    print("- probability_stationary: Probability of being stationary")

def main():
    """Run the complete pipeline"""
    print("🚀 Starting Stationarity Detection ML Pipeline")
    print(f"Working directory: {os.getcwd()}")
    
    # Check data directory
    if not os.path.exists("data"):
        print("❌ 'data' directory not found!")
        print("Please ensure your CSV files are in the 'data' directory")
        return False
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    pipeline_start = time.time()
    
    # Step 1: Feature Extraction
    if not run_feature_extraction():
        print("❌ Pipeline failed at feature extraction step")
        return False
    
    # Step 2: Model Training
    if not run_model_training():
        print("❌ Pipeline failed at model training step")
        return False
    
    # Step 3: Model Validation
    if not validate_models():
        print("❌ Pipeline failed at model validation step")
        return False
    
    # Success!
    total_time = time.time() - pipeline_start
    print(f"\n⏱️  Total pipeline time: {total_time:.2f} seconds")
    
    print_final_summary()
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n🎯 Ready to make predictions on new time series data!")