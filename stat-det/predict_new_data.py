import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats

class StationarityPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model_and_scaler()
    
    def load_model_and_scaler(self):
        """Load the best trained model and scaler"""
        try:
            # Load best model
            model_path = os.path.join(self.models_dir, "best_model.pkl")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature info
            feature_info_path = os.path.join(self.models_dir, "feature_info.json")
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                self.feature_columns = feature_info['feature_columns']
            
            # Load best model info
            best_info_path = os.path.join(self.models_dir, "best_model_info.json")
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
            
            print(f"Loaded model: {best_info['best_model_name']}")
            print(f"Model CV score: {best_info['best_score']:.4f}")
            print(f"Features expected: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_features_from_series(self, series):
        """Extract the same features used during training"""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(series)
        features['std'] = np.std(series)
        features['var'] = np.var(series)
        features['skewness'] = stats.skew(series)
        features['kurtosis'] = stats.kurtosis(series)
        features['min'] = np.min(series)
        features['max'] = np.max(series)
        features['range'] = np.max(series) - np.min(series)
        
        # Quantiles
        features['q25'] = np.percentile(series, 25)
        features['q75'] = np.percentile(series, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # First differences
        diff1 = np.diff(series)
        if len(diff1) > 0:
            features['diff_mean'] = np.mean(diff1)
            features['diff_std'] = np.std(diff1)
            features['diff_var'] = np.var(diff1)
        else:
            features['diff_mean'] = 0
            features['diff_std'] = 0
            features['diff_var'] = 0
            
        # Second differences
        diff2 = np.diff(diff1) if len(diff1) > 1 else [0]
        if len(diff2) > 0:
            features['diff2_mean'] = np.mean(diff2)
            features['diff2_std'] = np.std(diff2)
        else:
            features['diff2_mean'] = 0
            features['diff2_std'] = 0
            
        # Autocorrelation features
        if len(series) > 5:
            acf_1 = self.autocorr(series, 1)
            acf_5 = self.autocorr(series, min(5, len(series)//2))
            features['acf_1'] = acf_1 if not np.isnan(acf_1) else 0
            features['acf_5'] = acf_5 if not np.isnan(acf_5) else 0
        else:
            features['acf_1'] = 0
            features['acf_5'] = 0
            
        # Trend features
        x = np.arange(len(series))
        if len(series) > 2:
            slope, intercept, r_value, _, _ = stats.linregress(x, series)
            features['trend_slope'] = slope
            features['trend_r2'] = r_value**2
        else:
            features['trend_slope'] = 0
            features['trend_r2'] = 0
            
        # Length
        features['length'] = len(series)
        
        return features
    
    def autocorr(self, series, lag):
        """Calculate autocorrelation with given lag"""
        try:
            if len(series) <= lag:
                return 0
            c0 = np.var(series)
            if c0 == 0:
                return 0
            c_lag = np.mean((series[:-lag] - np.mean(series)) * (series[lag:] - np.mean(series)))
            return c_lag / c0
        except:
            return 0
    
    def predict_from_csv(self, csv_path):
        """Predict stationarity from a CSV file"""
        try:
            df = pd.read_csv(csv_path)
            if 'data' not in df.columns:
                raise ValueError("CSV must contain a 'data' column")
            
            series = df['data'].values
            series = series[~np.isnan(series)]  # Remove NaN values
            
            if len(series) < 5:
                raise ValueError("Time series too short (minimum 5 points required)")
            
            return self.predict_from_series(series)
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return None
    
    def predict_from_series(self, series):
        """Predict stationarity from a time series array"""
        try:
            # Extract features
            features = self.extract_features_from_series(series)
            
            # Convert to DataFrame with correct column order
            feature_df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Select and order features correctly
            X = feature_df[self.feature_columns]
            
            # Handle any NaN values
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
                prob_stationary = probabilities[1]  # Probability of being stationary
            else:
                confidence = None
                prob_stationary = None
            
            return {
                'prediction': int(prediction),
                'prediction_label': 'Stationary' if prediction == 1 else 'Non-stationary',
                'confidence': float(confidence) if confidence is not None else None,
                'probability_stationary': float(prob_stationary) if prob_stationary is not None else None,
                'series_length': len(series),
                'extracted_features': features
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_batch_csv_files(self, csv_files_list):
        """Predict for multiple CSV files"""
        results = []
        
        for csv_file in csv_files_list:
            print(f"Processing: {csv_file}")
            result = self.predict_from_csv(csv_file)
            if result is not None:
                result['filename'] = os.path.basename(csv_file)
                results.append(result)
        
        return results

def main():
    # Example usage
    models_dir = "models"
    
    # Check if models exist
    if not os.path.exists(os.path.join(models_dir, "best_model.pkl")):
        print("No trained model found! Please run the training script first.")
        return
    
    # Initialize predictor
    predictor = StationarityPredictor(models_dir)
    
    # Example 1: Predict from a single CSV file
    print("\n=== Example Usage ===")
    print("1. To predict from a CSV file:")
    print("   result = predictor.predict_from_csv('path/to/your/file.csv')")
    
    print("\n2. To predict from a time series array:")
    print("   series = np.array([1, 2, 3, 4, 5, ...])")
    print("   result = predictor.predict_from_series(series)")
    
    print("\n3. To predict for multiple files:")
    print("   files = ['file1.csv', 'file2.csv', ...]")
    print("   results = predictor.predict_batch_csv_files(files)")
    
    # Test with a synthetic example
    print("\n=== Testing with synthetic data ===")
    
    # Create a stationary series (white noise)
    np.random.seed(42)
    stationary_series = np.random.normal(0, 1, 100)
    result = predictor.predict_from_series(stationary_series)
    
    if result:
        print("Stationary test series:")
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  P(Stationary): {result['probability_stationary']:.4f}")
    
    # Create a non-stationary series (random walk)
    non_stationary_series = np.cumsum(np.random.normal(0, 1, 100))
    result = predictor.predict_from_series(non_stationary_series)
    
    if result:
        print("\nNon-stationary test series:")
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  P(Stationary): {result['probability_stationary']:.4f}")

if __name__ == "__main__":
    main()