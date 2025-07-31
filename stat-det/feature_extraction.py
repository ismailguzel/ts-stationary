import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Statistical feature extraction libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pickle

class TimeSeriesFeatureExtractor:
    def __init__(self):
        self.features = []
        self.labels = []
        
    def extract_statistical_features(self, series):
        """Extract statistical features from time series"""
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
            
        # Second differences
        diff2 = np.diff(diff1) if len(diff1) > 1 else [0]
        if len(diff2) > 0:
            features['diff2_mean'] = np.mean(diff2)
            features['diff2_std'] = np.std(diff2)
            
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
    
    def process_csv_file(self, filepath):
        """Process a single CSV file"""
        try:
            df = pd.read_csv(filepath)
            if 'data' in df.columns:
                series = df['data'].values
                # Remove any NaN values
                series = series[~np.isnan(series)]
                if len(series) > 5:  # Minimum length requirement
                    return self.extract_statistical_features(series)
            return None
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    def determine_label(self, filepath):
        """Determine if the time series is stationary or non-stationary based on folder structure"""
        path_str = str(filepath).lower()
        
        # Non-stationary folders
        if 'stochastic trend' in path_str or 'volatility' in path_str:
            return 0  # Non-stationary
        else:
            return 1  # Stationary
    
    def process_directory(self, data_dir):
        """Process all CSV files in the data directory"""
        csv_files = list(Path(data_dir).rglob("*.csv"))
        # Filter out metadata files
        csv_files = [f for f in csv_files if 'metadata' not in f.name.lower()]
        
        print(f"Found {len(csv_files)} CSV files to process...")
        
        features_list = []
        labels_list = []
        filenames_list = []
        
        for filepath in tqdm(csv_files, desc="Processing files"):
            features = self.process_csv_file(filepath)
            if features is not None:
                label = self.determine_label(filepath)
                
                features_list.append(features)
                labels_list.append(label)
                filenames_list.append(filepath.name)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        features_df['label'] = labels_list
        features_df['filename'] = filenames_list
        
        print(f"Successfully processed {len(features_df)} files")
        print(f"Stationary: {sum(labels_list)} | Non-stationary: {len(labels_list) - sum(labels_list)}")
        
        return features_df

def main():
    # Set paths
    data_dir = "data"
    output_dir = "models"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature extractor
    extractor = TimeSeriesFeatureExtractor()
    
    # Process all files and extract features
    print("Starting feature extraction...")
    features_df = extractor.process_directory(data_dir)
    
    # Save features
    features_df.to_csv(os.path.join(output_dir, "extracted_features.csv"), index=False)
    print(f"Features saved to {output_dir}/extracted_features.csv")
    
    # Prepare data for ML
    feature_columns = [col for col in features_df.columns if col not in ['label', 'filename']]
    X = features_df[feature_columns]
    y = features_df['label']
    
    # Handle any NaN values
    X = X.fillna(0)
    
    # Save feature names and preprocessing info
    with open(os.path.join(output_dir, "feature_info.json"), 'w') as f:
        json.dump({
            'feature_columns': feature_columns,
            'total_samples': len(X),
            'stationary_samples': sum(y),
            'non_stationary_samples': len(y) - sum(y)
        }, f, indent=2)
    
    print("Feature extraction completed!")
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Feature columns: {feature_columns}")

if __name__ == "__main__":
    main()