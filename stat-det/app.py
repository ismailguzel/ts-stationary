from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import json
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our predictor
from predict_new_data import StationarityPredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class WebStationarityPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.all_models = {}
        self.scaler = None
        self.feature_columns = None
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models"""
        model_files = {
            'RandomForest': 'randomforest_model.pkl',
            'GradientBoosting': 'gradientboosting_model.pkl',
            'LogisticRegression': 'logisticregression_model.pkl',
            'SVM': 'svm_model.pkl',
            'KNN': 'knn_model.pkl',
            'NaiveBayes': 'naivebayes_model.pkl',
            'DecisionTree': 'decisiontree_model.pkl'
        }
        
        # Load scaler
        with open(os.path.join(self.models_dir, "scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature info
        with open(os.path.join(self.models_dir, "feature_info.json"), 'r') as f:
            feature_info = json.load(f)
            self.feature_columns = feature_info['feature_columns']
        
        # Load all models
        for name, filename in model_files.items():
            try:
                with open(os.path.join(self.models_dir, filename), 'rb') as f:
                    self.all_models[name] = pickle.load(f)
                print(f"✅ Loaded {name}")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        
        print(f"Loaded {len(self.all_models)} models successfully")
    
    def predict_with_all_models(self, series):
        """Get predictions from all models"""
        # Use the same feature extraction from predict_new_data
        predictor = StationarityPredictor(self.models_dir)
        features = predictor.extract_features_from_series(series)
        
        # Convert to DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Select and order features correctly
        X = feature_df[self.feature_columns]
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        results = {}
        for name, model in self.all_models.items():
            try:
                prediction = model.predict(X_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_scaled)[0]
                    confidence = max(probabilities)
                    prob_stationary = probabilities[1]
                else:
                    confidence = None
                    prob_stationary = None
                
                results[name] = {
                    'prediction': int(prediction),
                    'prediction_label': 'Stationary' if prediction == 1 else 'Non-stationary',
                    'confidence': float(confidence) if confidence is not None else None,
                    'probability_stationary': float(prob_stationary) if prob_stationary is not None else None
                }
            except Exception as e:
                results[name] = {
                    'error': str(e)
                }
        
        return results, features

# Initialize predictor
try:
    web_predictor = WebStationarityPredictor()
    print("🚀 Web predictor initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize predictor: {e}")
    web_predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if web_predictor is None:
            return jsonify({'error': 'Models not loaded properly'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
            if 'data' not in df.columns:
                return jsonify({'error': 'CSV must contain a "data" column'}), 400
            
            series = df['data'].values
            series = series[~np.isnan(series)]  # Remove NaN values
            
            if len(series) < 5:
                return jsonify({'error': 'Time series too short (minimum 5 points required)'}), 400
            
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Get predictions from all models
        predictions, features = web_predictor.predict_with_all_models(series)
        
        # Add series info
        result = {
            'filename': file.filename,
            'series_length': len(series),
            'series_stats': {
                'mean': float(np.mean(series)),
                'std': float(np.std(series)),
                'min': float(np.min(series)),
                'max': float(np.max(series))
            },
            'predictions': predictions,
            'extracted_features': features,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        if 'data' not in df.columns:
            return jsonify({'error': 'CSV must contain a "data" column'}), 400
        
        series = df['data'].values
        series = series[~np.isnan(series)]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Time Series Analysis: {file.filename}', fontsize=16, fontweight='bold')
        
        # 1. Original Time Series
        axes[0, 0].plot(series, linewidth=1.5, color='blue')
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. First Differences
        diff1 = np.diff(series)
        axes[0, 1].plot(diff1, linewidth=1.5, color='red')
        axes[0, 1].set_title('First Differences')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Δ Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution
        axes[1, 0].hist(series, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('Value Distribution')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Autocorrelation Plot (simple version)
        lags = min(50, len(series) // 4)
        autocorr = []
        for lag in range(lags):
            if lag == 0:
                autocorr.append(1.0)
            else:
                if len(series) > lag:
                    c0 = np.var(series)
                    if c0 > 0:
                        c_lag = np.mean((series[:-lag] - np.mean(series)) * (series[lag:] - np.mean(series)))
                        autocorr.append(c_lag / c0)
                    else:
                        autocorr.append(0)
                else:
                    autocorr.append(0)
        
        axes[1, 1].plot(range(len(autocorr)), autocorr, 'o-', linewidth=1.5, markersize=3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Autocorrelation Function')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return jsonify({
            'plot': img_string,
            'filename': file.filename,
            'series_length': len(series)
        })
        
    except Exception as e:
        return jsonify({'error': f'Visualization error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(web_predictor.all_models) if web_predictor else 0,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🌐 Starting Stationarity Detection Web Server...")
    print("📊 Available at: http://localhost:5000")
    print("🔧 Models loaded:", len(web_predictor.all_models) if web_predictor else 0)
    app.run(debug=True, host='0.0.0.0', port=5000)