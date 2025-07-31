import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

class StationarityModelTrainer:
    def __init__(self, output_dir="models"):
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def initialize_models(self):
        """Initialize different ML models for training"""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000
            ),
            'SVM': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'NaiveBayes': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10, 
                random_state=42
            )
        }
    
    def load_data(self, features_file):
        """Load the extracted features"""
        print("Loading extracted features...")
        df = pd.read_csv(features_file)
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['label', 'filename']]
        X = df[feature_columns]
        y = df['label']
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        print(f"Loaded {len(X)} samples with {len(feature_columns)} features")
        print(f"Class distribution: Stationary={sum(y)}, Non-stationary={len(y)-sum(y)}")
        
        return X, y, feature_columns
    
    def train_and_evaluate_models(self, X, y):
        """Train and evaluate all models"""
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        with open(os.path.join(self.output_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        
        self.initialize_models()
        
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                # Test predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                test_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_auc': test_auc,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                print(f"Test Accuracy: {test_accuracy:.4f}")
                if test_auc:
                    print(f"Test AUC: {test_auc:.4f}")
                
                # Check if this is the best model
                if cv_scores.mean() > self.best_score:
                    self.best_score = cv_scores.mean()
                    self.best_model = (name, model)
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
    
    def save_models_and_results(self):
        """Save all trained models and results"""
        print("\nSaving models and results...")
        
        # Save all models
        for name, result in self.results.items():
            model_path = os.path.join(self.output_dir, f"{name.lower()}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save best model separately
        if self.best_model:
            best_name, best_model_obj = self.best_model
            best_model_path = os.path.join(self.output_dir, "best_model.pkl")
            with open(best_model_path, 'wb') as f:
                pickle.dump(best_model_obj, f)
            
            # Save best model info
            with open(os.path.join(self.output_dir, "best_model_info.json"), 'w') as f:
                json.dump({
                    'best_model_name': best_name,
                    'best_score': self.best_score,
                    'model_path': 'best_model.pkl'
                }, f, indent=2)
        
        # Prepare results for saving (remove model objects)
        results_to_save = {}
        for name, result in self.results.items():
            results_to_save[name] = {
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'test_accuracy': result['test_accuracy'],
                'test_auc': result['test_auc'],
                'classification_report': result['classification_report'],
                'confusion_matrix': result['confusion_matrix']
            }
        
        # Save results
        with open(os.path.join(self.output_dir, "training_results.json"), 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"All models saved to {self.output_dir}/")
        if self.best_model:
            print(f"Best model: {self.best_model[0]} (CV Score: {self.best_score:.4f})")
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if not self.results:
            print("No models were successfully trained!")
            return
        
        # Sort by CV score
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['cv_mean'], 
            reverse=True
        )
        
        print(f"{'Model':<20} {'CV Score':<12} {'Test Acc':<12} {'AUC':<10}")
        print("-" * 60)
        
        for name, result in sorted_results:
            cv_score = f"{result['cv_mean']:.4f}"
            test_acc = f"{result['test_accuracy']:.4f}"
            auc = f"{result['test_auc']:.4f}" if result['test_auc'] else "N/A"
            print(f"{name:<20} {cv_score:<12} {test_acc:<12} {auc:<10}")
        
        print("\nBest model details:")
        if self.best_model:
            best_name, _ = self.best_model
            best_result = self.results[best_name]
            print(f"Model: {best_name}")
            print(f"Cross-validation score: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']*2:.4f})")
            print(f"Test accuracy: {best_result['test_accuracy']:.4f}")
            if best_result['test_auc']:
                print(f"Test AUC: {best_result['test_auc']:.4f}")

def main():
    # Paths
    output_dir = "models"
    features_file = os.path.join(output_dir, "extracted_features.csv")
    
    # Check if features file exists
    if not os.path.exists(features_file):
        print(f"Features file not found: {features_file}")
        print("Please run the feature extraction script first!")
        return
    
    # Create trainer
    trainer = StationarityModelTrainer(output_dir)
    
    # Load data
    X, y, feature_columns = trainer.load_data(features_file)
    
    # Train models
    trainer.train_and_evaluate_models(X, y)
    
    # Save everything
    trainer.save_models_and_results()
    
    # Print summary
    trainer.print_summary()
    
    print(f"\nAll models and results saved to: {output_dir}/")
    print("Files created:")
    print("- best_model.pkl (best performing model)")
    print("- best_model_info.json (best model metadata)")
    print("- scaler.pkl (feature scaler)")
    print("- training_results.json (detailed results)")
    print("- individual model files (*_model.pkl)")

if __name__ == "__main__":
    main()