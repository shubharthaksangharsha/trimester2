# Model Loading Helper Script
# Generated on 2025-07-27 11:19:28

import joblib
import pickle
import pandas as pd
import numpy as np

def load_best_model():
    """Load the best performing model (Ridge Regression)"""
    model = joblib.load('models/ridge_regression_optimized_20250727_111928.pkl')
    scaler = joblib.load('models/robust_scaler_20250727_111928.pkl')
    
    with open('models/model_metadata_20250727_111928.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, metadata

def predict_traffic(model, scaler, features):
    """Make traffic predictions using the trained model"""
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    return predictions

# Example usage:
# model, scaler, metadata = load_best_model()
# print(f"Best model R²: {metadata['best_r2']}")
# print(f"Feature names: {metadata['feature_names']}")
