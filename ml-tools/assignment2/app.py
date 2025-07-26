from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

# Load the cleaned dataset
data = pd.read_csv('assignment2_data_2025_cleaned.csv')

# Define features and labels
label = 'label'
features = data.columns.tolist()
features.remove(label)

# Top 4 discriminative features based on T-score analysis
top_features = ['mean concave points', 'mean perimeter', 'area error', 'mean concavity']
top_feature_scores = {
    'mean concave points': 2.338,
    'mean perimeter': 1.928, 
    'area error': 1.737,
    'mean concavity': 1.684
}

# Load all models
models = {}
model_files = {
    'Random Baseline': 'model/random_baseline_20250726_200129.joblib',
    'SGD Baseline': 'model/sgd_baseline_20250726_200129.joblib', 
    'Random Forest': 'model/random_forest_20250726_200129.joblib',
    'Improved Random Forest': 'model/improved_random_forest_20250726_200129.joblib'
}

for name, filepath in model_files.items():
    if os.path.exists(filepath):
        models[name] = joblib.load(filepath)
        print(f"Loaded {name} from {filepath}")
    else:
        print(f"Warning: {filepath} not found")

# Model performance data from the notebook
model_performance = {
    'Random Baseline': {
        'accuracy': 0.5000, 'recall': 0.0769, 'precision': 0.0909, 'f1': 0.0833, 'auc': 0.3772
    },
    'SGD Baseline': {
        'accuracy': 0.8864, 'recall': 0.8462, 'precision': 0.7857, 'f1': 0.8148, 'auc': 0.9491  
    },
    'Random Forest': {
        'accuracy': 0.9318, 'recall': 0.7692, 'precision': 1.0000, 'f1': 0.8696, 'auc': 0.9715
    },
    'Improved Random Forest': {
        'accuracy': 0.9091, 'recall': 1.0000, 'precision': 0.7647, 'f1': 0.8667, 'auc': 0.9355
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return empty response with 204 No Content

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = request.json
        
        # Create feature vector with mean values for missing features
        feature_vector = []
        
        # Fill in provided features, use mean for missing ones
        for feature in features:
            if feature in input_data and input_data[feature] is not None and input_data[feature] != '':
                feature_vector.append(float(input_data[feature]))
            else:
                # Use mean value from dataset for missing features
                mean_val = data[feature].mean()
                feature_vector.append(float(mean_val))
        
        # Create pandas DataFrame with proper feature names to avoid sklearn warnings
        feature_df = pd.DataFrame([feature_vector], columns=features)
        
        # Make predictions with all models
        predictions = {}
        probabilities = {}
        
        # Ensure all 4 models are processed
        expected_models = ['Random Baseline', 'SGD Baseline', 'Random Forest', 'Improved Random Forest']
        
        for name in expected_models:
            if name in models:
                try:
                    model = models[name]
                    pred = model.predict(feature_df)[0]
                    
                    # Convert numpy types to Python native types for JSON serialization
                    if hasattr(pred, 'item'):
                        pred = pred.item()
                    
                    # Convert numerical predictions to meaningful labels
                    if pred == 1 or pred == '1':
                        predictions[name] = 'malignant'
                    elif pred == 0 or pred == '0':
                        predictions[name] = 'benign'
                    else:
                        predictions[name] = str(pred).lower()
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(feature_df)[0]
                        if len(prob) == 2:
                            probabilities[name] = {
                                'benign': float(prob[0]),
                                'malignant': float(prob[1])
                            }
                    elif hasattr(model, 'decision_function'):
                        decision = model.decision_function(feature_df)[0]
                        # Convert to probability using sigmoid
                        prob_malignant = 1 / (1 + np.exp(-float(decision)))
                        probabilities[name] = {
                            'benign': float(1 - prob_malignant),
                            'malignant': float(prob_malignant)
                        }
                    else:
                        # Default probabilities for models without probability methods
                        probabilities[name] = {
                            'benign': 0.5,
                            'malignant': 0.5
                        }
                except Exception as e:
                    print(f"Error with model {name}: {e}")
                    predictions[name] = "Error"
                    probabilities[name] = {'benign': 0.5, 'malignant': 0.5}
            else:
                print(f"Model {name} not found in loaded models")
                predictions[name] = "Model not found"
                probabilities[name] = {'benign': 0.5, 'malignant': 0.5}
        
        return jsonify({
            'predictions': predictions,
            'probabilities': probabilities,
            'input_features': input_data
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/feature_analysis')
def feature_analysis():
    # Calculate T-scores for all features
    malignant_samples = data[data[label] == 'malignant']
    benign_samples = data[data[label] == 'benign']
    
    t_scores = {}
    for feature in features:
        mean_malignant = malignant_samples[feature].mean()
        mean_benign = benign_samples[feature].mean()
        std_malignant = malignant_samples[feature].std()
        std_benign = benign_samples[feature].std()
        
        t_score = (mean_malignant - mean_benign) / (0.5 * (std_malignant + std_benign))
        t_scores[feature] = float(t_score)
    
    # Sort by absolute T-score
    sorted_features = sorted(t_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return jsonify({
        't_scores': t_scores,
        'sorted_features': sorted_features,
        'top_features': top_features,
        'feature_stats': {
            'malignant_means': malignant_samples[features].mean().to_dict(),
            'benign_means': benign_samples[features].mean().to_dict(),
            'malignant_stds': malignant_samples[features].std().to_dict(),
            'benign_stds': benign_samples[features].std().to_dict()
        }
    })

@app.route('/model_performance')
def get_model_performance():
    return jsonify(model_performance)

@app.route('/dataset_info')
def dataset_info():
    return jsonify({
        'total_samples': len(data),
        'features': features,
        'feature_count': len(features),
        'class_distribution': data[label].value_counts().to_dict(),
        'feature_ranges': {
            feature: {
                'min': float(data[feature].min()),
                'max': float(data[feature].max()),
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std())
            } for feature in features
        }
    })

@app.route('/visualization_data')
def visualization_data():
    # Prepare data for 3D visualization
    viz_data = {
        'samples': [],
        'features': top_features[:3]  # Use top 3 features for 3D plot
    }
    
    for _, row in data.iterrows():
        sample = {
            'label': row[label],
            'features': {feature: float(row[feature]) for feature in top_features[:3]}
        }
        viz_data['samples'].append(sample)
    
    return jsonify(viz_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 