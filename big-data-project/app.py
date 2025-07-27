#!/usr/bin/env python3
"""
Adelaide Traffic Prediction Dashboard
Interactive 3D Visualization with Machine Learning Integration

A sophisticated web application demonstrating advanced traffic prediction
capabilities using ThreeJS visualization and real-time data interaction.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import RobustScaler
import random

app = Flask(__name__)

# Simulate model loading (in real implementation, load the actual trained model)
class TrafficPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model_coefficients = {
            'VEHICLE_COUNT_ROLLING_MEAN_24': 170.99,
            'VEHICLE_COUNT_LAG_1': 50.64,
            'VEHICLE_COUNT_LAG_24': 26.95,
            'HOUR_COS': 17.80,
            'TRANSIT_TRIP_COUNT': 14.70,
            'DAY_OF_WEEK_SIN': 11.14,
            'IS_RAINY': 10.74,
            'IS_PEAK_HOUR': 9.91,
            'IS_NIGHT': 4.75,
            'DELAY_PEAK_INTERACTION': 3.57
        }
        
    def predict_traffic(self, features):
        """Simulate traffic prediction using our Ridge Regression model"""
        base_prediction = 800  # Base traffic volume
        
        # Apply feature coefficients
        for feature, coefficient in self.model_coefficients.items():
            if feature in features:
                base_prediction += features[feature] * coefficient * 0.1
        
        # Add some realistic noise
        noise = np.random.normal(0, 50)
        prediction = max(0, base_prediction + noise)
        
        return prediction

# Global model instance
predictor = TrafficPredictor()

# Adelaide intersection data with coordinates
ADELAIDE_INTERSECTIONS = {
    'INT1': {'name': 'King William St & North Tce', 'lat': -34.9205, 'lng': 138.5986, 'x': 0, 'z': 0},
    'INT2': {'name': 'North Tce & Pulteney St', 'lat': -34.9215, 'lng': 138.6035, 'x': 200, 'z': -50},
    'INT3': {'name': 'Rundle Mall & King William St', 'lat': -34.9240, 'lng': 138.5995, 'x': 50, 'z': -200},
    'INT4': {'name': 'Hindley St & King William St', 'lat': -34.9285, 'lng': 138.5985, 'x': -50, 'z': -400},
    'INT5': {'name': 'South Tce & King William St', 'lat': -34.9320, 'lng': 138.5980, 'x': -100, 'z': -600},
    'INT6': {'name': 'Grenfell St & Pulteney St', 'lat': -34.9250, 'lng': 138.6080, 'x': 300, 'z': -250},
    'INT7': {'name': 'Currie St & West Tce', 'lat': -34.9270, 'lng': 138.5920, 'x': -200, 'z': -300},
    'INT8': {'name': 'Franklin St & Morphett St', 'lat': -34.9300, 'lng': 138.5940, 'x': -150, 'z': -500},
    'INT9': {'name': 'Wakefield St & Pulteney St', 'lat': -34.9350, 'lng': 138.6050, 'x': 250, 'z': -700},
    'INT10': {'name': 'Gouger St & Morphett St', 'lat': -34.9380, 'lng': 138.5950, 'x': -100, 'z': -800},
}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/traffic/current')
def get_current_traffic():
    """Get current traffic data for all intersections"""
    current_time = datetime.now()
    hour = current_time.hour
    day_of_week = current_time.weekday()
    
    traffic_data = []
    
    for int_id, info in ADELAIDE_INTERSECTIONS.items():
        # Create realistic features
        features = {
            'HOUR_COS': np.cos(2 * np.pi * hour / 24),
            'DAY_OF_WEEK_SIN': np.sin(2 * np.pi * day_of_week / 7),
            'IS_PEAK_HOUR': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            'IS_NIGHT': 1 if hour < 6 or hour > 22 else 0,
            'IS_RAINY': random.choice([0, 1]),  # Simulate weather
            'TRANSIT_TRIP_COUNT': random.uniform(80, 120),
            'VEHICLE_COUNT_LAG_1': random.uniform(400, 1200),
            'VEHICLE_COUNT_LAG_24': random.uniform(500, 1000),
            'VEHICLE_COUNT_ROLLING_MEAN_24': random.uniform(600, 900),
            'DELAY_PEAK_INTERACTION': random.uniform(0, 5)
        }
        
        # Predict traffic
        predicted_volume = predictor.predict_traffic(features)
        
        # Determine congestion level
        if predicted_volume < 500:
            congestion_level = 'low'
            color = '#00ff00'
        elif predicted_volume < 800:
            congestion_level = 'medium'
            color = '#ffff00'
        else:
            congestion_level = 'high'
            color = '#ff0000'
        
        traffic_data.append({
            'id': int_id,
            'name': info['name'],
            'position': {'x': info['x'], 'y': 0, 'z': info['z']},
            'volume': round(predicted_volume),
            'congestion_level': congestion_level,
            'color': color,
            'features': features,
            'timestamp': current_time.isoformat()
        })
    
    return jsonify({
        'status': 'success',
        'data': traffic_data,
        'timestamp': current_time.isoformat(),
        'total_intersections': len(traffic_data)
    })

@app.route('/api/traffic/predict', methods=['POST'])
def predict_traffic():
    """Predict traffic for specific intersection and time"""
    data = request.json
    
    # Extract parameters
    intersection_id = data.get('intersection_id', 'INT1')
    target_hour = data.get('hour', datetime.now().hour)
    day_of_week = data.get('day_of_week', datetime.now().weekday())
    weather_conditions = data.get('weather', {})
    
    # Create features for prediction
    is_peak_hour = 1 if (7 <= target_hour <= 9) or (17 <= target_hour <= 19) else 0
    transit_delay = weather_conditions.get('transit_delay', 0)
    
    features = {
        'HOUR_COS': np.cos(2 * np.pi * target_hour / 24),
        'DAY_OF_WEEK_SIN': np.sin(2 * np.pi * day_of_week / 7),
        'IS_PEAK_HOUR': is_peak_hour,
        'IS_NIGHT': 1 if target_hour < 6 or target_hour > 22 else 0,
        'IS_RAINY': weather_conditions.get('is_rainy', 0),
        'TRANSIT_TRIP_COUNT': weather_conditions.get('transit_trips', 100),
        'VEHICLE_COUNT_LAG_1': data.get('historical_data', {}).get('lag_1', 700),
        'VEHICLE_COUNT_LAG_24': data.get('historical_data', {}).get('lag_24', 750),
        'VEHICLE_COUNT_ROLLING_MEAN_24': data.get('historical_data', {}).get('rolling_mean', 725),
        'DELAY_PEAK_INTERACTION': transit_delay * is_peak_hour
    }
    
    # Make prediction
    predicted_volume = predictor.predict_traffic(features)
    
    # Calculate confidence interval (simulate)
    confidence_lower = predicted_volume - 100
    confidence_upper = predicted_volume + 100
    
    return jsonify({
        'status': 'success',
        'prediction': {
            'intersection_id': intersection_id,
            'predicted_volume': round(predicted_volume),
            'confidence_interval': {
                'lower': round(confidence_lower),
                'upper': round(confidence_upper)
            },
            'hour': target_hour,
            'day_of_week': day_of_week,
            'features_used': features,
            'model_performance': {
                'rmse': 536.27,
                'mae': 462.05,
                'r_squared': 0.048,
                'mape': 89.0
            }
        }
    })

@app.route('/api/traffic/historical/<intersection_id>')
def get_historical_traffic(intersection_id):
    """Get historical traffic data for visualization"""
    # Generate realistic historical data
    current_time = datetime.now()
    historical_data = []
    
    for i in range(24):  # Last 24 hours
        time_point = current_time - timedelta(hours=i)
        hour = time_point.hour
        
        # Generate realistic traffic volume based on time
        base_volume = 400
        if 7 <= hour <= 9:  # Morning rush
            base_volume = 900
        elif 17 <= hour <= 19:  # Evening rush
            base_volume = 850
        elif 22 <= hour or hour <= 6:  # Night
            base_volume = 200
        
        # Add noise
        volume = base_volume + random.randint(-100, 100)
        
        historical_data.append({
            'timestamp': time_point.isoformat(),
            'hour': hour,
            'volume': max(0, volume)
        })
    
    return jsonify({
        'status': 'success',
        'intersection_id': intersection_id,
        'data': list(reversed(historical_data))  # Chronological order
    })

@app.route('/api/model/performance')
def get_model_performance():
    """Get model performance metrics"""
    return jsonify({
        'status': 'success',
        'model_info': {
            'name': 'Ridge Regression (Optimized)',
            'version': '1.0',
            'training_date': '2024-07-15',
            'performance_metrics': {
                'rmse': 536.27,
                'mae': 462.05,
                'r_squared': 0.048,
                'mape': 89.0
            },
            'feature_importance': predictor.model_coefficients,
            'hyperparameters': {
                'alpha': 10.0,
                'regularization': 'L2',
                'solver': 'auto'
            },
            'training_data': {
                'samples': 10000,
                'features': 15,
                'intersections': 20,
                'time_period': '2022-01-01 to 2022-12-31'
            }
        }
    })

@app.route('/api/weather/current')
def get_current_weather():
    """Get current weather conditions"""
    # Simulate current weather
    weather_conditions = random.choice([
        {'condition': 'sunny', 'temperature': 22, 'rainfall': 0, 'is_rainy': 0},
        {'condition': 'cloudy', 'temperature': 18, 'rainfall': 0, 'is_rainy': 0},
        {'condition': 'rainy', 'temperature': 15, 'rainfall': 5.2, 'is_rainy': 1},
        {'condition': 'partly_cloudy', 'temperature': 20, 'rainfall': 0, 'is_rainy': 0}
    ])
    
    return jsonify({
        'status': 'success',
        'weather': weather_conditions,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 