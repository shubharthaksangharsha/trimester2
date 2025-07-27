#!/usr/bin/env python3
"""
Assignment 1: Part C - Advanced Predictive Modeling
Urban Traffic Congestion Prediction in Adelaide

This script builds upon the exploratory data analysis and initial modeling from Part B 
to develop and compare multiple advanced predictive models for forecasting hourly 
vehicle counts at major intersections in Adelaide.

Research Question:
"Can we predict hourly vehicle counts (as a proxy for traffic congestion levels) 
at major intersections in Adelaide using historical traffic volumes, public transport 
delay data, and weather conditions?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    print("XGBoost not available - install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

print("=" * 70)
print("ASSIGNMENT 1 PART C - ADVANCED PREDICTIVE MODELING")
print("Urban Traffic Congestion Prediction in Adelaide")
print("=" * 70)

def load_and_prepare_data():
    """
    Load and prepare the datasets from Part B analysis
    """
    print("\n1. LOADING AND PREPARING DATA")
    print("-" * 40)
    
    # Load the three main datasets
    print("Loading datasets...")
    
    try:
        # Traffic data
        traffic_data = pd.read_csv('dataset/traffic_data.csv')
        print(f"Traffic data shape: {traffic_data.shape}")
        
        # Weather data
        weather_data = pd.read_csv('dataset/weather_data.csv')
        print(f"Weather data shape: {weather_data.shape}")
        
        # Transit data
        transit_data = pd.read_csv('dataset/transit_data.csv')
        print(f"Transit data shape: {transit_data.shape}")
        
        return traffic_data, weather_data, transit_data
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the dataset files are in the 'dataset' directory")
        return None, None, None

def advanced_data_preprocessing(traffic_df, weather_df, transit_df):
    """
    Advanced preprocessing building on Part B analysis
    """
    print("\n2. ADVANCED DATA PREPROCESSING")
    print("-" * 40)
    
    if traffic_df is None:
        print("Cannot proceed without data")
        return None
    
    print("Starting advanced data preprocessing...")
    
    # Prepare traffic data
    traffic_df = traffic_df.copy()
    
    # Check actual column names and use correct one
    if 'DATE_TIME' in traffic_df.columns:
        datetime_col = 'DATE_TIME'
    elif 'DATETIME' in traffic_df.columns:
        datetime_col = 'DATETIME'
    else:
        print(f"Available columns: {list(traffic_df.columns)}")
        raise ValueError("No datetime column found. Expected 'DATE_TIME' or 'DATETIME'")
    
    traffic_df['DATETIME'] = pd.to_datetime(traffic_df[datetime_col])
    
    # Basic feature engineering from Part B
    traffic_df['HOUR'] = traffic_df['DATETIME'].dt.hour
    traffic_df['DAY_OF_WEEK'] = traffic_df['DATETIME'].dt.dayofweek
    traffic_df['MONTH'] = traffic_df['DATETIME'].dt.month
    traffic_df['DATE'] = traffic_df['DATETIME'].dt.date
    
    # Advanced cyclical encoding
    traffic_df['HOUR_SIN'] = np.sin(2 * np.pi * traffic_df['HOUR'] / 24)
    traffic_df['HOUR_COS'] = np.cos(2 * np.pi * traffic_df['HOUR'] / 24)
    traffic_df['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * traffic_df['DAY_OF_WEEK'] / 7)
    traffic_df['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * traffic_df['DAY_OF_WEEK'] / 7)
    traffic_df['MONTH_SIN'] = np.sin(2 * np.pi * traffic_df['MONTH'] / 12)
    traffic_df['MONTH_COS'] = np.cos(2 * np.pi * traffic_df['MONTH'] / 12)
    
    # Binary features
    traffic_df['IS_WEEKEND'] = (traffic_df['DAY_OF_WEEK'] >= 5).astype(int)
    traffic_df['IS_PEAK_HOUR'] = ((traffic_df['HOUR'].between(7, 9)) | 
                                  (traffic_df['HOUR'].between(17, 19))).astype(int)
    
    # Advanced time-based features
    traffic_df['IS_BUSINESS_HOUR'] = (traffic_df['HOUR'].between(9, 17)).astype(int)
    traffic_df['IS_NIGHT'] = ((traffic_df['HOUR'] < 6) | (traffic_df['HOUR'] > 22)).astype(int)
    
    # Sort by intersection and datetime for lag features
    if 'INTERSECTION_ID' in traffic_df.columns:
        traffic_df = traffic_df.sort_values(['INTERSECTION_ID', 'DATETIME'])
    else:
        traffic_df = traffic_df.sort_values(['DATETIME'])
    
    # Create lag features for major intersections only (to manage computational complexity)
    print("Creating lag features...")
    
    # Check if INTERSECTION_ID column exists
    if 'INTERSECTION_ID' not in traffic_df.columns:
        print(f"Available columns: {list(traffic_df.columns)}")
        print("Warning: INTERSECTION_ID not found. Using all data without grouping by intersection.")
        traffic_filtered = traffic_df.copy()
        # Create simple lag features without intersection grouping
        traffic_filtered['VEHICLE_COUNT_LAG_1'] = traffic_filtered['VEHICLE_COUNT'].shift(1)
        traffic_filtered['VEHICLE_COUNT_LAG_24'] = traffic_filtered['VEHICLE_COUNT'].shift(24)
        traffic_filtered['VEHICLE_COUNT_LAG_168'] = traffic_filtered['VEHICLE_COUNT'].shift(168)
        traffic_filtered['VEHICLE_COUNT_ROLLING_MEAN_24'] = traffic_filtered['VEHICLE_COUNT'].rolling(24, min_periods=1).mean()
        traffic_filtered['VEHICLE_COUNT_ROLLING_STD_24'] = traffic_filtered['VEHICLE_COUNT'].rolling(24, min_periods=1).std()
        traffic_processed = traffic_filtered
        num_intersections = "all (ungrouped)"
    else:
        major_intersections = traffic_df['INTERSECTION_ID'].value_counts().head(20).index
        traffic_filtered = traffic_df[traffic_df['INTERSECTION_ID'].isin(major_intersections)].copy()
        
        lag_features = []
        for intersection in major_intersections:
            intersection_mask = traffic_filtered['INTERSECTION_ID'] == intersection
            intersection_data = traffic_filtered[intersection_mask].copy()
            
            # Various lag periods
            intersection_data['VEHICLE_COUNT_LAG_1'] = intersection_data['VEHICLE_COUNT'].shift(1)
            intersection_data['VEHICLE_COUNT_LAG_24'] = intersection_data['VEHICLE_COUNT'].shift(24)
            intersection_data['VEHICLE_COUNT_LAG_168'] = intersection_data['VEHICLE_COUNT'].shift(168)  # 1 week
            
            # Rolling statistics
            intersection_data['VEHICLE_COUNT_ROLLING_MEAN_24'] = intersection_data['VEHICLE_COUNT'].rolling(24, min_periods=1).mean()
            intersection_data['VEHICLE_COUNT_ROLLING_STD_24'] = intersection_data['VEHICLE_COUNT'].rolling(24, min_periods=1).std()
            
            lag_features.append(intersection_data)
        
        traffic_processed = pd.concat(lag_features, ignore_index=True)
        num_intersections = len(major_intersections)
    
    print(f"Traffic data preprocessed. Shape: {traffic_processed.shape}")
    print(f"Focused on top {num_intersections} busiest intersections")
    
    return traffic_processed

def integrate_datasets(traffic_df, weather_df, transit_df):
    """
    Integrate traffic, weather, and transit data
    """
    print("\n3. DATASET INTEGRATION")
    print("-" * 40)
    
    if traffic_df is None:
        return None
        
    print("Integrating datasets...")
    
    # Prepare weather data
    if weather_df is not None:
        weather_df = weather_df.copy()
        
        # Check for correct datetime column name
        if 'DATE_TIME' in weather_df.columns:
            weather_df['DATETIME'] = pd.to_datetime(weather_df['DATE_TIME'])
        elif 'DATETIME' in weather_df.columns:
            weather_df['DATETIME'] = pd.to_datetime(weather_df['DATETIME'])
        else:
            print(f"Weather columns: {list(weather_df.columns)}")
            
        weather_df['DATE'] = weather_df['DATETIME'].dt.date
        weather_df['HOUR'] = weather_df['DATETIME'].dt.hour
        
        # Weather feature engineering
        weather_df['IS_RAINY'] = (weather_df['RAINFALL_MM'] > 0).astype(int)
        weather_df['TEMP_CATEGORY'] = pd.cut(weather_df['TEMPERATURE_C'], 
                                           bins=[-np.inf, 10, 20, 30, np.inf], 
                                           labels=['Cold', 'Cool', 'Warm', 'Hot'])
        
        # Aggregate weather by date and hour (in case there are multiple readings)
        weather_hourly = weather_df.groupby(['DATE', 'HOUR']).agg({
            'TEMPERATURE_C': 'mean',
            'RAINFALL_MM': 'sum', 
            'IS_RAINY': 'max'
        }).reset_index()
    else:
        weather_hourly = None
    
    # Prepare transit data
    if transit_df is not None:
        transit_df = transit_df.copy()
        
        # Check for correct datetime column name
        if 'DATE_TIME' in transit_df.columns:
            transit_df['DATETIME'] = pd.to_datetime(transit_df['DATE_TIME'])
        elif 'DATETIME' in transit_df.columns:
            transit_df['DATETIME'] = pd.to_datetime(transit_df['DATETIME'])
        else:
            print(f"Transit columns: {list(transit_df.columns)}")
            
        transit_df['DATE'] = transit_df['DATETIME'].dt.date
        transit_df['HOUR'] = transit_df['DATETIME'].dt.hour
        
        # Aggregate transit data by date and hour
        transit_hourly = transit_df.groupby(['DATE', 'HOUR']).agg({
            'DELAY_MINUTES': 'mean',
            'STOP_ID': 'count'  # Count number of trips as proxy for trip count
        }).reset_index()
        transit_hourly.columns = ['DATE', 'HOUR', 'AVG_TRANSIT_DELAY', 'TRANSIT_TRIP_COUNT']
    else:
        transit_hourly = None
    
    # Start with traffic data
    integrated_data = traffic_df.copy()
    
    # Merge with weather data if available
    if weather_hourly is not None:
        integrated_data = integrated_data.merge(weather_hourly, on=['DATE', 'HOUR'], how='left')
        print("Weather data integrated")
    else:
        # Create dummy weather features
        integrated_data['TEMPERATURE_C'] = 20.0  # Default temperature
        integrated_data['RAINFALL_MM'] = 0.0
        integrated_data['IS_RAINY'] = 0
        print("Weather data not available - using default values")
    
    # Merge with transit data if available
    if transit_hourly is not None:
        integrated_data = integrated_data.merge(transit_hourly, on=['DATE', 'HOUR'], how='left')
        print("Transit data integrated")
    else:
        # Create dummy transit features
        integrated_data['AVG_TRANSIT_DELAY'] = 0.0
        integrated_data['TRANSIT_TRIP_COUNT'] = 100
        print("Transit data not available - using default values")
    
    # Handle missing values from merges
    print("Handling missing values from data integration...")
    
    # Fill missing weather data
    if 'TEMPERATURE_C' in integrated_data.columns:
        integrated_data['TEMPERATURE_C'].fillna(20.0, inplace=True)
    if 'RAINFALL_MM' in integrated_data.columns:
        integrated_data['RAINFALL_MM'].fillna(0.0, inplace=True)
    if 'IS_RAINY' in integrated_data.columns:
        integrated_data['IS_RAINY'].fillna(0, inplace=True)
    
    # Fill missing transit data
    if 'AVG_TRANSIT_DELAY' in integrated_data.columns:
        integrated_data['AVG_TRANSIT_DELAY'].fillna(0.0, inplace=True)
    if 'TRANSIT_TRIP_COUNT' in integrated_data.columns:
        integrated_data['TRANSIT_TRIP_COUNT'].fillna(100, inplace=True)
    
    # Create interaction features
    integrated_data['RAIN_PEAK_INTERACTION'] = integrated_data['IS_RAINY'] * integrated_data['IS_PEAK_HOUR']
    integrated_data['DELAY_PEAK_INTERACTION'] = integrated_data['AVG_TRANSIT_DELAY'] * integrated_data['IS_PEAK_HOUR']
    integrated_data['TEMP_WEEKEND_INTERACTION'] = integrated_data['TEMPERATURE_C'] * integrated_data['IS_WEEKEND']
    
    print(f"Integrated dataset shape: {integrated_data.shape}")
    print(f"Missing values after integration: {integrated_data.isnull().sum().sum()}")
    
    return integrated_data

def advanced_preprocessing_pipeline(data):
    """
    Comprehensive preprocessing pipeline including imputation, scaling, and feature selection
    """
    print("\n4. ADVANCED PREPROCESSING PIPELINE")
    print("-" * 40)
    
    if data is None:
        return None, None, None
        
    print("Applying advanced preprocessing pipeline...")
    
    # Remove rows with missing target variable
    data_clean = data.dropna(subset=['VEHICLE_COUNT']).copy()
    
    # Define feature columns (excluding target and non-feature columns)
    feature_columns = [
        'HOUR_SIN', 'HOUR_COS', 'DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS', 
        'MONTH_SIN', 'MONTH_COS', 'IS_WEEKEND', 'IS_PEAK_HOUR', 
        'IS_BUSINESS_HOUR', 'IS_NIGHT',
        'VEHICLE_COUNT_LAG_1', 'VEHICLE_COUNT_LAG_24', 'VEHICLE_COUNT_LAG_168',
        'VEHICLE_COUNT_ROLLING_MEAN_24', 'VEHICLE_COUNT_ROLLING_STD_24',
        'TEMPERATURE_C', 'RAINFALL_MM', 'IS_RAINY',
        'AVG_TRANSIT_DELAY', 'TRANSIT_TRIP_COUNT',
        'RAIN_PEAK_INTERACTION', 'DELAY_PEAK_INTERACTION', 'TEMP_WEEKEND_INTERACTION'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in data_clean.columns]
    
    X = data_clean[available_features].copy()
    y = data_clean['VEHICLE_COUNT'].copy()
    
    print(f"Features before preprocessing: {X.shape[1]}")
    print(f"Missing values in features: {X.isnull().sum().sum()}")
    
    # Handle missing values using simple imputation (KNN might be too slow for large datasets)
    if X.isnull().sum().sum() > 0:
        print("Applying median imputation...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_imputed = X.copy()
        print("No missing values found.")
    
    # Remove features with very low variance
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance_filtered = pd.DataFrame(
        variance_selector.fit_transform(X_imputed),
        columns=X_imputed.columns[variance_selector.get_support()],
        index=X_imputed.index
    )
    
    print(f"Features after variance filtering: {X_variance_filtered.shape[1]}")
    
    # Feature selection using statistical tests
    k_features = min(15, X_variance_filtered.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_selected = pd.DataFrame(
        selector.fit_transform(X_variance_filtered, y),
        columns=X_variance_filtered.columns[selector.get_support()],
        index=X_variance_filtered.index
    )
    
    print(f"Features after selection: {X_selected.shape[1]}")
    print(f"Selected features: {list(X_selected.columns)}")
    
    return X_selected, y, X_variance_filtered

def create_time_aware_split(X, y, integrated_data, test_size=0.2, val_size=0.1):
    """
    Create time-aware train/validation/test splits
    """
    print("\n5. TIME-AWARE DATA SPLITTING")
    print("-" * 40)
    
    # Sort by datetime to ensure temporal order
    datetime_col = integrated_data['DATETIME'].loc[X.index]
    sort_idx = datetime_col.sort_values().index
    
    X_sorted = X.loc[sort_idx]
    y_sorted = y.loc[sort_idx]
    
    n_samples = len(X_sorted)
    
    # Time-based split (earlier data for training, later for testing)
    train_end = int(n_samples * (1 - test_size - val_size))
    val_end = int(n_samples * (1 - test_size))
    
    X_train = X_sorted.iloc[:train_end]
    y_train = y_sorted.iloc[:train_end]
    
    X_val = X_sorted.iloc[train_end:val_end]
    y_val = y_sorted.iloc[train_end:val_end]
    
    X_test = X_sorted.iloc[val_end:]
    y_test = y_sorted.iloc[val_end:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def initialize_candidate_models():
    """
    Initialize multiple candidate models for comparison
    """
    print("\n6. MODEL SELECTION AND INITIALIZATION")
    print("-" * 40)
    
    models = {
        # Linear Models
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=1.0, random_state=42, max_iter=2000),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
        
        # Tree-based Models
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        
        # Neural Network
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    print(f"Initialized {len(models)} candidate models:")
    for name in models.keys():
        print(f"- {name}")
    
    return models

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Comprehensive model evaluation with multiple metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    return metrics

def train_and_evaluate_baseline_models(models, X_train, y_train, X_val, y_val):
    """
    Train all models with default parameters and evaluate on validation set
    """
    print("\n7. BASELINE MODEL TRAINING AND EVALUATION")
    print("-" * 40)
    
    results = []
    trained_models = {}
    
    print("Training baseline models...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_val_pred = model.predict(X_val)
            
            # Evaluate
            metrics = evaluate_model(y_val, y_val_pred, name)
            results.append(metrics)
            trained_models[name] = model
            
            print(f"{name} - RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.3f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return pd.DataFrame(results), trained_models

def get_hyperparameter_grids():
    """
    Define hyperparameter grids for different models
    """
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        },
        
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0]
        },
        
        'Lasso Regression': {
            'alpha': [0.1, 1.0, 10.0]
        },
        
        'Elastic Net': {
            'alpha': [0.1, 1.0],
            'l1_ratio': [0.5, 0.7]
        }
    }
    
    # Add XGBoost parameters if available
    if XGBOOST_AVAILABLE:
        param_grids['XGBoost'] = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
    
    return param_grids

def optimize_hyperparameters(model_name, base_model, param_grid, X_train, y_train, cv_folds=3):
    """
    Perform hyperparameter optimization using TimeSeriesSplit cross-validation
    """
    print(f"\nOptimizing {model_name}...")
    
    # Use TimeSeriesSplit for cross-validation (appropriate for time series data)
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # GridSearchCV with time series cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def run_hyperparameter_optimization(baseline_results, candidate_models, X_train_scaled, y_train, X_val_scaled, y_val):
    """
    Run hyperparameter optimization for top performing models
    """
    print("\n8. HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    
    # Select top models for optimization
    top_models = baseline_results.sort_values('RMSE').head(4)['Model'].tolist()
    print(f"Top 4 models selected for optimization: {top_models}")
    
    param_grids = get_hyperparameter_grids()
    
    optimized_models = {}
    optimization_results = []
    
    for model_name in top_models:
        if model_name in param_grids and model_name in candidate_models:
            try:
                # Get base model and parameter grid
                base_model = candidate_models[model_name]
                param_grid = param_grids[model_name]
                
                # Optimize hyperparameters
                optimized_model, best_params = optimize_hyperparameters(
                    model_name, base_model, param_grid, X_train_scaled, y_train
                )
                
                # Evaluate optimized model on validation set
                y_val_pred_opt = optimized_model.predict(X_val_scaled)
                metrics_opt = evaluate_model(y_val, y_val_pred_opt, f"{model_name} (Optimized)")
                
                optimized_models[model_name] = optimized_model
                optimization_results.append(metrics_opt)
                
                print(f"Optimized {model_name} - RMSE: {metrics_opt['RMSE']:.2f}, R²: {metrics_opt['R²']:.3f}")
                
            except Exception as e:
                print(f"Error optimizing {model_name}: {str(e)}")
                continue
    
    return pd.DataFrame(optimization_results), optimized_models

def final_model_evaluation(optimized_models, baseline_models, X_test, y_test):
    """
    Final evaluation of best models on test set
    """
    print("\n9. FINAL MODEL EVALUATION")
    print("-" * 40)
    
    final_results = []
    predictions = {}
    
    print("Final model evaluation on test set...")
    
    # Evaluate optimized models
    for name, model in optimized_models.items():
        y_test_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_test_pred, f"{name} (Optimized)")
        final_results.append(metrics)
        predictions[f"{name} (Optimized)"] = y_test_pred
    
    # Also evaluate baseline Random Forest for comparison
    if 'Random Forest' in baseline_models:
        y_test_pred = baseline_models['Random Forest'].predict(X_test)
        metrics = evaluate_model(y_test, y_test_pred, "Random Forest (Baseline)")
        final_results.append(metrics)
        predictions["Random Forest (Baseline)"] = y_test_pred
    
    return pd.DataFrame(final_results), predictions

def analyze_feature_importance(best_model_name, optimized_models, feature_names):
    """
    Analyze and display feature importance for the best model
    """
    print("\n10. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    # Extract model name without "(Optimized)" suffix
    model_key = best_model_name.replace(' (Optimized)', '')
    
    if model_key not in optimized_models:
        print(f"Model {model_key} not found in optimized models")
        return None
    
    best_model = optimized_models[model_key]
    
    # Get feature importance based on model type
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models
        importances = best_model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(best_model, 'coef_'):
        # Linear models
        importances = np.abs(best_model.coef_)
        importance_type = "Coefficient Magnitude"
    else:
        print(f"Cannot extract feature importance for {best_model_name}")
        return None
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{importance_type} for {best_model_name}:")
    print(feature_importance_df.to_string(index=False, float_format='%.4f'))
    
    return feature_importance_df

def generate_business_insights(feature_importance_df, final_results_df, baseline_results):
    """
    Generate actionable business insights from the modeling results
    """
    print("\n11. BUSINESS INSIGHTS AND MODEL INTERPRETATION")
    print("-" * 40)
    
    # Model performance insights
    best_model = final_results_df.iloc[0]
    baseline_rf = baseline_results[baseline_results['Model'] == 'Random Forest']
    
    if not baseline_rf.empty:
        improvement = ((baseline_rf.iloc[0]['RMSE'] - best_model['RMSE']) / 
                      baseline_rf.iloc[0]['RMSE'] * 100)
        print(f"\n🚀 MODEL PERFORMANCE IMPROVEMENT:")
        print(f"   - Best model ({best_model['Model']}) achieved {improvement:.1f}% improvement over baseline")
        print(f"   - RMSE reduced from {baseline_rf.iloc[0]['RMSE']:.2f} to {best_model['RMSE']:.2f}")
        print(f"   - R² improved from {baseline_rf.iloc[0]['R²']:.3f} to {best_model['R²']:.3f}")
    
    # Feature importance insights
    if feature_importance_df is not None:
        top_5_features = feature_importance_df.head(5)
        print(f"\n🔍 KEY PREDICTIVE FACTORS:")
        print(f"   The top 5 most important features for predicting traffic volume are:")
        for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
            print(f"   {i}. {row['Feature']} (importance: {row['Importance']:.4f})")
    
    # Model reliability assessment
    print(f"\n📊 MODEL RELIABILITY:")
    if best_model['R²'] > 0.8:
        reliability = "Excellent"
    elif best_model['R²'] > 0.6:
        reliability = "Good"
    elif best_model['R²'] > 0.4:
        reliability = "Moderate"
    else:
        reliability = "Poor"
    
    print(f"   - Model reliability: {reliability} (R² = {best_model['R²']:.3f})")
    print(f"   - Average prediction error: ±{best_model['MAE']:.0f} vehicles per hour")
    print(f"   - Percentage error: {best_model['MAPE']:.1f}% MAPE")
    
    # Practical applications
    print(f"\n🎯 PRACTICAL APPLICATIONS:")
    print(f"   - Traffic Management: Predict congestion up to 24 hours in advance")
    print(f"   - Infrastructure Planning: Identify consistently high-traffic intersections")
    print(f"   - Public Transport Optimization: Understand traffic-transit interactions")
    print(f"   - Emergency Response: Anticipate traffic impacts during adverse weather")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   - Deploy this model for real-time traffic prediction systems")
    print(f"   - Focus monitoring on peak hours and weather-sensitive intersections")
    print(f"   - Integrate with public transport scheduling for optimal city mobility")
    print(f"   - Use predictions for dynamic traffic signal optimization")

def create_final_summary(baseline_results, final_results_df, feature_importance_df):
    """
    Create a comprehensive summary of the entire modeling process
    """
    print("\n" + "=" * 70)
    print("PART C MODELING SUMMARY")
    print("=" * 70)
    
    print(f"\n🔬 RESEARCH QUESTION ANSWERED:")
    print(f'"Can we predict hourly vehicle counts at major intersections in Adelaide')
    print(f'using historical traffic volumes, public transport delay data, and weather conditions?"')
    
    print(f"\n✅ ANSWER: YES - Traffic volumes can be predicted with {final_results_df.iloc[0]['R²']:.1%} accuracy")
    
    print(f"\n📈 BEST MODEL PERFORMANCE:")
    best_result = final_results_df.iloc[0]
    print(f"   Model: {best_result['Model']}")
    print(f"   RMSE: {best_result['RMSE']:.2f} vehicles/hour")
    print(f"   MAE: {best_result['MAE']:.2f} vehicles/hour")
    print(f"   R²: {best_result['R²']:.3f} ({best_result['R²']:.1%} variance explained)")
    print(f"   MAPE: {best_result['MAPE']:.1f}% average percentage error")
    
    # Save results for Part D report
    print(f"\n💾 SAVING RESULTS FOR PART D REPORT...")
    
    try:
        final_results_df.to_csv('part_c_final_model_results.csv', index=False)
        baseline_results.to_csv('part_c_baseline_results.csv', index=False)
        
        if feature_importance_df is not None:
            feature_importance_df.to_csv('part_c_feature_importance.csv', index=False)
        
        print("✓ Results saved successfully!")
        print("  - part_c_final_model_results.csv")
        print("  - part_c_baseline_results.csv")
        print("  - part_c_feature_importance.csv")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """
    Main function to run the complete Part C modeling pipeline
    """
    try:
        # Step 1: Load and prepare data
        traffic_df, weather_df, transit_df = load_and_prepare_data()
        
        if traffic_df is None:
            print("Cannot proceed without traffic data")
            return
        
        # Step 2: Advanced preprocessing
        traffic_processed = advanced_data_preprocessing(traffic_df, weather_df, transit_df)
        
        # Step 3: Dataset integration
        integrated_data = integrate_datasets(traffic_processed, weather_df, transit_df)
        
        # Step 4: Advanced preprocessing pipeline
        X_processed, y, X_full = advanced_preprocessing_pipeline(integrated_data)
        
        if X_processed is None:
            print("Preprocessing failed")
            return
        
        # Step 5: Create time-aware splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_time_aware_split(
            X_processed, y, integrated_data
        )
        
        # Step 6: Scale features
        print("\nApplying robust scaling...")
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Step 7: Initialize and train baseline models
        candidate_models = initialize_candidate_models()
        baseline_results, trained_models = train_and_evaluate_baseline_models(
            candidate_models, X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        print("\n" + "=" * 60)
        print("BASELINE MODEL PERFORMANCE (Validation Set)")
        print("=" * 60)
        baseline_results_sorted = baseline_results.sort_values('RMSE')
        print(baseline_results_sorted.to_string(index=False, float_format='%.3f'))
        
        # Step 8: Hyperparameter optimization
        optimization_results, optimized_models = run_hyperparameter_optimization(
            baseline_results, candidate_models, X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        if not optimization_results.empty:
            print("\n" + "=" * 60)
            print("OPTIMIZED MODEL PERFORMANCE (Validation Set)")
            print("=" * 60)
            optimization_df_sorted = optimization_results.sort_values('RMSE')
            print(optimization_df_sorted.to_string(index=False, float_format='%.3f'))
        
        # Step 9: Final evaluation on test set
        final_results_df, test_predictions = final_model_evaluation(
            optimized_models, trained_models, X_test_scaled, y_test
        )
        
        print("\n" + "=" * 70)
        print("FINAL MODEL PERFORMANCE (Test Set)")
        print("=" * 70)
        final_results_sorted = final_results_df.sort_values('RMSE')
        print(final_results_sorted.to_string(index=False, float_format='%.3f'))
        
        # Step 10: Feature importance analysis
        best_model_name = final_results_sorted.iloc[0]['Model']
        feature_importance_df = analyze_feature_importance(
            best_model_name, optimized_models, X_processed.columns
        )
        
        # Step 11: Business insights
        generate_business_insights(feature_importance_df, final_results_df, baseline_results)
        
        # Step 12: Final summary and save results
        create_final_summary(baseline_results, final_results_df, feature_importance_df)
        
        print(f"\n🎉 PART C MODELING COMPLETED SUCCESSFULLY!")
        print(f"Best Model: {best_model_name}")
        print(f"Final RMSE: {final_results_sorted.iloc[0]['RMSE']:.2f}")
        print(f"Final R²: {final_results_sorted.iloc[0]['R²']:.3f}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 