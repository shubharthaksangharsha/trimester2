import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Seoul Bike Rental Prediction", page_icon="🚲", layout="wide")

# Custom CSS - Force dark mode only
st.markdown("""
<style>
    /* Force dark mode */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #121212 !important;
        color: white !important;
    }
    
    /* Override any light theme elements */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        border-right: 1px solid #333 !important;
    }
    
    /* Disable theme switcher */
    [data-testid="baseButton-secondary"] {
        display: none !important;
    }
    
    /* Core styling */
    .title {
        color: #ffffff;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subtitle {
        color: #E0E0E0;
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2F80ED;
        font-size: 1.8rem;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3082F5;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: rgba(79, 79, 79, 0.2);
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Text visibility improvements - force white text everywhere */
    p, li, div, span, label {
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: bold;
    }
    
    /* Fix sidebar text */
    .st-emotion-cache-16idsys p, .st-emotion-cache-16idsys span, .st-emotion-cache-16idsys div {
        color: white !important;
    }
    
    /* Fix radio buttons in sidebar */
    .st-emotion-cache-16idsys label {
        color: white !important;
    }
    
    /* Fix metrics */
    [data-testid="stMetricValue"] {
        color: #4ECDC4 !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: white !important;
    }
    
    /* Fix tables and dataframes */
    [data-testid="stTable"] {
        color: white !important;
        background-color: rgba(50, 50, 50, 0.3) !important;
    }
    .dataframe {
        color: white !important;
    }
    th {
        background-color: #1E3A5F !important;
        color: white !important;
    }
    td {
        color: white !important;
    }
    
    /* Fix status messages like info, success, etc */
    .stAlert {
        background-color: rgba(50, 50, 50, 0.7) !important;
        color: white !important;
    }
    [data-baseweb="notification"] {
        background-color: #1E3A5F !important;
    }
    
    /* Fix expander */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Make model performance box stand out */
    div[style*="background-color:#3082F5"] {
        background-color: #3082F5 !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    div[style*="background-color:#3082F5"] h3 {
        color: white !important;
        font-weight: bold !important;
    }
    div[style*="background-color:#3082F5"] p {
        color: white !important;
    }
    div[style*="background-color:#3082F5"] strong {
        color: #E2F0FF !important;
        font-weight: bold !important;
    }
    
    /* Fix markdown text */
    .st-emotion-cache-nahz7x p {
        color: white !important;
    }
    
    /* Fix tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: pre-wrap;
        background-color: rgba(79, 79, 79, 0.2);
        border-radius: 4px;
        color: white;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3082F5 !important;
        color: white !important;
    }
    
    /* Fix progress bars */
    .stProgress > div > div {
        background-color: #3082F5 !important;
    }
    
    /* Force dark buttons */
    button {
        background-color: #1E3A5F !important;
        color: white !important;
    }
    
    /* Fix light theme elements */
    [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    
    /* Keep info messages visible */
    .stAlert p {
        opacity: 1 !important;
    }
    
    /* Fix code blocks */
    code {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for weekday conversion
def date_is_weekday(datestring):
    dsplit = datestring.split('/')
    wday = datetime.datetime(int(dsplit[2]), int(dsplit[1]), int(dsplit[0])).weekday()
    return int(wday <= 4)

# Function to clean data
@st.cache_data
def load_and_clean_data():
    # Load original data
    bike_data = pd.read_csv('dataset/SeoulBikeData.csv', encoding='unicode_escape')
    
    # Make a copy of the original data
    cleaned_bike_data = bike_data.copy()
    
    # Remove rows where business is closed and delete the 'Functioning Day' column
    cleaned_bike_data = cleaned_bike_data[cleaned_bike_data['Functioning Day'] == 'Yes']
    cleaned_bike_data = cleaned_bike_data.drop(['Functioning Day'], axis=1)
    
    # Convert seasons to one-hot encoded format
    seasons_dummies = pd.get_dummies(cleaned_bike_data['Seasons'], prefix='Season')
    cleaned_bike_data = pd.concat([cleaned_bike_data, seasons_dummies], axis=1)
    cleaned_bike_data = cleaned_bike_data.drop(['Seasons'], axis=1)
    
    # Convert Date to binary Weekday feature
    cleaned_bike_data['Weekday'] = cleaned_bike_data['Date'].apply(date_is_weekday)
    cleaned_bike_data = cleaned_bike_data.drop(['Date'], axis=1)
    
    # Convert Holiday feature to binary
    cleaned_bike_data['Holiday'] = cleaned_bike_data['Holiday'].map({'Holiday': 1, 'No Holiday': 0})
    
    # Fix extreme outliers
    # Temperature
    cleaned_bike_data['Temperature (C)'] = cleaned_bike_data['Temperature (C)'].clip(-20, 40)
    
    # Humidity
    cleaned_bike_data['Humidity (%)'] = cleaned_bike_data['Humidity (%)'].clip(0, 100)
    
    # Wind speed
    cleaned_bike_data['Wind speed (m/s)'] = cleaned_bike_data['Wind speed (m/s)'].clip(0, 30)
    
    # Bike count
    cleaned_bike_data['Rented Bike Count'] = cleaned_bike_data['Rented Bike Count'].clip(upper=8000)
    
    # Clean other numerical columns with standard statistical outlier detection
    numerical_cols = cleaned_bike_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col not in ['Temperature (C)', 'Humidity (%)', 'Wind speed (m/s)', 'Rented Bike Count']:
            mean = cleaned_bike_data[col].mean()
            std = cleaned_bike_data[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            cleaned_bike_data[col] = cleaned_bike_data[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Fix negative values
    for col in numerical_cols:
        if col not in ['Temperature (C)', 'Dew point temperature (C)']:  # These can be negative
            cleaned_bike_data[col] = cleaned_bike_data[col].clip(lower=0)
    
    return bike_data, cleaned_bike_data

# Function to perform preprocessing and model training
@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_and_train_models(cleaned_data):
    # Create a placeholder for progress bars and status updates
    progress_placeholder = st.empty()
    status_text = st.empty()
    
    try:
        # Initialize progress
        status_text.text("Step 1/7: Preparing data...")
        progress_bar = progress_placeholder.progress(0)
        
        # Split features and target
        X = cleaned_data.drop(['Rented Bike Count'], axis=1)
        y = cleaned_data['Rented Bike Count']
        
        # Check for non-numeric values and convert them
        status_text.text("Step 2/7: Cleaning non-numeric values...")
        progress_bar.progress(15)
        
        non_numeric_counts = {}
        for col in X.columns:
            # Try to convert to numeric, set errors='coerce' to convert problematic values to NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Fill NaN values with column median
        for col in X.columns:
            if X[col].isna().sum() > 0:
                non_numeric_counts[col] = X[col].isna().sum()
                X[col] = X[col].fillna(X[col].median())
        
        # Report non-numeric values at the end to keep progress flowing
        for col, count in non_numeric_counts.items():
            st.info(f"Found {count} non-numeric values in column '{col}'. Replaced with median.")
        
        # Split train/test
        status_text.text("Step 3/7: Splitting data into train/test sets...")
        progress_bar.progress(25)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        binary_cols = ['Holiday', 'Weekday', 'Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']
        numeric_cols = [col for col in numeric_cols if col not in binary_cols]
        
        # Feature engineering
        status_text.text("Step 4/7: Performing feature engineering...")
        progress_bar.progress(35)
        X_engineered = X_train.copy()
        X_test_engineered = X_test.copy()
        
        # Add polynomial features
        poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_cols = ['Temperature (C)', 'Humidity (%)']
        poly_df = pd.DataFrame(
            poly_features.fit_transform(X_train[poly_cols]), 
            columns=[f"{col}_poly_{i}" for i, col in enumerate(poly_features.get_feature_names_out(poly_cols))]
        )
        poly_df_test = pd.DataFrame(
            poly_features.transform(X_test[poly_cols]),
            columns=[f"{col}_poly_{i}" for i, col in enumerate(poly_features.get_feature_names_out(poly_cols))]
        )
        
        X_engineered = pd.concat([X_engineered.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
        X_test_engineered = pd.concat([X_test_engineered.reset_index(drop=True), poly_df_test.reset_index(drop=True)], axis=1)
        
        # Create time interactions
        progress_bar.progress(45)
        for season in ['Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']:
            X_engineered[f'Hour_{season}'] = X_engineered['Hour'] * X_engineered[season]
            X_test_engineered[f'Hour_{season}'] = X_test_engineered['Hour'] * X_test_engineered[season]
        
        # Create time period features
        def categorize_hour(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
                
        X_engineered['TimePeriod'] = X_engineered['Hour'].apply(categorize_hour)
        X_test_engineered['TimePeriod'] = X_test_engineered['Hour'].apply(categorize_hour)
        
        time_period_dummies = pd.get_dummies(X_engineered['TimePeriod'], prefix='TimePeriod')
        time_period_dummies_test = pd.get_dummies(X_test_engineered['TimePeriod'], prefix='TimePeriod')
        
        X_engineered = pd.concat([X_engineered, time_period_dummies], axis=1)
        X_test_engineered = pd.concat([X_test_engineered, time_period_dummies_test], axis=1)
        X_engineered = X_engineered.drop('TimePeriod', axis=1)
        X_test_engineered = X_test_engineered.drop('TimePeriod', axis=1)
        
        # Feature selection with RandomForest
        status_text.text("Step 5/7: Selecting important features...")
        progress_bar.progress(55)
        feature_selector = RandomForestRegressor(n_estimators=50, random_state=42)
        feature_selector.fit(X_engineered, y_train)
        
        feature_importances = feature_selector.feature_importances_
        features_df = pd.DataFrame({
            'feature': X_engineered.columns,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_n = 20
        selected_features = features_df.head(top_n)['feature'].values
        
        X_selected = X_engineered[selected_features]
        X_test_selected = X_test_engineered[selected_features]
        
        # Preprocessing pipeline
        progress_bar.progress(65)
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), list(selected_features))
            ])
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=10.0),
            'SVR': SVR(C=100.0, epsilon=0.01, kernel='rbf'),
            'Random Forest': RandomForestRegressor(
                n_estimators=413, 
                max_depth=25,
                min_samples_split=13,
                min_samples_leaf=1,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
        }
        
        # Train models and get results
        status_text.text("Step 6/7: Training models...")
        progress_bar.progress(75)
        results = {}
        model_count = len(models)
        
        for i, (name, model) in enumerate(models.items()):
            sub_progress = int(75 + (i / model_count) * 20)
            status_text.text(f"Training model {i+1}/{model_count}: {name}...")
            progress_bar.progress(sub_progress)
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Fit model
            pipeline.fit(X_selected, y_train)
            
            # Make predictions
            y_train_pred = pipeline.predict(X_selected)
            y_test_pred = pipeline.predict(X_test_selected)
            
            # Calculate RMSE
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Store results
            results[name] = {
                'pipeline': pipeline,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
        
        # Complete progress
        status_text.text("Step 7/7: Finalizing results...")
        progress_bar.progress(100)
        
        # Clear progress indicators when done
        progress_placeholder.empty()
        status_text.empty()
        
        return results, features_df, X_selected, X_test_selected, y_train, y_test
    
    except Exception as e:
        # In case of error, clear progress indicators
        progress_placeholder.empty()
        status_text.empty()
        # Re-raise the exception
        raise e

# Main App
def main():
    st.markdown('<h1 class="title">Seoul Bike Rental Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analysis and prediction model for bike rental demand</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading and cleaning data...'):
        original_data, cleaned_data = load_and_clean_data()
    
    # Train models with a more descriptive message
    st.markdown("### Model Training")
    st.info("Loading and training models (this may take a few minutes on first run, but will be cached for subsequent views).")
    
    # Get or train the models
    model_results, feature_importance_df, X_selected, X_test_selected, y_train, y_test = preprocess_and_train_models(cleaned_data)
    
    # After successful training/loading
    st.success("Models ready! Navigate through the sidebar to explore results.")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Model Performance", "Feature Importance", "Predictions"])

    if page == "Overview":
        display_overview_page(original_data, cleaned_data, model_results)
    
    elif page == "Data Exploration":
        display_data_exploration(original_data, cleaned_data)
    
    elif page == "Model Performance":
        display_model_performance(model_results)
    
    elif page == "Feature Importance":
        display_feature_importance(feature_importance_df, model_results)
    
    elif page == "Predictions":
        display_predictions(model_results, y_test)

def display_overview_page(original_data, cleaned_data, model_results):
    st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Records", f"{len(original_data):,}")
    
    with col2:
        st.metric("Features", f"{len(original_data.columns) - 1}")
    
    with col3:
        best_model = min(model_results.items(), key=lambda x: x[1]['test_rmse'])
        st.metric("Best Model", best_model[0])
    
    with col4:
        improvement = ((646.17 - best_model[1]['test_rmse']) / 646.17) * 100
        st.metric("Improvement over Baseline", f"{improvement:.1f}%")
    
    # Project summary
    st.markdown("""
    ### About This Project
    
    This dashboard visualizes a machine learning project for predicting bike rental demand in Seoul. 
    The project follows these key steps:
    
    1. **Data Cleaning**: Handling missing values, outliers, and feature encoding
    2. **Feature Engineering**: Creating polynomial features and time-based interactions
    3. **Model Training**: Testing multiple algorithms including Linear Regression, SVR, and ensemble methods
    4. **Model Evaluation**: Measuring performance using RMSE and visualizing predictions
    
    Navigate through the sidebar to explore different aspects of the project.
    """)
    
    # Show improvement visualization
    st.markdown('<h3 class="section-header">Model Performance Improvement</h3>', unsafe_allow_html=True)
    
    baseline_rmse = 646.17
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()) + ['Baseline (Mean)'],
        'RMSE': [results['test_rmse'] for results in model_results.values()] + [baseline_rmse]
    })
    
    results_df = results_df.sort_values('RMSE')
    
    fig = px.bar(
        results_df,
        x='Model',
        y='RMSE',
        color='Model',
        title='Model Performance Comparison (Lower is Better)',
        height=500
    )
    
    fig.update_layout(xaxis_title="Models", yaxis_title="RMSE (Error)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show executive summary
    st.markdown('<h3 class="section-header">Executive Summary</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    The bike rental prediction model achieved a **{improvement:.1f}%** improvement over the baseline approach.
    
    **Key Findings**:
    
    - Hour of the day and weather conditions (especially temperature) are the strongest predictors
    - Random Forest provided the best performance with an RMSE of {best_model[1]['test_rmse']:.2f}
    - Feature engineering significantly improved model accuracy, especially adding polynomial features
    - Time-based patterns show highest demand during commuting hours and favorable weather conditions
    """)

def display_data_exploration(original_data, cleaned_data):
    st.markdown('<h2 class="section-header">Data Exploration</h2>', unsafe_allow_html=True)
    
    # Data shape - using columns directly without empty containers
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="background-color:rgba(79, 79, 79, 0.2);">
            <p style='color:white; font-weight:bold;'>Original Data Shape:</p>
            <p style='color:#A2D5F2; font-size:1.1rem;'>{original_data.shape[0]} rows × {original_data.shape[1]} columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="background-color:rgba(79, 79, 79, 0.2);">
            <p style='color:white; font-weight:bold;'>Cleaned Data Shape:</p>
            <p style='color:#A2D5F2; font-size:1.1rem;'>{cleaned_data.shape[0]} rows × {cleaned_data.shape[1]} columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data tabs
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Time Patterns", "Correlation Analysis"])
    
    with tab1:
        st.markdown('<h3 style="color:white;">Distribution of Rental Counts</h3>', unsafe_allow_html=True)
        
        fig = px.histogram(
            cleaned_data,
            x="Rented Bike Count",
            marginal="box",
            title="Distribution of Bike Rentals",
            color_discrete_sequence=['#2F80ED'],
            height=500
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(50,50,50,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(color='white', size=18)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<h3 style="color:white;">Weather Factor Distributions</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                cleaned_data,
                x="Temperature (C)",
                title="Temperature Distribution",
                color_discrete_sequence=['#2F80ED'],
                height=400
            )
            fig.update_layout(
                plot_bgcolor='rgba(50,50,50,0.1)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(color='white', size=16)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                cleaned_data,
                x="Humidity (%)",
                title="Humidity Distribution",
                color_discrete_sequence=['#2F80ED'],
                height=400
            )
            fig.update_layout(
                plot_bgcolor='rgba(50,50,50,0.1)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(color='white', size=16)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<h3 style="color:white;">Hourly Rental Patterns</h3>', unsafe_allow_html=True)
        
        hourly_avg = cleaned_data.groupby('Hour')['Rented Bike Count'].mean().reset_index()
        fig = px.line(
            hourly_avg,
            x='Hour',
            y='Rented Bike Count',
            title='Average Hourly Bike Rentals',
            markers=True,
            height=500
        )
        fig.update_layout(
            xaxis_title="Hour of Day", 
            yaxis_title="Average Rentals",
            plot_bgcolor='rgba(50,50,50,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(color='white', size=18)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<h3 style="color:white;">Rentals by Day Type</h3>', unsafe_allow_html=True)
        
        # Create day type column for visualization - FIX: use correct comparison syntax
        viz_data = cleaned_data.copy()
        
        def determine_day_type(row):
            if row['Holiday'] == 1:
                return 'Holiday'
            elif row['Weekday'] == 1:
                return 'Weekday'
            else:
                return 'Weekend'
        
        viz_data['Day Type'] = viz_data.apply(determine_day_type, axis=1)
        
        # Use observed=True to avoid FutureWarning
        day_type_avg = viz_data.groupby(['Day Type', 'Hour'], observed=True)['Rented Bike Count'].mean().reset_index()
        
        fig = px.line(
            day_type_avg,
            x='Hour',
            y='Rented Bike Count',
            color='Day Type',
            title='Hourly Rentals by Day Type',
            markers=True,
            height=500
        )
        fig.update_layout(
            xaxis_title="Hour of Day", 
            yaxis_title="Average Rentals",
            plot_bgcolor='rgba(50,50,50,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(color='white', size=18)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h3 style="color:white;">Correlation Heatmap</h3>', unsafe_allow_html=True)
        
        # Select only numerical columns for correlation
        numeric_data = cleaned_data.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr = numeric_data.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr,
            text_auto=".2f",
            title="Feature Correlation Matrix",
            height=700,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(
            plot_bgcolor='rgba(50,50,50,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(color='white', size=18)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with target
        st.markdown('<h3 style="color:white;">Correlation with Rental Count</h3>', unsafe_allow_html=True)
        
        target_corr = corr['Rented Bike Count'].sort_values(ascending=False).drop('Rented Bike Count')
        top_corr = target_corr.head(10)
        bottom_corr = target_corr.tail(5)
        
        fig = px.bar(
            x=top_corr.values, 
            y=top_corr.index,
            orientation='h',
            title='Top Correlations with Rental Count',
            color=top_corr.values,
            color_continuous_scale='RdBu_r',
            height=500
        )
        fig.update_layout(
            xaxis_title="Correlation", 
            yaxis_title="Features",
            plot_bgcolor='rgba(50,50,50,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(color='white', size=18)
        )
        st.plotly_chart(fig, use_container_width=True)

def display_model_performance(model_results):
    st.markdown('<h2 class="section-header">Model Performance</h2>', unsafe_allow_html=True)
    
    # Create a dataframe of model performance metrics
    performance_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Training RMSE': [results['train_rmse'] for results in model_results.values()],
        'Testing RMSE': [results['test_rmse'] for results in model_results.values()]
    })
    
    # Add baseline for comparison
    baseline_rmse = 646.17
    performance_df = pd.concat([
        performance_df,
        pd.DataFrame({
            'Model': ['Baseline (Mean)'],
            'Training RMSE': [baseline_rmse],
            'Testing RMSE': [baseline_rmse]
        })
    ])
    
    # Sort by testing RMSE
    performance_df = performance_df.sort_values('Testing RMSE')
    
    # Calculate improvement percentage
    performance_df['Improvement (%)'] = ((baseline_rmse - performance_df['Testing RMSE']) / baseline_rmse * 100).round(2)
    
    # Display the best model
    best_model = performance_df.iloc[0]['Model']
    best_rmse = performance_df.iloc[0]['Testing RMSE']
    best_improvement = performance_df.iloc[0]['Improvement (%)']
    
    st.markdown(f"""
    <div style="background-color:#3082F5; padding:20px; border-radius:5px; margin-bottom:20px; color:white;">
        <h3 style="margin-top:0; color:white; font-weight:bold;">Best Performing Model</h3>
        <p><strong style="color:#E2F0FF;">{best_model}</strong> achieved the lowest error with RMSE of <strong style="color:#E2F0FF;">{best_rmse:.2f}</strong></p>
        <p>This represents a <strong style="color:#E2F0FF;">{best_improvement:.2f}%</strong> improvement over the baseline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance comparison visualization
    st.markdown('<h3 style="color:white;">Model Performance Comparison</h3>', unsafe_allow_html=True)
    
    # Bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=performance_df['Model'],
        y=performance_df['Testing RMSE'],
        name='Testing RMSE',
        marker_color='crimson'
    ))
    
    fig.add_trace(go.Bar(
        x=performance_df['Model'],
        y=performance_df['Training RMSE'],
        name='Training RMSE',
        marker_color='royalblue'
    ))
    
    fig.update_layout(
        title='RMSE Comparison (Lower is Better)',
        xaxis_title='Model',
        yaxis_title='RMSE',
        barmode='group',
        height=600,
        plot_bgcolor='rgba(50,50,50,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.markdown("### Detailed Performance Metrics")
    
    # Format the dataframe for display without styling (to avoid the non-unique index error)
    # Set Model as index to ensure uniqueness
    formatted_df = performance_df.copy()
    formatted_df['Training RMSE'] = formatted_df['Training RMSE'].round(2)
    formatted_df['Testing RMSE'] = formatted_df['Testing RMSE'].round(2)
    formatted_df['Improvement (%)'] = formatted_df['Improvement (%)'].round(2).astype(str) + '%'
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Overfitting analysis
    st.markdown("### Overfitting Analysis")
    
    performance_df['Overfitting Ratio'] = performance_df['Testing RMSE'] / performance_df['Training RMSE']
    
    fig = px.bar(
        performance_df,
        x='Model',
        y='Overfitting Ratio',
        color='Overfitting Ratio',
        title='Overfitting Analysis (Closer to 1 is better)',
        height=500,
        color_continuous_scale='RdYlGn_r'
    )
    
    # Add a horizontal line at y=1
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(performance_df)-0.5,
        y0=1,
        y1=1,
        line=dict(color="black", width=2, dash="dash"),
    )
    
    fig.update_layout(xaxis_title="Model", yaxis_title="Test RMSE / Train RMSE")
    st.plotly_chart(fig, use_container_width=True)

def display_feature_importance(feature_importance_df, model_results):
    st.markdown('<h2 class="section-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    # Get feature importances from Random Forest
    rf_model = model_results.get('Random Forest', None)
    
    if rf_model:
        rf_pipeline = rf_model['pipeline']
        rf_importance = rf_pipeline.named_steps['model'].feature_importances_
        
        # Get the feature names
        feature_names = feature_importance_df['feature'].values
        top_n = min(20, len(feature_names))
        top_features = feature_importance_df.head(top_n)
        
        # Plot feature importance
        st.markdown("### Random Forest Feature Importance")
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Features by Importance',
            height=600,
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature categories analysis
        st.markdown("### Feature Categories Analysis")
        
        # Group features by category
        categories = {
            'Time': [f for f in feature_names if 'Hour' in f or 'TimePeriod' in f],
            'Weather': [f for f in feature_names if any(w in f for w in ['Temperature', 'Humidity', 'Wind', 'Visibility', 'Dew'])],
            'Season': [f for f in feature_names if 'Season' in f],
            'Other': [f for f in feature_names if not any(c in f for c in ['Hour', 'TimePeriod', 'Temperature', 'Humidity', 'Wind', 'Visibility', 'Dew', 'Season'])]
        }
        
        category_importance = {}
        for category, features in categories.items():
            # Get indices of these features in the feature_names array
            feature_indices = [i for i, f in enumerate(feature_names) if f in features]
            
            # Sum importance of these features
            if feature_indices:
                importance_sum = sum(rf_importance[i] for i in feature_indices if i < len(rf_importance))
                category_importance[category] = importance_sum
        
        # Create a dataframe for visualization
        category_df = pd.DataFrame({
            'Category': list(category_importance.keys()),
            'Importance': list(category_importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Plot category importance
        fig = px.pie(
            category_df,
            values='Importance',
            names='Category',
            title='Feature Importance by Category',
            height=500,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Random Forest model not found in results.")
    
    # Time-based feature importance
    st.markdown("### Time-Related Feature Importance")
    
    time_features = feature_importance_df[feature_importance_df['feature'].str.contains('Hour|TimePeriod')].head(10)
    
    fig = px.bar(
        time_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Time-Related Feature Importance',
        height=500,
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)
    
    # Weather-related feature importance
    st.markdown("### Weather-Related Feature Importance")
    
    weather_features = feature_importance_df[
        feature_importance_df['feature'].str.contains('Temperature|Humidity|Wind|Visibility|Dew|Rain|Snow|Solar')
    ].head(10)
    
    fig = px.bar(
        weather_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Weather-Related Feature Importance',
        height=500,
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)

def display_predictions(model_results, y_test):
    st.markdown('<h2 class="section-header">Prediction Analysis</h2>', unsafe_allow_html=True)
    
    # Select model for analysis
    models = list(model_results.keys())
    selected_model = st.selectbox("Select Model for Analysis", models, index=models.index('Random Forest') if 'Random Forest' in models else 0)
    
    model_data = model_results[selected_model]
    y_pred = model_data['y_test_pred']
    
    # Actual vs Predicted
    st.markdown("### Actual vs Predicted Values")
    
    # Create dataframe for visualization
    pred_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    
    # Add error
    pred_df['Error'] = pred_df['Actual'] - pred_df['Predicted']
    pred_df['AbsError'] = abs(pred_df['Error'])
    
    # Scatter plot
    fig = px.scatter(
        pred_df,
        x='Actual',
        y='Predicted',
        color='AbsError',
        title=f'{selected_model}: Actual vs Predicted Bike Rentals',
        height=600,
        color_continuous_scale='RdYlGn_r',
        opacity=0.7
    )
    
    # Add perfect prediction line
    max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
    min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='black')
        )
    )
    
    fig.update_layout(
        xaxis_title="Actual Bike Rentals",
        yaxis_title="Predicted Bike Rentals",
        coloraxis_colorbar=dict(title="Absolute Error")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error distribution
    st.markdown("### Error Distribution")
    
    fig = px.histogram(
        pred_df,
        x='Error',
        marginal='box',
        color_discrete_sequence=['#2F80ED'],
        title=f'{selected_model}: Prediction Error Distribution',
        height=500
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(xaxis_title="Prediction Error (Actual - Predicted)", yaxis_title="Count")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RMSE", f"{model_data['test_rmse']:.2f}")
    
    with col2:
        mae = np.mean(np.abs(pred_df['Error']))
        st.metric("MAE", f"{mae:.2f}")
    
    with col3:
        mape = np.mean(np.abs(pred_df['Error'] / pred_df['Actual']) * 100)
        st.metric("MAPE", f"{mape:.2f}%")
    
    # Error by prediction range
    st.markdown("### Error Analysis by Rental Count Range")
    
    # Create bins for analysis
    bins = [0, 500, 1000, 2000, 3000, 8000]
    labels = ['0-500', '501-1000', '1001-2000', '2001-3000', '3001+']
    
    pred_df['RentalBin'] = pd.cut(pred_df['Actual'], bins=bins, labels=labels)
    
    # Calculate metrics by bin
    bin_metrics = pred_df.groupby('RentalBin', observed=True).agg({
        'Actual': 'count',
        'Error': ['mean', 'std'],
        'AbsError': 'mean'
    })
    
    bin_metrics.columns = ['Count', 'Mean Error', 'Error StdDev', 'Mean Abs Error']
    bin_metrics = bin_metrics.reset_index()
    
    # Create a grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=bin_metrics['RentalBin'],
        y=bin_metrics['Mean Abs Error'],
        name='Mean Absolute Error',
        marker_color='crimson'
    ))
    
    fig.add_trace(go.Bar(
        x=bin_metrics['RentalBin'],
        y=bin_metrics['Error StdDev'],
        name='Error Standard Deviation',
        marker_color='royalblue'
    ))
    
    fig.update_layout(
        title='Error Metrics by Rental Count Range',
        xaxis_title='Rental Count Range',
        yaxis_title='Error Metric Value',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display count distribution
    fig = px.bar(
        bin_metrics,
        x='RentalBin',
        y='Count',
        title='Sample Count by Rental Range',
        color='Count',
        height=400
    )
    
    fig.update_layout(xaxis_title="Rental Count Range", yaxis_title="Number of Samples")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model learning curve
    st.markdown("### Learning Curves")
    
    if selected_model in ['Random Forest', 'Gradient Boosting']:
        st.markdown("""
        The learning curve shows how the model's performance improves with more training data:
        
        - **Red line**: Training score - shows how well the model fits the training data
        - **Green line**: Cross-validation score - shows how well the model generalizes
        
        A good model will have both lines converge to a similar value as training data increases.
        """)
        
        # Display image (using a placeholder since we can't directly show the actual learning curve)
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png", 
                 caption="Learning Curve Example (Placeholder)", 
                 use_container_width=True)
    else:
        st.info(f"Learning curve visualization is not available for {selected_model}")

if __name__ == "__main__":
    main() 