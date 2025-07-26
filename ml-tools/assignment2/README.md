# 🔬 Breast Cancer Classification Demo

An interactive web application showcasing machine learning models for breast cancer classification. This app demonstrates comprehensive ML workflow including data analysis, model training, feature analysis, and interactive predictions.

## ✨ Features

### 🤖 **Live Prediction Interface**
- Interactive form for inputting cell measurement features
- Real-time predictions from 4 different trained models
- Sample data loading for malignant and benign cases
- Probability visualizations with animated progress bars

### 📊 **Feature Analysis Dashboard**
- T-score analysis showing discriminative power of each feature
- Interactive bar charts with the top 10 most important features
- Statistical insights into feature distributions

### 📈 **Model Performance Comparison**
- Comprehensive metrics for all 4 models:
  - Random Baseline (DummyClassifier)
  - SGD Baseline
  - Random Forest
  - Improved Random Forest (optimized for clinical requirements)
- Performance metrics: Accuracy, Recall, Precision, F1-Score, AUC

### 🌐 **3D Interactive Visualization**
- Three.js-powered 3D scatter plot of feature space
- Interactive camera controls with mouse movement
- Color-coded data points (red for malignant, blue for benign)
- Animation controls and camera reset functionality

## 🏥 **Clinical Context**

This project addresses a real-world medical scenario where:
- **Requirement 1**: ≥90% probability of detecting malignant cancer when present
- **Requirement 2**: ≤20% false positive rate (benign cases incorrectly labeled as malignant)

The final optimized model achieves:
- ✅ **100% recall** (detects all malignant cases)
- ✅ **12.9% false positive rate** (well within acceptable limits)

## 🚀 **Getting Started**

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd assignment2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## 🧠 **Models Included**

### 1. **Random Baseline**
- DummyClassifier for establishing baseline performance
- Demonstrates the challenge of the classification problem

### 2. **SGD Baseline**
- Stochastic Gradient Descent classifier
- Simple linear model with good initial performance

### 3. **Random Forest**
- Ensemble method with excellent accuracy
- Optimized through GridSearchCV hyperparameter tuning

### 4. **Improved Random Forest**
- Specifically tuned to meet clinical requirements
- Uses class weighting and threshold optimization
- Achieves perfect recall while maintaining low false positive rate

## 📊 **Dataset Information**

- **Total Samples**: 220 histological cell images
- **Features**: 20 cell measurement features including:
  - Cell size measurements (radius, perimeter, area)
  - Cell shape characteristics (compactness, concavity, concave points)
  - Texture and symmetry measurements
  - Error measurements for each feature
- **Classes**: Malignant (65 samples) and Benign (155 samples)

## 🎯 **Top Discriminative Features**

Based on T-score analysis:
1. **Mean Concave Points** (T-score: 2.338)
2. **Mean Perimeter** (T-score: 1.928)
3. **Area Error** (T-score: 1.737)
4. **Mean Concavity** (T-score: 1.684)

## 🎨 **Technology Stack**

### Backend
- **Flask**: Web framework
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **NumPy**: Numerical computations
- **joblib**: Model serialization

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive functionality
- **Three.js**: 3D visualizations
- **GSAP**: Smooth animations
- **Chart.js**: Data visualization
- **Plotly**: Interactive charts

### Design Features
- **Gradient backgrounds** with glassmorphism effects
- **Smooth animations** using GSAP
- **Responsive design** for all screen sizes
- **Interactive 3D visualizations**
- **Real-time data updates**

## 🔧 **File Structure**

```
assignment2/
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── A2_student_2025.ipynb             # Original Jupyter notebook
├── assignment2_data_2025_cleaned.csv # Cleaned dataset
├── templates/
│   └── index.html                    # Main web interface
└── model/                            # Trained models
    ├── random_baseline_20250726_200129.joblib
    ├── sgd_baseline_20250726_200129.joblib
    ├── random_forest_20250726_200129.joblib
    └── improved_random_forest_20250726_200129.joblib
```

## 🎮 **Usage Guide**

### Making Predictions
1. Switch to the "Live Prediction" tab
2. Enter cell measurement values in the feature input form
3. Use "Load Sample Data" buttons for quick testing
4. Click "Predict Classification" to see results from all models

### Exploring Features
1. Navigate to "Feature Analysis" tab
2. View T-score rankings for all features
3. Understand which features best separate malignant from benign cases

### Comparing Models
1. Check the "Model Performance" tab
2. Compare accuracy, recall, precision, F1-score, and AUC across models
3. Understand the trade-offs between different approaches

### 3D Visualization
1. Go to "3D Visualization" tab
2. Interact with the 3D scatter plot using mouse movement
3. Toggle animation and reset camera as needed

## 🏆 **Key Achievements**

- ✅ Comprehensive ML pipeline from data cleaning to deployment
- ✅ Multiple model comparison with hyperparameter optimization
- ✅ Clinical requirements successfully met
- ✅ Interactive web-based demonstration
- ✅ Modern, responsive UI with 3D visualizations
- ✅ Real-time prediction capabilities

## 📝 **License**

This project is for educational purposes as part of a machine learning assignment demonstrating best practices in medical AI applications.

---
