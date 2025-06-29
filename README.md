# Seoul Bike Rental Prediction

## Overview
This project implements a machine learning solution to predict bike rental demand in Seoul based on weather conditions and other factors. The system includes data exploration, model training, and an interactive Streamlit dashboard for visualization and prediction.

**Created by: Shubharthak Sangharsha ([Portfolio](https://devshubh.me))**

## Live App
**Web App Link**: [mlt-a1.devshubh.me](https://mlt-a1.devshubh.me)

## Resources
- **GitHub Repository**: [GitHub Link](https://github.com/shubharthaksangharsha/seoul-bike-rental-prediction) 
- **Dataset**: [Download from Google Drive](https://drive.google.com/file/placeholder) *(placeholder - replace with actual link)*
- **Project Report**: See [PDF Document](ml-tools/assingment1/pdf/A1-a1944839-shubharthak-2025-ml-tools.pdf)

## Features
- Interactive data exploration of Seoul bike rental patterns
- Multiple machine learning models:
  - Linear Regression
  - Ridge Regression
  - Support Vector Regression (SVR)
  - Random Forest
  - Gradient Boosting
  - Voting Regressor
  - Stacking Regressor
- Feature importance analysis
- Model performance visualization
- Prediction analysis with error metrics

## Dataset Information
The project uses the Seoul Bike Sharing Demand dataset, which includes:
- Weather conditions (Temperature, Humidity, Wind speed, Visibility, Dew point, Solar radiation, Snowfall, Rainfall)
- Seasonal information (Hour, Day, Month, Season)
- Holiday information
- Bike rental count (target variable)

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
```bash
git clone https://github.com/shubharthaksangharsha/seoul-bike-rental-prediction.git
cd seoul-bike-rental-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App Locally

### Using Python
```bash
cd ml-tools/assingment1
streamlit run app.py
```

### Using Batch File (Windows)
```bash
cd ml-tools/assingment1
run_app.bat
```


## Project Structure
```
└── ml-tools/
    └── assingment1/
        ├── A1-a1944839-shubharthak-2025-ml-tools.ipynb  # Main notebook with analysis
        ├── app/
        │   └── app.py                                   # Streamlit app code (development version)
        ├── app.py                                       # Streamlit app main file
        ├── dataset/
        │   ├── CleanedSeoulBikeData.csv                # Preprocessed data
        │   └── SeoulBikeData.csv                       # Raw data
        ├── pdf/
        │   └── A1-a1944839-shubharthak-2025-ml-tools.pdf # Project report
        ├── requirements.txt                             # Python dependencies
        ├── run_app.bat                                  # Windows batch file to run app
```

## How to Use the App
1. Open the web app
2. Navigate through the different sections:
   - **Overview**: General patterns in bike rental data
   - **Data Exploration**: Interactive visualizations of relationships between variables
   - **Model Performance**: Comparison of different machine learning models
   - **Feature Importance**: Analysis of which factors most impact bike rental demand
   - **Predictions**: View model predictions and errors

## Contributing
This project was developed as part of the Machine Learning Tools course. For any questions or improvements, please open an issue on GitHub.

## License
This project is open source and available under the MIT License. 