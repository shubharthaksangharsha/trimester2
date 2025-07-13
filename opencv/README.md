# Computer Vision Explorer

An interactive Streamlit application demonstrating computer vision concepts including feature matching, homography estimation, image retrieval, and epipolar geometry.

## Features

- **Feature Detection and Matching**: Visualize ORB features and matches between images
- **Homography Estimation**: Compare least squares and RANSAC methods for homography estimation
- **Image Retrieval**: Perform content-based image retrieval across multiple datasets
- **Epipolar Geometry**: Visualize epipolar lines and understand fundamental matrix computation

## Datasets

The application works with the following datasets:

1. **Book Covers**: Collection of book cover images in different viewing conditions
2. **Landmarks**: Famous landmarks photographed from different viewpoints
3. **Museum Paintings**: Paintings captured with varying lighting and perspectives

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

The app will open in your default web browser.

## Deploying to AWS/Oracle

### AWS Deployment

1. Create an EC2 instance
2. Install required dependencies
3. Clone this repository
4. Run the app with:
   ```
   streamlit run app.py --server.port 8501 --server.enableCORS false
   ```

### Oracle Cloud Deployment

1. Create a Compute instance
2. Install required dependencies
3. Clone this repository
4. Configure firewall rules to allow traffic on port 8501
5. Run the app with:
   ```
   streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
   ```

## Credits

This application was developed as part of a Computer Vision assignment, featuring implementations of:

- Feature detection and matching with ORB
- Homography estimation with least squares and RANSAC
- Content-based image retrieval
- Fundamental matrix computation and epipolar geometry visualization 