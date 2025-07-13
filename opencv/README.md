# Computer Vision Explorer

An interactive Streamlit application demonstrating computer vision concepts including feature matching, homography estimation, image retrieval, and epipolar geometry.

## About the App

This application provides a visual and interactive exploration of fundamental computer vision techniques. Users can explore different datasets and observe how various algorithms perform on real-world image data.

**Live Demo:** [https://a2-cv.devshubh.me/](https://a2-cv.devshubh.me/)

## Creator

Developed by [Shubharthak Sangharasha](https://devshubh.me/) as part of the Computer Vision course assignment.

## Resources

- **Jupyter Notebook:** [Assignment_2_Notebook.ipynb](https://github.com/shubharthaksangharsha/trimester2/blob/main/opencv/Assignment_2_Notebook.ipynb)
- **PDF Report:** [Coming Soon]
- **GitHub Repository:** [https://github.com/shubharthaksangharsha/trimester2/tree/main/opencv](https://github.com/shubharthaksangharsha/trimester2/tree/main/opencv)

## Datasets

The application works with the following datasets:

1. **[Book Covers](https://github.com/shubharthaksangharsha/trimester2/blob/main/opencv/A2_smvs/book_covers)**: Collection of book cover images in different viewing conditions
2. **[Landmarks](https://github.com/shubharthaksangharsha/trimester2/blob/main/opencv/A2_smvs/landmarks)**: Famous landmarks photographed from different viewpoints
3. **[Museum Paintings](https://github.com/shubharthaksangharsha/trimester2/blob/main/opencv/A2_smvs/museum_paintings)**: Paintings captured with varying lighting and perspectives

Each dataset contains both query and reference images to demonstrate matching and retrieval tasks.

## Assignment Questions

### Q1: Feature Matching and Homography Estimation

This section implements robust feature detection, matching, and homography estimation between image pairs. The solution includes:

- **Feature Detection**: Using ORB (Oriented FAST and Rotated BRIEF) detector to identify up to 1000 keypoints in each image
- **Feature Matching**: Implementing Brute Force matching with Hamming distance for binary descriptors
- **Ratio Test**: Filtering matches using Lowe's ratio test to eliminate ambiguous matches (with adjustable ratio threshold)
- **Homography Estimation**: 
  - Least Squares method for basic estimation
  - RANSAC (Random Sample Consensus) for robust estimation with outlier rejection
  - Comparative analysis of different RANSAC thresholds (1.0, 3.0, 5.0)
- **Performance Metrics**:
  - Inlier counts and ratios to evaluate matching quality
  - Visual verification through projection of reference image outlines onto query images

The implementation demonstrates significant improvement in matching accuracy when using RANSAC (achieving inlier ratios of up to 0.84) compared to the basic least squares method (inlier ratio of 0.09). The code also includes parameter tuning to handle challenging cases, showing how increasing feature counts and adjusting ratio thresholds can improve results for difficult image pairs.

### Q2: Content-Based Image Retrieval

This section implements a comprehensive image retrieval system that can identify the most similar reference images for a given query. The solution features:

- **Feature-Based Representation**: Using ORB features to create distinctive representations of images
- **Efficient Matching Pipeline**:
  1. Extract keypoints and descriptors from all reference images
  2. Match query image descriptors to each reference image
  3. Apply ratio test to filter good matches
  4. Compute homography using RANSAC for geometric verification
  5. Rank reference images by inlier count
  
- **Performance Evaluation**:
  - Achieved 100% accuracy on the book covers dataset (20/20 correct matches)
  - Implemented evaluation with external queries (images not in the dataset)
  - Analysis of false positives and threshold sensitivity
  
- **Robustness Improvements**:
  - Adjustable match threshold to reduce false positives
  - Precision-recall analysis for optimal threshold selection
  - Alternative matching using fundamental matrix for viewpoint-invariant retrieval

The system demonstrates the effectiveness of combining feature matching with geometric verification for reliable image retrieval, achieving excellent results even with challenging viewpoint changes and partial occlusions.

### Q3: Epipolar Geometry

This section explores the geometric relationship between multiple views of the same scene through epipolar geometry. The implementation includes:

- **Fundamental Matrix Estimation**:
  - 8-point algorithm for basic estimation
  - RANSAC-based robust estimation to handle outliers
  - Comparative analysis showing RANSAC's superior performance (average epipolar error of 0.001107 vs 0.022626 for 8-point)

- **Epipolar Line Visualization**:
  - Computation of corresponding epipolar lines between image pairs
  - Visual representation showing the geometric constraints between views
  - Testing on multiple landmark pairs (Eiffel Tower, Colosseum) with varying viewpoints

- **Quantitative Evaluation**:
  - Calculation of average epipolar error to measure estimation accuracy
  - Analysis of inlier ratios for different estimation methods
  - Demonstration of epipolar geometry's role in multi-view constraints

The implementation provides insights into the fundamental geometric relationships in stereo vision and how they can be used for 3D reconstruction, camera calibration, and view synthesis applications.

## Jupyter Notebook

The [`Assignment_2_Notebook.ipynb`](https://github.com/shubharthaksangharsha/trimester2/blob/main/opencv/Assignment_2_Notebook.ipynb) contains detailed explanations, code implementations, and visualizations for all the questions. It serves as both documentation and a tutorial for the computer vision concepts implemented in the application. 