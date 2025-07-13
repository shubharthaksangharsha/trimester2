import streamlit as st
import os

def render_page():
    st.markdown('<h2 class="sub-header">Welcome to the Computer Vision Explorer 🚀</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
    This interactive application demonstrates key concepts from computer vision. Use the navigation above to explore different modules, from fundamental feature matching to advanced augmented imagery.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
        <h3>What You Can Do Here</h3>
        <ul>
            <li><b>Feature Matching:</b> Visualize how ORB algorithms find corresponding points between images.</li>
            <li><b>Homography:</b> See how to map one image's perspective onto another using projective transformations.</li>
            <li><b>Image Retrieval:</b> Find a query image within a large dataset using feature-based matching.</li>
            <li><b>Epipolar Geometry:</b> Understand the 3D relationship between two camera views.</li>
            <li><b>Innovation:</b> Experience an interactive augmented reality visualization demo.</li>
        </ul>
        <p>Developed by <a href="https://devshubh.me" target="_blank">Shubharthak Sangharasha</a></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # Use a placeholder if the image is not found
        image_path = "book.png"
        if os.path.exists(image_path):
            st.image(image_path, caption="Powered by OpenCV & Streamlit", use_column_width=True)
        else:
            st.image("https://placehold.co/400x300/161B22/EAEAEA?text=Computer+Vision+Explorer", caption="Computer Vision Explorer", use_column_width=True)

    # Dataset information
    st.markdown("""
    <div class="card">
    <h3>Dataset Information</h3>
    <p>This application uses three datasets for demonstration:</p>
    
    <h4>Book Covers</h4>
    <p>A collection of book cover images photographed under different viewing conditions, angles, and lighting. Great for homography estimation and image retrieval tasks.</p>
    
    <h4>Landmarks</h4>
    <p>Famous landmarks photographed from different viewpoints, ideal for epipolar geometry demonstrations and feature matching.</p>
    
    <h4>Museum Paintings</h4>
    <p>Paintings captured with varying lighting and perspectives, challenging for image retrieval algorithms.</p>
    
    <p>Each dataset contains reference images and query images with various transformations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Technical information
    st.markdown("""
    <div class="card">
    <h3>Technical Implementation</h3>
    <p>This application demonstrates several computer vision techniques:</p>
    <ul>
        <li><b>ORB Feature Detection:</b> Scale and rotation invariant feature detection based on FAST keypoints and BRIEF descriptors</li>
        <li><b>RANSAC:</b> Random Sample Consensus for robust model fitting with outlier rejection</li>
        <li><b>Homography Estimation:</b> Finding the projective transformation between image planes</li>
        <li><b>Fundamental Matrix:</b> Encodes the epipolar geometry between two views</li>
        <li><b>Augmented Visualization:</b> Blending and warping techniques for interactive image manipulation</li>
    </ul>
    <p>Built with Python, OpenCV, and Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)

    # Getting started guide
    st.markdown("""
    <div class="highlight">
    <h3>Getting Started</h3>
    <p>To begin exploring:</p>
    <ol>
        <li>Select a module from the navigation bar above</li>
        <li>Choose a dataset and sample images</li>
        <li>Adjust parameters to see how they affect the results</li>
        <li>Explore the visualizations and metrics</li>
    </ol>
    <p>For the best experience, start with Feature Matching and progress through the modules.</p>
    </div>
    """, unsafe_allow_html=True)

