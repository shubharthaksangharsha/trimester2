import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import io
import streamlit.components.v1 as components

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"

# Initialize theme preference
if 'theme' not in st.session_state:
    st.session_state['theme'] = "light"

# Function to toggle theme
def toggle_theme():
    st.session_state['theme'] = "dark" if st.session_state['theme'] == "light" else "light"

# Configure the page
st.set_page_config(
    layout="wide", 
    page_title="Computer Vision Explorer", 
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Get current theme
current_theme = st.session_state['theme']

# Apply custom CSS for better aesthetics with theme support
st.markdown(f"""
<style>
    :root {{
        --primary-color-light: #4A90E2;
        --primary-color-dark: #58A6FF;
        --background-color-light: #F5F7FA;
        --background-color-dark: #1E2125;
        --secondary-background-light: #FFFFFF;
        --secondary-background-dark: #2C3035;
        --text-color-light: #333333;
        --text-color-dark: #EAEAEA;
        --text-color-light-subtle: #757575;
        --text-color-dark-subtle: #A0A0A0;
        --border-color-light: #DCE1E7;
        --border-color-dark: #444950;
    }}

    /* Light Theme */
    body.light-mode {{
        --primary-color: var(--primary-color-light);
        --background-color: var(--background-color-light);
        --secondary-background: var(--secondary-background-light);
        --text-color: var(--text-color-light);
        --text-color-subtle: var(--text-color-light-subtle);
        --border-color: var(--border-color-light);
    }}

    /* Dark Theme */
    body.dark-mode {{
        --primary-color: var(--primary-color-dark);
        --background-color: var(--background-color-dark);
        --secondary-background: var(--secondary-background-dark);
        --text-color: var(--text-color-dark);
        --text-color-subtle: var(--text-color-dark-subtle);
        --border-color: var(--border-color-dark);
    }}

    /* Apply base styles */
    body {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}
    
    .stApp {{
        background-color: var(--background-color);
    }}

    h1, h2, h3, h4, h5, p, li, td, th, label, .stMarkdown {{
        color: var(--text-color) !important;
    }}

    .main-header {{
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
        color: var(--primary-color);
    }}

    .sub-header {{
        font-size: 1.8rem;
        font-weight: 500;
        color: var(--primary-color);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.5rem;
    }}

    .card, .highlight {{
        background-color: var(--secondary-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }}
    
    .highlight {{
        border-left: 5px solid var(--primary-color);
    }}

    /* Navigation buttons */
    .nav-button {{
        background: transparent;
        color: var(--text-color-subtle) !important;
        border: 2px solid transparent;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
        text-align: center;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }}

    .nav-button:hover {{
        color: var(--primary-color) !important;
        background-color: var(--secondary-background);
    }}

    .nav-button-active {{
        background-color: var(--primary-color);
        color: white !important;
        font-weight: bold;
    }}

    /* Theme Toggle */
    .theme-toggle {{
        background-color: var(--secondary-background);
        color: var(--text-color-subtle) !important;
        border-radius: 50%;
        width: 45px !important;
        height: 45px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 20px;
        border: 1px solid var(--border-color);
    }}

    /* Form and Button Elements */
    .stButton > button {{
        background-color: var(--primary-color);
        color: white !important;
        font-weight: 500;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }}

    .stSelectbox > div, .stTextInput > div > div > input {{
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px;
    }}
    
    div[data-baseweb="select"] > div {{
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }}

    .stSlider {{
        color: var(--text-color) !important;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        font-size: 0.9rem;
        color: var(--text-color-subtle);
    }}

    .footer a {{
        color: var(--primary-color) !important;
        text-decoration: none;
        font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)

# JavaScript to apply the theme class to the body
components.html(
    f"""
    <script>
    const body = window.parent.document.querySelector('body');
    body.classList.remove('light-mode', 'dark-mode');
    body.classList.add('{current_theme}-mode');
    </script>
    """,
    height=0,
)

# Functions for feature detection and matching
def detect_and_match(img1, img2, nfeatures=1000, ratio=0.8):
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    
    return kp1, kp2, des1, des2, good_matches

def visualize_matches(img1, img2, kp1, kp2, matches):
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

def find_homography(kp1, kp2, good_matches, method=0, ransac_thresh=3.0):
    # Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
    
    return H, mask, src_pts, dst_pts

def draw_outline(img1, img2, H):
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    # Transform points
    dst = cv2.perspectiveTransform(pts, H)
    
    # Draw outline
    img_out = img2.copy()
    img_out = cv2.polylines(img_out, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    
    return img_out

def compute_fundamental_matrix(pts1, pts2, method=cv2.FM_RANSAC, threshold=3.0):
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, method, threshold, 0.99)
    
    return F, mask

def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=10):
    # Select random subset of points
    idx = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)
    pts1_subset = pts1[idx]
    pts2_subset = pts2[idx]
    
    # Compute epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts2_subset.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1_subset.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    # Draw lines on images
    img1_color = cv2.cvtColor(img1.copy(), cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2.copy(), cv2.COLOR_GRAY2BGR)
    
    colors = np.random.randint(0, 255, (len(pts1_subset), 3)).tolist()
    
    # Draw lines on img1
    h, w = img1.shape
    for i, (line, pt, color) in enumerate(zip(lines1, pts1_subset, colors)):
        x0, y0 = 0, int(-line[2]/line[1])
        x1, y1 = w, int(-(line[2] + line[0]*w)/line[1])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 5, color, -1)
    
    # Draw lines on img2
    h, w = img2.shape
    for i, (line, pt, color) in enumerate(zip(lines2, pts2_subset, colors)):
        x0, y0 = 0, int(-line[2]/line[1])
        x1, y1 = w, int(-(line[2] + line[0]*w)/line[1])
        img2_color = cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
        img2_color = cv2.circle(img2_color, (int(pt[0]), int(pt[1])), 5, color, -1)
    
    return img1_color, img2_color

# Add this function near the top of your file, after imports but before the main function
def check_dataset_paths(dataset_path):
    """Check if dataset paths exist and provide informative errors"""
    reference_path = f"{dataset_path}/Reference"
    query_path = f"{dataset_path}/Query"
    
    errors = []
    if not os.path.exists(dataset_path):
        errors.append(f"Dataset path '{dataset_path}' does not exist.")
    elif not os.path.exists(reference_path):
        errors.append(f"Reference folder '{reference_path}' does not exist.")
    elif not os.path.exists(query_path):
        errors.append(f"Query folder '{query_path}' does not exist.")
    elif len(os.listdir(reference_path)) == 0:
        errors.append(f"Reference folder '{reference_path}' is empty.")
    elif len(os.listdir(query_path)) == 0:
        errors.append(f"Query folder '{query_path}' is empty.")
    
    return errors

# Navigation header
def display_navigation():
    st.markdown('<h1 class="main-header">Computer Vision Explorer</h1>', unsafe_allow_html=True)
    
    # Create horizontal navigation with 6 columns (adding Innovation)
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 0.5])
    
    # Define pages and their corresponding columns
    pages = {
        "Home": col1,
        "Feature Matching": col2,
        "Homography Estimation": col3,
        "Image Retrieval": col4,
        "Epipolar Geometry": col5,
        "Innovation": col6
    }
    
    # Create buttons for each page
    for page_name, col in pages.items():
        button_class = "nav-button nav-button-active" if st.session_state.current_page == page_name else "nav-button nav-button-inactive"
        if col.button(page_name, key=f"nav_{page_name}"):
            st.session_state.current_page = page_name
            st.experimental_rerun()
    
    # Theme toggle button with more visible styling
    theme_icon = "🌙" if st.session_state['theme'] == "light" else "☀️"
    if col7.button(theme_icon, key="theme_toggle", help="Toggle between light and dark mode"):
        toggle_theme()
        st.experimental_rerun()

# Add a new function for the innovation tab
def interactive_augmented_imagery(img1, img2):
    """
    Create an interactive augmented imagery demo combining multiple computer vision techniques
    """
    # Create ORB detector for feature matching
    orb = cv2.ORB_create(nfeatures=1500)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Match descriptors using KNN
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Extract points from good matches
    if len(good_matches) >= 10:
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography using RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Check if homography estimation was successful
            if H is None:
                return None, None, None, len(good_matches), 0, None
            
            # Get dimensions
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Create panorama canvas
            pts1 = np.float32([[0,0], [0,h1-1], [w1-1,h1-1], [w1-1,0]]).reshape(-1, 1, 2)
            pts2 = cv2.perspectiveTransform(pts1, H)
            
            # Combine images for visualization
            result_img = img2.copy()
            
            # Create mask for blending
            blend_mask = np.zeros(img2.shape, dtype=np.uint8)
            cv2.fillPoly(blend_mask, [np.int32(pts2)], (255, 255, 255))
            
            # Apply transformation to create overlay
            warped = cv2.warpPerspective(img1, H, (w2, h2))
            
            # Blend the images with transparency
            alpha = 0.7
            beta = 1.0 - alpha
            
            # If images are grayscale, convert to BGR for blending
            if len(img2.shape) == 2:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
                warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            
            # Create composite image with blending
            composite = cv2.addWeighted(result_img, alpha, warped, beta, 0)
            
            # Extract inliers for visualization - Fix for mask handling
            mask_flat = mask.ravel()
            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask_flat[i] > 0]
            
            # Draw matches between images
            matched_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:15], None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Draw transformed outline on query image
            img_outline = img2.copy()
            if len(img_outline.shape) == 2:
                img_outline = cv2.cvtColor(img_outline, cv2.COLOR_GRAY2BGR)
            img_outline = cv2.polylines(img_outline, [np.int32(pts2)], True, (0, 255, 0), 2, cv2.LINE_AA)
            
            return composite, matched_img, img_outline, len(good_matches), np.sum(mask), H
        except Exception as e:
            print(f"Error in homography estimation: {e}")
            return None, None, None, len(good_matches), 0, None
    
    return None, None, None, len(good_matches), 0, None

# Main app
def main():
    # Display the horizontal navigation
    display_navigation()
    
    # Home page
    if st.session_state.current_page == "Home":
        st.markdown('<h2 class="sub-header">Welcome to Computer Vision Explorer</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="highlight">
            <h3>About This App</h3>
            <p>This interactive application demonstrates key concepts from computer vision, including:</p>
            <ul>
                <li><b>Feature detection and matching</b> with ORB</li>
                <li><b>Homography estimation</b> with least squares and RANSAC</li>
                <li><b>Content-based image retrieval</b></li>
                <li><b>Epipolar geometry</b> and fundamental matrix computation</li>
            </ul>
            <p>Use the navigation at the top to explore different sections.</p>
            <p>Developed by <a href="https://devshubh.me" target="_blank">Shubharthak Sangharasha</a> | 
            <a href="https://github.com/shubharthaksangharsha/trimester2/tree/main/opencv" target="_blank">GitHub Repository</a></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            try:
                st.image("book.png", caption="Sample image", use_column_width=True)
            except Exception as e:
                # If book.png doesn't exist, show a placeholder
                st.warning("Sample image 'book.png' not found. Using placeholder instead.")
                # Create a simple placeholder using matplotlib
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.text(0.5, 0.5, 'Computer\nVision\nExplorer', horizontalalignment='center', 
                            verticalalignment='center', fontsize=20)
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as ex:
                    st.error(f"Could not create placeholder: {str(ex)}")
        
        st.markdown("""
        <div class="card">
        <h3>Dataset Information</h3>
        <p>This application uses three datasets:</p>
        <ol>
            <li><b>Book Covers</b>: Collection of book cover images in different viewing conditions</li>
            <li><b>Landmarks</b>: Famous landmarks photographed from different viewpoints</li>
            <li><b>Museum Paintings</b>: Paintings captured with varying lighting and perspectives</li>
        </ol>
        <p>Each dataset contains reference images and query images with various transformations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Matching
    elif st.session_state.current_page == "Feature Matching":
        st.markdown('<h2 class="sub-header">Feature Detection and Matching</h2>', unsafe_allow_html=True)
        
        # Dataset selection
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"])
        
        if dataset == "Book Covers":
            dataset_path = "A2_smvs/book_covers"
        elif dataset == "Landmarks":
            dataset_path = "A2_smvs/landmarks"
        else:
            dataset_path = "A2_smvs/museum_paintings"
        
        # Check dataset paths
        errors = check_dataset_paths(dataset_path)
        if errors:
            for error in errors:
                st.error(error)
            st.error("Please make sure the dataset folders exist at the correct paths.")
            st.info("Expected structure: A2_smvs/[dataset]/Reference and A2_smvs/[dataset]/Query")
            return
        
        # Image selection
        ref_files = sorted([f for f in os.listdir(f"{dataset_path}/Reference") if f.endswith('.jpg')])
        query_files = sorted([f for f in os.listdir(f"{dataset_path}/Query") if f.endswith('.jpg')])
        
        col1, col2 = st.columns(2)
        
        with col1:
            ref_img_idx = st.selectbox("Select Reference Image", range(len(ref_files)), 
                                     format_func=lambda i: ref_files[i])
        with col2:
            query_img_idx = st.selectbox("Select Query Image", range(len(query_files)), 
                                       format_func=lambda i: query_files[i])
        
        # Load images
        ref_img = cv2.imread(f"{dataset_path}/Reference/{ref_files[ref_img_idx]}", 0)
        query_img = cv2.imread(f"{dataset_path}/Query/{query_files[query_img_idx]}", 0)
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            nfeatures = st.slider("Number of Features", 100, 2000, 1000, 100)
        with col2:
            ratio_thresh = st.slider("Ratio Test Threshold", 0.5, 0.95, 0.8, 0.05)
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_detection = st.button("Detect and Match Features")
        
        if run_detection:
            with st.spinner("Detecting features and computing matches..."):
                kp1, kp2, des1, des2, good_matches = detect_and_match(ref_img, query_img, nfeatures, ratio_thresh)
                matches_img = visualize_matches(ref_img, query_img, kp1, kp2, good_matches)
                
                # Display images with keypoints
                ref_img_kp = cv2.drawKeypoints(ref_img, kp1, None, color=(0, 255, 0), flags=0)
                query_img_kp = cv2.drawKeypoints(query_img, kp2, None, color=(0, 255, 0), flags=0)
                
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"Found {len(kp1)} keypoints in reference image")
                st.markdown(f"Found {len(kp2)} keypoints in query image")
                st.markdown(f"Found {len(good_matches)} good matches after ratio test")
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(ref_img_kp, caption="Reference Image with Keypoints", use_column_width=True, channels="BGR")
                with col2:
                    st.image(query_img_kp, caption="Query Image with Keypoints", use_column_width=True, channels="BGR")
                
                st.image(matches_img, caption="Feature Matches", use_column_width=True, channels="BGR")
    
    # Homography Estimation
    elif st.session_state.current_page == "Homography Estimation":
        st.markdown('<h2 class="sub-header">Homography Estimation</h2>', unsafe_allow_html=True)
        
        # Dataset selection
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"])
        
        if dataset == "Book Covers":
            dataset_path = "A2_smvs/book_covers"
        elif dataset == "Landmarks":
            dataset_path = "A2_smvs/landmarks"
        else:
            dataset_path = "A2_smvs/museum_paintings"

        # Check dataset paths
        errors = check_dataset_paths(dataset_path)
        if errors:
            for error in errors:
                st.error(error)
            st.error("Please make sure the dataset folders exist at the correct paths.")
            st.info("Expected structure: A2_smvs/[dataset]/Reference and A2_smvs/[dataset]/Query")
            return
        
        # Image selection
        ref_files = sorted([f for f in os.listdir(f"{dataset_path}/Reference") if f.endswith('.jpg')])
        query_files = sorted([f for f in os.listdir(f"{dataset_path}/Query") if f.endswith('.jpg')])
        
        col1, col2 = st.columns(2)
        
        with col1:
            ref_img_idx = st.selectbox("Select Reference Image", range(len(ref_files)), 
                                     format_func=lambda i: ref_files[i])
        with col2:
            query_img_idx = st.selectbox("Select Query Image", range(len(query_files)), 
                                       format_func=lambda i: query_files[i])
        
        # Load images
        ref_img = cv2.imread(f"{dataset_path}/Reference/{ref_files[ref_img_idx]}", 0)
        query_img = cv2.imread(f"{dataset_path}/Query/{query_files[query_img_idx]}", 0)
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Method", ["Least Squares", "RANSAC"])
            method_val = 0 if method == "Least Squares" else cv2.RANSAC
        with col2:
            ratio_thresh = st.slider("Ratio Test Threshold", 0.5, 0.95, 0.8, 0.05)
        with col3:
            ransac_thresh = st.slider("RANSAC Threshold", 1.0, 10.0, 3.0, 0.5)
        
        run_homography = st.button("Estimate Homography")
        
        if run_homography:
            with st.spinner("Estimating homography..."):
                # Detect and match features
                kp1, kp2, des1, des2, good_matches = detect_and_match(ref_img, query_img, 1000, ratio_thresh)
                
                # Find homography
                H, mask, src_pts, dst_pts = find_homography(kp1, kp2, good_matches, method_val, ransac_thresh)
                
                # Draw outline
                img_outline = draw_outline(ref_img, query_img, H)
                
                # Draw matches with inliers
                matches_mask = mask.ravel().tolist()
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=None,
                    matchesMask=matches_mask,
                    flags=2
                )
                img_matches = cv2.drawMatches(ref_img, kp1, query_img, kp2, good_matches, None, **draw_params)
                
                # Results
                inliers = np.sum(mask)
                inlier_ratio = inliers / len(good_matches) if good_matches else 0
                
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"Found {len(good_matches)} good matches after ratio test")
                st.markdown(f"Inliers: {inliers} out of {len(good_matches)} ({inlier_ratio:.4f})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("#### Homography Matrix")
                if H is not None:
                    st.text(str(H))
                else:
                    st.text("No homography matrix found")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_outline, caption="Reference Image Outline Projected to Query", use_column_width=True)
                with col2:
                    st.image(img_matches, caption="Inlier Matches", use_column_width=True, channels="BGR")
    
    # Image Retrieval
    elif st.session_state.current_page == "Image Retrieval":
        st.markdown('<h2 class="sub-header">Content-Based Image Retrieval</h2>', unsafe_allow_html=True)
        
        # Dataset selection
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"])
        
        if dataset == "Book Covers":
            dataset_path = "A2_smvs/book_covers"
        elif dataset == "Landmarks":
            dataset_path = "A2_smvs/landmarks"
        else:
            dataset_path = "A2_smvs/museum_paintings"

        # Check dataset paths
        errors = check_dataset_paths(dataset_path)
        if errors:
            for error in errors:
                st.error(error)
            st.error("Please make sure the dataset folders exist at the correct paths.")
            st.info("Expected structure: A2_smvs/[dataset]/Reference and A2_smvs/[dataset]/Query")
            return
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            match_threshold = st.slider("Match Threshold", 5, 30, 15, 1)
        with col2:
            max_results = st.slider("Maximum Results", 1, 10, 5, 1)
        with col3:
            is_external = st.checkbox("Use External Query")
        
        # Image selection
        if is_external:
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Read the uploaded image
                image_bytes = uploaded_file.getvalue()
                query_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 0)
                query_path = None
                true_id = "external"
            else:
                st.warning("Please upload an image")
                return
        else:
            query_files = sorted([f for f in os.listdir(f"{dataset_path}/Query") if f.endswith('.jpg')])
            query_idx = st.selectbox("Select Query Image", range(len(query_files)), 
                                    format_func=lambda i: query_files[i])
            query_path = f"{dataset_path}/Query/{query_files[query_idx]}"
            query_img = cv2.imread(query_path, 0)
            true_id = query_files[query_idx].split('.')[0]
        
        run_retrieval = st.button("Retrieve Similar Images")
        
        if run_retrieval and query_img is not None:
            with st.spinner("Searching for matches..."):
                # Get reference images
                ref_path = f"{dataset_path}/Reference"
                ref_files = sorted([f for f in os.listdir(ref_path) if f.endswith('.jpg')])
                
                # Detect features in query
                orb = cv2.ORB_create(nfeatures=1000)
                kp_query, des_query = orb.detectAndCompute(query_img, None)
                
                # Create matcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                
                # Match against all reference images
                results = []
                
                for ref_file in ref_files:
                    ref_id = ref_file.split('.')[0]
                    ref_img = cv2.imread(f"{ref_path}/{ref_file}", 0)
                    
                    # Detect features in reference
                    kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
                    
                    try:
                        # Match descriptors
                        matches = bf.knnMatch(des_query, des_ref, k=2)
                        
                        # Apply ratio test
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.8 * n.distance:
                                good_matches.append(m)
                        
                        # If enough matches, compute homography
                        if len(good_matches) >= 10:
                            # Extract points
                            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            # Find homography with RANSAC
                            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                            
                            # Count inliers
                            inliers = np.sum(mask) if mask is not None else 0
                            results.append((ref_id, ref_file, inliers, len(good_matches)))
                        else:
                            results.append((ref_id, ref_file, 0, len(good_matches)))
                    except Exception as e:
                        results.append((ref_id, ref_file, 0, 0))
                
                # Sort by inlier count (descending)
                results.sort(key=lambda x: x[2], reverse=True)
                
                # Check if any results exceed threshold
                is_found = results[0][2] >= match_threshold
                
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown(f"### Query Image")
                st.image(query_img, caption="Query Image", width=300)
                
                if is_found:
                    st.markdown(f"### Retrieved Images")
                    st.markdown(f"Found {len([r for r in results if r[2] >= match_threshold])} images with scores above threshold")
                    
                    # Show top matches
                    top_matches = results[:max_results]
                    for i, (ref_id, ref_file, inliers, total_matches) in enumerate(top_matches):
                        col1, col2 = st.columns([1, 3])
                        
                        ref_img = cv2.imread(f"{ref_path}/{ref_file}", 0)
                        
                        with col1:
                            st.image(ref_img, caption=f"Match #{i+1}", width=200)
                        
                        with col2:
                            is_correct = ref_id == true_id
                            st.markdown(f"**ID**: {ref_id}")
                            st.markdown(f"**Score**: {inliers} inliers out of {total_matches} matches")
                            st.markdown(f"**Correct Match**: {'✅' if is_correct else '❌'}")
                else:
                    st.warning("No matches found above the threshold")
                    
                    # Show best match anyway
                    if results[0][2] > 0:
                        best_id, best_file, best_inliers, best_total = results[0]
                        st.markdown("#### Best Match (below threshold)")
                        col1, col2 = st.columns([1, 3])
                        
                        ref_img = cv2.imread(f"{ref_path}/{best_file}", 0)
                        
                        with col1:
                            st.image(ref_img, caption="Best Match", width=200)
                        
                        with col2:
                            is_correct = best_id == true_id
                            st.markdown(f"**ID**: {best_id}")
                            st.markdown(f"**Score**: {best_inliers} inliers out of {best_total} matches")
                            st.markdown(f"**Correct Match**: {'✅' if is_correct else '❌'}")
        
    # Epipolar Geometry
    elif st.session_state.current_page == "Epipolar Geometry":
        st.markdown('<h2 class="sub-header">Epipolar Geometry</h2>', unsafe_allow_html=True)
        
        # Dataset selection - using landmarks dataset as it's best for epipolar geometry
        dataset_path = "A2_smvs/landmarks"

        # Check dataset paths
        errors = check_dataset_paths(dataset_path)
        if errors:
            for error in errors:
                st.error(error)
            st.error("Please make sure the dataset folders exist at the correct paths.")
            st.info("Expected structure: A2_smvs/[dataset]/Reference and A2_smvs/[dataset]/Query")
            return
        
        # Image selection
        ref_files = sorted([f for f in os.listdir(f"{dataset_path}/Reference") if f.endswith('.jpg')])
        query_files = sorted([f for f in os.listdir(f"{dataset_path}/Query") if f.endswith('.jpg')])
        
        col1, col2 = st.columns(2)
        
        with col1:
            ref_img_idx = st.selectbox("Select Reference Image", range(len(ref_files)), 
                                     format_func=lambda i: ref_files[i])
        with col2:
            query_img_idx = st.selectbox("Select Query Image", range(len(query_files)), 
                                       format_func=lambda i: query_files[i])
        
        # Load images
        ref_img = cv2.imread(f"{dataset_path}/Reference/{ref_files[ref_img_idx]}", 0)
        query_img = cv2.imread(f"{dataset_path}/Query/{query_files[query_img_idx]}", 0)
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Method", ["8-point Algorithm", "RANSAC"])
            method_val = cv2.FM_8POINT if method == "8-point Algorithm" else cv2.FM_RANSAC
        with col2:
            num_lines = st.slider("Number of Epipolar Lines", 5, 20, 10, 1)
        with col3:
            ransac_thresh = st.slider("RANSAC Threshold", 0.5, 5.0, 1.0, 0.5)
        
        run_epipolar = st.button("Compute Epipolar Geometry")
        
        if run_epipolar:
            with st.spinner("Computing fundamental matrix and epipolar lines..."):
                # Detect and match features
                kp1, kp2, des1, des2, good_matches = detect_and_match(ref_img, query_img, 2000, 0.75)
                
                if len(good_matches) < 8:
                    st.error(f"Not enough good matches found: {len(good_matches)}. Need at least 8.")
                    return
                
                # Extract points
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                # Compute fundamental matrix
                F, mask = compute_fundamental_matrix(pts1, pts2, method_val, ransac_thresh)
                
                if F is None or mask is None:
                    st.error("Failed to compute fundamental matrix")
                    return
                
                # Count inliers
                inliers = np.sum(mask)
                inlier_ratio = inliers / len(good_matches)
                
                # Draw epipolar lines
                img1_lines, img2_lines = draw_epipolar_lines(ref_img, query_img, pts1, pts2, F, num_lines)
                
                # Display results
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"Found {len(good_matches)} good matches")
                st.markdown(f"Inliers: {inliers} out of {len(good_matches)} ({inlier_ratio:.4f})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("#### Fundamental Matrix")
                if F is not None:
                    st.text(str(F))
                else:
                    st.text("No fundamental matrix found")
                
                # Calculate epipolar error
                error_sum = 0
                for i in range(len(pts1)):
                    pt1 = np.array([pts1[i][0], pts1[i][1], 1])
                    pt2 = np.array([pts2[i][0], pts2[i][1], 1])
                    error = abs(np.dot(pt2, np.dot(F, pt1)))
                    error_sum += error
                
                avg_error = error_sum / len(pts1)
                st.markdown(f"Average Epipolar Error: {avg_error:.6f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1_lines, caption="Reference Image with Epipolar Lines", use_column_width=True, channels="BGR")
                with col2:
                    st.image(img2_lines, caption="Query Image with Epipolar Lines", use_column_width=True, channels="BGR")
                
                # Show matches
                matches_mask = mask.ravel().tolist()
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=None,
                    matchesMask=matches_mask,
                    flags=2
                )
                img_matches = cv2.drawMatches(ref_img, kp1, query_img, kp2, good_matches, None, **draw_params)
                st.image(img_matches, caption="Inlier Matches", use_column_width=True, channels="BGR")
        
    # Innovation Lab
    elif st.session_state.current_page == "Innovation":
        st.markdown('<h2 class="sub-header">Innovation Lab: Interactive Augmented Imagery</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight">
        <h3>About This Innovation</h3>
        <p>This interactive demo combines multiple computer vision techniques to create an augmented reality experience:</p>
        <ul>
            <li><b>Feature Detection & Matching</b>: Identifies corresponding points between images</li>
            <li><b>Homography Estimation</b>: Maps the reference image onto the query image plane</li>
            <li><b>Image Blending</b>: Creates a composite visualization with adjustable transparency</li>
            <li><b>Real-time Interaction</b>: Adjust parameters and see results instantly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset selection
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"], key="innovation_dataset")
        
        if dataset == "Book Covers":
            dataset_path = "A2_smvs/book_covers"
        elif dataset == "Landmarks":
            dataset_path = "A2_smvs/landmarks"
        else:
            dataset_path = "A2_smvs/museum_paintings"
        
        # Check dataset paths
        errors = check_dataset_paths(dataset_path)
        if errors:
            for error in errors:
                st.error(error)
            st.error("Please make sure the dataset folders exist at the correct paths.")
            st.info("Expected structure: A2_smvs/[dataset]/Reference and A2_smvs/[dataset]/Query")
            return
        
        # Image selection
        ref_files = sorted([f for f in os.listdir(f"{dataset_path}/Reference") if f.endswith('.jpg')])
        query_files = sorted([f for f in os.listdir(f"{dataset_path}/Query") if f.endswith('.jpg')])
        
        col1, col2 = st.columns(2)
        
        with col1:
            ref_img_idx = st.selectbox("Select Reference Image", range(len(ref_files)), 
                                     format_func=lambda i: ref_files[i], key="innovation_ref")
        with col2:
            query_img_idx = st.selectbox("Select Query Image", range(len(query_files)), 
                                       format_func=lambda i: query_files[i], key="innovation_query")
        
        # Load images
        ref_img = cv2.imread(f"{dataset_path}/Reference/{ref_files[ref_img_idx]}", 0)
        query_img = cv2.imread(f"{dataset_path}/Query/{query_files[query_img_idx]}", 0)
        
        # Interactive parameters
        st.markdown("### Adjust Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            blend_alpha = st.slider("Transparency", 0.0, 1.0, 0.5, 0.05, key="innovation_alpha")
        
        with col2:
            color_mode = st.selectbox("Visualization Mode", ["Augmented Reality", "Heat Map", "Wireframe"], key="innovation_mode")
        
        # Process and display
        st.markdown("### Interactive Visualization")
        with st.spinner("Creating augmented imagery..."):
            # Create augmented imagery
            composite, matched_img, img_outline, num_matches, num_inliers, homography_matrix = interactive_augmented_imagery(ref_img, query_img)
            
            if composite is not None and homography_matrix is not None:
                # Apply adjustments based on user parameters
                if color_mode == "Heat Map":
                    # Convert to heatmap visualization
                    composite = cv2.applyColorMap(composite, cv2.COLORMAP_JET)
                elif color_mode == "Wireframe":
                    # Create a wireframe visualization
                    composite = img_outline
                
                # Apply transparency adjustment
                if color_mode == "Augmented Reality":
                    try:
                        # Create images with adjusted transparency
                        h, w = ref_img.shape
                        if len(ref_img.shape) == 2:
                            ref_img_color = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        else:
                            ref_img_color = ref_img
                            
                        if len(query_img.shape) == 2:
                            query_img_color = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
                        else:
                            query_img_color = query_img
                            
                        # Warp reference image using the homography matrix
                        warped = cv2.warpPerspective(ref_img_color, homography_matrix, (query_img_color.shape[1], query_img_color.shape[0]))
                        
                        # Blend with adjusted alpha
                        composite = cv2.addWeighted(query_img_color, 1-blend_alpha, warped, blend_alpha, 0)
                    except Exception as e:
                        st.error(f"Error creating augmented reality view: {str(e)}")
                        # Use the original composite as fallback
                
                # Display the results
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"Found {num_matches} good matches")
                st.markdown(f"Identified {num_inliers} inliers for transformation")
                st.markdown(f"Inlier ratio: {num_inliers/num_matches if num_matches > 0 else 0:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(composite, caption=f"{color_mode} Visualization", use_column_width=True, channels="BGR")
                with col2:
                    st.image(matched_img, caption="Feature Matching", use_column_width=True, channels="BGR")
                
                # Add interactive 3D effect using matplotlib
                st.markdown("### 3D Perspective Visualization")
                
                # Create 3D visualization separately to avoid reference issues
                try:
                    # Import matplotlib inside the try block to ensure it's available
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    # Detect features again for 3D visualization
                    orb = cv2.ORB_create(nfeatures=1500)
                    kp1, des1 = orb.detectAndCompute(ref_img, None)
                    kp2, des2 = orb.detectAndCompute(query_img, None)
                    
                    # Match descriptors
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                    matches = bf.knnMatch(des1, des2, k=2)
                    
                    # Apply ratio test
                    viz_good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            viz_good_matches.append(m)
                    
                    # Create 3D perspective plot
                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Extract points for 3D visualization
                    # Make sure we have enough matches for visualization
                    num_points_to_show = min(20, len(viz_good_matches))
                    if num_points_to_show > 0:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in viz_good_matches[:num_points_to_show]]).reshape(-1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in viz_good_matches[:num_points_to_show]]).reshape(-1, 2)
                        
                        # Create z coordinates
                        z_src = np.zeros(len(src_pts))
                        z_dst = np.ones(len(dst_pts)) * 10
                        
                        # Plot points
                        ax.scatter(src_pts[:, 0], src_pts[:, 1], z_src, c='r', marker='o', label='Reference Points')
                        ax.scatter(dst_pts[:, 0], dst_pts[:, 1], z_dst, c='b', marker='^', label='Query Points')
                        
                        # Plot connection lines
                        for i in range(len(src_pts)):
                            ax.plot([src_pts[i, 0], dst_pts[i, 0]], 
                                    [src_pts[i, 1], dst_pts[i, 1]], 
                                    [z_src[i], z_dst[i]], 'g-', alpha=0.3)
                        
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title('3D Perspective of Feature Matches')
                        ax.legend()
                        
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough matches to create 3D visualization")
                except Exception as e:
                    st.error(f"Error creating 3D visualization: {str(e)}")
            else:
                st.error("Not enough good matches found to create the visualization. Try different images or adjust parameters.")
        
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Computer Vision Explorer | Developed by <a href="https://devshubh.me" target="_blank">Shubharthak Sangharasha</a> | 
        <a href="https://github.com/shubharthaksangharsha/trimester2/tree/main/opencv" target="_blank">GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main() 