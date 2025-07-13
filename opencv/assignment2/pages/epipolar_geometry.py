import streamlit as st
import cv2
import os
import numpy as np
from utils import check_dataset_paths, detect_and_match, compute_fundamental_matrix, draw_epipolar_lines

def render_page():
    st.markdown('<h2 class="sub-header">Epipolar Geometry</h2>', unsafe_allow_html=True)
    st.info("This visualization works best with images of the same scene from different viewpoints, such as the 'Landmarks' dataset.")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Default to landmarks as it's most suitable
        dataset_options = ["Landmarks", "Book Covers", "Museum Paintings"]
        dataset = st.selectbox("Select Dataset", dataset_options, index=0, key="epipolar_dataset")
        dataset_path = f"A2_smvs/{dataset.lower().replace(' ', '_')}"

        if not check_dataset_paths(dataset_path):
            st.markdown('</div>', unsafe_allow_html=True)
            return

        ref_path = os.path.join(dataset_path, "Reference")
        query_path = os.path.join(dataset_path, "Query")
        ref_files = sorted([f for f in os.listdir(ref_path) if f.endswith(('.jpg', '.png'))])
        query_files = sorted([f for f in os.listdir(query_path) if f.endswith(('.jpg', '.png'))])

        col1, col2 = st.columns(2)
        with col1:
            ref_choice = st.selectbox("Select Reference Image", ref_files, key="epipolar_ref_image")
            ref_img = cv2.imread(os.path.join(ref_path, ref_choice), 0)
            if ref_img is not None:
                st.image(ref_img, caption=f"Reference: {ref_choice}", use_column_width=True)
        with col2:
            query_choice = st.selectbox("Select Query Image", query_files, key="epipolar_query_image")
            query_img = cv2.imread(os.path.join(query_path, query_choice), 0)
            if query_img is not None:
                st.image(query_img, caption=f"Query: {query_choice}", use_column_width=True)

        if ref_img is None or query_img is None:
            st.error("Failed to load one or both images.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Parameters")
        
        param_cols = st.columns([1, 1, 1])
        with param_cols[0]:
            method_str = st.selectbox("Method", ["RANSAC", "8-Point Algorithm"], key="epipolar_method")
            method = cv2.FM_RANSAC if method_str == "RANSAC" else cv2.FM_8POINT
        with param_cols[1]:
            num_lines = st.slider("Number of Epipolar Lines", 5, 25, 10, 1, key="epipolar_num_lines")
        with param_cols[2]:
            threshold = st.slider("RANSAC Threshold", 0.5, 5.0, 1.0, 0.5, key="epipolar_threshold",
                                 help="Distance threshold for RANSAC (pixels)",
                                 disabled=(method != cv2.FM_RANSAC))

        if st.button("Compute Epipolar Geometry", use_container_width=True, key="epipolar_compute_btn"):
            with st.spinner("Processing..."):
                kp1, kp2, _, _, good_matches = detect_and_match(ref_img, query_img, nfeatures=2000)
                
                if len(good_matches) < 8:
                    st.error(f"Not enough matches found to compute the Fundamental Matrix. Found {len(good_matches)}, need at least 8.")
                else:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                    F, mask = compute_fundamental_matrix(pts1, pts2, method, threshold)

                    st.markdown("---")
                    st.subheader("Results")

                    if F is None:
                        st.error("Fundamental Matrix estimation failed.")
                    else:
                        # Show fundamental matrix
                        with st.expander("Fundamental Matrix", expanded=True):
                            st.code(np.round(F, 4))
                        
                        # Calculate epipolar error
                        if mask is not None:
                            inliers = np.sum(mask)
                            inlier_ratio = inliers / len(pts1) if len(pts1) > 0 else 0
                            
                            # Get inlier points
                            mask_flat = mask.ravel().tolist() if mask is not None else []
                            pts1_inliers = pts1[np.array(mask_flat).astype(bool)] if len(mask_flat) > 0 else []
                            pts2_inliers = pts2[np.array(mask_flat).astype(bool)] if len(mask_flat) > 0 else []
                            
                            # Calculate epipolar error
                            error_sum = 0
                            if len(pts1_inliers) > 0:
                                for i in range(len(pts1_inliers)):
                                    pt1 = np.array([pts1_inliers[i][0], pts1_inliers[i][1], 1])
                                    pt2 = np.array([pts2_inliers[i][0], pts2_inliers[i][1], 1])
                                    error = abs(np.dot(pt2, np.dot(F, pt1)))
                                    error_sum += error
                                
                                avg_error = error_sum / len(pts1_inliers)
                                
                                metric_cols = st.columns(3)
                                metric_cols[0].metric("Matches", len(pts1))
                                metric_cols[1].metric("Inliers", int(inliers))
                                metric_cols[2].metric("Avg. Epipolar Error", f"{avg_error:.6f}")
                        
                        # Draw epipolar lines
                        img1_lines, img2_lines = draw_epipolar_lines(ref_img, query_img, pts1, pts2, F, num_lines)

                        st.markdown("#### Epipolar Lines Visualization")
                        viz_cols = st.columns(2)
                        with viz_cols[0]:
                            st.image(img1_lines, caption="Epipolar Lines on Reference Image", use_column_width=True)
                        with viz_cols[1]:
                            st.image(img2_lines, caption="Epipolar Lines on Query Image", use_column_width=True)
                        
                        # Draw matches with inliers highlighted
                        st.markdown("#### Feature Matches")
                        if mask is not None:
                            matches_mask = mask.ravel().tolist()
                            img_matches = cv2.drawMatches(ref_img, kp1, query_img, kp2, good_matches, None, 
                                                        matchColor=(0, 255, 0),
                                                        singlePointColor=None,
                                                        matchesMask=matches_mask,
                                                        flags=2)
                            st.image(img_matches, caption="Inlier Matches", use_column_width=True)
                        
                        # Show epipole visualization if possible
                        try:
                            # Calculate epipoles
                            st.markdown("#### Epipoles")
                            
                            # Get epipoles by computing the null space of F and F.T
                            U, S, Vh = np.linalg.svd(F)
                            epipole1 = Vh[-1]  # Last row of Vh is right null vector
                            epipole1 = epipole1 / epipole1[2]  # Normalize
                            
                            U, S, Vh = np.linalg.svd(F.T)
                            epipole2 = Vh[-1]  # Last row of Vh is right null vector
                            epipole2 = epipole2 / epipole2[2]  # Normalize
                            
                            # Create copies for drawing
                            img1_with_epipole = cv2.cvtColor(ref_img.copy(), cv2.COLOR_GRAY2BGR)
                            img2_with_epipole = cv2.cvtColor(query_img.copy(), cv2.COLOR_GRAY2BGR)
                            
                            # Check if epipoles are within image boundaries
                            h1, w1 = ref_img.shape[:2]
                            h2, w2 = query_img.shape[:2]
                            
                            # Draw epipole on first image if within bounds
                            e1_x, e1_y = int(epipole1[0]), int(epipole1[1])
                            e1_in_bounds = 0 <= e1_x < w1 and 0 <= e1_y < h1
                            
                            e2_x, e2_y = int(epipole2[0]), int(epipole2[1])
                            e2_in_bounds = 0 <= e2_x < w2 and 0 <= e2_y < h2
                            
                            if e1_in_bounds:
                                cv2.circle(img1_with_epipole, (e1_x, e1_y), 10, (0, 0, 255), -1)
                                cv2.circle(img1_with_epipole, (e1_x, e1_y), 5, (255, 255, 255), -1)
                            
                            if e2_in_bounds:
                                cv2.circle(img2_with_epipole, (e2_x, e2_y), 10, (0, 0, 255), -1)
                                cv2.circle(img2_with_epipole, (e2_x, e2_y), 5, (255, 255, 255), -1)
                            
                            epipole_cols = st.columns(2)
                            with epipole_cols[0]:
                                if e1_in_bounds:
                                    st.image(img1_with_epipole, caption=f"Reference Image Epipole ({e1_x}, {e1_y})", use_column_width=True)
                                else:
                                    st.info(f"Epipole 1 is outside image bounds: ({e1_x}, {e1_y})")
                            
                            with epipole_cols[1]:
                                if e2_in_bounds:
                                    st.image(img2_with_epipole, caption=f"Query Image Epipole ({e2_x}, {e2_y})", use_column_width=True)
                                else:
                                    st.info(f"Epipole 2 is outside image bounds: ({e2_x}, {e2_y})")
                                    
                        except Exception as e:
                            st.warning(f"Could not compute or display epipoles: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
