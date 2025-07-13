import streamlit as st
import cv2
import os
import numpy as np
from utils import check_dataset_paths, detect_and_match, find_homography, draw_outline

def render_page():
    st.markdown('<h2 class="sub-header">Homography Estimation</h2>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"], key="homography_dataset")
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
            ref_choice = st.selectbox("Select Reference Image", ref_files, key="homography_ref_image")
            ref_img = cv2.imread(os.path.join(ref_path, ref_choice), 0)
            if ref_img is not None:
                st.image(ref_img, caption=f"Reference: {ref_choice}", use_column_width=True)
        with col2:
            query_choice = st.selectbox("Select Query Image", query_files, key="homography_query_image")
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
        
        # Responsive parameter layout
        param_cols = st.columns([1, 1, 1])
        with param_cols[0]:
            method_str = st.selectbox("Method", ["RANSAC", "Least Squares (LMEDS)"], key="homography_method")
            method = cv2.RANSAC if method_str == "RANSAC" else cv2.LMEDS
        with param_cols[1]:
            ratio_thresh = st.slider("Ratio Threshold", 0.5, 0.95, 0.8, 0.05, key="homography_ratio")
        with param_cols[2]:
            ransac_thresh = st.slider("RANSAC Threshold", 1.0, 10.0, 5.0, 0.5, key="homography_ransac_thresh", disabled=(method != cv2.RANSAC))

        if st.button("Estimate Homography", use_container_width=True, key="homography_estimate_btn"):
            with st.spinner("Processing..."):
                kp1, kp2, _, _, good_matches = detect_and_match(ref_img, query_img, ratio_thresh=ratio_thresh)
                
                if len(good_matches) < 4:
                    st.error("Not enough matches found to compute homography. Try different images or parameters.")
                else:
                    H, mask, src_pts, dst_pts = find_homography(kp1, kp2, good_matches, method, ransac_thresh)

                    st.markdown("---")
                    st.subheader("Results")

                    if H is None:
                        st.error("Homography estimation failed.")
                    else:
                        inliers = np.sum(mask)
                        inlier_ratio = inliers / len(good_matches) if len(good_matches) > 0 else 0
                        
                        metric_cols = st.columns(3)
                        metric_cols[0].metric("Good Matches", len(good_matches))
                        metric_cols[1].metric("Inliers", int(inliers))
                        metric_cols[2].metric("Inlier Ratio", f"{inlier_ratio:.2%}")

                        # Display homography matrix
                        with st.expander("Homography Matrix", expanded=True):
                            st.code(np.round(H, 4))

                        # Draw outline of reference image on query image
                        img_outline = draw_outline(ref_img, query_img, H)
                        
                        # Draw matches with inliers highlighted
                        matches_mask = mask.ravel().tolist()
                        img_matches = cv2.drawMatches(ref_img, kp1, query_img, kp2, good_matches, None, 
                                                      matchColor=(0, 255, 0), 
                                                      singlePointColor=None, 
                                                      matchesMask=matches_mask, 
                                                      flags=2)
                        
                        # Calculate and display reprojection error
                        if len(src_pts) > 0:
                            # Transform source points
                            src_pts_transformed = cv2.perspectiveTransform(src_pts, H)
                            
                            # Calculate errors
                            errors = np.sqrt(np.sum((dst_pts - src_pts_transformed)**2, axis=2))
                            inlier_errors = errors[np.array(matches_mask).astype(bool)]
                            
                            if len(inlier_errors) > 0:
                                avg_error = np.mean(inlier_errors)
                                max_error = np.max(inlier_errors)
                                
                                st.metric("Average Reprojection Error", f"{avg_error:.2f} pixels")
                                st.metric("Maximum Reprojection Error", f"{max_error:.2f} pixels")

                        # Display result images
                        st.markdown("#### Visualization")
                        result_cols = st.columns(2)
                        with result_cols[0]:
                            st.image(img_outline, caption="Reference Outline on Query Image", use_column_width=True, channels="BGR")
                        with result_cols[1]:
                            st.image(img_matches, caption="Inlier Matches", use_column_width=True)
                            
                        # Add perspective warping visualization
                        try:
                            st.markdown("#### Perspective Transformation")
                            
                            # Warp the reference image to match the query image
                            h, w = query_img.shape[:2]
                            warped = cv2.warpPerspective(ref_img, H, (w, h))
                            
                            # Create a side-by-side comparison
                            warp_cols = st.columns(2)
                            with warp_cols[0]:
                                st.image(warped, caption="Reference Warped to Query Perspective", use_column_width=True)
                            with warp_cols[1]:
                                # Create a blended overlay
                                if len(query_img.shape) == 2:
                                    query_color = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
                                    warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                                else:
                                    query_color = query_img.copy()
                                    warped_color = warped.copy()
                                    
                                blend = cv2.addWeighted(query_color, 0.5, warped_color, 0.5, 0)
                                st.image(blend, caption="Overlay of Query and Warped Reference", use_column_width=True, channels="BGR")
                        except Exception as e:
                            st.warning(f"Could not create warping visualization: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

