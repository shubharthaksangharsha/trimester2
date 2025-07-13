import streamlit as st
import cv2
import os
import numpy as np
from utils import check_dataset_paths, detect_and_match, visualize_matches

def render_page():
    st.markdown('<h2 class="sub-header">Feature Detection and Matching</h2>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"], key="feature_dataset")
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
            ref_choice = st.selectbox("Select Reference Image", ref_files, key="feature_ref_image")
            ref_img = cv2.imread(os.path.join(ref_path, ref_choice), 0)
            if ref_img is not None:
                st.image(ref_img, caption=f"Reference: {ref_choice}", use_column_width=True)
        with col2:
            query_choice = st.selectbox("Select Query Image", query_files, key="feature_query_image")
            query_img = cv2.imread(os.path.join(query_path, query_choice), 0)
            if query_img is not None:
                st.image(query_img, caption=f"Query: {query_choice}", use_column_width=True)

        if ref_img is None or query_img is None:
            st.error("Failed to load one or both images.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Parameters")
        
        # Create responsive parameter controls based on screen size
        use_cols = st.columns(3)
        with use_cols[0]:
            nfeatures = st.slider("Number of Features (ORB)", 100, 5000, 1000, 100, key="feature_nfeatures")
        with use_cols[1]:
            ratio_thresh = st.slider("Ratio Test Threshold", 0.5, 0.95, 0.8, 0.05, key="feature_ratio")
        with use_cols[2]:
            st.write("")
            st.write("")
            detect_button = st.button("Detect and Match Features", use_container_width=True, key="feature_detect_btn")
        
        if detect_button:
            with st.spinner("Processing..."):
                kp1, kp2, des1, des2, good_matches = detect_and_match(ref_img, query_img, nfeatures, ratio_thresh)
                
                st.markdown("---")
                st.subheader("Results")
                
                metric_cols = st.columns(3)
                metric_cols[0].metric("Reference Keypoints", len(kp1))
                metric_cols[1].metric("Query Keypoints", len(kp2))
                metric_cols[2].metric("Good Matches", len(good_matches))

                # Draw keypoints on images
                ref_img_kp = cv2.drawKeypoints(ref_img, kp1, None, color=(0, 255, 0), flags=0)
                query_img_kp = cv2.drawKeypoints(query_img, kp2, None, color=(0, 255, 0), flags=0)

                kp_cols = st.columns(2)
                with kp_cols[0]:
                    st.image(ref_img_kp, caption="Reference Image with Keypoints", use_column_width=True)
                with kp_cols[1]:
                    st.image(query_img_kp, caption="Query Image with Keypoints", use_column_width=True)

                # Draw matches
                matches_img = visualize_matches(ref_img, query_img, kp1, kp2, good_matches)
                
                st.markdown("#### Feature Matches")
                st.image(matches_img, caption=f"{len(good_matches)} Good Matches", use_column_width=True)
                
                # Feature distribution visualization
                st.markdown("#### Feature Distribution")
                try:
                    # Create a heatmap of keypoint distribution
                    h, w = ref_img.shape[:2]
                    ref_heatmap = np.zeros((h, w), dtype=np.uint8)
                    query_heatmap = np.zeros((query_img.shape[:2]), dtype=np.uint8)
                    
                    # Add keypoints to heatmap
                    for kp in kp1:
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(ref_heatmap, (x, y), 5, 255, -1)
                    
                    for kp in kp2:
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        if 0 <= x < query_img.shape[1] and 0 <= y < query_img.shape[0]:
                            cv2.circle(query_heatmap, (x, y), 5, 255, -1)
                    
                    # Apply color map
                    ref_heatmap = cv2.applyColorMap(ref_heatmap, cv2.COLORMAP_JET)
                    query_heatmap = cv2.applyColorMap(query_heatmap, cv2.COLORMAP_JET)
                    
                    # Blend with original image
                    ref_blend = cv2.addWeighted(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR), 0.7, ref_heatmap, 0.3, 0)
                    query_blend = cv2.addWeighted(cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR), 0.7, query_heatmap, 0.3, 0)
                    
                    heat_cols = st.columns(2)
                    with heat_cols[0]:
                        st.image(ref_blend, caption="Reference Feature Distribution", use_column_width=True, channels="BGR")
                    with heat_cols[1]:
                        st.image(query_blend, caption="Query Feature Distribution", use_column_width=True, channels="BGR")
                except Exception as e:
                    st.warning(f"Could not create feature distribution visualization: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
