import streamlit as st
import cv2
import os
import numpy as np
from utils import check_dataset_paths, detect_and_match, find_homography
import matplotlib.pyplot as plt

def render_page():
    st.markdown('<h2 class="sub-header">Content-Based Image Retrieval</h2>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"], key="retrieval_dataset")
        dataset_path = f"A2_smvs/{dataset.lower().replace(' ', '_')}"

        if not check_dataset_paths(dataset_path):
            st.markdown('</div>', unsafe_allow_html=True)
            return

        is_external = st.checkbox("Use External Query Image", key="retrieval_external")
        query_img = None
        true_id = "external"
        query_path = None

        if is_external:
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="retrieval_upload")
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                query_img = cv2.imdecode(file_bytes, 0)
                st.image(query_img, caption="Uploaded Query Image", use_column_width=True)
        else:
            query_path = os.path.join(dataset_path, "Query")
            query_files = sorted([f for f in os.listdir(query_path) if f.endswith(('.jpg', '.png'))])
            query_choice = st.selectbox("Select Query Image", query_files, key="retrieval_query_image")
            query_path_full = os.path.join(query_path, query_choice)
            query_img = cv2.imread(query_path_full, 0)
            true_id = os.path.splitext(query_choice)[0]
            
            if query_img is not None:
                st.image(query_img, caption=f"Query: {query_choice}", use_column_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Search Parameters")
        
        # Create responsive parameter layout
        param_cols = st.columns(3)
        with param_cols[0]:
            inlier_threshold = st.slider("Minimum Inliers", 5, 50, 15, 1, key="retrieval_inliers",
                                      help="Minimum number of inliers required for a match")
        with param_cols[1]:
            ratio_thresh = st.slider("Ratio Threshold", 0.6, 0.95, 0.8, 0.05, key="retrieval_ratio",
                                  help="Ratio test threshold for feature matching")
        with param_cols[2]:
            max_results = st.slider("Maximum Results", 1, 10, 5, 1, key="retrieval_max_results",
                                 help="Maximum number of results to display")

        if st.button("Retrieve Similar Images", use_container_width=True, key="retrieval_search_btn"):
            if query_img is None:
                st.warning("Please select or upload a query image.")
            else:
                with st.spinner("Searching database..."):
                    ref_path = os.path.join(dataset_path, "Reference")
                    ref_files = sorted([f for f in os.listdir(ref_path) if f.endswith(('.jpg', '.png'))])
                    
                    # Detect features in query image
                    orb = cv2.ORB_create(nfeatures=1500)
                    kp_query, des_query = orb.detectAndCompute(query_img, None)
                    
                    if kp_query is None or len(kp_query) < 10 or des_query is None:
                        st.error("Not enough features detected in query image.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        return

                    # Create progress bar for search operation
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []
                    for i, ref_file in enumerate(ref_files):
                        # Update progress
                        progress = int((i + 1) / len(ref_files) * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing image {i+1}/{len(ref_files)}: {ref_file}")
                        
                        # Load reference image
                        ref_img = cv2.imread(os.path.join(ref_path, ref_file), 0)
                        if ref_img is None:
                            continue
                        
                        # Match features
                        kp1, kp2, _, _, good_matches = detect_and_match(query_img, ref_img, ratio_thresh=ratio_thresh)
                        
                        # If enough matches found, compute homography
                        if len(good_matches) >= 10:
                            H, mask, _, _ = find_homography(kp1, kp2, good_matches)
                            
                            # Count inliers
                            inliers = np.sum(mask) if mask is not None else 0
                            
                            # Store result
                            ref_id = os.path.splitext(ref_file)[0]
                            results.append((ref_id, ref_file, inliers, len(good_matches)))
                        else:
                            ref_id = os.path.splitext(ref_file)[0]
                            results.append((ref_id, ref_file, 0, len(good_matches)))

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Sort results by inlier count (descending)
                    results.sort(key=lambda x: x[2], reverse=True)
                    
                    st.markdown("---")
                    st.subheader("Search Results")
                    
                    # Display query image
                    st.markdown("#### Query Image")
                    st.image(query_img, caption="Query Image", width=300)

                    # Filter results by threshold
                    matches_above_threshold = [r for r in results if r[2] >= inlier_threshold]
                    
                    if not matches_above_threshold:
                        st.warning(f"No matches found with {inlier_threshold} or more inliers.")
                        
                        # Show best match anyway
                        if results:
                            st.markdown("#### Best Match (Below Threshold)")
                            best_match = results[0]
                            best_id, best_file, best_inliers, total_matches = best_match
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                best_img = cv2.imread(os.path.join(ref_path, best_file))
                                st.image(best_img, caption=f"{best_file}", width=200, channels="BGR")
                            with col2:
                                is_correct = (best_id == true_id)
                                st.markdown(f"**ID:** {best_id}")
                                st.markdown(f"**Score:** {best_inliers} inliers out of {total_matches} matches")
                                st.markdown(f"**Correct Match:** {'✓' if is_correct else '✗'}")
                    else:
                        st.success(f"Found {len(matches_above_threshold)} matches with {inlier_threshold}+ inliers.")
                        
                        # Create a visualization of match scores
                        try:
                            if len(matches_above_threshold) >= 2:
                                scores = [match[2] for match in matches_above_threshold[:10]]
                                labels = [match[0] for match in matches_above_threshold[:10]]
                                
                                # Create a horizontal bar chart
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.barh(labels, scores, color='skyblue')
                                ax.set_xlabel('Inlier Count')
                                ax.set_title('Top Match Scores')
                                
                                st.pyplot(fig)
                        except Exception:
                            # If matplotlib import fails, skip the chart
                            pass

                        # Display top matches
                        st.markdown("#### Top Matches")
                        for i, (match_id, match_file, inliers, total_matches) in enumerate(matches_above_threshold[:max_results]):
                            st.markdown("---")
                            st.markdown(f"**Match #{i+1}**")
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                match_img = cv2.imread(os.path.join(ref_path, match_file))
                                st.image(match_img, caption=f"{match_file}", width=200, channels="BGR")
                            with col2:
                                is_correct = (match_id == true_id)
                                st.markdown(f"**ID:** {match_id}")
                                st.markdown(f"**Score:** {inliers} inliers out of {total_matches} matches")
                                st.markdown(f"**Inlier Ratio:** {inliers/total_matches:.2%}")
                                st.markdown(f"**Correct Match:** {'✓' if is_correct else '✗'}")
                                
                                # Show feature matches
                                if st.button(f"Show Detailed Match #{i+1}", key=f"show_match_{i}"):
                                    with st.spinner("Computing matches..."):
                                        # Load image and match again to display
                                        ref_img = cv2.imread(os.path.join(ref_path, match_file), 0)
                                        kp1, kp2, _, _, good_matches = detect_and_match(query_img, ref_img, ratio_thresh=ratio_thresh)
                                        H, mask, _, _ = find_homography(kp1, kp2, good_matches)
                                        
                                        if H is not None and mask is not None:
                                            # Draw matches with inliers highlighted
                                            matches_mask = mask.ravel().tolist()
                                            img_matches = cv2.drawMatches(query_img, kp1, ref_img, kp2, 
                                                                        good_matches, None, 
                                                                        matchColor=(0, 255, 0), 
                                                                        singlePointColor=None, 
                                                                        matchesMask=matches_mask, 
                                                                        flags=2)
                                            
                                            st.image(img_matches, caption="Inlier Matches", use_column_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
