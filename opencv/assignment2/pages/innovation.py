import streamlit as st
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import check_dataset_paths, detect_and_match, find_homography, interactive_augmented_imagery

def render_page():
    st.markdown('<h2 class="sub-header">Innovation: Interactive Augmented Imagery 🔬</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
    This demo combines feature matching and homography to overlay a reference image onto a query image. Adjust the transparency to see the effect in real-time!
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        dataset = st.selectbox("Select Dataset", ["Book Covers", "Landmarks", "Museum Paintings"], key="inno_dataset")
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
            ref_choice = st.selectbox("Select Reference Image", ref_files, key="inno_ref")
        with col2:
            query_choice = st.selectbox("Select Query Image", query_files, key="inno_query")

        ref_img = cv2.imread(os.path.join(ref_path, ref_choice), 0)
        query_img = cv2.imread(os.path.join(query_path, query_choice), 0)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Interactive Parameters")

        col1, col2 = st.columns(2)
        with col1:
            blend_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.6, 0.05)
        with col2:
            color_mode = st.selectbox("Visualization Mode", ["Augmented Reality", "Heat Map", "Wireframe"])

        with st.spinner("Updating visualization..."):
            composite, matched_img, outline_img, num_matches, num_inliers, H = interactive_augmented_imagery(ref_img, query_img, blend_alpha)

            if H is not None and composite is not None:
                # Modify visualization based on selected mode
                if color_mode == "Heat Map":
                    try:
                        # Apply color map for heat map visualization
                        composite = cv2.applyColorMap(cv2.convertScaleAbs(composite, alpha=0.8), cv2.COLORMAP_JET)
                    except Exception as e:
                        st.warning(f"Error applying heat map: {e}")
                elif color_mode == "Wireframe":
                    # Use outline image for wireframe view
                    composite = outline_img

                st.markdown("#### Interactive Augmented View")
                st.image(composite, use_column_width=True, channels="BGR")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Feature Matches")
                    st.image(matched_img, use_column_width=True, channels="BGR")
                with col2:
                    st.markdown("#### Match Statistics")
                    st.markdown(f"""
                    <div class='highlight' style='padding: 1rem;'>
                    <p><strong>Good Matches:</strong> {num_matches}</p>
                    <p><strong>Inliers:</strong> {num_inliers}</p>
                    <p><strong>Inlier Ratio:</strong> {num_inliers/num_matches:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Create 3D visualization
                try:
                    st.markdown("#### 3D Perspective Visualization")
                    
                    # Extract points for visualization from matched features
                    kp1, kp2, _, _, good_matches = detect_and_match(ref_img, query_img, nfeatures=1500)
                    
                    if H is not None and len(good_matches) >= 10:
                        # Create 3D plot
                        fig = plt.figure(figsize=(10, 6))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # Get points from matches
                        num_points = min(15, len(good_matches))
                        src_pts = np.float32([kp1[good_matches[i].queryIdx].pt for i in range(num_points)])
                        dst_pts = np.float32([kp2[good_matches[i].trainIdx].pt for i in range(num_points)])
                        
                        # Create Z coordinates for 3D effect
                        z_src = np.zeros(len(src_pts))
                        z_dst = np.ones(len(dst_pts)) * 10
                        
                        # Plot points
                        ax.scatter(src_pts[:, 0], src_pts[:, 1], z_src, c='r', marker='o', label='Reference Points')
                        ax.scatter(dst_pts[:, 0], dst_pts[:, 1], z_dst, c='b', marker='^', label='Query Points')
                        
                        # Plot connecting lines
                        for i in range(len(src_pts)):
                            ax.plot([src_pts[i, 0], dst_pts[i, 0]],
                                    [src_pts[i, 1], dst_pts[i, 1]],
                                    [z_src[i], z_dst[i]], 'g-', alpha=0.5)
                        
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title('3D Perspective of Feature Matches')
                        ax.legend()
                        
                        # Set view angle for better visualization
                        ax.view_init(elev=20, azim=-35)
                        
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not create 3D visualization: {e}")
            else:
                st.error("Could not compute a stable homography for these images. Try another pair.")
        
        st.markdown("</div>", unsafe_allow_html=True)
