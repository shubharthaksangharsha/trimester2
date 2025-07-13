import cv2
import numpy as np
import streamlit as st
import os

def check_dataset_paths(dataset_path):
    """
    Checks if the specified dataset path and its subdirectories exist.
    Displays an error in Streamlit if paths are missing.
    Returns True if paths are valid, False otherwise.
    """
    ref_path = os.path.join(dataset_path, "Reference")
    query_path = os.path.join(dataset_path, "Query")

    if not os.path.exists(dataset_path):
        st.error(f"Dataset path does not exist: '{dataset_path}'")
        return False
    if not os.path.exists(ref_path) or not os.path.isdir(ref_path):
        st.error(f"Reference folder not found at: '{ref_path}'")
        return False
    if not os.path.exists(query_path) or not os.path.isdir(query_path):
        st.error(f"Query folder not found at: '{query_path}'")
        return False
    if not os.listdir(ref_path):
        st.warning(f"The Reference folder is empty: '{ref_path}'")
    if not os.listdir(query_path):
        st.warning(f"The Query folder is empty: '{query_path}'")
        
    return True

def detect_and_match(img1, img2, nfeatures=1000, ratio_thresh=0.8):
    """
    Detects features using ORB and matches them using BFMatcher with a ratio test.
    """
    if img1 is None or img2 is None:
        st.error("One or both images are invalid.")
        return [], [], None, None, []
        
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        st.warning("Could not compute descriptors for one or both images.")
        return kp1, kp2, des1, des2, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        st.warning(f"Error during matching: {e}")
        return kp1, kp2, des1, des2, []

    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    return kp1, kp2, des1, des2, good_matches

def visualize_matches(img1, img2, kp1, kp2, matches):
    """
    Draws the matched features between two images.
    """
    if img1 is None or img2 is None or len(matches) == 0:
        # Create a blank image with a message instead of failing
        height = max(img1.shape[0] if img1 is not None else 0, 
                    img2.shape[0] if img2 is not None else 0, 300)
        width = 600
        blank_img = np.ones((height, width), dtype=np.uint8) * 255
        text = "Not enough matches to visualize"
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - textsize[0]) // 2
        text_y = (height + textsize[1]) // 2
        return cv2.putText(cv2.cvtColor(blank_img, cv2.COLOR_GRAY2BGR), 
                          text, (text_x, text_y), font, 1, (0, 0, 255), 2)
    
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def find_homography(kp1, kp2, good_matches, method=cv2.RANSAC, ransac_thresh=5.0):
    """
    Finds the homography matrix between two sets of points.
    """
    if len(good_matches) < 4:
        return None, None, [], []
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    try:
        H, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
        return H, mask, src_pts, dst_pts
    except cv2.error as e:
        st.warning(f"Error computing homography: {e}")
        return None, None, src_pts, dst_pts

def draw_outline(img1, img2, H):
    """
    Draws the transformed outline of the first image onto the second image.
    """
    if H is None or img1 is None or img2 is None:
        # Return the second image as is or a placeholder
        if img2 is not None:
            return cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
        else:
            blank_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
            cv2.putText(blank_img, "No homography found", (50, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return blank_img
        
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    try:
        dst = cv2.perspectiveTransform(pts, H)
        
        # Ensure output is color
        if len(img2.shape) == 2:
            img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img_out = img2.copy()
        
        return cv2.polylines(img_out, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    except cv2.error as e:
        st.warning(f"Error drawing outline: {e}")
        return cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()

def compute_fundamental_matrix(pts1, pts2, method=cv2.FM_RANSAC, threshold=3.0):
    """
    Computes the fundamental matrix from two sets of corresponding points.
    """
    if len(pts1) < 8:
        return None, None
    
    try:
        F, mask = cv2.findFundamentalMat(pts1, pts2, method, threshold, 0.99)
        return F, mask
    except cv2.error as e:
        st.warning(f"Error computing fundamental matrix: {e}")
        return None, None

def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=10):
    """
    Draws epipolar lines on both images.
    """
    if F is None or F.shape != (3, 3):
        # Return the original images if fundamental matrix is invalid
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
        return img1_color, img2_color
        
    try:
        # Select a random subset of points for cleaner visualization
        idx = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)
        pts1_sub, pts2_sub = pts1[idx], pts2[idx]
        
        # Compute epipolar lines for points in the other image
        lines1 = cv2.computeCorrespondEpilines(pts2_sub.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(pts1_sub.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
        colors = np.random.randint(0, 255, (num_lines, 3)).tolist()

        # Draw lines on the first image
        h1, w1 = img1.shape[:2]
        for r, pt, color in zip(lines1, pts1_sub, colors):
            # Avoid division by zero
            if abs(r[1]) < 1e-10:
                continue
                
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [w1, -(r[2]+r[0]*w1)/r[1]])
            img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
            img1_color = cv2.circle(img1_color, tuple(map(int, pt)), 5, color, -1)

        # Draw lines on the second image
        h2, w2 = img2.shape[:2]
        for r, pt, color in zip(lines2, pts2_sub, colors):
            # Avoid division by zero
            if abs(r[1]) < 1e-10:
                continue
                
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [w2, -(r[2]+r[0]*w2)/r[1]])
            img2_color = cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
            img2_color = cv2.circle(img2_color, tuple(map(int, pt)), 5, color, -1)
            
        return img1_color, img2_color
    except Exception as e:
        st.warning(f"Error drawing epipolar lines: {e}")
        # Return original images on error
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
        return img1_color, img2_color

def interactive_augmented_imagery(img1, img2, blend_alpha=0.5):
    """
    Create an interactive augmented imagery visualization blending reference and query images.
    """
    if img1 is None or img2 is None:
        st.error("One or both images are invalid.")
        return None, None, None, 0, 0, None
        
    try:
        # Detect features
        kp1, kp2, des1, des2, good_matches = detect_and_match(img1, img2, nfeatures=1500)
        
        # If not enough matches, return early
        if len(good_matches) < 10:
            return None, None, None, len(good_matches), 0, None
            
        # Find homography
        H, mask, _, _ = find_homography(kp1, kp2, good_matches)
        
        if H is None:
            return None, None, None, len(good_matches), 0, None
            
        # Create visualization
        h2, w2 = img2.shape[:2]
        
        # Warp reference image
        if len(img1.shape) == 2:
            ref_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            ref_color = img1.copy()
            
        warped = cv2.warpPerspective(ref_color, H, (w2, h2))
        
        # Prepare query image
        if len(img2.shape) == 2:
            query_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            query_color = img2.copy()
            
        # Create blended result
        composite = cv2.addWeighted(query_color, 1-blend_alpha, warped, blend_alpha, 0)
        
        # Create matches visualization
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] > 0]
        matched_img = visualize_matches(img1, img2, kp1, kp2, inlier_matches[:15])
        
        # Create outline visualization
        outline_img = draw_outline(img1, img2, H)
        
        return composite, matched_img, outline_img, len(good_matches), np.sum(mask), H
        
    except Exception as e:
        st.warning(f"Error in augmented imagery: {e}")
        return None, None, None, 0, 0, None
