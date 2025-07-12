import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import draw_outline, draw_inliers

def load_images(ref_path, query_path):
    """
    Load reference and query images as grayscale
    
    Inputs:
        ref_path: path to reference image
        query_path: path to query image
        
    Returns:
        img1: reference image
        img2: query image
    """
    img1 = cv2.imread(ref_path, 0)
    img2 = cv2.imread(query_path, 0)
    
    if img1 is None:
        raise FileNotFoundError(f"Could not load reference image from {ref_path}")
    if img2 is None:
        raise FileNotFoundError(f"Could not load query image from {query_path}")
    
    return img1, img2

def detect_orb_features(img, nfeatures=500, scaleFactor=1.2, nlevels=8):
    """
    Detect ORB features in an image
    
    Inputs:
        img: input image
        nfeatures: max number of features to retain
        scaleFactor: pyramid decimation ratio
        nlevels: number of pyramid levels
        
    Returns:
        kp: keypoints
        des: descriptors
    """
    # Create ORB detector with parameters
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=scaleFactor,
        nlevels=nlevels
    )
    
    # Detect keypoints
    kp = orb.detect(img, None)
    
    # Compute descriptors
    kp, des = orb.compute(img, kp)
    
    return kp, des

def draw_keypoints(img, kp):
    """
    Draw keypoints on an image
    
    Inputs:
        img: input image
        kp: detected keypoints
        
    Returns:
        img_kp: image with keypoints drawn
    """
    img_kp = cv2.drawKeypoints(
        img, kp, None, 
        color=(0, 255, 0), 
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    
    return img_kp

def match_features(des1, des2, k=2, ratio_thresh=0.8):
    """
    Match features using Brute Force matcher with Hamming distance
    
    Inputs:
        des1: descriptors from first image
        des2: descriptors from second image
        k: number of nearest neighbors to find
        ratio_thresh: threshold for ratio test
        
    Returns:
        good_matches: filtered matches after ratio test
    """
    # Create BFMatcher object with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Match descriptors using KNN
    matches = bf.knnMatch(des1, des2, k=k)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    return good_matches

def draw_matches(img1, kp1, img2, kp2, matches, max_matches=100):
    """
    Draw matches between images
    
    Inputs:
        img1, img2: input images
        kp1, kp2: keypoints
        matches: matches to draw
        max_matches: maximum number of matches to draw
        
    Returns:
        img_matches: image with matches drawn
    """
    try:
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take only the best matches
        matches = matches[:max_matches]
        
        # Create a simple side-by-side image manually
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create a result image
        height = max(h1, h2)
        width = w1 + w2
        
        # Ensure we create a color image for drawing
        if len(img1.shape) == 2:  # Grayscale
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            result = np.zeros((height, width, 3), dtype=np.uint8)
        else:  # Already color
            img1_color = img1.copy()
            img2_color = img2.copy()
            result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Copy the images
        result[:h1, :w1] = img1_color
        result[:h2, w1:w1+w2] = img2_color
        
        # Draw matches manually
        for match in matches:
            # Get the keypoints from the match
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            
            # Get x,y coordinates
            x1, y1 = int(kp1[img1_idx].pt[0]), int(kp1[img1_idx].pt[1])
            x2, y2 = int(kp2[img2_idx].pt[0]), int(kp2[img2_idx].pt[1])
            
            # Draw circles
            cv2.circle(result, (x1, y1), 4, (0, 255, 0), 1)
            cv2.circle(result, (x2 + w1, y2), 4, (0, 255, 0), 1)
            
            # Connect with lines (random colors for better visibility)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(result, (x1, y1), (x2 + w1, y2), color, 1)
        
        return result
    except Exception as e:
        print(f"Warning: Could not draw matches: {e}")
        # Create a simple side-by-side image as fallback
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if len(img1.shape) == 2:  # Grayscale
            result = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
            result[:h1, :w1] = img1
            result[:h2, w1:w1+w2] = img2
        else:  # Color
            result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            result[:h1, :w1] = img1
            result[:h2, w1:w1+w2] = img2
            
        return result

def estimate_homography(kp1, kp2, good_matches, method=cv2.RANSAC, ransac_thresh=3.0):
    """
    Estimate homography between two images
    
    Inputs:
        kp1, kp2: keypoints from both images
        good_matches: filtered matches
        method: method to use for homography estimation
        ransac_thresh: threshold for RANSAC
        
    Returns:
        H: homography matrix
        mask: inlier mask
        src_pts: source points
        dst_pts: destination points
    """
    # Extract location of good matches
    if len(good_matches) < 4:
        raise ValueError(f"Not enough good matches: {len(good_matches)}")
    
    # Create lists to store the points
    src_pts = []
    dst_pts = []
    
    # Extract point coordinates from keypoints
    for match in good_matches:
        # Check if the attributes exist
        if not hasattr(match, 'queryIdx') or not hasattr(match, 'trainIdx'):
            raise ValueError("Match object does not have required attributes")
        
        # Get the keypoints for each match
        try:
            query_pt = kp1[match.queryIdx].pt
            train_pt = kp2[match.trainIdx].pt
            
            if query_pt is None or train_pt is None:
                continue
                
            # Add to lists
            src_pts.append([query_pt[0], query_pt[1]])
            dst_pts.append([train_pt[0], train_pt[1]])
        except Exception as e:
            continue  # Skip invalid points
    
    if len(src_pts) < 4 or len(dst_pts) < 4:
        raise ValueError(f"Not enough valid points for homography: {len(src_pts)}")
    
    # Convert to numpy arrays of the right shape and type
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
    
    # Ensure mask is a numpy array
    if mask is None:
        mask = np.zeros(len(src_pts), dtype=np.uint8)
    
    # Reshape for consistency with other functions
    src_pts = src_pts.reshape(-1, 1, 2)
    dst_pts = dst_pts.reshape(-1, 1, 2)
    
    return H, mask, src_pts, dst_pts

def run_orb_matching(ref_path, query_path, nfeatures=500, ratio_thresh=0.8, method=cv2.RANSAC, ransac_thresh=3.0):
    """
    Run the entire ORB matching pipeline
    
    Inputs:
        ref_path: path to reference image
        query_path: path to query image
        nfeatures: number of features for ORB
        ratio_thresh: threshold for ratio test
        method: method for homography estimation
        ransac_thresh: threshold for RANSAC
        
    Returns:
        results: dictionary containing all results
    """
    # Load images
    img1, img2 = load_images(ref_path, query_path)
    
    # Detect features
    kp1, des1 = detect_orb_features(img1, nfeatures=nfeatures)
    kp2, des2 = detect_orb_features(img2, nfeatures=nfeatures)
    
    # Draw keypoints
    img1_kp = draw_keypoints(img1, kp1)
    img2_kp = draw_keypoints(img2, kp2)
    
    # Match features
    good_matches = match_features(des1, des2, ratio_thresh=ratio_thresh)
    
    # Draw matches
    try:
        img_matches = draw_matches(img1, kp1, img2, kp2, good_matches)
    except Exception as e:
        print(f"Warning: Could not draw matches: {e}")
        # Create a simple side-by-side image as fallback
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if len(img1.shape) == 2:  # Grayscale
            img_matches = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
            img_matches[:h1, :w1] = img1
            img_matches[:h2, w1:w1+w2] = img2
        else:  # Color
            img_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            img_matches[:h1, :w1] = img1
            img_matches[:h2, w1:w1+w2] = img2
    
    # Estimate homography
    try:
        H, mask, src_pts, dst_pts = estimate_homography(
            kp1, kp2, good_matches, 
            method=method, 
            ransac_thresh=ransac_thresh
        )
        
        if H is None:
            raise ValueError("Homography matrix is None")
        
        # Draw outline
        img_outline = draw_outline(img1, img2, H)
        
        # Draw inliers
        img_inliers = draw_inliers(img1, kp1, img2, kp2, good_matches, mask)
        
        # Calculate inliers count
        inlier_count = np.sum(mask) if mask is not None else 0
        
        # Success
        success = True
    except Exception as e:
        print(f"Homography estimation failed: {e}")
        H = None
        mask = None
        img_outline = img2.copy()
        img_inliers = None
        inlier_count = 0
        success = False
    
    # Collect results
    results = {
        'ref_img': img1,
        'query_img': img2,
        'ref_keypoints': kp1,
        'query_keypoints': kp2,
        'ref_descriptors': des1,
        'query_descriptors': des2,
        'ref_kp_img': img1_kp,
        'query_kp_img': img2_kp,
        'good_matches': good_matches,
        'matches_img': img_matches,
        'homography': H,
        'mask': mask,
        'outline_img': img_outline,
        'inliers_img': img_inliers,
        'inlier_count': inlier_count,
        'success': success,
        'num_matches': len(good_matches)
    }
    
    return results

def compare_parameters(ref_path, query_path):
    """
    Compare different parameter settings and their effect on matching
    
    Inputs:
        ref_path: path to reference image
        query_path: path to query image
        
    Returns:
        results: dictionary with results for different parameter settings
    """
    # Parameter variations to test
    nfeatures_list = [500, 1000, 2000]
    ratio_thresh_list = [0.6, 0.7, 0.8, 0.9]
    
    # Store results
    results = {}
    
    # Test different feature counts
    for nf in nfeatures_list:
        key = f"nfeatures_{nf}"
        results[key] = run_orb_matching(ref_path, query_path, nfeatures=nf)
    
    # Test different ratio thresholds
    for rt in ratio_thresh_list:
        key = f"ratio_{rt}"
        results[key] = run_orb_matching(ref_path, query_path, ratio_thresh=rt)
    
    # Test different homography methods
    results["method_0"] = run_orb_matching(ref_path, query_path, method=0)  # Least squares
    results["method_RANSAC"] = run_orb_matching(ref_path, query_path, method=cv2.RANSAC)
    
    # Test different RANSAC thresholds
    for thresh in [1.0, 3.0, 5.0]:
        key = f"ransac_thresh_{thresh}"
        results[key] = run_orb_matching(ref_path, query_path, ransac_thresh=thresh)
    
    return results

if __name__ == "__main__":
    # Example usage
    ref_path = "A2_smvs/book_covers/Reference/001.jpg"
    query_path = "A2_smvs/book_covers/Query/001.jpg"
    
    # Run ORB matching
    results = run_orb_matching(ref_path, query_path)
    
    # Display results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(results['ref_kp_img'])
    plt.title("Reference Image with Keypoints")
    
    plt.subplot(2, 2, 2)
    plt.imshow(results['query_kp_img'])
    plt.title("Query Image with Keypoints")
    
    plt.subplot(2, 2, 3)
    plt.imshow(results['matches_img'])
    plt.title(f"Matches (Count: {results['num_matches']})")
    
    plt.subplot(2, 2, 4)
    plt.imshow(results['outline_img'])
    plt.title("Reference Outline on Query")
    
    plt.tight_layout()
    plt.show()