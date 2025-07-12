import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from q1 import load_images, detect_orb_features, match_features, draw_matches, estimate_homography, run_orb_matching
from helper_functions import draw_outline, draw_inliers

def test_basic_functionality():
    """Test the basic functionality of the q1.py implementation"""
    
    # Define paths
    ref_path = "A2_smvs/book_covers/Reference/001.jpg"
    query_path = "A2_smvs/book_covers/Query/001.jpg"
    
    # Check if files exist
    if not os.path.exists(ref_path) or not os.path.exists(query_path):
        print(f"Test files not found. Please make sure the A2_smvs dataset is available.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for: {ref_path} and {query_path}")
        return False
    
    try:
        # Test loading images
        print("Testing image loading...")
        img1, img2 = load_images(ref_path, query_path)
        assert img1 is not None and img2 is not None
        print("[PASS] Image loading successful")
        
        # Test feature detection
        print("Testing feature detection...")
        kp1, des1 = detect_orb_features(img1)
        kp2, des2 = detect_orb_features(img2)
        assert len(kp1) > 0 and des1 is not None
        assert len(kp2) > 0 and des2 is not None
        print(f"[PASS] Feature detection successful ({len(kp1)} features in reference, {len(kp2)} in query)")
        
        # Test feature matching
        print("Testing feature matching...")
        good_matches = match_features(des1, des2)
        assert len(good_matches) > 0
        print(f"[PASS] Feature matching successful ({len(good_matches)} good matches)")
        
        # Test homography estimation
        print("Testing homography estimation...")
        if len(good_matches) >= 4:
            H, mask, src_pts, dst_pts = estimate_homography(kp1, kp2, good_matches)
            assert H is not None and mask is not None
            inliers = np.sum(mask)
            print(f"[PASS] Homography estimation successful ({inliers} inliers)")
            
            # Test outline drawing
            print("Testing outline drawing...")
            outline_img = draw_outline(img1, img2, H)
            assert outline_img is not None
            print("[PASS] Outline drawing successful")
            
            # Test inliers drawing
            print("Testing inliers drawing...")
            inliers_img = draw_inliers(img1, kp1, img2, kp2, good_matches, mask)
            assert inliers_img is not None
            print("[PASS] Inliers drawing successful")
        else:
            print("[WARNING] Not enough good matches for homography estimation")
        
        # Test full pipeline
        print("Testing full pipeline...")
        results = run_orb_matching(ref_path, query_path)
        assert results is not None
        print("[PASS] Full pipeline successful")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_variations():
    """Test different parameter variations to see their effect"""
    
    # Define paths
    ref_path = "A2_smvs/book_covers/Reference/001.jpg"
    query_path = "A2_smvs/book_covers/Query/001.jpg"
    
    # Check if files exist
    if not os.path.exists(ref_path) or not os.path.exists(query_path):
        print(f"Test files not found. Please make sure the A2_smvs dataset is available.")
        return
    
    # Test different numbers of features
    print("\nTesting different feature counts...")
    for nfeatures in [100, 500, 2000]:
        results = run_orb_matching(ref_path, query_path, nfeatures=nfeatures)
        print(f"nfeatures={nfeatures}: {len(results['good_matches'])} good matches, {results['inlier_count']} inliers")
    
    # Test different ratio thresholds
    print("\nTesting different ratio thresholds...")
    for ratio_thresh in [0.6, 0.7, 0.8, 0.9]:
        results = run_orb_matching(ref_path, query_path, ratio_thresh=ratio_thresh)
        print(f"ratio_thresh={ratio_thresh}: {len(results['good_matches'])} good matches, {results['inlier_count']} inliers")
    
    # Test different homography methods
    print("\nTesting different homography methods...")
    for method, name in [(0, "Least Squares"), (cv2.RANSAC, "RANSAC")]:
        results = run_orb_matching(ref_path, query_path, method=method)
        print(f"method={name}: {len(results['good_matches'])} good matches, {results['inlier_count']} inliers")
    
    # Test different RANSAC thresholds
    print("\nTesting different RANSAC thresholds...")
    for ransac_thresh in [1.0, 3.0, 5.0]:
        results = run_orb_matching(ref_path, query_path, ransac_thresh=ransac_thresh)
        print(f"ransac_thresh={ransac_thresh}: {len(results['good_matches'])} good matches, {results['inlier_count']} inliers")

def visual_test():
    """Run a visual test to display the results"""
    
    # Define paths
    ref_path = "A2_smvs/book_covers/Reference/001.jpg"
    query_path = "A2_smvs/book_covers/Query/001.jpg"
    
    # Check if files exist
    if not os.path.exists(ref_path) or not os.path.exists(query_path):
        print(f"Test files not found. Please make sure the A2_smvs dataset is available.")
        return
    
    # Run matching
    results = run_orb_matching(ref_path, query_path)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(results['ref_img'], cmap='gray')
    plt.title("Reference Image")
    
    plt.subplot(2, 2, 2)
    plt.imshow(results['query_img'], cmap='gray')
    plt.title("Query Image")
    
    plt.subplot(2, 2, 3)
    plt.imshow(results['matches_img'])
    plt.title(f"Matches (Count: {results['num_matches']})")
    
    plt.subplot(2, 2, 4)
    plt.imshow(results['outline_img'])
    plt.title(f"Outline (Inliers: {results['inlier_count']})")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Testing q1.py Implementation ===")
    
    # Basic functionality test
    success = test_basic_functionality()
    
    if success:
        print("\n=== All basic tests passed! ===")
        
        # Parameter variation tests
        test_parameter_variations()
        
        # Visual test
        visual_test()
    else:
        print("\n[FAIL] Some tests failed. Please fix the errors before proceeding.") 