import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from q1 import run_orb_matching, compare_parameters

def analyze_image_pair(ref_path, query_path, title, description):
    """
    Analyze a specific image pair and display detailed results
    
    Inputs:
        ref_path: path to reference image
        query_path: path to query image
        title: title for the analysis
        description: description of the test case
        
    Returns:
        results: dictionary with results
    """
    print(f"=== {title} ===")
    print(f"Description: {description}")
    print(f"Reference: {ref_path}")
    print(f"Query: {query_path}")
    
    # Run matching with default parameters
    results = run_orb_matching(ref_path, query_path)
    
    # Print statistics
    print(f"Number of keypoints in reference: {len(results['ref_keypoints'])}")
    print(f"Number of keypoints in query: {len(results['query_keypoints'])}")
    print(f"Number of good matches: {len(results['good_matches'])}")
    
    if results['success']:
        print(f"Homography estimation successful")
        print(f"Number of inliers: {results['inlier_count']}")
        inlier_ratio = results['inlier_count'] / len(results['good_matches'])
        print(f"Inlier ratio: {inlier_ratio:.2f}")
    else:
        print(f"Homography estimation failed")
    
    # Display results
    try:
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(results['ref_img'], cmap='gray')
        plt.title("Reference Image")
        
        plt.subplot(2, 3, 2)
        plt.imshow(results['query_img'], cmap='gray')
        plt.title("Query Image")
        
        plt.subplot(2, 3, 3)
        plt.imshow(results['matches_img'])
        plt.title(f"Matches (Count: {results['num_matches']})")
        
        plt.subplot(2, 3, 4)
        plt.imshow(results['ref_kp_img'])
        plt.title(f"Reference Keypoints ({len(results['ref_keypoints'])})")
        
        plt.subplot(2, 3, 5)
        plt.imshow(results['query_kp_img'])
        plt.title(f"Query Keypoints ({len(results['query_keypoints'])})")
        
        plt.subplot(2, 3, 6)
        if results['inliers_img'] is not None:
            plt.imshow(results['inliers_img'])
            plt.title(f"Inliers ({results['inlier_count']})")
        else:
            plt.imshow(results['query_img'], cmap='gray')
            plt.title("Homography Failed")
        
        plt.tight_layout()
        plt.show()
        
        # Show outline separately
        if results['success']:
            plt.figure(figsize=(10, 8))
            plt.imshow(results['outline_img'])
            plt.title("Reference Outline on Query")
            plt.show()
    except Exception as e:
        print(f"Could not display results: {e}")
    
    return results

def try_improve_matching(ref_path, query_path, title, description, 
                        nfeatures=2000, 
                        ratio_thresh=0.75,
                        method=cv2.RANSAC,
                        ransac_thresh=3.0):
    """
    Try to improve matching for a difficult case
    
    Inputs:
        ref_path: path to reference image
        query_path: path to query image
        title: title for the analysis
        description: description of why improvements are needed
        nfeatures: number of features to detect
        ratio_thresh: threshold for ratio test
        method: homography estimation method
        ransac_thresh: threshold for RANSAC
        
    Returns:
        results: dictionary with results
    """
    print(f"=== IMPROVEMENT ATTEMPT: {title} ===")
    print(f"Description: {description}")
    print(f"Reference: {ref_path}")
    print(f"Query: {query_path}")
    print(f"Parameters: nfeatures={nfeatures}, ratio_thresh={ratio_thresh}, ransac_thresh={ransac_thresh}")
    
    # Run matching with custom parameters
    results = run_orb_matching(
        ref_path, query_path,
        nfeatures=nfeatures,
        ratio_thresh=ratio_thresh,
        method=method,
        ransac_thresh=ransac_thresh
    )
    
    # Print statistics
    print(f"Number of keypoints in reference: {len(results['ref_keypoints'])}")
    print(f"Number of keypoints in query: {len(results['query_keypoints'])}")
    print(f"Number of good matches: {len(results['good_matches'])}")
    
    if results['success']:
        print(f"Homography estimation successful")
        print(f"Number of inliers: {results['inlier_count']}")
        inlier_ratio = results['inlier_count'] / len(results['good_matches'])
        print(f"Inlier ratio: {inlier_ratio:.2f}")
    else:
        print(f"Homography estimation failed")
    
    # Display results
    try:
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(results['matches_img'])
        plt.title(f"Matches (Count: {results['num_matches']})")
        
        plt.subplot(2, 2, 2)
        if results['inliers_img'] is not None:
            plt.imshow(results['inliers_img'])
            plt.title(f"Inliers ({results['inlier_count']})")
        else:
            plt.imshow(results['query_img'], cmap='gray')
            plt.title("Homography Failed")
        
        plt.subplot(2, 2, 3)
        plt.imshow(results['ref_kp_img'])
        plt.title(f"Reference Keypoints ({len(results['ref_keypoints'])})")
        
        plt.subplot(2, 2, 4)
        plt.imshow(results['query_kp_img'])
        plt.title(f"Query Keypoints ({len(results['query_keypoints'])})")
        
        plt.tight_layout()
        plt.show()
        
        # Show outline separately
        if results['success']:
            plt.figure(figsize=(10, 8))
            plt.imshow(results['outline_img'])
            plt.title("Reference Outline on Query")
            plt.show()
    except Exception as e:
        print(f"Could not display results: {e}")
    
    return results

def test_multiple_categories():
    """
    Test image pairs from different categories (books, paintings, landmarks)
    to compare difficulty levels and find success/failure cases
    
    Returns:
        results: dictionary with results for each category
    """
    categories = [
        ('book_covers', '001'),  # Expected to be easiest
        ('museum_paintings', '001'),  # Medium difficulty
        ('landmarks', '001')  # Expected to be hardest
    ]
    
    results = {}
    
    for category, image_id in categories:
        ref_path = f"A2_smvs/{category}/Reference/{image_id}.jpg"
        query_path = f"A2_smvs/{category}/Query/{image_id}.jpg"
        
        if os.path.exists(ref_path) and os.path.exists(query_path):
            title = f"{category.replace('_', ' ').title()} {image_id}"
            description = f"Testing matching difficulty for {category}"
            
            print(f"\n\n{'='*50}")
            print(f"CATEGORY: {title}")
            print(f"{'='*50}\n")
            
            results[category] = analyze_image_pair(ref_path, query_path, title, description)
        else:
            print(f"Files not found for category {category}, image {image_id}")
    
    return results

def find_failure_cases():
    """
    Search for failure cases in each category
    
    Returns:
        failure_cases: list of paths to failure cases
    """
    categories = ['book_covers', 'museum_paintings', 'landmarks']
    failure_cases = []
    success_cases = []
    
    for category in categories:
        ref_dir = f"A2_smvs/{category}/Reference"
        query_dir = f"A2_smvs/{category}/Query"
        
        if not os.path.exists(ref_dir) or not os.path.exists(query_dir):
            print(f"Directory not found for category {category}")
            continue
        
        # Get all image IDs
        image_ids = [f.split('.')[0] for f in os.listdir(ref_dir) if f.endswith('.jpg')]
        
        # Test a subset (to save time)
        for image_id in image_ids[:5]:  # Test first 5 images in each category
            ref_path = f"{ref_dir}/{image_id}.jpg"
            query_path = f"{query_dir}/{image_id}.jpg"
            
            if os.path.exists(ref_path) and os.path.exists(query_path):
                # Run matching with default parameters
                try:
                    results = run_orb_matching(ref_path, query_path)
                    
                    if results['success'] and results['inlier_count'] >= 10:
                        success_cases.append((category, image_id, ref_path, query_path, results['inlier_count']))
                    else:
                        failure_cases.append((category, image_id, ref_path, query_path))
                except Exception as e:
                    print(f"Error processing {category}/{image_id}: {e}")
                    failure_cases.append((category, image_id, ref_path, query_path))
    
    # Print results
    print("\n=== SUCCESS CASES ===")
    for category, image_id, ref_path, query_path, inlier_count in success_cases:
        print(f"{category}/{image_id}: {inlier_count} inliers")
    
    print("\n=== FAILURE CASES ===")
    for category, image_id, ref_path, query_path in failure_cases:
        print(f"{category}/{image_id}")
    
    return success_cases, failure_cases

if __name__ == "__main__":
    # Test images from different categories to find success and failure cases
    success_cases, failure_cases = find_failure_cases()
    
    # Analyze one success case in detail
    if success_cases:
        category, image_id, ref_path, query_path, _ = success_cases[0]
        analyze_image_pair(
            ref_path, query_path, 
            f"Success Case: {category}/{image_id}", 
            "This is a case where default parameters work well"
        )
    
    # Analyze one failure case in detail
    if failure_cases:
        category, image_id, ref_path, query_path = failure_cases[0]
        analyze_image_pair(
            ref_path, query_path, 
            f"Failure Case: {category}/{image_id}", 
            "This is a case where default parameters fail"
        )
        
        # Try to improve the failure case
        try_improve_matching(
            ref_path, query_path,
            f"Improved: {category}/{image_id}",
            "Attempting to improve matching with adjusted parameters",
            nfeatures=3000,
            ratio_thresh=0.85,
            ransac_thresh=5.0
        ) 