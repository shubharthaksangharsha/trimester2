import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import draw_outline, draw_inliers
import glob
from collections import defaultdict

def load_image(image_path):
    """Load an image from file path"""
    img = cv2.imread(image_path, 0)  # Load as grayscale
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return img

def extract_features(img, nfeatures=2000):
    """Extract ORB features from an image"""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def match_features(des1, des2, ratio_thresh=0.8):
    """Match features between two images using ratio test"""
    if des1 is None or des2 is None:
        return []
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    try:
        # Match descriptors using KNN
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        return good_matches
    except Exception as e:
        print(f"Error in matching: {e}")
        return []

def estimate_homography(kp1, kp2, good_matches, ransac_thresh=3.0):
    """Estimate homography and count inliers"""
    if len(good_matches) < 4:
        return None, 0, None
    
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    
    # Count inliers
    inlier_count = np.sum(mask) if mask is not None else 0
    
    return H, inlier_count, mask

def load_dataset(base_dir, categories=None, max_images=None):
    """
    Load reference and query images from dataset
    
    Inputs:
        base_dir: Base directory containing the dataset
        categories: List of categories to load (e.g., ['book_covers', 'landmarks'])
        max_images: Maximum number of images to load per category (for testing)
        
    Returns:
        ref_images: Dictionary of {image_id: {'path': path, 'image': image, 'category': category}}
        query_images: Dictionary of {image_id: {'path': path, 'image': image, 'category': category}}
        
    The image_id format is 'category_imagenumber' (e.g., 'book_covers_001')
    """
    if categories is None:
        categories = ['book_covers', 'museum_paintings', 'landmarks']
    
    ref_images = {}
    query_images = {}
    
    for category in categories:
        ref_dir = os.path.join(base_dir, category, 'Reference')
        query_dir = os.path.join(base_dir, category, 'Query')
        
        if not os.path.exists(ref_dir) or not os.path.exists(query_dir):
            print(f"Warning: Directory not found for {category}")
            continue
        
        # Load reference images
        image_count = 0
        for img_path in glob.glob(os.path.join(ref_dir, '*.jpg')):
            if max_images is not None and image_count >= max_images:
                break
                
            img_name = os.path.basename(img_path).split('.')[0]
            image_id = f"{category}_{img_name}"
            try:
                img = load_image(img_path)
                ref_images[image_id] = {
                    'path': img_path,
                    'image': img,
                    'category': category
                }
                image_count += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        # Reset counter for query images
        image_count = 0
        # Load query images
        for img_path in glob.glob(os.path.join(query_dir, '*.jpg')):
            if max_images is not None and image_count >= max_images:
                break
                
            img_name = os.path.basename(img_path).split('.')[0]
            image_id = f"{category}_{img_name}"
            try:
                img = load_image(img_path)
                query_images[image_id] = {
                    'path': img_path,
                    'image': img,
                    'category': category,
                    'ground_truth': image_id  # Same ID means it's the same object
                }
                image_count += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return ref_images, query_images


def precompute_features(images_dict, nfeatures=2000):
    """
    Precompute features for all images in a dictionary
    
    Inputs:
        images_dict: Dictionary of images as returned by load_dataset
        nfeatures: Number of features to extract
        
    Returns:
        Dictionary with added 'keypoints' and 'descriptors' for each image
    """
    result = {}
    
    for image_id, img_data in images_dict.items():
        try:
            kp, des = extract_features(img_data['image'], nfeatures=nfeatures)
            
            # Add features to the dictionary
            result[image_id] = {
                **img_data,
                'keypoints': kp,
                'descriptors': des
            }
        except Exception as e:
            print(f"Error extracting features for {image_id}: {e}")
            # Still add the image to the result without features
            result[image_id] = {
                **img_data,
                'keypoints': None,
                'descriptors': None
            }
    
    return result

def identify_query_image(query_image_data, ref_images, ratio_thresh=0.8, ransac_thresh=3.0, inlier_threshold=10):
    """
    Identify a query image by matching it against all reference images
    
    Inputs:
        query_image_data: Dictionary with query image data including keypoints and descriptors
        ref_images: Dictionary of reference images with precomputed features
        ratio_thresh: Threshold for ratio test
        ransac_thresh: Threshold for RANSAC
        inlier_threshold: Minimum number of inliers to consider a match valid
        
    Returns:
        match_scores: Dictionary of {ref_id: inlier_count}
        best_match: ID of the best matching reference image or None if no good match
        matched_mask: Inlier mask for the best match
        matched_H: Homography for the best match
    """
    query_kp = query_image_data['keypoints']
    query_des = query_image_data['descriptors']
    
    if query_kp is None or query_des is None:
        return {}, None, None, None
    
    match_scores = {}
    best_match = None
    best_score = 0
    matched_mask = None
    matched_H = None
    
    for ref_id, ref_data in ref_images.items():
        ref_kp = ref_data['keypoints']
        ref_des = ref_data['descriptors']
        
        if ref_kp is None or ref_des is None:
            continue
        
        # Match features
        good_matches = match_features(query_des, ref_des, ratio_thresh=ratio_thresh)
        
        # Estimate homography and count inliers
        if len(good_matches) >= 4:
            H, inlier_count, mask = estimate_homography(query_kp, ref_kp, good_matches, ransac_thresh=ransac_thresh)
            match_scores[ref_id] = inlier_count
            
            if inlier_count > best_score:
                best_score = inlier_count
                best_match = ref_id
                matched_mask = mask
                matched_H = H
        else:
            match_scores[ref_id] = 0
    
    # If best score is below threshold, consider it as no match
    if best_score < inlier_threshold:
        best_match = None
    
    return match_scores, best_match, matched_mask, matched_H

def evaluate_matching(query_images, ref_images, ratio_thresh=0.8, ransac_thresh=3.0, inlier_threshold=10):
    """
    Evaluate the matching performance on a set of query images
    
    Inputs:
        query_images: Dictionary of query images with precomputed features
        ref_images: Dictionary of reference images with precomputed features
        ratio_thresh: Threshold for ratio test
        ransac_thresh: Threshold for RANSAC
        inlier_threshold: Minimum inliers to consider a match valid
        
    Returns:
        results: Dictionary of results for each query
        accuracy: Overall accuracy (percentage of correct matches)
        correct_matches: Number of correct matches
        total_queries: Total number of queries
    """
    results = {}
    correct_matches = 0
    total_queries = 0
    
    for query_id, query_data in query_images.items():
        ground_truth = query_data['ground_truth']
        
        # Identify the query image
        match_scores, best_match, mask, H = identify_query_image(
            query_data, ref_images, 
            ratio_thresh=ratio_thresh, 
            ransac_thresh=ransac_thresh, 
            inlier_threshold=inlier_threshold
        )
        
        # Store results
        results[query_id] = {
            'match_scores': match_scores,
            'best_match': best_match,
            'ground_truth': ground_truth,
            'correct': best_match == ground_truth,
            'homography': H,
            'mask': mask
        }
        
        # Calculate accuracy
        if best_match == ground_truth:
            correct_matches += 1
        
        total_queries += 1
    
    accuracy = correct_matches / total_queries if total_queries > 0 else 0
    
    return results, accuracy, correct_matches, total_queries

def evaluate_top_k(results, k=3):
    """Evaluate top-k accuracy from the results"""
    correct_in_top_k = 0
    total_queries = len(results)
    
    for query_id, result in results.items():
        ground_truth = result['ground_truth']
        match_scores = result['match_scores']
        
        # Sort matches by score in descending order
        sorted_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_matches = [match_id for match_id, score in sorted_matches[:k]]
        
        if ground_truth in top_k_matches:
            correct_in_top_k += 1
    
    top_k_accuracy = correct_in_top_k / total_queries if total_queries > 0 else 0
    return top_k_accuracy, correct_in_top_k, total_queries

def visualize_matches(query_data, ref_data, best_match, mask=None, H=None):
    """
    Visualize the matching results
    
    Inputs:
        query_data: Dictionary with query image data
        ref_data: Dictionary with reference image data for the best match
        best_match: ID of the best matching reference image
        mask: Inlier mask for the match
        H: Homography matrix
        
    Returns:
        fig: Matplotlib figure with visualization
    """
    query_img = query_data['image']
    query_kp = query_data['keypoints']
    ref_img = ref_data['image']
    ref_kp = ref_data['keypoints']
    
    # Create a figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Display query image
    axes[0].imshow(query_img, cmap='gray')
    axes[0].set_title(f"Query Image: {os.path.basename(query_data['path'])}")
    axes[0].axis('off')
    
    # Display reference image
    axes[1].imshow(ref_img, cmap='gray')
    axes[1].set_title(f"Best Match: {os.path.basename(ref_data['path'])}")
    axes[1].axis('off')
    
    # Display the query image with the outline of the reference image
    if H is not None:
        try:
            # Draw outline of reference on query image
            h, w = ref_img.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, H)
            
            # Create a copy of query image and draw outline
            img_with_outline = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
            img_with_outline = cv2.polylines(img_with_outline, [np.int32(dst)], True, (0,255,0), 3)
            
            axes[2].imshow(cv2.cvtColor(img_with_outline, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Reference Outline on Query")
        except Exception as e:
            print(f"Error drawing outline: {e}")
            axes[2].imshow(query_img, cmap='gray')
            axes[2].set_title("Error drawing outline")
    else:
        axes[2].imshow(query_img, cmap='gray')
        axes[2].set_title("No valid homography")
    
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def evaluate_category_performance(results):
    """
    Analyze performance by category
    
    Returns:
        category_stats: Dictionary with stats per category
    """
    category_results = defaultdict(list)
    
    for query_id, result in results.items():
        category = query_id.split('_')[0]
        category_results[category].append(result['correct'])
    
    category_stats = {}
    for category, correct_list in category_results.items():
        total = len(correct_list)
        correct = sum(correct_list)
        accuracy = correct / total if total > 0 else 0
        
        category_stats[category] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    return category_stats

def run_image_search(base_dir="A2_smvs", 
                    categories=None, 
                    nfeatures=2000, 
                    ratio_thresh=0.8, 
                    ransac_thresh=3.0, 
                    inlier_threshold=10,
                    max_images=10,
                    include_extra_queries=False):
    """
    Run the complete image search pipeline
    
    Inputs:
        base_dir: Base directory for dataset
        categories: List of categories to include
        nfeatures: Number of features for ORB
        ratio_thresh: Threshold for ratio test
        ransac_thresh: Threshold for RANSAC
        inlier_threshold: Minimum inliers to consider a match valid
        include_extra_queries: Whether to include extra query images
        
    Returns:
        Dictionary with all results and statistics
    """
    # Load dataset
    ref_images, query_images = load_dataset(base_dir, categories)
    
    print(f"Loaded {len(ref_images)} reference images and {len(query_images)} query images")
    
    # Precompute features
    print("Precomputing features for reference images...")
    ref_images_with_features = precompute_features(ref_images, nfeatures=nfeatures)
    
    print("Precomputing features for query images...")
    query_images_with_features = precompute_features(query_images, nfeatures=nfeatures)
    
    # Evaluate matching
    print("Evaluating matching performance...")
    results, accuracy, correct_matches, total_queries = evaluate_matching(
        query_images_with_features, ref_images_with_features,
        ratio_thresh=ratio_thresh, ransac_thresh=ransac_thresh, inlier_threshold=inlier_threshold
    )
    
    # Evaluate top-k accuracy
    top_k_accuracy, correct_in_top_k, _ = evaluate_top_k(results, k=3)
    
    # Analyze performance by category
    category_stats = evaluate_category_performance(results)
    
    # Prepare final results
    final_results = {
        'results': results,
        'accuracy': accuracy,
        'correct_matches': correct_matches,
        'total_queries': total_queries,
        'top_k_accuracy': top_k_accuracy,
        'correct_in_top_k': correct_in_top_k,
        'category_stats': category_stats,
        'ref_images': ref_images_with_features,
        'query_images': query_images_with_features,
        'params': {
            'nfeatures': nfeatures,
            'ratio_thresh': ratio_thresh,
            'ransac_thresh': ransac_thresh,
            'inlier_threshold': inlier_threshold
        }
    }
    
    return final_results

def visualize_results(search_results, num_examples=3, include_failures=True):
    """
    Visualize some example results
    
    Inputs:
        search_results: Results from run_image_search
        num_examples: Number of examples to visualize
        include_failures: Whether to include failure cases
        
    Returns:
        None (displays plots)
    """
    results = search_results['results']
    ref_images = search_results['ref_images']
    query_images = search_results['query_images']
    
    # Separate successes and failures
    successes = [qid for qid, res in results.items() if res['correct']]
    failures = [qid for qid, res in results.items() if not res['correct']]
    
    # Visualize some successful matches
    if successes:
        print(f"\n=== Successful Matches ({len(successes)}/{len(results)}) ===")
        for i, query_id in enumerate(successes[:num_examples]):
            result = results[query_id]
            best_match = result['best_match']
            
            if best_match and best_match in ref_images:
                query_data = query_images[query_id]
                ref_data = ref_images[best_match]
                
                print(f"\nQuery: {query_id}")
                print(f"Matched with: {best_match}")
                print(f"Match Score: {result['match_scores'][best_match]} inliers")
                
                # Visualize the match
                fig = visualize_matches(query_data, ref_data, best_match, result['mask'], result['homography'])
                plt.show()
    
    # Visualize some failures
    if failures and include_failures:
        print(f"\n=== Failed Matches ({len(failures)}/{len(results)}) ===")
        for i, query_id in enumerate(failures[:num_examples]):
            result = results[query_id]
            best_match = result['best_match']
            ground_truth = result['ground_truth']
            
            print(f"\nQuery: {query_id}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Best Match: {best_match}")
            
            if best_match and best_match in ref_images:
                # Show the incorrect match
                query_data = query_images[query_id]
                ref_data = ref_images[best_match]
                
                # Get match scores
                match_scores = result['match_scores']
                sorted_scores = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
                
                print("Top 3 matches:")
                for match_id, score in sorted_scores[:3]:
                    print(f"- {match_id}: {score} inliers")
                
                # Check ground truth rank
                ground_truth_rank = None
                for rank, (match_id, score) in enumerate(sorted_scores):
                    if match_id == ground_truth:
                        ground_truth_rank = rank + 1
                        break
                
                if ground_truth_rank:
                    print(f"Ground truth rank: {ground_truth_rank}")
                else:
                    print("Ground truth not found in matches")
                
                # Visualize the incorrect match
                fig = visualize_matches(query_data, ref_data, best_match, result['mask'], result['homography'])
                plt.show()

def analyze_parameters(base_dir="A2_smvs", categories=None):
    """
    Analyze the effect of different parameters on matching performance
    
    Returns:
        Dictionary with parameter analysis results
    """
    if categories is None:
        categories = ['book_covers']  # Use a smaller dataset for parameter testing
    
    print("Starting parameter analysis...")
    
    # Test different feature counts
    nfeatures_list = [500, 1000, 2000, 3000]
    nfeatures_results = {}
    
    for nf in nfeatures_list:
        print(f"Testing nfeatures={nf}")
        results = run_image_search(base_dir, categories, nfeatures=nf)
        nfeatures_results[nf] = {
            'accuracy': results['accuracy'],
            'top_k_accuracy': results['top_k_accuracy']
        }
    
    # Test different ratio thresholds
    ratio_list = [0.6, 0.7, 0.8, 0.9]
    ratio_results = {}
    
    for rt in ratio_list:
        print(f"Testing ratio_thresh={rt}")
        results = run_image_search(base_dir, categories, ratio_thresh=rt)
        ratio_results[rt] = {
            'accuracy': results['accuracy'],
            'top_k_accuracy': results['top_k_accuracy']
        }
    
    # Test different RANSAC thresholds
    ransac_list = [1.0, 2.0, 3.0, 5.0]
    ransac_results = {}
    
    for rs in ransac_list:
        print(f"Testing ransac_thresh={rs}")
        results = run_image_search(base_dir, categories, ransac_thresh=rs)
        ransac_results[rs] = {
            'accuracy': results['accuracy'],
            'top_k_accuracy': results['top_k_accuracy']
        }
    
    # Test different inlier thresholds
    inlier_list = [5, 10, 15, 20]
    inlier_results = {}
    
    for it in inlier_list:
        print(f"Testing inlier_threshold={it}")
        results = run_image_search(base_dir, categories, inlier_threshold=it)
        inlier_results[it] = {
            'accuracy': results['accuracy'],
            'top_k_accuracy': results['top_k_accuracy']
        }
    
    # Compile all results
    parameter_analysis = {
        'nfeatures': nfeatures_results,
        'ratio_thresh': ratio_results,
        'ransac_thresh': ransac_results,
        'inlier_threshold': inlier_results
    }
    
    return parameter_analysis

def plot_parameter_analysis(parameter_analysis):
    """Plot the results of parameter analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot nfeatures results
    nf_values = list(parameter_analysis['nfeatures'].keys())
    nf_accuracy = [parameter_analysis['nfeatures'][nf]['accuracy'] for nf in nf_values]
    nf_top_k = [parameter_analysis['nfeatures'][nf]['top_k_accuracy'] for nf in nf_values]
    
    axes[0, 0].plot(nf_values, nf_accuracy, 'o-', label='Accuracy')
    axes[0, 0].plot(nf_values, nf_top_k, 's--', label='Top-3 Accuracy')
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Effect of Feature Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot ratio threshold results
    rt_values = list(parameter_analysis['ratio_thresh'].keys())
    rt_accuracy = [parameter_analysis['ratio_thresh'][rt]['accuracy'] for rt in rt_values]
    rt_top_k = [parameter_analysis['ratio_thresh'][rt]['top_k_accuracy'] for rt in rt_values]
    
    axes[0, 1].plot(rt_values, rt_accuracy, 'o-', label='Accuracy')
    axes[0, 1].plot(rt_values, rt_top_k, 's--', label='Top-3 Accuracy')
    axes[0, 1].set_xlabel('Ratio Threshold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Effect of Ratio Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot RANSAC threshold results
    rs_values = list(parameter_analysis['ransac_thresh'].keys())
    rs_accuracy = [parameter_analysis['ransac_thresh'][rs]['accuracy'] for rs in rs_values]
    rs_top_k = [parameter_analysis['ransac_thresh'][rs]['top_k_accuracy'] for rs in rs_values]
    
    axes[1, 0].plot(rs_values, rs_accuracy, 'o-', label='Accuracy')
    axes[1, 0].plot(rs_values, rs_top_k, 's--', label='Top-3 Accuracy')
    axes[1, 0].set_xlabel('RANSAC Threshold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Effect of RANSAC Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot inlier threshold results
    it_values = list(parameter_analysis['inlier_threshold'].keys())
    it_accuracy = [parameter_analysis['inlier_threshold'][it]['accuracy'] for it in it_values]
    it_top_k = [parameter_analysis['inlier_threshold'][it]['top_k_accuracy'] for it in it_values]
    
    axes[1, 1].plot(it_values, it_accuracy, 'o-', label='Accuracy')
    axes[1, 1].plot(it_values, it_top_k, 's--', label='Top-3 Accuracy')
    axes[1, 1].set_xlabel('Inlier Threshold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Effect of Inlier Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Example usage
    base_dir = "A2_smvs"
    
    # Run image search on book covers
    print("Running image search on book covers...")
    book_results = run_image_search(base_dir, categories=['book_covers'])
    
    # Visualize some results
    visualize_results(book_results, num_examples=2)
    
    # Run image search on museum paintings
    print("\nRunning image search on museum paintings...")
    painting_results = run_image_search(base_dir, categories=['museum_paintings'])
    
    # Visualize some results
    visualize_results(painting_results, num_examples=2)
    
    # Run image search on landmarks
    print("\nRunning image search on landmarks...")
    landmark_results = run_image_search(base_dir, categories=['landmarks'])
    
    # Visualize some results
    visualize_results(landmark_results, num_examples=2)
    
    # Compare performance across categories
    print("\n=== Performance Comparison ===")
    print(f"Book covers: {book_results['accuracy']:.2f} ({book_results['correct_matches']}/{book_results['total_queries']})")
    print(f"Museum paintings: {painting_results['accuracy']:.2f} ({painting_results['correct_matches']}/{painting_results['total_queries']})")
    print(f"Landmarks: {landmark_results['accuracy']:.2f} ({landmark_results['correct_matches']}/{landmark_results['total_queries']})")
    
    # Analyze parameters (optional, time-consuming)
    # param_analysis = analyze_parameters(base_dir)
    # plot_parameter_analysis(param_analysis) 