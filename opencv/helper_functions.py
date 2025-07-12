import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_outline(ref, query, model):
    """
    Draw outline of reference image in the query image.
    
    Inputs:
        ref: reference image
        query: query image
        model: estimated transformation from query to reference image
    
    Returns:
        img: query image with reference outline drawn
    """
    try:
        h, w = ref.shape[:2]
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, model)

        img = query.copy()
        img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        return img
    except Exception as e:
        print(f"Warning: Could not draw outline: {e}")
        return query.copy()

def draw_inliers(img1, img2, kp1, kp2, matches, matchesMask):
    """
    Draw inliers between images
    
    Inputs:
        img1: reference image
        img2: query image
        kp1: keypoints from reference image
        kp2: keypoints from query image
        matches: list of (good) matches after ratio test
        matchesMask: Inlier mask returned in cv2.findHomography()
    
    Returns:
        img3: image with inliers drawn
    """
    try:
        # Make sure images are valid
        if img1 is None or img2 is None:
            raise ValueError("Input images cannot be None")
            
        # Check image dimensions
        if len(img1.shape) < 2 or len(img2.shape) < 2:
            raise ValueError("Input images must have valid dimensions")
            
        # Get image dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create a result image
        height = max(h1, h2)
        width = w1 + w2
        
        # Convert grayscale images to color for drawing
        if len(img1.shape) == 2:  # Grayscale
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            result = np.zeros((height, width, 3), dtype=np.uint8)
        else:  # Already color
            img1_color = img1.copy()
            img2_color = img2.copy()
            result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Copy the images into the result
        result[:h1, :w1] = img1_color
        result[:h2, w1:w1+w2] = img2_color
        
        # Convert matchesMask to list if it's a numpy array
        if matchesMask is not None:
            if isinstance(matchesMask, np.ndarray):
                matchesMask = matchesMask.ravel().tolist()
        else:
            # If matchesMask is None, create a list of ones
            matchesMask = [1] * len(matches)
        
        # Draw inlier matches manually
        for idx, match in enumerate(matches):
            if idx < len(matchesMask) and matchesMask[idx]:
                # Get the keypoints from the match
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx
                
                # Get x,y coordinates
                x1, y1 = int(kp1[img1_idx].pt[0]), int(kp1[img1_idx].pt[1])
                x2, y2 = int(kp2[img2_idx].pt[0]), int(kp2[img2_idx].pt[1])
                
                # Draw circles
                cv2.circle(result, (x1, y1), 4, (0, 255, 0), 1)
                cv2.circle(result, (x2 + w1, y2), 4, (0, 255, 0), 1)
                
                # Connect with lines
                cv2.line(result, (x1, y1), (x2 + w1, y2), (0, 255, 0), 1)
        
        return result
    except Exception as e:
        print(f"Warning: Could not draw inliers: {e}")
        # Create a simple side-by-side image as fallback
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Create a simple side-by-side image
            if len(img1.shape) == 2:  # Grayscale
                result = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
                result[:h1, :w1] = img1
                result[:h2, w1:w1+w2] = img2
            else:  # Color
                result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                result[:h1, :w1] = img1
                result[:h2, w1:w1+w2] = img2
                
            return result
        except:
            # Last resort - just return the first image
            return img1

def drawlines(img1, img2, lines, pts1, pts2):
    """
    Draw epipolar lines on images
    
    Inputs:
        img1: image on which we draw the epipolar lines
        img2: image where the points are defined for visualizing epilines in image 1
        lines: corresponding epilines
        pts1, pts2: good matches in image 1 and 2
    
    Returns:
        img1, img2: images with epipolar lines drawn
    """
    try:
        # Check if inputs are valid
        if img1 is None or img2 is None or lines is None or pts1 is None or pts2 is None:
            return img1, img2
            
        r, c = img1.shape
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
            img1_color = cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
            img2_color = cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)
        
        return img1_color, img2_color
    except Exception as e:
        print(f"Warning: Could not draw epipolar lines: {e}")
        try:
            return cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        except:
            return img1, img2 