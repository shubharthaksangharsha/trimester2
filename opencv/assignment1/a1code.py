### Supporting code for Computer Vision Assignment 1
### See "Assignment_1_Notebook.ipynb" for instructions

import math
import numpy as np
from skimage import io


def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    out = io.imread(img_path).astype(np.float64) / 255.0
    
    return out


def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    
    if image.ndim == 3:
        height, width, channels = image.shape
        print(f"Image height: {height}, width: {width}, channels: {channels}")
    else:
        height, width = image.shape
        print(f"Image height: {height}, width: {width}, channels: 1")


def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    end_row = start_row + num_rows
    end_col = start_col + num_cols
    out = image[start_row:end_row, start_col:end_col]

    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = np.clip(factor * (image - 0.5) + 0.5, 0.0, 1.0)

    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols = input_image.shape[0], input_image.shape[1]
    row_ratio = input_rows / output_rows
    col_ratio = input_cols / output_cols

    out_r_indices = np.arange(output_rows)
    out_c_indices = np.arange(output_cols)

    in_r_indices = np.floor(out_r_indices * row_ratio).astype(int)
    in_c_indices = np.floor(out_c_indices * col_ratio).astype(int)
    
    if input_image.ndim == 3:
        out = input_image[in_r_indices[:, np.newaxis], in_c_indices]
    else:
        out = input_image[in_r_indices[:, np.newaxis], in_c_indices]

    return out


def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    out = np.mean(input_image, axis=2)

    return out

def binary(grey_img, threshold):
    """Convert a greyscale image to a binary mask with threshold.

                    x_out = 0, if x_in < threshold
                    x_out = 1, if x_in >= threshold

    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        threshold (float): The threshold used for binarization, and the value range of threshold is from 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = (grey_img >= threshold).astype(float)
    
    return out


def xcorr2D(image, kernel):
    """ Cross correlation of a 2D image with a 2D kernel.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros_like(image)

    pad_height = Hk // 2
    pad_width = Wk // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(kernel * padded_image[i:i+Hk, j:j+Wk])

    return out


def test_xcorr2D():
    """ A simple test for your 2D cross-correlation function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = xcorr2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2, 4:7] = 1
    expected_output[3, 4:7] = 1
    expected_output[4, 2:4] = 1
    expected_output[4, 4] = 3
    expected_output[4, 5:7] = 2
    expected_output[5, 2:4] = 1
    expected_output[5, 4] = 2
    expected_output[5, 5:7] = 1
    expected_output[6, 2:4] = 1
    expected_output[6, 4] = 2
    expected_output[6, 5:7] = 1

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."

def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    flipped_kernel = np.flip(np.flip(kernel, 0), 1)
    out = xcorr2D(image, flipped_kernel)

    return out


def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of an image with a kernel. 
    This function is a wrapper for conv2D. It detects if an image is greyscale or
    RGB and applies the convolution to each channel separately.

    Args:
        image: numpy array of shape (Hi, Wi) or (Hi, Wi, 3).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi) or (Hi, Wi, 3).
    """
    if image.ndim == 3:
        h, w, d = image.shape
        out = np.zeros((h, w, d))
        for i in range(d):
            out[:, :, i] = conv2D(image[:, :, i], kernel)
    elif image.ndim == 2:
        out = conv2D(image, kernel)
    else:
        raise ValueError("Image must be 2D or 3D")
        
    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
    
    
def LoG2D(size, sigma):

    """
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing LoG filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    
    term1 = (x**2 + y**2) / (2 * sigma**2)
    log = -1 / (np.pi * sigma**4) * (1 - term1) * np.exp(-term1)
    
    return log - np.mean(log) # Normalize to sum to zero
