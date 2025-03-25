# -*- coding: utf-8 -*-
"""
Image processing utilities for Stego-AI.

This module provides functions for loading, saving, and processing
images used in steganographic operations.
"""

import os
import io
import logging
import numpy as np
from PIL import Image
import torch
import cv2

# Set up logging
logger = logging.getLogger(__name__)


def read_image(file_path, normalize=False):
    """
    Read an image from file.
    
    Args:
        file_path (str): Path to image file
        normalize (bool): Whether to normalize to [-1, 1] range
        
    Returns:
        numpy.ndarray: Image array in RGB format with values in [0, 255] or [-1, 1]
    """
    try:
        # Read image
        img = Image.open(file_path).convert('RGB')
        img_array = np.array(img)
        
        # Normalize if requested
        if normalize:
            img_array = normalize_image(img_array)
            
        return img_array
    
    except Exception as e:
        logger.error(f"Error reading image {file_path}: {e}")
        raise


def save_image(img_array, file_path, quality=95):
    """
    Save an image to file.
    
    Args:
        img_array (numpy.ndarray): Image array in RGB format
        file_path (str): Path to save image
        quality (int): JPEG quality (0-100) if saving as JPEG
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convert to PIL image
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Determine format from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # Save image with appropriate format
        if ext == '.jpg' or ext == '.jpeg':
            img.save(file_path, quality=quality, subsampling=0)
        elif ext == '.png':
            img.save(file_path, compress_level=1)
        elif ext == '.webp':
            img.save(file_path, quality=quality)
        elif ext == '.bmp':
            img.save(file_path)
        else:
            # Default to PNG
            logger.warning(f"Unknown format for {file_path}, saving as PNG")
            file_path = os.path.splitext(file_path)[0] + '.png'
            img.save(file_path, compress_level=1)
    
    except Exception as e:
        logger.error(f"Error saving image to {file_path}: {e}")
        raise


def normalize_image(img_array):
    """
    Normalize image array to [-1, 1] range.
    
    Args:
        img_array (numpy.ndarray): Image array in [0, 255] range
        
    Returns:
        numpy.ndarray: Normalized image array in [-1, 1] range
    """
    # Ensure array is float
    img_float = img_array.astype(np.float32)
    
    # Normalize to [-1, 1]
    img_norm = (img_float / 127.5) - 1.0
    
    return img_norm


def denormalize_image(img_array):
    """
    Denormalize image array from [-1, 1] to [0, 255] range.
    
    Args:
        img_array (numpy.ndarray): Normalized image array in [-1, 1] range
        
    Returns:
        numpy.ndarray: Image array in [0, 255] range
    """
    # Convert from [-1, 1] to [0, 255]
    img_denorm = ((img_array + 1.0) * 127.5).astype(np.uint8)
    
    return img_denorm


def get_image_dimensions(file_path):
    """
    Get image dimensions.
    
    Args:
        file_path (str): Path to image file
        
    Returns:
        tuple: (width, height) of the image
    """
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Error getting image dimensions for {file_path}: {e}")
        raise


def estimate_image_quality(file_path):
    """
    Estimate image quality.
    
    Args:
        file_path (str): Path to image file
        
    Returns:
        float: Estimated quality (0.0-1.0)
    """
    try:
        # Read image with OpenCV for quality estimation
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (a measure of sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Normalize to 0-1 range (empirical values)
        quality = min(max(laplacian_var / 500.0, 0.0), 1.0)
        
        return quality
    
    except Exception as e:
        logger.error(f"Error estimating image quality for {file_path}: {e}")
        # Return a default quality value
        return 0.5


def is_image_png_or_bmp(file_path):
    """
    Check if an image is PNG or BMP (lossless formats).
    
    Args:
        file_path (str): Path to image file
        
    Returns:
        bool: True if image is PNG or BMP, False otherwise
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.png', '.bmp']
    except Exception:
        return False


def image_to_tensor(img_array, add_batch=True):
    """
    Convert image array to PyTorch tensor.
    
    Args:
        img_array (numpy.ndarray): Image array (H, W, C)
        add_batch (bool): Whether to add batch dimension
        
    Returns:
        torch.Tensor: Image tensor (N, C, H, W) or (C, H, W)
    """
    # Convert to float if not already
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(img_array)
    
    # Rearrange dimensions from (H, W, C) to (C, H, W)
    tensor = tensor.permute(2, 0, 1)
    
    # Add batch dimension if requested
    if add_batch:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor to image array.
    
    Args:
        tensor (torch.Tensor): Image tensor (N, C, H, W) or (C, H, W)
        
    Returns:
        numpy.ndarray: Image array (H, W, C)
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy and rearrange dimensions
    img_array = tensor.numpy()
    img_array = np.transpose(img_array, (1, 2, 0))
    
    return img_array


def extract_image_features(img_array):
    """
    Extract basic image features.
    
    Args:
        img_array (numpy.ndarray): Image array
        
    Returns:
        dict: Dictionary of image features
    """
    # Ensure array is in the right format
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Basic features
    features = {
        'shape': img_array.shape,
        'mean': np.mean(img_array, axis=(0, 1)),
        'std': np.std(img_array, axis=(0, 1)),
        'min': np.min(img_array, axis=(0, 1)),
        'max': np.max(img_array, axis=(0, 1)),
    }
    
    # Texture features (using Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var'] = laplacian.var()
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    features['edge_ratio'] = np.count_nonzero(edges) / edges.size
    
    return features


def resize_image(img_array, target_size):
    """
    Resize image while preserving aspect ratio.
    
    Args:
        img_array (numpy.ndarray): Image array
        target_size (tuple): Target (width, height)
        
    Returns:
        numpy.ndarray: Resized image array
    """
    # Get current dimensions
    height, width = img_array.shape[:2]
    target_width, target_height = target_size
    
    # Calculate aspect ratios
    aspect_ratio = width / height
    target_aspect = target_width / target_height
    
    # Determine new dimensions
    if aspect_ratio > target_aspect:
        # Width constrained
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Height constrained
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    
    # Resize image
    resized = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create new image with target size (letterboxing)
    result = np.zeros((target_height, target_width, img_array.shape[2]), dtype=img_array.dtype)
    
    # Calculate offsets for centering
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image
    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return result


def crop_image(img_array, crop_size):
    """
    Crop image to target size from center.
    
    Args:
        img_array (numpy.ndarray): Image array
        crop_size (tuple): Target (width, height)
        
    Returns:
        numpy.ndarray: Cropped image array
    """
    # Get current dimensions
    height, width = img_array.shape[:2]
    crop_width, crop_height = crop_size
    
    # Calculate crop coordinates
    x_start = (width - crop_width) // 2
    y_start = (height - crop_height) // 2
    
    # Crop image
    cropped = img_array[y_start:y_start+crop_height, x_start:x_start+crop_width]
    
    return cropped


def detect_faces(img_array):
    """
    Detect faces in image.
    
    Args:
        img_array (numpy.ndarray): Image array
        
    Returns:
        list: List of face bounding boxes (x, y, w, h)
    """
    # Load face cascade
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        logger.error(f"Error loading face cascade: {e}")
        return []
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces.tolist() if len(faces) > 0 else []


def compress_image_to_jpeg(img_array, quality=95):
    """
    Compress image using JPEG.
    
    Args:
        img_array (numpy.ndarray): Image array
        quality (int): JPEG quality (0-100)
        
    Returns:
        numpy.ndarray: Compressed image array
    """
    # Convert to uint8 if not already
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # Create in-memory buffer
    buffer = io.BytesIO()
    
    # Save as JPEG
    img = Image.fromarray(img_array)
    img.save(buffer, format='JPEG', quality=quality)
    
    # Load back
    buffer.seek(0)
    compressed = np.array(Image.open(buffer))
    
    return compressed


def estimate_jpeg_quality(file_path):
    """
    Estimate JPEG quality.
    
    Args:
        file_path (str): Path to JPEG image
        
    Returns:
        int: Estimated quality (0-100)
    """
    try:
        # Check if file is JPEG
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.jpg', '.jpeg']:
            return None
        
        # Read JPEG file
        with open(file_path, 'rb') as f:
            jpeg_data = f.read()
        
        # Find DQT (Define Quantization Table) markers
        dqt_marker = b'\xFF\xDB'
        dqt_positions = []
        idx = 0
        while True:
            idx = jpeg_data.find(dqt_marker, idx)
            if idx == -1:
                break
            dqt_positions.append(idx)
            idx += 2
        
        if not dqt_positions:
            return None
        
        # Extract quantization tables
        q_tables = []
        for pos in dqt_positions:
            # Read table length
            length = (jpeg_data[pos+2] << 8) + jpeg_data[pos+3]
            # Read table
            table_data = jpeg_data[pos+4:pos+2+length]
            q_tables.append(table_data)
        
        # Simple quality estimation based on average values
        avg_values = [sum(table) / len(table) for table in q_tables]
        avg_q = sum(avg_values) / len(avg_values)
        
        # Convert to quality scale (approximate)
        if avg_q <= 1:
            quality = 100
        elif avg_q >= 50:
            quality = 1
        else:
            quality = int(100 - avg_q * 2)
        
        return quality
    
    except Exception as e:
        logger.error(f"Error estimating JPEG quality for {file_path}: {e}")
        return None


def apply_blur(img_array, radius=3):
    """
    Apply Gaussian blur to image.
    
    Args:
        img_array (numpy.ndarray): Image array
        radius (int): Blur radius
        
    Returns:
        numpy.ndarray: Blurred image array
    """
    return cv2.GaussianBlur(img_array.astype(np.float32), (radius*2+1, radius*2+1), 0)


def apply_random_noise(img_array, std=0.01):
    """
    Apply random noise to image.
    
    Args:
        img_array (numpy.ndarray): Image array (normalized to [-1, 1])
        std (float): Standard deviation of noise
        
    Returns:
        numpy.ndarray: Noisy image array
    """
    # Generate noise
    noise = np.random.normal(0, std, img_array.shape)
    
    # Add noise
    noisy = img_array + noise
    
    # Clip to valid range
    return np.clip(noisy, -1.0, 1.0)


def psnr(original, compressed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original (torch.Tensor or numpy.ndarray): Original image
        compressed (torch.Tensor or numpy.ndarray): Compressed image
        
    Returns:
        float: PSNR value in dB
    """
    # Convert to numpy if needed
    if isinstance(original, torch.Tensor):
        if original.is_cuda:
            original = original.cpu()
        original = original.detach().numpy()
    
    if isinstance(compressed, torch.Tensor):
        if compressed.is_cuda:
            compressed = compressed.cpu()
        compressed = compressed.detach().numpy()
    
    # Calculate MSE
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_value = 2.0  # For [-1, 1] range
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    
    return psnr


def ssim(original, compressed):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        original (torch.Tensor or numpy.ndarray): Original image
        compressed (torch.Tensor or numpy.ndarray): Compressed image
        
    Returns:
        float: SSIM value
    """
    # Constants for stability
    K1 = 0.01
    K2 = 0.03
    L = 2.0  # Dynamic range for [-1, 1] images
    
    # Convert to numpy if needed
    if isinstance(original, torch.Tensor):
        if original.is_cuda:
            original = original.cpu()
        original = original.detach().numpy()
    
    if isinstance(compressed, torch.Tensor):
        if compressed.is_cuda:
            compressed = compressed.cpu()
        compressed = compressed.detach().numpy()
    
    # Calculate statistics
    mu1 = np.mean(original)
    mu2 = np.mean(compressed)
    
    sigma1_sq = np.var(original)
    sigma2_sq = np.var(compressed)
    
    # Calculate covariance
    sigma12 = np.mean((original - mu1) * (compressed - mu2))
    
    # Calculate SSIM
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = num / den
    
    return ssim