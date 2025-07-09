"""
Image Preprocessing for Lunar DEM Generation

This module handles image preprocessing tasks including loading, resizing,
noise reduction, and enhancement for lunar surface images.
"""

import numpy as np
import cv2
from PIL import Image
import os
from skimage import filters, restoration, exposure
from scipy import ndimage


class ImagePreprocessor:
    """
    Image preprocessing class for lunar surface images.
    
    Handles various preprocessing tasks including:
    - Image loading and format conversion
    - Noise reduction and enhancement
    - Resizing and interpolation
    - Contrast and brightness adjustment
    """
    
    def __init__(self):
        """Initialize the image preprocessor."""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
    def load_image(self, image_path):
        """
        Load an image from file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check file extension
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {ext}")
            
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def preprocess(self, image, target_size=None, enhance_contrast=True,
                  reduce_noise=True, normalize=True):
        """
        Preprocess lunar image for DEM generation.
        
        Args:
            image (numpy.ndarray): Input image
            target_size (tuple): Target size (height, width) for resizing
            enhance_contrast (bool): Whether to enhance contrast
            reduce_noise (bool): Whether to reduce noise
            normalize (bool): Whether to normalize pixel values
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        processed_image = image.copy()
        
        # Convert to grayscale if needed
        if len(processed_image.shape) == 3:
            processed_image = self._convert_to_grayscale(processed_image)
            
        # Resize if target size is specified
        if target_size is not None:
            processed_image = self._resize_image(processed_image, target_size)
            
        # Reduce noise
        if reduce_noise:
            processed_image = self._reduce_noise(processed_image)
            
        # Enhance contrast
        if enhance_contrast:
            processed_image = self._enhance_contrast(processed_image)
            
        # Normalize
        if normalize:
            processed_image = self._normalize_image(processed_image)
            
        return processed_image
    
    def _convert_to_grayscale(self, image):
        """
        Convert RGB image to grayscale.
        
        Args:
            image (numpy.ndarray): RGB image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        # Use luminance-weighted conversion for better lunar surface representation
        # Lunar surface is best represented using luminance rather than simple averaging
        weights = np.array([0.299, 0.587, 0.114])  # Standard luminance weights
        grayscale = np.dot(image, weights)
        
        return grayscale.astype(np.uint8)
    
    def _resize_image(self, image, target_size):
        """
        Resize image to target size.
        
        Args:
            image (numpy.ndarray): Input image
            target_size (tuple or int): Target size (height, width) or single dimension
            
        Returns:
            numpy.ndarray: Resized image
        """
        if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            height, width = target_size
        else:
            height = width = target_size
            
        # Use cubic interpolation for better quality
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def _reduce_noise(self, image):
        """
        Reduce noise in the image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Denoised image
        """
        # Apply bilateral filter for edge-preserving noise reduction
        denoised = cv2.bilateralFilter(
            image.astype(np.float32),
            d=15,  # Diameter of pixel neighborhood
            sigmaColor=75,  # Filter sigma in color space
            sigmaSpace=75   # Filter sigma in coordinate space
        )
        
        # Apply non-local means denoising for additional noise reduction
        denoised = cv2.fastNlMeansDenoising(
            denoised.astype(np.uint8),
            h=10,  # Filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return denoised
    
    def _enhance_contrast(self, image):
        """
        Enhance image contrast.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Contrast-enhanced image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Apply gamma correction for better lunar surface visibility
        gamma = 0.8  # Slightly darker for lunar surface
        enhanced = np.power(enhanced / 255.0, gamma) * 255
        enhanced = enhanced.astype(np.uint8)
        
        return enhanced
    
    def _normalize_image(self, image):
        """
        Normalize image pixel values.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Normalized image
        """
        # Normalize to [0, 255] range
        normalized = cv2.normalize(
            image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        
        return normalized.astype(np.uint8)
    
    def extract_region_of_interest(self, image, roi_coords):
        """
        Extract region of interest from image.
        
        Args:
            image (numpy.ndarray): Input image
            roi_coords (tuple): ROI coordinates (x, y, width, height)
            
        Returns:
            numpy.ndarray: Extracted ROI
        """
        x, y, w, h = roi_coords
        roi = image[y:y+h, x:x+w]
        
        return roi
    
    def apply_mask(self, image, mask):
        """
        Apply mask to image.
        
        Args:
            image (numpy.ndarray): Input image
            mask (numpy.ndarray): Binary mask
            
        Returns:
            numpy.ndarray: Masked image
        """
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape")
            
        masked_image = image.copy()
        masked_image[mask == 0] = 0
        
        return masked_image
    
    def detect_shadows(self, image, threshold=0.3):
        """
        Detect shadow regions in lunar image.
        
        Args:
            image (numpy.ndarray): Input image
            threshold (float): Shadow detection threshold
            
        Returns:
            numpy.ndarray: Binary shadow mask
        """
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_float, (5, 5), 0)
        
        # Compute local mean and standard deviation
        mean_filter = cv2.blur(blurred, (15, 15))
        var_filter = cv2.blur(blurred**2, (15, 15)) - mean_filter**2
        std_filter = np.sqrt(np.maximum(var_filter, 0))
        
        # Detect shadows based on low intensity and low variance
        shadow_mask = (blurred < threshold) & (std_filter < 0.05)
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(
            shadow_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        
        return shadow_mask.astype(bool)
    
    def enhance_lunar_features(self, image):
        """
        Enhance lunar surface features for better DEM generation.
        
        Args:
            image (numpy.ndarray): Input lunar image
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        # Apply unsharp masking to enhance edges
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Apply edge enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(unsharp_mask, -1, kernel)
        
        # Normalize result
        enhanced = cv2.normalize(
            enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        
        return enhanced.astype(np.uint8)
    
    def correct_illumination(self, image, solar_angle=None):
        """
        Correct for non-uniform illumination.
        
        Args:
            image (numpy.ndarray): Input image
            solar_angle (float): Solar incidence angle in radians
            
        Returns:
            numpy.ndarray: Illumination-corrected image
        """
        # Estimate illumination pattern using low-pass filtering
        illumination = cv2.GaussianBlur(image.astype(np.float32), (51, 51), 0)
        
        # Normalize illumination pattern
        illumination = illumination / np.max(illumination)
        
        # Correct image by dividing by illumination pattern
        corrected = image.astype(np.float32) / (illumination + 1e-6)
        
        # Normalize result
        corrected = cv2.normalize(
            corrected, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        
        return corrected.astype(np.uint8)
    
    def create_synthetic_lunar_image(self, size=(512, 512), crater_density=0.1):
        """
        Create synthetic lunar image for testing.
        
        Args:
            size (tuple): Image size (height, width)
            crater_density (float): Density of craters
            
        Returns:
            numpy.ndarray: Synthetic lunar image
        """
        height, width = size
        
        # Create base lunar surface
        base_surface = np.random.normal(128, 20, size).astype(np.uint8)
        
        # Add craters
        num_craters = int(crater_density * height * width / 1000)
        
        for _ in range(num_craters):
            # Random crater parameters
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)
            radius = np.random.randint(5, 30)
            
            # Create crater (dark circular region)
            y, x = np.ogrid[:height, :width]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            # Darken crater region
            base_surface[mask] = np.clip(base_surface[mask] - 50, 0, 255)
            
            # Add crater rim (brighter ring)
            rim_mask = ((x - cx)**2 + (y - cy)**2 <= (radius + 2)**2) & \
                      ((x - cx)**2 + (y - cy)**2 > radius**2)
            base_surface[rim_mask] = np.clip(base_surface[rim_mask] + 30, 0, 255)
        
        # Add noise
        noise = np.random.normal(0, 5, size).astype(np.int16)
        base_surface = np.clip(base_surface.astype(np.int16) + noise, 0, 255)
        
        return base_surface.astype(np.uint8) 