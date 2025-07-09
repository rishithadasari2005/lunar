"""
Lunar Digital Elevation Model Generation using Photoclinometry

This module implements the main LunarDEMGenerator class for generating
high-resolution lunar DEMs using shape from shading techniques.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from photometric_models import HapkeModel, LommelSeeligerModel, LambertianModel
from preprocessing import ImagePreprocessor
from validation import DEMValidator
from visualization import DEMVisualizer
from utils import compute_gradients, integrate_gradients, apply_smoothing


class LunarDEMGenerator:
    """
    Main class for generating lunar Digital Elevation Models using photoclinometry.
    
    This class implements shape from shading techniques specifically optimized
    for lunar surface characteristics, including multiple photometric models
    and advanced noise reduction algorithms.
    """
    
    def __init__(self, photometric_model="hapke", **model_params):
        """
        Initialize the Lunar DEM Generator.
        
        Args:
            photometric_model (str): Type of photometric model to use
                                    ('hapke', 'lommel_seeliger', 'lambertian')
            **model_params: Parameters for the photometric model
        """
        self.preprocessor = ImagePreprocessor()
        self.validator = DEMValidator()
        self.visualizer = DEMVisualizer()
        
        # Initialize photometric model
        self.set_photometric_model(photometric_model, **model_params)
        
        # Default parameters
        self.resolution = 1000
        self.smoothing_factor = 0.1
        self.noise_reduction = True
        self.max_iterations = 100
        self.tolerance = 1e-6
        
        # Solar illumination parameters (default values for lunar surface)
        self.solar_zenith = np.radians(30)  # 30 degrees from zenith
        self.solar_azimuth = np.radians(0)  # North direction
        
    def set_photometric_model(self, model_type, **params):
        """
        Set the photometric model for reflectance calculations.
        
        Args:
            model_type (str): Type of photometric model
            **params: Model-specific parameters
        """
        if model_type.lower() == "hapke":
            self.photometric_model = HapkeModel(**params)
        elif model_type.lower() == "lommel_seeliger":
            self.photometric_model = LommelSeeligerModel(**params)
        elif model_type.lower() == "lambertian":
            self.photometric_model = LambertianModel(**params)
        else:
            raise ValueError(f"Unknown photometric model: {model_type}")
            
    def generate_dem(self, image_path, resolution=None, smoothing_factor=None,
                    noise_reduction=None, **kwargs):
        """
        Generate a Digital Elevation Model from a lunar image.
        
        Args:
            image_path (str): Path to the lunar image
            resolution (int): Target resolution for the DEM
            smoothing_factor (float): Smoothing parameter (0-1)
            noise_reduction (bool): Whether to apply noise reduction
            **kwargs: Additional parameters
            
        Returns:
            numpy.ndarray: Generated DEM
        """
        # Update parameters if provided
        if resolution is not None:
            self.resolution = resolution
        if smoothing_factor is not None:
            self.smoothing_factor = smoothing_factor
        if noise_reduction is not None:
            self.noise_reduction = noise_reduction
            
        print("Loading and preprocessing lunar image...")
        
        # Load and preprocess image
        image = self.preprocessor.load_image(image_path)
        processed_image = self.preprocessor.preprocess(image, self.resolution)
        
        # Convert to grayscale if needed
        if len(processed_image.shape) == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            
        # Normalize image to [0, 1]
        processed_image = processed_image.astype(np.float64) / 255.0
        
        print("Computing surface gradients...")
        
        # Compute gradients using photoclinometry
        gradients = self._compute_surface_gradients(processed_image)
        
        print("Integrating gradients to generate DEM...")
        
        # Integrate gradients to get surface heights
        dem = self._integrate_gradients(gradients)
        
        # Apply smoothing and noise reduction
        if self.smoothing_factor > 0:
            dem = apply_smoothing(dem, self.smoothing_factor)
            
        if self.noise_reduction:
            dem = self._apply_noise_reduction(dem)
            
        print("DEM generation completed!")
        
        return dem
    
    def _compute_surface_gradients(self, image):
        """
        Compute surface gradients using photoclinometry.
        
        Args:
            image (numpy.ndarray): Preprocessed lunar image
            
        Returns:
            tuple: (gradient_x, gradient_y) surface gradients
        """
        height, width = image.shape
        
        # Initialize gradient arrays
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        
        # Compute gradients for each pixel
        for i in tqdm(range(height), desc="Computing gradients"):
            for j in range(width):
                if i > 0 and i < height - 1 and j > 0 and j < width - 1:
                    # Get local neighborhood
                    neighborhood = image[i-1:i+2, j-1:j+2]
                    
                    # Compute gradients using photoclinometry
                    gx, gy = self._photoclinometry_gradients(
                        neighborhood, i, j, image
                    )
                    
                    grad_x[i, j] = gx
                    grad_y[i, j] = gy
                    
        return grad_x, grad_y
    
    def _photoclinometry_gradients(self, neighborhood, i, j, full_image):
        """
        Compute gradients using photoclinometry for a single pixel.
        
        Args:
            neighborhood (numpy.ndarray): 3x3 neighborhood around pixel
            i, j (int): Pixel coordinates
            full_image (numpy.ndarray): Full image array
            
        Returns:
            tuple: (gradient_x, gradient_y)
        """
        # Get current pixel intensity
        intensity = full_image[i, j]
        
        # Define objective function for gradient optimization
        def objective_function(gradients):
            gx, gy = gradients
            
            # Compute surface normal from gradients
            normal = np.array([-gx, -gy, 1.0])
            normal = normal / np.linalg.norm(normal)
            
            # Compute solar direction
            solar_dir = np.array([
                np.sin(self.solar_zenith) * np.cos(self.solar_azimuth),
                np.sin(self.solar_zenith) * np.sin(self.solar_azimuth),
                np.cos(self.solar_zenith)
            ])
            
            # Compute viewing direction (assume nadir viewing)
            view_dir = np.array([0, 0, 1])
            
            # Compute predicted intensity using photometric model
            predicted_intensity = self.photometric_model.compute_reflectance(
                normal, solar_dir, view_dir
            )
            
            # Return squared error
            return (predicted_intensity - intensity) ** 2
        
        # Optimize gradients
        try:
            result = minimize(
                objective_function,
                x0=[0.0, 0.0],
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0), (-1.0, 1.0)],
                options={'maxiter': 50}
            )
            
            if result.success:
                return result.x[0], result.x[1]
            else:
                # Fallback to simple gradient computation
                return self._simple_gradients(neighborhood)
                
        except:
            # Fallback to simple gradient computation
            return self._simple_gradients(neighborhood)
    
    def _simple_gradients(self, neighborhood):
        """
        Compute simple gradients as fallback method.
        
        Args:
            neighborhood (numpy.ndarray): 3x3 neighborhood
            
        Returns:
            tuple: (gradient_x, gradient_y)
        """
        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        gx = np.sum(neighborhood * sobel_x)
        gy = np.sum(neighborhood * sobel_y)
        
        return gx, gy
    
    def _integrate_gradients(self, gradients):
        """
        Integrate surface gradients to generate DEM.
        
        Args:
            gradients (tuple): (gradient_x, gradient_y)
            
        Returns:
            numpy.ndarray: Integrated DEM
        """
        grad_x, grad_y = gradients
        
        # Use Frankot-Chellappa algorithm for integration
        dem = integrate_gradients(grad_x, grad_y)
        
        return dem
    
    def _apply_noise_reduction(self, dem):
        """
        Apply noise reduction to the DEM.
        
        Args:
            dem (numpy.ndarray): Input DEM
            
        Returns:
            numpy.ndarray: Denoised DEM
        """
        # Apply bilateral filter for edge-preserving smoothing
        dem_denoised = cv2.bilateralFilter(
            dem.astype(np.float32), 
            d=15, 
            sigmaColor=0.1, 
            sigmaSpace=1.5
        )
        
        # Apply median filter to remove outliers
        dem_denoised = ndimage.median_filter(dem_denoised, size=3)
        
        return dem_denoised
    
    def validate_dem(self, dem, reference_data=None):
        """
        Validate the generated DEM against reference data.
        
        Args:
            dem (numpy.ndarray): Generated DEM
            reference_data (numpy.ndarray): Reference DEM for validation
            
        Returns:
            dict: Validation metrics
        """
        return self.validator.validate(dem, reference_data)
    
    def visualize_3d(self, dem, save_path=None, **kwargs):
        """
        Create 3D visualization of the DEM.
        
        Args:
            dem (numpy.ndarray): DEM to visualize
            save_path (str): Path to save visualization
            **kwargs: Additional visualization parameters
        """
        return self.visualizer.visualize_3d(dem, save_path, **kwargs)
    
    def plot_contours(self, dem, levels=20, **kwargs):
        """
        Create contour plot of the DEM.
        
        Args:
            dem (numpy.ndarray): DEM to plot
            levels (int): Number of contour levels
            **kwargs: Additional plotting parameters
        """
        return self.visualizer.plot_contours(dem, levels, **kwargs)
    
    def save_dem(self, dem, filepath, format='npy'):
        """
        Save DEM to file.
        
        Args:
            dem (numpy.ndarray): DEM to save
            filepath (str): Output file path
            format (str): File format ('npy', 'tiff', 'csv')
        """
        if format.lower() == 'npy':
            np.save(filepath, dem)
        elif format.lower() == 'tiff':
            import rasterio
            with rasterio.open(
                filepath, 'w',
                driver='GTiff',
                height=dem.shape[0],
                width=dem.shape[1],
                count=1,
                dtype=dem.dtype
            ) as dst:
                dst.write(dem, 1)
        elif format.lower() == 'csv':
            np.savetxt(filepath, dem, delimiter=',')
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"DEM saved to {filepath}")
    
    def batch_process(self, image_paths, output_dir, **kwargs):
        """
        Process multiple lunar images in batch.
        
        Args:
            image_paths (list): List of image file paths
            output_dir (str): Output directory for DEMs
            **kwargs: Additional parameters for DEM generation
            
        Returns:
            list: List of generated DEMs
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        dems = []
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Batch processing")):
            try:
                # Generate DEM
                dem = self.generate_dem(image_path, **kwargs)
                
                # Save DEM
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_dem.npy")
                self.save_dem(dem, output_path)
                
                dems.append(dem)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
                
        return dems
    
    def plot_surface(self, dem, title="Lunar DEM Surface", save_path=None, figsize=(12, 8)):
        """
        Create 2D surface plot of the DEM.
        
        Args:
            dem (numpy.ndarray): DEM to plot
            title (str): Plot title
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        return self.visualizer.plot_surface(dem, title=title, save_path=save_path, figsize=figsize)
