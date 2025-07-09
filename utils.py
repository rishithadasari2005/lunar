"""
Utility Functions for Lunar DEM Generation

This module provides utility functions for gradient computation, integration,
smoothing, and other common operations used in photoclinometry.
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq


def compute_gradients(image, method='sobel'):
    """
    Compute image gradients using various methods.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Gradient computation method ('sobel', 'prewitt', 'central')
        
    Returns:
        tuple: (gradient_x, gradient_y) gradients
    """
    if method.lower() == 'sobel':
        grad_x = ndimage.sobel(image, axis=1)
        grad_y = ndimage.sobel(image, axis=0)
    elif method.lower() == 'prewitt':
        grad_x = ndimage.prewitt(image, axis=1)
        grad_y = ndimage.prewitt(image, axis=0)
    elif method.lower() == 'central':
        # Central difference
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
    else:
        raise ValueError(f"Unknown gradient method: {method}")
        
    return grad_x, grad_y


def integrate_gradients(grad_x, grad_y, method='frankot_chellappa'):
    """
    Integrate gradients to recover surface heights.
    
    Args:
        grad_x (numpy.ndarray): X-gradient
        grad_y (numpy.ndarray): Y-gradient
        method (str): Integration method ('frankot_chellappa', 'poisson')
        
    Returns:
        numpy.ndarray: Integrated surface heights
    """
    if method.lower() == 'frankot_chellappa':
        return _frankot_chellappa_integration(grad_x, grad_y)
    elif method.lower() == 'poisson':
        return _poisson_integration(grad_x, grad_y)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def _frankot_chellappa_integration(grad_x, grad_y):
    """
    Frankot-Chellappa algorithm for gradient integration.
    
    Args:
        grad_x (numpy.ndarray): X-gradient
        grad_y (numpy.ndarray): Y-gradient
        
    Returns:
        numpy.ndarray: Integrated surface heights
    """
    height, width = grad_x.shape
    
    # Create frequency grids
    u = fftfreq(width)
    v = fftfreq(height)
    
    # Create 2D frequency grids
    U, V = np.meshgrid(u, v)
    
    # Avoid division by zero
    denominator = U**2 + V**2
    denominator[0, 0] = 1.0  # Set DC component to 1
    
    # Compute Fourier transforms of gradients
    Gx = fft2(grad_x)
    Gy = fft2(grad_y)
    
    # Frankot-Chellappa formula
    Z = (-1j * U * Gx - 1j * V * Gy) / denominator
    
    # Inverse Fourier transform
    surface = np.real(ifft2(Z))
    
    return surface


def _poisson_integration(grad_x, grad_y):
    """
    Poisson equation solver for gradient integration.
    
    Args:
        grad_x (numpy.ndarray): X-gradient
        grad_y (numpy.ndarray): Y-gradient
        
    Returns:
        numpy.ndarray: Integrated surface heights
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    
    height, width = grad_x.shape
    
    # Compute divergence
    div_grad = np.gradient(grad_x, axis=1) + np.gradient(grad_y, axis=0)
    
    # Create Laplacian matrix
    n = height * width
    diagonals = [-4 * np.ones(n)]
    offsets = [0]
    
    # Add off-diagonal elements for 4-connectivity
    for offset in [-1, 1, -width, width]:
        if offset == -1:  # Left neighbor
            mask = np.arange(n) % width != 0
        elif offset == 1:  # Right neighbor
            mask = np.arange(n) % width != width - 1
        elif offset == -width:  # Top neighbor
            mask = np.arange(n) >= width
        else:  # Bottom neighbor
            mask = np.arange(n) < n - width
            
        diagonals.append(np.ones(n)[mask])
        offsets.append(offset)
    
    # Create sparse matrix
    A = diags(diagonals, offsets, shape=(n, n), format='csr')
    
    # Solve Poisson equation
    b = div_grad.flatten()
    z = spsolve(A, b)
    
    return z.reshape(height, width)


def apply_smoothing(surface, factor=0.1, method='gaussian'):
    """
    Apply smoothing to surface.
    
    Args:
        surface (numpy.ndarray): Input surface
        factor (float): Smoothing factor (0-1)
        method (str): Smoothing method ('gaussian', 'bilateral', 'median')
        
    Returns:
        numpy.ndarray: Smoothed surface
    """
    if method.lower() == 'gaussian':
        sigma = factor * 5.0  # Scale factor to sigma
        smoothed = ndimage.gaussian_filter(surface, sigma=sigma)
    elif method.lower() == 'bilateral':
        # Bilateral filter for edge-preserving smoothing
        import cv2
        smoothed = cv2.bilateralFilter(
            surface.astype(np.float32),
            d=15,
            sigmaColor=factor * 75,
            sigmaSpace=factor * 75
        )
    elif method.lower() == 'median':
        size = int(factor * 10) + 1
        smoothed = ndimage.median_filter(surface, size=size)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
        
    return smoothed


def normalize_surface(surface, method='minmax'):
    """
    Normalize surface heights.
    
    Args:
        surface (numpy.ndarray): Input surface
        method (str): Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        numpy.ndarray: Normalized surface
    """
    if method.lower() == 'minmax':
        # Min-max normalization to [0, 1]
        normalized = (surface - np.min(surface)) / (np.max(surface) - np.min(surface))
    elif method.lower() == 'zscore':
        # Z-score normalization
        normalized = (surface - np.mean(surface)) / np.std(surface)
    elif method.lower() == 'robust':
        # Robust normalization using median and MAD
        median = np.median(surface)
        mad = np.median(np.abs(surface - median))
        normalized = (surface - median) / (mad + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    return normalized


def compute_surface_statistics(surface):
    """
    Compute surface statistics.
    
    Args:
        surface (numpy.ndarray): Input surface
        
    Returns:
        dict: Surface statistics
    """
    stats = {
        'mean': np.mean(surface),
        'std': np.std(surface),
        'min': np.min(surface),
        'max': np.max(surface),
        'median': np.median(surface),
        'skewness': _compute_skewness(surface),
        'kurtosis': _compute_kurtosis(surface),
        'roughness': np.std(surface),
        'range': np.max(surface) - np.min(surface)
    }
    
    return stats


def _compute_skewness(surface):
    """Compute skewness of surface."""
    mean = np.mean(surface)
    std = np.std(surface)
    if std == 0:
        return 0
    return np.mean(((surface - mean) / std) ** 3)


def _compute_kurtosis(surface):
    """Compute kurtosis of surface."""
    mean = np.mean(surface)
    std = np.std(surface)
    if std == 0:
        return 0
    return np.mean(((surface - mean) / std) ** 4) - 3


def detect_outliers(surface, method='iqr', threshold=1.5):
    """
    Detect outliers in surface data.
    
    Args:
        surface (numpy.ndarray): Input surface
        method (str): Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold (float): Detection threshold
        
    Returns:
        numpy.ndarray: Boolean mask of outliers
    """
    if method.lower() == 'iqr':
        # Interquartile range method
        q1 = np.percentile(surface, 25)
        q3 = np.percentile(surface, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (surface < lower_bound) | (surface > upper_bound)
        
    elif method.lower() == 'zscore':
        # Z-score method
        z_scores = np.abs((surface - np.mean(surface)) / np.std(surface))
        outliers = z_scores > threshold
        
    elif method.lower() == 'isolation_forest':
        # Isolation Forest method
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(surface.reshape(-1, 1)) == -1
        outliers = outliers.reshape(surface.shape)
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
        
    return outliers


def fill_gaps(surface, mask=None, method='interpolation'):
    """
    Fill gaps in surface data.
    
    Args:
        surface (numpy.ndarray): Input surface
        mask (numpy.ndarray): Boolean mask of gaps (True = gap)
        method (str): Gap filling method ('interpolation', 'inpainting', 'nearest')
        
    Returns:
        numpy.ndarray: Surface with filled gaps
    """
    if mask is None:
        # Assume NaN values are gaps
        mask = np.isnan(surface)
        
    filled_surface = surface.copy()
    
    if method.lower() == 'interpolation':
        # Linear interpolation
        from scipy.interpolate import griddata
        
        # Get valid points
        y, x = np.mgrid[0:surface.shape[0], 0:surface.shape[1]]
        valid_points = ~mask
        points = np.column_stack([x[valid_points], y[valid_points]])
        values = surface[valid_points]
        
        # Interpolate
        grid_points = np.column_stack([x[mask], y[mask]])
        filled_values = griddata(points, values, grid_points, method='linear')
        filled_surface[mask] = filled_values
        
    elif method.lower() == 'inpainting':
        # Inpainting using OpenCV
        import cv2
        mask_uint8 = mask.astype(np.uint8) * 255
        filled_surface = cv2.inpaint(
            surface.astype(np.float32), mask_uint8, 3, cv2.INPAINT_TELEA
        )
        
    elif method.lower() == 'nearest':
        # Nearest neighbor interpolation
        from scipy.interpolate import griddata
        
        y, x = np.mgrid[0:surface.shape[0], 0:surface.shape[1]]
        valid_points = ~mask
        points = np.column_stack([x[valid_points], y[valid_points]])
        values = surface[valid_points]
        
        grid_points = np.column_stack([x[mask], y[mask]])
        filled_values = griddata(points, values, grid_points, method='nearest')
        filled_surface[mask] = filled_values
        
    else:
        raise ValueError(f"Unknown gap filling method: {method}")
        
    return filled_surface


def compute_slope_aspect(surface, pixel_size=1.0):
    """
    Compute slope and aspect from surface.
    
    Args:
        surface (numpy.ndarray): Input surface
        pixel_size (float): Pixel size in meters
        
    Returns:
        tuple: (slope, aspect) in degrees
    """
    # Compute gradients
    grad_x, grad_y = np.gradient(surface, pixel_size)
    
    # Compute slope (in degrees)
    slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180 / np.pi
    
    # Compute aspect (in degrees)
    aspect = np.arctan2(grad_y, grad_x) * 180 / np.pi
    aspect = (aspect + 360) % 360  # Convert to 0-360 range
    
    return slope, aspect


def create_synthetic_dem(size=(256, 256), features='craters', noise_level=0.1):
    """
    Create synthetic DEM for testing.
    
    Args:
        size (tuple): DEM size (height, width)
        features (str): Type of features ('craters', 'ridges', 'random')
        noise_level (float): Noise level (0-1)
        
    Returns:
        numpy.ndarray: Synthetic DEM
    """
    height, width = size
    
    # Create base surface
    dem = np.zeros(size)
    
    if features.lower() == 'craters':
        # Add craters
        num_craters = int(height * width / 1000)
        
        for _ in range(num_craters):
            # Random crater parameters
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)
            radius = np.random.randint(5, 20)
            depth = np.random.uniform(0.5, 2.0)
            
            # Create crater
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Crater shape (parabolic)
            crater = depth * (1 - (distance / radius)**2)
            crater[distance > radius] = 0
            
            dem -= crater
            
    elif features.lower() == 'ridges':
        # Add ridges
        num_ridges = int(height / 20)
        
        for _ in range(num_ridges):
            # Random ridge parameters
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            length = np.random.randint(20, 50)
            angle = np.random.uniform(0, 2 * np.pi)
            height_ridge = np.random.uniform(1.0, 3.0)
            
            # Create ridge
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Draw line
            num_points = max(length, 1)
            x_points = np.linspace(start_x, end_x, num_points)
            y_points = np.linspace(start_y, end_y, num_points)
            
            for i in range(len(x_points)):
                x, y = int(x_points[i]), int(y_points[i])
                if 0 <= x < width and 0 <= y < height:
                    # Add Gaussian ridge
                    y_grid, x_grid = np.ogrid[:height, :width]
                    ridge = height_ridge * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / 10)
                    dem += ridge
                    
    elif features.lower() == 'random':
        # Random surface
        dem = np.random.normal(0, 1, size)
        
    # Add noise
    noise = np.random.normal(0, noise_level, size)
    dem += noise
    
    return dem 