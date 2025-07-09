"""
DEM Validation and Quality Assessment

This module provides tools for validating generated DEMs against reference data
and assessing quality metrics for lunar surface reconstruction.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class DEMValidator:
    """
    DEM validation and quality assessment class.
    
    Provides methods for:
    - Comparing DEMs with reference data
    - Computing quality metrics
    - Statistical analysis
    - Error assessment
    """
    
    def __init__(self):
        """Initialize the DEM validator."""
        self.metrics = {}
        
    def validate(self, dem, reference_dem=None, mask=None):
        """
        Validate DEM against reference data.
        
        Args:
            dem (numpy.ndarray): Generated DEM
            reference_dem (numpy.ndarray): Reference DEM for comparison
            mask (numpy.ndarray): Mask for regions to validate
            
        Returns:
            dict: Validation metrics
        """
        self.metrics = {}
        
        # Basic DEM statistics
        self.metrics.update(self._compute_basic_statistics(dem, mask))
        
        # Validate against reference if provided
        if reference_dem is not None:
            self.metrics.update(self._compare_with_reference(dem, reference_dem, mask))
            
        # Quality assessment
        self.metrics.update(self._assess_quality(dem, mask))
        
        return self.metrics
    
    def _compute_basic_statistics(self, dem, mask=None):
        """
        Compute basic statistics for the DEM.
        
        Args:
            dem (numpy.ndarray): DEM array
            mask (numpy.ndarray): Optional mask
            
        Returns:
            dict: Basic statistics
        """
        if mask is not None:
            dem_masked = dem[mask]
        else:
            dem_masked = dem.flatten()
            
        stats_dict = {
            'mean_elevation': np.mean(dem_masked),
            'std_elevation': np.std(dem_masked),
            'min_elevation': np.min(dem_masked),
            'max_elevation': np.max(dem_masked),
            'median_elevation': np.median(dem_masked),
            'elevation_range': np.max(dem_masked) - np.min(dem_masked),
            'total_pixels': len(dem_masked),
            'valid_pixels': np.sum(~np.isnan(dem_masked))
        }
        
        return stats_dict
    
    def _compare_with_reference(self, dem, reference_dem, mask=None):
        """
        Compare DEM with reference data.
        
        Args:
            dem (numpy.ndarray): Generated DEM
            reference_dem (numpy.ndarray): Reference DEM
            mask (numpy.ndarray): Optional mask
            
        Returns:
            dict: Comparison metrics
        """
        # Ensure same shape
        if dem.shape != reference_dem.shape:
            raise ValueError("DEM and reference DEM must have the same shape")
            
        # Apply mask if provided
        if mask is not None:
            dem_masked = dem[mask]
            ref_masked = reference_dem[mask]
        else:
            dem_masked = dem.flatten()
            ref_masked = reference_dem.flatten()
            
        # Remove NaN values
        valid_mask = ~(np.isnan(dem_masked) | np.isnan(ref_masked))
        dem_valid = dem_masked[valid_mask]
        ref_valid = ref_masked[valid_mask]
        
        if len(dem_valid) == 0:
            return {'error': 'No valid data for comparison'}
            
        # Compute error metrics
        errors = dem_valid - ref_valid
        
        comparison_metrics = {
            'rmse': np.sqrt(mean_squared_error(ref_valid, dem_valid)),
            'mae': mean_absolute_error(ref_valid, dem_valid),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'median_error': np.median(errors),
            'correlation': np.corrcoef(dem_valid, ref_valid)[0, 1],
            'r_squared': self._compute_r_squared(ref_valid, dem_valid),
            'bias': np.mean(errors),
            'relative_error': np.mean(np.abs(errors) / (np.abs(ref_valid) + 1e-6)) * 100
        }
        
        return comparison_metrics
    
    def _assess_quality(self, dem, mask=None):
        """
        Assess DEM quality using various metrics.
        
        Args:
            dem (numpy.ndarray): DEM array
            mask (numpy.ndarray): Optional mask
            
        Returns:
            dict: Quality metrics
        """
        if mask is not None:
            dem_masked = dem[mask]
        else:
            dem_masked = dem.flatten()
            
        # Remove NaN values
        dem_valid = dem_masked[~np.isnan(dem_masked)]
        
        if len(dem_valid) == 0:
            return {'quality_error': 'No valid data for quality assessment'}
            
        # Compute gradients for smoothness assessment
        grad_x, grad_y = np.gradient(dem)
        
        if mask is not None:
            grad_x = grad_x[mask]
            grad_y = grad_y[mask]
            
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        quality_metrics = {
            'smoothness': np.mean(grad_magnitude),
            'gradient_std': np.std(grad_magnitude),
            'max_gradient': np.max(grad_magnitude),
            'data_completeness': len(dem_valid) / len(dem_masked),
            'surface_roughness': np.std(dem_valid),
            'signal_to_noise': np.mean(dem_valid) / (np.std(dem_valid) + 1e-6)
        }
        
        return quality_metrics
    
    def _compute_r_squared(self, y_true, y_pred):
        """
        Compute R-squared coefficient of determination.
        
        Args:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values
            
        Returns:
            float: R-squared value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    
    def plot_validation_results(self, dem, reference_dem=None, save_path=None):
        """
        Plot validation results.
        
        Args:
            dem (numpy.ndarray): Generated DEM
            reference_dem (numpy.ndarray): Reference DEM
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot DEM
        im1 = axes[0, 0].imshow(dem, cmap='terrain')
        axes[0, 0].set_title('Generated DEM')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot reference DEM if available
        if reference_dem is not None:
            im2 = axes[0, 1].imshow(reference_dem, cmap='terrain')
            axes[0, 1].set_title('Reference DEM')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Plot difference
            diff = dem - reference_dem
            im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-np.std(diff), vmax=np.std(diff))
            axes[1, 0].set_title('Difference (Generated - Reference)')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Scatter plot
            valid_mask = ~(np.isnan(dem.flatten()) | np.isnan(reference_dem.flatten()))
            dem_flat = dem.flatten()[valid_mask]
            ref_flat = reference_dem.flatten()[valid_mask]
            
            axes[1, 1].scatter(ref_flat, dem_flat, alpha=0.5, s=1)
            axes[1, 1].plot([ref_flat.min(), ref_flat.max()], 
                           [ref_flat.min(), ref_flat.max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Reference DEM')
            axes[1, 1].set_ylabel('Generated DEM')
            axes[1, 1].set_title('Scatter Plot')
            
        else:
            # Plot histogram
            axes[0, 1].hist(dem.flatten(), bins=50, alpha=0.7)
            axes[0, 1].set_title('DEM Histogram')
            axes[0, 1].set_xlabel('Elevation')
            axes[0, 1].set_ylabel('Frequency')
            
            # Plot gradient magnitude
            grad_x, grad_y = np.gradient(dem)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            im3 = axes[1, 0].imshow(grad_mag, cmap='hot')
            axes[1, 0].set_title('Gradient Magnitude')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Plot 3D surface
            from mpl_toolkits.mplot3d import Axes3D
            y, x = np.mgrid[0:dem.shape[0]:50j, 0:dem.shape[1]:50j]
            axes[1, 1].remove()
            ax3d = fig.add_subplot(2, 2, 4, projection='3d')
            surf = ax3d.plot_surface(x, y, dem[::dem.shape[0]//50, ::dem.shape[1]//50], 
                                   cmap='terrain', alpha=0.8)
            ax3d.set_title('3D Surface')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def generate_validation_report(self, output_path=None):
        """
        Generate a comprehensive validation report.
        
        Args:
            output_path (str): Path to save report
            
        Returns:
            str: Report text
        """
        if not self.metrics:
            return "No validation metrics available. Run validate() first."
            
        report = "LUNAR DEM VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Basic statistics
        if 'mean_elevation' in self.metrics:
            report += "BASIC STATISTICS:\n"
            report += "-" * 20 + "\n"
            report += f"Mean elevation: {self.metrics['mean_elevation']:.3f}\n"
            report += f"Std elevation: {self.metrics['std_elevation']:.3f}\n"
            report += f"Elevation range: {self.metrics['elevation_range']:.3f}\n"
            report += f"Valid pixels: {self.metrics['valid_pixels']}/{self.metrics['total_pixels']}\n\n"
        
        # Comparison metrics
        if 'rmse' in self.metrics:
            report += "COMPARISON METRICS:\n"
            report += "-" * 20 + "\n"
            report += f"RMSE: {self.metrics['rmse']:.3f}\n"
            report += f"MAE: {self.metrics['mae']:.3f}\n"
            report += f"Correlation: {self.metrics['correlation']:.3f}\n"
            report += f"R-squared: {self.metrics['r_squared']:.3f}\n"
            report += f"Bias: {self.metrics['bias']:.3f}\n"
            report += f"Relative error: {self.metrics['relative_error']:.2f}%\n\n"
        
        # Quality metrics
        if 'smoothness' in self.metrics:
            report += "QUALITY METRICS:\n"
            report += "-" * 20 + "\n"
            report += f"Smoothness: {self.metrics['smoothness']:.3f}\n"
            report += f"Surface roughness: {self.metrics['surface_roughness']:.3f}\n"
            report += f"Signal-to-noise ratio: {self.metrics['signal_to_noise']:.3f}\n"
            report += f"Data completeness: {self.metrics['data_completeness']:.3f}\n\n"
        
        # Overall assessment
        report += "OVERALL ASSESSMENT:\n"
        report += "-" * 20 + "\n"
        
        if 'rmse' in self.metrics:
            if self.metrics['rmse'] < 0.1:
                report += "✓ Excellent accuracy (RMSE < 0.1)\n"
            elif self.metrics['rmse'] < 0.3:
                report += "✓ Good accuracy (RMSE < 0.3)\n"
            elif self.metrics['rmse'] < 0.5:
                report += "○ Acceptable accuracy (RMSE < 0.5)\n"
            else:
                report += "✗ Poor accuracy (RMSE >= 0.5)\n"
                
        if 'correlation' in self.metrics:
            if self.metrics['correlation'] > 0.9:
                report += "✓ Excellent correlation (> 0.9)\n"
            elif self.metrics['correlation'] > 0.7:
                report += "✓ Good correlation (> 0.7)\n"
            elif self.metrics['correlation'] > 0.5:
                report += "○ Acceptable correlation (> 0.5)\n"
            else:
                report += "✗ Poor correlation (<= 0.5)\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
                
        return report
    
    def validate_lunar_features(self, dem, feature_locations=None):
        """
        Validate specific lunar features (craters, ridges, etc.).
        
        Args:
            dem (numpy.ndarray): Generated DEM
            feature_locations (list): List of feature coordinates [(x, y), ...]
            
        Returns:
            dict: Feature validation metrics
        """
        feature_metrics = {}
        
        if feature_locations is None:
            return feature_metrics
            
        for i, (x, y) in enumerate(feature_locations):
            if 0 <= x < dem.shape[1] and 0 <= y < dem.shape[0]:
                # Extract local region around feature
                x_min = max(0, x - 10)
                x_max = min(dem.shape[1], x + 10)
                y_min = max(0, y - 10)
                y_max = min(dem.shape[0], y + 10)
                
                local_dem = dem[y_min:y_max, x_min:x_max]
                
                # Compute feature metrics
                feature_metrics[f'feature_{i}'] = {
                    'location': (x, y),
                    'local_mean': np.mean(local_dem),
                    'local_std': np.std(local_dem),
                    'local_range': np.max(local_dem) - np.min(local_dem),
                    'feature_height': dem[y, x]
                }
        
        return feature_metrics 