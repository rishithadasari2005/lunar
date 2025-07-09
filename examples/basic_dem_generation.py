"""
Basic Lunar DEM Generation Example

This script demonstrates the basic usage of the lunar DEM generation system
using synthetic lunar images.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunar_dem import LunarDEMGenerator
from preprocessing import ImagePreprocessor
from utils import create_synthetic_dem


def main():
    """Main function demonstrating basic DEM generation."""
    
    print("=== Lunar DEM Generation Example ===")
    print("Creating synthetic lunar image...")
    
    # Create synthetic lunar image
    preprocessor = ImagePreprocessor()
    synthetic_image = preprocessor.create_synthetic_lunar_image(
        size=(256, 256), 
        crater_density=0.15
    )
    
    # Save synthetic image for reference
    import cv2
    cv2.imwrite('synthetic_lunar_image.jpg', synthetic_image)
    print("Synthetic lunar image saved as 'synthetic_lunar_image.jpg'")
    
    # Initialize DEM generator with Hapke photometric model
    print("\nInitializing DEM generator...")
    generator = LunarDEMGenerator(
        photometric_model="hapke",
        w=0.12,  # Single scattering albedo
        b=0.5,   # Backscatter parameter
        c=0.1,   # Forward scatter parameter
        h=0.1    # Opposition effect parameter
    )
    
    # Generate DEM
    print("Generating DEM from synthetic lunar image...")
    dem = generator.generate_dem(
        'synthetic_lunar_image.jpg',
        resolution=256,
        smoothing_factor=0.1,
        noise_reduction=True
    )
    
    print(f"DEM generated successfully! Shape: {dem.shape}")
    
    # Save DEM
    generator.save_dem(dem, 'generated_dem.npy')
    print("DEM saved as 'generated_dem.npy'")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 2D surface plot
    generator.plot_surface(dem, title="Generated Lunar DEM")
    plt.savefig('dem_surface_plot.png', dpi=300, bbox_inches='tight')
    print("Surface plot saved as 'dem_surface_plot.png'")
    
    # Contour plot
    generator.plot_contours(dem, levels=15, title="Lunar DEM Contours")
    plt.savefig('dem_contour_plot.png', dpi=300, bbox_inches='tight')
    print("Contour plot saved as 'dem_contour_plot.png'")
    
    # 3D visualization
    generator.visualize_3d(dem, save_path='dem_3d_visualization.html')
    print("3D visualization saved as 'dem_3d_visualization.html'")
    
    # Validate DEM
    print("\nValidating DEM...")
    validation_metrics = generator.validate_dem(dem)
    
    print("\nValidation Results:")
    for metric, value in validation_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Generate validation report
    report = generator.validator.generate_validation_report()
    print(f"\nValidation Report:\n{report}")
    
    # Save validation report
    with open('validation_report.txt', 'w') as f:
        f.write(report)
    print("Validation report saved as 'validation_report.txt'")
    
    print("\n=== Example completed successfully! ===")
    print("Generated files:")
    print("  - synthetic_lunar_image.jpg")
    print("  - generated_dem.npy")
    print("  - dem_surface_plot.png")
    print("  - dem_contour_plot.png")
    print("  - dem_3d_visualization.html")
    print("  - validation_report.txt")


if __name__ == "__main__":
    main() 