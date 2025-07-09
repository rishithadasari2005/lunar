"""
Photometric Model Comparison Example

This script compares different photometric models (Hapke, Lommel-Seeliger, Lambertian)
for lunar DEM generation and analyzes their performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunar_dem import LunarDEMGenerator
from preprocessing import ImagePreprocessor
from photometric_models import HapkeModel, LommelSeeligerModel, LambertianModel


def compare_photometric_models():
    """Compare different photometric models."""
    
    print("=== Photometric Model Comparison ===")
    
    # Create synthetic lunar image
    preprocessor = ImagePreprocessor()
    synthetic_image = preprocessor.create_synthetic_lunar_image(
        size=(128, 128), 
        crater_density=0.2
    )
    
    # Save synthetic image
    import cv2
    cv2.imwrite('comparison_lunar_image.jpg', synthetic_image)
    
    # Define photometric models to compare
    models = {
        'Hapke': {
            'type': 'hapke',
            'params': {'w': 0.12, 'b': 0.5, 'c': 0.1, 'h': 0.1}
        },
        'Lommel-Seeliger': {
            'type': 'lommel_seeliger',
            'params': {'albedo': 0.12}
        },
        'Lambertian': {
            'type': 'lambertian',
            'params': {'albedo': 0.12}
        }
    }
    
    results = {}
    
    # Generate DEMs using different models
    for model_name, model_config in models.items():
        print(f"\nGenerating DEM using {model_name} model...")
        
        # Initialize generator with specific model
        generator = LunarDEMGenerator(
            photometric_model=model_config['type'],
            **model_config['params']
        )
        
        # Generate DEM
        dem = generator.generate_dem(
            'comparison_lunar_image.jpg',
            resolution=128,
            smoothing_factor=0.1,
            noise_reduction=True
        )
        
        # Validate DEM
        validation_metrics = generator.validate_dem(dem)
        
        results[model_name] = {
            'dem': dem,
            'metrics': validation_metrics,
            'generator': generator
        }
        
        print(f"  {model_name} DEM generated successfully!")
    
    # Compare results
    print("\n=== Model Comparison Results ===")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (model_name, result) in enumerate(results.items()):
        # Plot DEM
        im = axes[0, i].imshow(result['dem'], cmap='terrain')
        axes[0, i].set_title(f'{model_name} DEM')
        axes[0, i].set_xlabel('X Coordinate')
        axes[0, i].set_ylabel('Y Coordinate')
        plt.colorbar(im, ax=axes[0, i])
        
        # Plot histogram
        axes[1, i].hist(result['dem'].flatten(), bins=30, alpha=0.7)
        axes[1, i].set_title(f'{model_name} Elevation Distribution')
        axes[1, i].set_xlabel('Elevation')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('photometric_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'photometric_model_comparison.png'")
    
    # Print metrics comparison
    print("\nMetrics Comparison:")
    print("-" * 80)
    print(f"{'Model':<15} {'Mean Elev':<12} {'Std Elev':<12} {'Smoothness':<12} {'Roughness':<12}")
    print("-" * 80)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:<15} {metrics['mean_elevation']:<12.4f} "
              f"{metrics['std_elevation']:<12.4f} {metrics['smoothness']:<12.4f} "
              f"{metrics['surface_roughness']:<12.4f}")
    
    # Save detailed comparison report
    with open('photometric_comparison_report.txt', 'w') as f:
        f.write("PHOTOMETRIC MODEL COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name} MODEL:\n")
            f.write("-" * 20 + "\n")
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
            f.write("\n")
    
    print("\nDetailed comparison report saved as 'photometric_comparison_report.txt'")
    
    return results


def analyze_model_performance(results):
    """Analyze the performance of different models."""
    
    print("\n=== Performance Analysis ===")
    
    # Extract key metrics
    metrics_to_compare = ['mean_elevation', 'std_elevation', 'smoothness', 'surface_roughness']
    
    # Create performance comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, metric in enumerate(metrics_to_compare):
        row = i // 2
        col = i % 2
        
        model_names = list(results.keys())
        metric_values = [results[model]['metrics'][metric] for model in model_names]
        
        bars = axes[row, col].bar(model_names, metric_values)
        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axes[row, col].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                              f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance analysis plot saved as 'model_performance_analysis.png'")
    
    # Determine best model for each metric
    print("\nBest Model for Each Metric:")
    for metric in metrics_to_compare:
        best_model = min(results.keys(), 
                        key=lambda x: results[x]['metrics'][metric])
        best_value = results[best_model]['metrics'][metric]
        print(f"  {metric}: {best_model} ({best_value:.4f})")


def main():
    """Main function."""
    
    # Compare photometric models
    results = compare_photometric_models()
    
    # Analyze performance
    analyze_model_performance(results)
    
    print("\n=== Comparison completed successfully! ===")
    print("Generated files:")
    print("  - comparison_lunar_image.jpg")
    print("  - photometric_model_comparison.png")
    print("  - model_performance_analysis.png")
    print("  - photometric_comparison_report.txt")


if __name__ == "__main__":
    main() 