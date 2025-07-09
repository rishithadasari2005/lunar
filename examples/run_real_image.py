"""
Run DEM generation on a real lunar image.
Place your real lunar image (e.g., real_lunar_image.jpg) in the examples/ directory before running this script.
"""

import sys
import os
from lunar_dem import LunarDEMGenerator

# Path to your real lunar image
image_path = "examples/real_lunar_image.jpg"  # Change this to your image filename

if not os.path.exists(image_path):
    print(f"Image not found: {image_path}\nPlease place your real lunar image in the examples/ directory and update the filename if needed.")
    sys.exit(1)

# Initialize the DEM generator
generator = LunarDEMGenerator(photometric_model="hapke", w=0.12, b=0.5, c=0.1, h=0.1)

# Generate DEM
dem = generator.generate_dem(image_path, resolution=512, smoothing_factor=0.1, noise_reduction=True)

# Visualize and save results
generator.plot_surface(dem, title="DEM from Real Lunar Image")
generator.visualize_3d(dem, save_path="real_lunar_dem_3d.html")
generator.save_dem(dem, "real_lunar_dem.npy")

print("DEM generation from real image complete! Results saved as 'real_lunar_dem.npy' and 'real_lunar_dem_3d.html'.") 