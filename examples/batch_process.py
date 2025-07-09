"""
Batch process all .jpg images in the examples/ directory to generate DEMs.
"""

import glob
import os
from lunar_dem import LunarDEMGenerator

# Find all .jpg images in examples/
image_paths = glob.glob("examples/*.jpg")
output_dir = "output_dems"

if not image_paths:
    print("No .jpg images found in examples/. Please add images to batch process.")
else:
    generator = LunarDEMGenerator(photometric_model="hapke", w=0.12, b=0.5, c=0.1, h=0.1)
    generator.batch_process(image_paths, output_dir=output_dir, resolution=256, smoothing_factor=0.1, noise_reduction=True)
    print(f"Batch processing complete! DEMs saved in {output_dir}/") 