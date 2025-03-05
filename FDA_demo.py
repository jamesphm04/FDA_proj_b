import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import FDA_source_to_target_np

# Load images
im_src = Image.open("demo_images/source.png").convert('RGB')
im_trg = Image.open("demo_images/target.png").convert('RGB')

# Resize images to 1024x1024
im_src = im_src.resize((1024, 1024), Image.BICUBIC)
im_trg = im_trg.resize((1024, 1024), Image.BICUBIC)

# Convert images to NumPy arrays (float32)
im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

# Transpose to (C, H, W) format
im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

# Apply FDA domain adaptation
src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.01)

# Convert back to (H, W, C) format
src_in_trg = src_in_trg.transpose((1, 2, 0))

# Normalize the image data to 0-1 range
src_in_trg = np.clip(src_in_trg / 255.0, 0, 1)

# Ensure output directory exists
save_path = "demo_images/src_in_tar.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the result using Matplotlib
plt.imsave(save_path, src_in_trg)
print(f"Image saved at {save_path}")

# Alternative: Save using PIL (uncomment if needed)
# Image.fromarray((src_in_trg * 255).astype(np.uint8)).save(save_path)
