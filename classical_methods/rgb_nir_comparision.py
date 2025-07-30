import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, normalized_mutual_information


rgb = cv2.imread('/home/krangana/2_Codes/1_Traditional_methods/Test_images/underexposed_RGB.JPG')  # Replace with your RGB image path
nir = cv2.imread('/home/krangana/2_Codes/1_Traditional_methods/Test_images/underexposed_NIR.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your NIR image path

# Downsize RGB to match NIR dimensions
nir_height, nir_width = nir.shape
rgb_resized = cv2.resize(rgb, (nir_width, nir_height), interpolation=cv2.INTER_AREA)

# Convert resized RGB to grayscale for template matching
gray_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)

#NCC
def normalized_cross_correlation(img1, img2):
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)

    # Compute cross-correlation
    cross_corr = np.correlate(img1_norm.flatten(), img2_norm.flatten(), mode='valid')

    # Normalize cross-correlation
    ncc = cross_corr / (np.linalg.norm(img1_norm) * np.linalg.norm(img2_norm))

    return ncc

# Compute comparison metrics
ssim_val = ssim(gray_rgb, nir, channel_axis=0)
ncc_val = normalized_cross_correlation(gray_rgb, nir)
mi = normalized_mutual_information(gray_rgb.ravel(), nir.ravel())

print(f"SSIM: {ssim_val}")
print(f"Mutual Information: {mi}")
print(f"Normalized Cross-Correlation (NCC): {ncc_val}")