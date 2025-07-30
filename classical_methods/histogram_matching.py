import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio, normalized_mutual_information
from scipy.stats import entropy
# from sklearn.metrics import jaccard_score

# Load RGB and NIR images
rgb_path = "/home/krangana/2_Codes/1_Traditional_methods/Test_images/underexposed.jpg"
nir_path = "/home/krangana/2_Codes/1_Traditional_methods/Test_images/underexposed_NIR.jpg"
rgb_img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

# Resize RGB image to match NIR image size
rgb_img_resized = cv2.resize(rgb_img, (nir_img.shape[1], nir_img.shape[0]))

# Expand NIR image to 3 channels
nir_img_3ch = cv2.merge([nir_img, nir_img, nir_img])

# Compute histograms for the resized RGB and NIR images
hist_rgb, _ = np.histogram(rgb_img_resized.flatten(), 256, [0, 256])
hist_nir, _ = np.histogram(nir_img.flatten(), 256, [0, 256])

# Normalize histograms
hist_rgb_norm = hist_rgb / np.sum(hist_rgb)
hist_nir_norm = hist_nir / np.sum(hist_nir)

# Compute cumulative distribution functions (CDFs)
cdf_rgb = np.cumsum(hist_rgb_norm)
cdf_nir = np.cumsum(hist_nir_norm)

# Perform histogram matching
lut = np.interp(cdf_rgb, cdf_nir, np.arange(256))

# Apply transformation to RGB image
matched_rgb = lut[rgb_img_resized]

# Convert back to uint8
matched_rgb = np.uint8(matched_rgb)

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
mse_val = mean_squared_error(matched_rgb, nir_img)
psnr_val = peak_signal_noise_ratio(matched_rgb, nir_img)
ssim_val = ssim(matched_rgb, nir_img, channel_axis=0)
mutual_info = entropy(hist_rgb_norm, hist_nir_norm)
ncc_val = normalized_cross_correlation(matched_rgb, nir_img)
mi = normalized_mutual_information(matched_rgb.ravel(), nir_img.ravel())
matched_rgb_normalized = matched_rgb / np.max(matched_rgb)
nir_img_normalized = nir_img / np.max(nir_img)
dice_coeff = 2 * np.sum(matched_rgb_normalized * nir_img_normalized) / (np.sum(matched_rgb_normalized) + np.sum(nir_img_normalized))

# Display the results
print(f"SSIM: {ssim_val}")
print(f"Mutual Information: {mi}")
print(f"Normalized Cross-Correlation (NCC): {ncc_val}")
print(f"MSE: {mse_val}")
print(f"PSNR: {psnr_val} dB")
print(f"Dice Coefficient: {dice_coeff}")

# Save the registered image
# img_name = rgb_path.split("/")[-1]
# output_path = "/home/krangana/2_Codes/1_Traditional_methods/outputs/histogram_matching/new/Registered_" + img_name
# cv2.imwrite(output_path, matched_rgb)

# Display the original and registered images
plt.figure(figsize=(15, 8))
plt.subplot(1, 3, 1)
plt.imshow(rgb_img_resized, cmap='gray')#, cv2.COLOR_BGR2RGB))
plt.title('Resized RGB Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(nir_img, cmap='gray')#, cv2.COLOR_BGR2RGB))
plt.title('NIR Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(matched_rgb, cmap='gray')#, cv2.COLOR_BGR2RGB))
plt.title('Registered Image (RGB)')
plt.axis('off')

plt.show()