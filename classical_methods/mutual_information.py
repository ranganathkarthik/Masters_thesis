
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, color, transform
from skimage.registration import phase_cross_correlation
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio, normalized_mutual_information
from scipy.stats import entropy
from scipy.ndimage import shift as nd_shift


# Load images
rgb_path = '/home/krangana/2_Codes/1_Traditional_methods/Test_images/overexposed_RGB.JPG'
nir_path = '/home/krangana/2_Codes/1_Traditional_methods/Test_images/underexposed_NIR.jpg'
rgb_image = io.imread(rgb_path)  # Replace 'rgb_image.jpg' with the path to your RGB image
nir_image_gray = io.imread(nir_path)  # Replace 'nir_image.jpg' with the path to your NIR image

# rgb_image_gray = color.rgb2gray(rgb_image)

# Resize RGB image to match NIR image size
rgb_image_resized = transform.resize(rgb_image, (nir_image_gray.shape[0], nir_image_gray.shape[1]), anti_aliasing=True)
rgb_image_resized=transform.rotate(rgb_image_resized, 180)

# Convert to grayscale
rgb_image_gray = color.rgb2gray(rgb_image_resized)
# nir_image_gray = color.rgb2gray(nir_image)

# Stack NIR image into three channels
# nir_image_stacked = np.stack([nir_image_gray] * 3, axis=-1)

# Define phase cross-correlation optimization function
# def pcc_cost(shift):
#     shifted_rgb = np.roll(rgb_image_gray, shift.astype(int), axis=(0, 1))
#     shift_estimate, _, _ = phase_cross_correlation(shifted_rgb, nir_image_gray)
#     return np.linalg.norm(shift_estimate)  # Minimize shift error

def mutual_information_cost(shift_values):
    shifted_rgb = nd_shift(rgb_image_gray.astype(np.float64), shift=shift_values, mode='constant', cval=0)
    return -normalized_mutual_information(shifted_rgb.ravel(), nir_image_gray.ravel())  # Negative MI (minimization)

# Optimize shift using scipy.optimize
initial_shift = np.array([0, 0])  # Initial guess
# result = minimize(pcc_cost, initial_shift, method='Powell')
# best_shift = result.x.astype(int)
result = minimize(mutual_information_cost, initial_shift, method='Powell')
best_shift = result.x
best_shift = np.round(best_shift).astype(int)

# Apply best translation
# registered_rgb_image = np.roll(rgb_image_gray, best_shift, axis=(0, 1))
registered_rgb_image = nd_shift(rgb_image_gray.astype(np.float64), shift=best_shift, mode='constant', cval=0, order=1)

# Ensure pixel values are between 0 and 255
registered_rgb_image = (registered_rgb_image * 255).astype(np.uint8)

# Compute final phase correlation shift
detected_shift, _, _ = phase_cross_correlation(registered_rgb_image, nir_image_gray)

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
mse_val = mean_squared_error(registered_rgb_image, nir_image_gray)
psnr_val = peak_signal_noise_ratio(registered_rgb_image, nir_image_gray)
ssim_val = ssim(registered_rgb_image, nir_image_gray, channel_axis=0)
# ssim_val = ssim(registered_rgb_image, nir_image_gray)#, data_range=None, channel_axis=0)
ncc_val = normalized_cross_correlation(registered_rgb_image, nir_image_gray)
mi = normalized_mutual_information(registered_rgb_image.ravel(), nir_image_gray.ravel())
dice_coeff = 2 * np.sum(registered_rgb_image * nir_image_gray) / (np.sum(registered_rgb_image) + np.sum(nir_image_gray))

# Display the results
print(f"SSIM: {ssim_val}")
print(f"Mutual Information: {mi}")
print(f"Normalized Cross-Correlation (NCC): {ncc_val}")
print(f"MSE: {mse_val}")
print(f"PSNR: {psnr_val} dB")
print(f"Dice Coefficient: {dice_coeff}")

# Save the registered image


# Define output path
# img_name = rgb_path.split("\\")[-1]
# output_path = rf"D:\thesis\2_Codes\1_Traditional_methods\outputs\mutual_information\Registered_{img_name}"
# img_name = rgb_path.split("/")[-1]
# output_path = "/home/krangana/2_Codes/1_Traditional_methods/outputs/mutual_information/new/Registered_" + img_name

# Save using cv2
# cv2.imwrite(output_path, registered_rgb_image)

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(rgb_image_gray, cmap='gray')
ax[0].set_title('Original RGB Image')
ax[1].imshow(registered_rgb_image, cmap='gray')
ax[1].set_title('Registered RGB Image')
ax[2].imshow(nir_image_gray, cmap='gray')
ax[2].set_title('NIR Image')
plt.show()

print(f"Optimized Shift: {best_shift}")
print(f"Detected Shift After Registration: {detected_shift}")