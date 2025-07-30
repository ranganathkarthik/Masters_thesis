import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio, normalized_mutual_information

# Load the RGB and NIR images
rgb_path = '/home/krangana/2_Codes/1_Traditional_methods/Test_images/overexposed_RGB.JPG'
rgb = cv2.imread(rgb_path)  # Replace with your RGB image path
nir = cv2.imread('/home/krangana/2_Codes/1_Traditional_methods/Test_images/overexposed_NIR.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your NIR image path

# Downsize RGB to match NIR dimensions
nir_height, nir_width = nir.shape
rgb_resized = cv2.resize(rgb, (nir_width, nir_height), interpolation=cv2.INTER_AREA)

# Convert resized RGB to grayscale for template matching
gray_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)

# Define a template from the NIR image (center region)
template_size = 960  # Adjust based on your image size
y = (nir_height - template_size) // 2
x = (nir_width - template_size) // 2
nir_template = nir[y:y+template_size, x:x+template_size]

# Perform template matching on the grayscale RGB image
result = cv2.matchTemplate(gray_rgb, nir_template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

# Calculate the translation offset (dx, dy)
dx = x - max_loc[0]
dy = y - max_loc[1]

# Apply translation to align the RGB image with NIR
translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
aligned_rgb = cv2.warpAffine(rgb_resized, translation_matrix, (nir_width, nir_height))

# Convert the aligned RGB image to grayscale
aligned_rgb_gray = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)

# Optional: Crop to the overlapping region to remove black borders
aligned_rgb_cropped = aligned_rgb_gray[max(0, dy):min(nir_height, nir_height + dy),
                                      max(0, dx):min(nir_width, nir_width + dx)]
nir_cropped = nir[max(0, -dy):min(nir_height, nir_height - dy),
                 max(0, -dx):min(nir_width, nir_width - dx)]

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
ssim_val = ssim(aligned_rgb_gray, nir, channel_axis=0)
ncc_val = normalized_cross_correlation(aligned_rgb_gray, nir)
mi = normalized_mutual_information(aligned_rgb_gray.ravel(), nir.ravel())

print(f"SSIM: {ssim_val}")
print(f"Mutual Information: {mi}")
print(f"Normalized Cross-Correlation (NCC): {ncc_val}")

cv2.imshow('Aligned rgb', aligned_rgb_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img_name = rgb_path.split("/")[-1]
# # Save the aligned grayscale RGB image
# cv2.imwrite('/home/krangana/2_Codes/1_Traditional_methods/outputs/template_matching/new_new/Registered_' + img_name, aligned_rgb_gray)

# Optional: Save the cropped images
# cv2.imwrite('/home/krangana/2_Codes/1_Traditional_methods/outputs/template_matching/aligned_rgb_cropped_gray.jpg', aligned_rgb_cropped)
# cv2.imwrite('/home/krangana/2_Codes/1_Traditional_methods/outputs/template_matching/nir_cropped.jpg', nir_cropped)