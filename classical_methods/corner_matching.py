import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio, normalized_mutual_information
from scipy.stats import entropy

rgb_path = '/home/krangana/2_Codes/1_Traditional_methods/Test_images/properexposure_RGB.JPG'
nir_path = '/home/krangana/2_Codes/1_Traditional_methods/Test_images/properexposure_NIR.jpg'

# Load RGB and NIR images
rgb_img = cv2.imread(rgb_path)
nir_img = cv2.imread(nir_path)

def increase_brightness(image, alpha=1.0, beta=20):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

nir_img= increase_brightness(nir_img, alpha=1.88, beta=64.44)

# rgb_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_140458_0550_RGB.JPG')
# nir_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_140458_0550_NIR.jpg')

# rgb_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_160101_002800_0180_RGB.JPG')
# nir_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_160101_002800_0180_NIR.jpg')

# Convert images to grayscale
rgb_gray_fullSize = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
rgb_gray = cv2.resize(rgb_gray_fullSize, (960, 1280), interpolation= cv2.INTER_LINEAR)
nir_gray = cv2.cvtColor(nir_img, cv2.COLOR_BGR2GRAY)

# Detect corners using Shi-Tomasi corner detector
max_corners = 200
quality_level = 0.2
min_distance = 25
rgb_corners = cv2.goodFeaturesToTrack(rgb_gray, max_corners, quality_level, min_distance)
nir_corners = cv2.goodFeaturesToTrack(nir_gray, max_corners, quality_level, min_distance)

# Convert corners to float32
rgb_corners = np.float32(rgb_corners)
nir_corners = np.float32(nir_corners)

# Find corresponding points between RGB and NIR images
matches = []
for rgb_corner in rgb_corners:
    min_distance = float('inf')
    min_nir_corner = None
    for nir_corner in nir_corners:
        distance = np.linalg.norm(nir_corner - rgb_corner)
        if distance < min_distance:
            min_distance = distance
            min_nir_corner = nir_corner
    if min_distance < 10 and min_nir_corner is not None:
        matches.append((rgb_corner, min_nir_corner))
    else:
        print("No match found for RGB corner:", rgb_corner)

# Estimate transformation using RANSAC
src_points = np.float32([m[0] for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([m[1] for m in matches]).reshape(-1, 1, 2)
transform_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Warp RGB image to align with NIR image
registered_image = cv2.warpPerspective(rgb_img, transform_matrix, (nir_img.shape[1], nir_img.shape[0]))

# Display registered image
cv2.imshow(registered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute comparison metrics
mse_val = mean_squared_error(registered_image, nir_gray)
psnr_val = peak_signal_noise_ratio(registered_image, nir_gray)
ssim_val = ssim(registered_image, nir_gray, channel_axis=0)
mutual_info = entropy(registered_image, nir_gray)
mi = normalized_mutual_information(registered_image.ravel(), nir_gray.ravel())
dice_coeff = 2 * np.sum(registered_image * nir_gray) / (np.sum(nir_gray) + np.sum(nir_gray))

# Display the results
print(f"MSE: {mse_val}")
print(f"PSNR: {psnr_val} dB")
print(f"SSIM: {ssim_val}")
print(f"Mutual Information: {mi}")
print(f"Dice Coefficient: {dice_coeff}")

# Save the registered image
# img_name = rgb_path.split("\\")[-1]
# output_path = rf"D:\thesis\2_Codes\1_Traditional_methods\outputs\corner_matching\Registered_{img_name}"
# # output_path_2= rf"D:\thesis\2_Codes\1_Traditional_methods\outputs\point_matching\keypts_rgb_nir_{img_name}"
# cv2.imwrite(output_path, registered_image)
# # cv2.imwrite(output_path_2, matched_img)