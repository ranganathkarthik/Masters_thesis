import cv2
import numpy as np
#from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import normalized_mutual_information
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio, normalized_mutual_information
# rgb_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_141741_0482_RGB.JPG')
# nir_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_141741_0482_NIR.jpg')

# rgb_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_140458_0550_RGB.JPG')
# nir_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_140458_0550_NIR.jpg')

#rgb_img = cv2.imread('/home/krangana/2_Codes/1_Traditional_methods/Test_images/IMG_160101_002800_0180_RGB.JPG')
rgb_path = '/home/krangana/2_Codes/1_Traditional_methods/Test_images/properexposure_RGB.JPG'
nir_path = '/home/krangana/2_Codes/1_Traditional_methods/Test_images/properexposure_NIR.jpg'
rgb_img = cv2.imread(rgb_path)
nir_img = cv2.imread(nir_path)
rgb_gray_fullSize = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
# rgb_gray = rgb_gray_fullSize
rgb_gray = cv2.resize(rgb_gray_fullSize, (nir_img.shape[1], nir_img.shape[0]), interpolation= cv2.INTER_LINEAR)
nir_gray = cv2.cvtColor(nir_img, cv2.COLOR_BGR2GRAY)
# Compute DFT(descrete fourier transform)
dft_rgb = np.fft.fft2(rgb_gray)
dft_nir = np.fft.fft2(nir_gray)
# Calculate cross-power spectrum
cross_power_spectrum = (dft_rgb) * np.conj(dft_nir)
# Compute phase correlation
phase_correlation = np.fft.ifft2(cross_power_spectrum / np.abs(cross_power_spectrum))
# Find peak location
peak_location = np.unravel_index(np.argmax(phase_correlation), phase_correlation.shape)

# Refine peak location
# search_radius = 10  # Define the search radius around the peak
# neighborhood = phase_correlation[max(0, peak_location[0]-search_radius):min(phase_correlation.shape[0], peak_location[0]+search_radius),
#                                  max(0, peak_location[1]-search_radius):min(phase_correlation.shape[1], peak_location[1]+search_radius)]
# refined_peak_location = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
# refined_peak_location = (refined_peak_location[0] + max(0, peak_location[0]-search_radius),
#                          refined_peak_location[1] + max(0, peak_location[1]-search_radius))
# Calculate translation
rows, cols = rgb_gray.shape
# translation_x = peak_location[1] - rows //2
# translation_y = peak_location[0] - cols //2
# translation_x = peak_location[1] - (cols // 2)
# translation_y = (peak_location[0] - (rows // 2))

if peak_location[1] > cols // 2:
    translation_x = peak_location[1] - cols
else:
    translation_x = peak_location[1]

if peak_location[0] > rows // 2:
    translation_y = peak_location[0] - rows
else:
    translation_y = peak_location[0]

print(f"Corrected Translation: ({translation_x}, {translation_y})")

#translate image
translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
registered_img = cv2.warpAffine(rgb_gray, translation_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

# Shift image
# registered_img = np.roll(rgb_gray, (translation_x, translation_y))

# Display registered images
#cv2.imshow('RGB_GRAY', rgb_gray)
#cv2.imshow('Registered Image', registered_rgb_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# # Load the registered and unregistered RGB image
# reg_img = cv2.imread('drive/MyDrive/Colab Notebooks/Output_images/regIMG_160101_002800_0180.jpg', cv2.IMREAD_GRAYSCALE)
# unreg_RGB_img = rgb_gray
# resize_unreg_RGB_img = cv2.resize(unreg_RGB_img, (960, 1280), interpolation= cv2.INTER_LINEAR)
# # cv2_imshow(resize_unreg_RGB_img)

# Compute SSIM
ssim_index = ssim(registered_img, nir_gray)

print(f"SSIM between registered RGB image and NIR image: {ssim_index}")
def normalized_cross_correlation(img1, img2):
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)

    # Compute cross-correlation
    cross_corr = np.correlate(img1_norm.flatten(), img2_norm.flatten(), mode='valid')

    # Normalize cross-correlation
    ncc = cross_corr / (np.linalg.norm(img1_norm) * np.linalg.norm(img2_norm))

    return ncc
# Compute NCC
ncc_value = normalized_cross_correlation(registered_img, rgb_gray)
print("Normalized Cross-Correlation between registered and unregistered RGB image:", ncc_value)
ncc_value = normalized_cross_correlation(registered_img, nir_gray)
print("Normalized Cross-Correlation between registered and unregistered NIR image:", ncc_value)
# Compute mutual information
mutual_info_value = normalized_mutual_information(registered_img.ravel(), rgb_gray.ravel())

# Compute comparison metrics
mse_val = mean_squared_error(registered_img, nir_gray)
psnr_val = peak_signal_noise_ratio(registered_img, nir_gray)
ssim_val = ssim(registered_img, nir_gray, channel_axis=0)


# Display the results
print(f"SSIM: {ssim_val}")
print(f"Mutual Information: {mutual_info_value}")
print(f"Normalized Cross-Correlation (NCC): {ncc_value}")
print(f"MSE: {mse_val}")
print(f"PSNR: {psnr_val} dB")
# print(f"Dice Coefficient: {dice_coeff}")

# Save the registered image
img_name = rgb_path.split("/")[-1]
output_path = "/home/krangana/2_Codes/1_Traditional_methods/outputs/fourier_based/new/Registered_" + img_name
cv2.imwrite(output_path, registered_img)

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(registered_img, cmap='gray')
ax[0].set_title('Registered Image')
ax[1].imshow(rgb_gray, cmap='gray')
ax[1].set_title('RGB Image')
plt.show()

print(f"Mutual Information: {mutual_info_value}")