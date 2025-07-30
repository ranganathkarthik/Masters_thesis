import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio, normalized_mutual_information
from scipy.stats import entropy

rgb_path = "/home/krangana/2_Codes/1_Traditional_methods/Test_images/overexposed_RGB.JPG"
nir_path = "/home/krangana/2_Codes/1_Traditional_methods/Test_images/overexposed_NIR.jpg"

# rgb_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_141741_0482_RGB.JPG')
# nir_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_141741_0482_NIR.jpg')
# rgb_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_140458_0550_RGB.JPG')
# nir_img = cv2.imread('drive/MyDrive/Colab Notebooks/Test_images/IMG_231107_140458_0550_NIR.jpg')
rgb_img = cv2.imread(rgb_path)
nir_img = cv2.imread(nir_path)
#img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
rgb_gray = cv2.resize(rgb_gray, (960, 1280), interpolation= cv2.INTER_LINEAR)
nir_gray = cv2.cvtColor(nir_img, cv2.COLOR_BGR2GRAY)

# Increase brightness of the NIR image
def increase_brightness(image, alpha=1.0, beta=20):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# nir_image_bright = increase_brightness(nir_image, alpha=2.0, beta=160)
nir_gray = increase_brightness(nir_gray, alpha=1.0, beta=80)
# Initiate ORB detector
orb = cv2.ORB_create(50)  #Registration works with at least 50 points

# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(rgb_gray, None)  #kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(nir_gray, None)
#Brute-Force matcher takes the descriptor of one feature in first set and is
#matched with all other features in second set using some distance calculation.
# create Matcher object

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
matched_img = cv2.drawMatches(rgb_gray,kp1, nir_gray, kp2, matches[:10], None)

#cv2_imshow(matched_img)
#cv2.waitKey(0)

plt.imshow(matched_img)
plt.axis("off")
plt.show()
points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
height, width = rgb_gray.shape
reg_img = cv2.warpPerspective(rgb_gray, h, (width, height))  #Applies a perspective transformation to an image.
print("Estimated homography : \n",  h)

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

# Save the registered image
# img_name = rgb_path.split("/")[-1]
# output_path = "/home/krangana/2_Codes/1_Traditional_methods/outputs/point_matching/new/Registered_" + img_name
# output_path_2= "/home/krangana/2_Codes/1_Traditional_methods/outputs/point_matching/new/keypts_rgb_nir_" + img_name
# cv2.imwrite(output_path, reg_img)
# cv2.imwrite(output_path_2, matched_img)

plt.imshow(reg_img)
plt.axis("off")
plt.show()

# Compute comparison metrics
mse_val = mean_squared_error(reg_img, nir_gray)
psnr_val = peak_signal_noise_ratio(reg_img, nir_gray)
ssim_val = ssim(reg_img, nir_gray, channel_axis=0)
ncc_val = normalized_cross_correlation(reg_img, nir_gray)
mutual_info = entropy(reg_img, nir_gray)
mi = normalized_mutual_information(reg_img.ravel(), nir_gray.ravel())
dice_coeff = 2 * np.sum(reg_img * nir_gray) / (np.sum(nir_gray) + np.sum(nir_gray))

# Display the results
print(f"SSIM: {ssim_val}")
print(f"Mutual Information: {mi}")
print(f"Normalized Cross-Correlation (NCC): {ncc_val}")
print(f"MSE: {mse_val}")
print(f"PSNR: {psnr_val} dB")
print(f"Dice Coefficient: {dice_coeff}")