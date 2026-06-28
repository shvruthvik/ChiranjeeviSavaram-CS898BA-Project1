import cv2
import os

# Loading the normalized image
image = cv2.imread("hw2_output/normalized_color.jpg")

if image is None:
    print("Error: Could not load normalized image.")
    exit()

# Converting the normalized image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Implementing Otsu's global thresholding
otsu_value, otsu_mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Implementing adaptive Gaussian thresholding
adaptive_mask = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,
    5
)

# Creating segmented foreground images using the masks
otsu_segmented = cv2.bitwise_and(image, image, mask=otsu_mask)
adaptive_segmented = cv2.bitwise_and(image, image, mask=adaptive_mask)

# Saving masks and segmented outputs
cv2.imwrite("hw2_output/otsu_mask.jpg", otsu_mask)
cv2.imwrite("hw2_output/otsu_segmented.jpg", otsu_segmented)

cv2.imwrite("hw2_output/adaptive_mask.jpg", adaptive_mask)
cv2.imwrite("hw2_output/adaptive_segmented.jpg", adaptive_segmented)

print("Threshold segmentation completed.")
print("Otsu threshold value:", otsu_value)
print("Saving Otsu and adaptive masks and segmented images.")