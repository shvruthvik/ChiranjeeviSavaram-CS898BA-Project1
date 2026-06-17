import cv2
import os

# Load the image
img = cv2.imread("images/alien.jpg")

if img is None:
    print("Image not found. Please check the image path.")
    exit()

# Make output folder if it does not exist
if not os.path.exists("output"):
    os.makedirs("output")

# Save original image
cv2.imwrite("output/01_original.jpg", img)

# Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/02_grayscale.jpg", gray)

# Binary image
# I used 127 as a middle threshold value
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("output/03_binary.jpg", binary)

# HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite("output/04_hsv.jpg", hsv)

# LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imwrite("output/05_lab.jpg", lab)

# HLS color space
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
cv2.imwrite("output/06_hls.jpg", hls)

# Histogram equalization on the V channel of HSV
h, s, v = cv2.split(hsv)

v_equalized = cv2.equalizeHist(v)

hsv_equalized = cv2.merge((h, s, v_equalized))

# Converting the normalized HSV image back to BGR
normalized_img = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)
cv2.imwrite("output/07_normalized_rgb.jpg", normalized_img)

print("Finished creating converted images.")
print("Total images saved: 7")

