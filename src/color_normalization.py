import cv2
import os

# Loading the original image
image = cv2.imread("images/alien.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()

# Creating output folder to save the normalized color image
output_folder = "hw2_output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# BGR format is used because OpenCV loads images in BGR format by default
blue, green, red = cv2.split(image)

# Applying histogram equalization to each channel
blue_equalized = cv2.equalizeHist(blue)
green_equalized = cv2.equalizeHist(green)
red_equalized = cv2.equalizeHist(red)

# Merging the equalized channels
normalized_image = cv2.merge((blue_equalized,green_equalized,red_equalized))

# Saving the normalized color image in the output folder
cv2.imwrite(
    os.path.join(output_folder, "normalized_color.jpg"),
    normalized_image
)

print("Multi-channel color normalization completed and saved as 'normalized_color.jpg' in the output folder.")