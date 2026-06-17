import cv2
import numpy as np
from scipy.stats import mode, skew

# Loading the original image
img = cv2.imread("images/alien.jpg")

# Check if image loaded successfully
if img is None:
    print("Image could not be loaded.")
    exit()

# Separating image channels
blue = img[:, :, 0]
green = img[:, :, 1]
red = img[:, :, 2]

channels = [blue, green, red]
names = ["Blue", "Green", "Red"]

# Calculate statistics for each channel
for i in range(len(channels)):

    data = channels[i].flatten()

    print("\n" + names[i] + " Channel")
    print("------------------------")

    print("Minimum:", np.min(data))
    print("Maximum:", np.max(data))
    print("Average:", np.mean(data))
    print("Median:", np.median(data))
    print("Mode:", mode(data, keepdims=True).mode[0])
    print("Skew:", skew(data))
    print("Range:", np.max(data) - np.min(data))
    print("Standard Deviation:", np.std(data))
    print("Variance:", np.var(data))