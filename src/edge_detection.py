import cv2
import numpy as np
import os

# Read the subset that was selected for edge detection
selected_images = []

with open("subset1.txt", "r") as file:
    for line in file:
        selected_images.append(line.strip())

# Create output folder for edge detection results
if not os.path.exists("output_edges"):
    os.makedirs("output_edges")

for file_name in selected_images:

    img = cv2.imread("output/" + file_name)

    if img is None:
        print("Could not read:", file_name)
        continue

    image_name = os.path.splitext(file_name)[0]

    # Save the input image also, so we have before and after images
    cv2.imwrite("output_edges/" + image_name + "_input.jpg", img)

    # Convert image to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite("output_edges/" + image_name + "_sobel.jpg", sobel)

    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite("output_edges/" + image_name + "_laplacian.jpg", laplacian)

    # Canny
    canny = cv2.Canny(gray, 100, 200)
    cv2.imwrite("output_edges/" + image_name + "_canny.jpg", canny)

    # Prewitt
    prewitt_x_kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    prewitt_y_kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    prewitt_x = cv2.filter2D(gray, -1, prewitt_x_kernel)
    prewitt_y = cv2.filter2D(gray, -1, prewitt_y_kernel)
    prewitt = prewitt_x + prewitt_y
    cv2.imwrite("output_edges/" + image_name + "_prewitt.jpg", prewitt)

print("Edge detection completed.")
print("Images processed:", len(selected_images))
print("Edge images created:", len(selected_images) * 4)
print("Total images in output_edges should be:", len(selected_images) * 5)