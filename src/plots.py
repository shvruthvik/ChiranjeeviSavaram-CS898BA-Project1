import cv2
import os
import numpy as np
import random

# Read selected subset
selected_images = []

with open("subset1.txt", "r") as file:
    for line in file:
        selected_images.append(line.strip())

# Create folders
if not os.path.exists("plots"):
    os.makedirs("plots")

if not os.path.exists("readme_plots"):
    os.makedirs("readme_plots")


def add_title(img, title):
    # Make all images same size
    img = cv2.resize(img, (220, 160))

    # Add white space for title
    title_area = np.ones((35, 220, 3), dtype=np.uint8) * 255

    cv2.putText(
        title_area,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    return np.vstack((title_area, img))


plot_files = []

for file_name in selected_images:

    image_name = os.path.splitext(file_name)[0]

    input_img = cv2.imread("output_edges/" + image_name + "_input.jpg")
    sobel = cv2.imread("output_edges/" + image_name + "_sobel.jpg")
    laplacian = cv2.imread("output_edges/" + image_name + "_laplacian.jpg")
    canny = cv2.imread("output_edges/" + image_name + "_canny.jpg")
    prewitt = cv2.imread("output_edges/" + image_name + "_prewitt.jpg")

    if input_img is None:
        print("Missing image:", file_name)
        continue

    input_img = add_title(input_img, "Input")
    sobel = add_title(sobel, "Sobel")
    laplacian = add_title(laplacian, "Laplacian")
    canny = add_title(canny, "Canny")
    prewitt = add_title(prewitt, "Prewitt")

    # Put 5 images side by side
    final_plot = np.hstack((input_img, sobel, laplacian, canny, prewitt))

    output_name = "plots/" + image_name + "_plot.jpg"
    cv2.imwrite(output_name, final_plot)

    plot_files.append(output_name)

# Randomly select 6 plots for README
random_plots = random.sample(plot_files, 6)

count = 1

for plot in random_plots:
    img = cv2.imread(plot)
    cv2.imwrite("readme_plots/plot" + str(count) + ".jpg", img)
    count = count + 1

print("Plots created:", len(plot_files))
print("README plots selected: 6")