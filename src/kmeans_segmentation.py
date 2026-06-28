import cv2
import numpy as np
import os

# Loading normalized image
image = cv2.imread("hw2_output/normalized_color.jpg")

if image is None:
    print("Error: Could not load normalized image.")
    exit()

# Converting the normalized image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Testing K values from 3 to 5
k_values = [3, 4, 5]

for k in k_values:

    # Reshaping image for K-Means
    pixel_values = hsv_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-Means stopping criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Applying K-Means clustering
    compactness, labels, centers = cv2.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Converting labels back to image shape
    labels_image = labels.reshape((image.shape[0], image.shape[1]))

    value_channel_centers = centers[:, 2]
    figure_cluster = np.argmin(value_channel_centers)

    # Creating the binary mask for selected cluster
    mask = np.zeros(labels_image.shape, dtype=np.uint8)
    mask[labels_image == figure_cluster] = 255

    # Creating segmented foreground extraction
    segmented = cv2.bitwise_and(image, image, mask=mask)

    # Saving the output files
    cv2.imwrite("hw2_output/kmeans_k" + str(k) + "_mask.jpg", mask)
    cv2.imwrite("hw2_output/kmeans_k" + str(k) + "_segmented.jpg", segmented)

    print("K-Means completed for K =", k)
    print("Selected cluster:", figure_cluster)

print("K-Means segmentation finished.")