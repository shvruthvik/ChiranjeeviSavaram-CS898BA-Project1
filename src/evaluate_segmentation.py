import cv2
import numpy as np
import os

# Loading the images from output and images folders
original = cv2.imread("images/alien.jpg")
normalized = cv2.imread("hw2_output/normalized_color.jpg")

otsu_mask = cv2.imread("hw2_output/otsu_mask.jpg", 0)
adaptive_mask = cv2.imread("hw2_output/adaptive_mask.jpg", 0)
kmeans_mask = cv2.imread("hw2_output/kmeans_k3_mask.jpg", 0)

# Loading the manually created reference mask
reference_mask = cv2.imread("hw2_output/reference_mask.jpg", 0)

if original is None or normalized is None:
    print("Error: Original or normalized image not found.")
    exit()

if otsu_mask is None or adaptive_mask is None or kmeans_mask is None:
    print("Error: One or more segmentation masks are missing.")
    exit()

if reference_mask is None:
    print("Error: Reference mask not found.")
    exit()


def clean_mask(mask):
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def calculate_iou(predicted, reference):
    predicted = clean_mask(predicted)
    reference = clean_mask(reference)

    intersection = np.logical_and(predicted == 255, reference == 255)
    union = np.logical_or(predicted == 255, reference == 255)

    if np.sum(union) == 0:
        return 0

    return np.sum(intersection) / np.sum(union)


def calculate_dice(predicted, reference):
    predicted = clean_mask(predicted)
    reference = clean_mask(reference)

    intersection = np.logical_and(predicted == 255, reference == 255)

    predicted_area = np.sum(predicted == 255)
    reference_area = np.sum(reference == 255)

    if predicted_area + reference_area == 0:
        return 0

    return (2 * np.sum(intersection)) / (predicted_area + reference_area)


# Calculating IoU and Dice scores
otsu_iou = calculate_iou(otsu_mask, reference_mask)
adaptive_iou = calculate_iou(adaptive_mask, reference_mask)
kmeans_iou = calculate_iou(kmeans_mask, reference_mask)

otsu_dice = calculate_dice(otsu_mask, reference_mask)
adaptive_dice = calculate_dice(adaptive_mask, reference_mask)
kmeans_dice = calculate_dice(kmeans_mask, reference_mask)

print("Segmentation Evaluation Results:\n")
print("Otsu IoU:", round(otsu_iou, 4))
print("Otsu Dice:", round(otsu_dice, 4))
print()
print("Adaptive IoU:", round(adaptive_iou, 4))
print("Adaptive Dice:", round(adaptive_dice, 4))
print()
print("K-Means K=3 IoU:", round(kmeans_iou, 4))
print("K-Means K=3 Dice:", round(kmeans_dice, 4))


def mask_to_color(mask):
    return cv2.cvtColor(clean_mask(mask), cv2.COLOR_GRAY2BGR)


def add_title(img, title):
    img = cv2.resize(img, (220, 160))

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


# Create final comparison plot
original_plot = add_title(original, "Original")
normalized_plot = add_title(normalized, "Normalized")
reference_plot = add_title(mask_to_color(reference_mask), "Reference")
otsu_plot = add_title(mask_to_color(otsu_mask), "Otsu")
adaptive_plot = add_title(mask_to_color(adaptive_mask), "Adaptive")
kmeans_plot = add_title(mask_to_color(kmeans_mask), "KMeans K=3")

comparison_plot = np.hstack((
    original_plot,
    normalized_plot,
    reference_plot,
    otsu_plot,
    adaptive_plot,
    kmeans_plot
))

cv2.imwrite("hw2_output/final_segmentation_comparison.jpg", comparison_plot)

print()
print("Final comparison plot saved as hw2_output/final_segmentation_comparison.jpg")