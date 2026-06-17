import cv2
import numpy as np

image_files = [
    "01_original.jpg",
    "02_grayscale.jpg",
    "03_binary.jpg",
    "04_hsv.jpg",
    "05_lab.jpg",
    "06_hls.jpg",
    "07_normalized_rgb.jpg"
]

for i in range(len(image_files)):

    img = cv2.imread("output/" + image_files[i])

    rows, cols = img.shape[:2]

    # Rotate image
    angle = 20 + (i * 15)

    rotation_matrix = cv2.getRotationMatrix2D(
        (cols / 2, rows / 2),
        angle,
        1
    )

    rotated_img = cv2.warpAffine(
        img,
        rotation_matrix,
        (cols, rows)
    )

    cv2.imwrite(
        "output/rotated_" + str(i + 1) + ".jpg",
        rotated_img
    )

    # Translate image
    x_shift = 10 + (i * 5)
    y_shift = 15 + (i * 5)

    translation_matrix = np.float32([
        [1, 0, x_shift],
        [0, 1, y_shift]
    ])

    translated_img = cv2.warpAffine(
        img,
        translation_matrix,
        (cols, rows)
    )

    cv2.imwrite(
        "output/translated_" + str(i + 1) + ".jpg",
        translated_img
    )

print("All affine transformations arecompleted.")