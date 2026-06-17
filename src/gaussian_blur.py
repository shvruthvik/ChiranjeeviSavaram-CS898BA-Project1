import cv2
import os

# Sigma values required in the assignment
sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# Find all original images in the output folder
image_files = []

for file_name in os.listdir("output"):

    # Only use jpg files that have not already been blurred
    if file_name.endswith(".jpg") and "_blur_" not in file_name:
        image_files.append(file_name)

# Apply Gaussian blur to every image
for file_name in image_files:

    img = cv2.imread("output/" + file_name)

    if img is None:
        continue

    # Remove .jpg from filename
    image_name = os.path.splitext(file_name)[0]

    # Create blurred versions using each sigma value
    for sigma in sigma_values:

        blurred_img = cv2.GaussianBlur(
            img,
            (0, 0),
            sigmaX=sigma
        )

        output_file = (
            "output/"
            + image_name
            + "_blur_"
            + str(sigma)
            + ".jpg"
        )

        cv2.imwrite(output_file, blurred_img)

print("Gaussian blur completed.")
print("Blurred images created:", len(image_files) * len(sigma_values))