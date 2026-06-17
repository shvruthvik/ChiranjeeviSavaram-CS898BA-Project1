import os
import random

# Getting all jpg images from output folder
image_files = []

for file_name in os.listdir("output"):

    if file_name.endswith(".jpg"):
        image_files.append(file_name)

# Shuffling images randomly
random.shuffle(image_files)

# Splitting into 4 equal subsets
subset_size = len(image_files) // 4

subset1 = image_files[0:subset_size]
subset2 = image_files[subset_size:subset_size * 2]
subset3 = image_files[subset_size * 2:subset_size * 3]
subset4 = image_files[subset_size * 3:]

print("Subset 1:", len(subset1))
print("Subset 2:", len(subset2))
print("Subset 3:", len(subset3))
print("Subset 4:", len(subset4))