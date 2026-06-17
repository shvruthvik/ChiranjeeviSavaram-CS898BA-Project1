# CS898BA Project 1

# Student Information

**Name:** Chiranjeevi Venkata Shiva Ruthvik Savaram
**WSU ID:** Y247K345

## Project Overview

This is Homework 1 in CS898BA – Image Analysis and Computer Vision. The objective of this assignment was to perform image analysis and image processing using Python and OpenCV. Various image processing techniques were applied to the provided image, including image statistics, color space conversions, histogram equalization, affine transformations, Gaussian blur, and edge detection. And we also created plots for images for each technique.

## Software and Libraries Used

* Python
* OpenCV
* NumPy
* SciPy

## How to Run the Program

1. Clone or download the project repository.
2. Open the project folder in VS Code.
3. Install the required libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

4. Place the homework image(alien.jpg) inside the `images` folder.
5. Run the commands in the following order in the terminal:

```bash
python src/image_statistics.py
python src/conversions.py
python src/affine_transformations.py
python src/gaussian_blur.py
python src/image_subsets.py
python src/edge_detection.py
python src/plots.py
```

## Tasks Completed

* Calculated image statistics for each RGB channel
* Converted the image to grayscale
* Created a binary image
* Converted the image to HSV, LAB, and HLS color spaces
* Applied histogram equalization to the V channel of the HSV image
* Converted the normalized image back to RGB
* Performed affine transformations using rotations and translations
* Applied Gaussian blur using multiple sigma values
* Created four random subsets containing 42 images each
* Applied Sobel edge detection
* Applied Laplacian edge detection
* Applied Canny edge detection
* Applied Prewitt edge detection
* Generated comparison plots for edge detection results

## Results

The project successfully generated all required images for the assignment. Different image processing techniques were applied to better understand how image characteristics change throughout the processing pipeline. The edge detection methods produced very different outputs, making it possible to compare their effectiveness on the provided image set.

## Gaussian Blur Analysis

Gaussian blur was applied using sigma values of 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, and 3.5.

As the sigma value increased, the images became smoother and less detailed. Lower sigma values preserved most image features while reducing minor noise. Higher sigma values produced stronger blurring effects and removed more image details. While larger sigma values helped reduce noise, they also made object boundaries less visible.

## Edge Detection Analysis

Four edge detection techniques were applied to the selected subset of images.

### Sobel

Sobel detected major edges and object boundaries clearly while maintaining a good balance between detail and noise reduction.

**Advantages**
* Produces strong edge responses
* Less sensitive to noise
* Easy to interpret

**Disadvantages**
* May miss very fine details
* Produces thicker edges than some methods

### Laplacian

Laplacian detected fine intensity changes and highlighted many image details.

**Advantages**
* Detects edges in all directions
* Captures fine image details

**Disadvantages**
* Sensitive to image noise
* Produces extra edge responses in some images

### Canny

Canny generated thin and precise edges through a multi-stage detection process.

**Advantages**
* Produces clean and well-defined edges
* Reduces false edge detections

**Disadvantages**
* Depends on threshold selection
* Some weaker edges may not be detected

### Prewitt

Prewitt produced results similar to Sobel but generally showed weaker edge responses.

**Advantages**
* Simple implementation
* Detects major edges reasonably well

**Disadvantages**
* More sensitive to noise than Sobel
* Edge responses are generally weaker

## Best Performing Method

Based on visual comparison of the generated outputs, Sobel provided the most useful and consistent edge detection results for this dataset. It highlighted the major object boundaries while reducing unnecessary noise. Although Canny produced cleaner edges in some cases, Sobel provided a better balance between edge visibility and detail preservation across the image set. Even though prewitt provided a similar amount of good images as Sobel, Sobel clearer and more visible images.

## Sample Comparison Plots

The following six plots were randomly selected from the forty-two generated comparison plots.

### Plot 1

![Plot 1](readme_plots/plot1.jpg)

### Plot 2

![Plot 2](readme_plots/plot2.jpg)

### Plot 3

![Plot 3](readme_plots/plot3.jpg)

### Plot 4

![Plot 4](readme_plots/plot4.jpg)

### Plot 5

![Plot 5](readme_plots/plot5.jpg)

### Plot 6

![Plot 6](readme_plots/plot6.jpg)

## Conclusion

This project demonstrated several fundamental image processing and computer vision techniques using OpenCV. The assignment included image statistics, color space conversions, histogram equalization, affine transformations, Gaussian blurring, subset creation, and edge detection. The generated outputs showed how different processing techniques affect image appearance and feature extraction. Overall, Sobel produced the most effective edge detection results for the selected image subset.