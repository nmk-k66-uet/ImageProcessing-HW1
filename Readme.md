# Image Processing with OpenCV

This Python Project demonstrates various image processing technics which I learned from the INT3404E Image Processing course.

## HW1:

## Functions

The script includes the following functions:

- `load_image(image_path)`: Loads an image from the specified file path and converts it from BGR to RGB color space.
- `display_image(image, title="Image")`: Displays an image with a specified title.
- `grayscale_image(image)`: Converts an image to grayscale using the formula `p = 0.299R + 0.587G + 0.114B`.
- `save_image(image, output_path)`: Saves an image to the specified output path.
- `flip_image(image)`: Flips an image horizontally.
- `rotate_image(image, angle)`: Rotates an image by a specified angle in degrees.

## Usage

The script is executed as follows:

1. An image is loaded from file.
2. The original image is displayed.
3. The image is converted to grayscale and displayed.
4. The grayscale image is saved to file.
5. The grayscale image is flipped horizontally and displayed.
6. The flipped image is saved to file.
7. The grayscale image is rotated by 45 degrees and displayed.
8. The rotated image is saved to file.

## Dependencies

This script requires the following libraries:

- OpenCV
- Matplotlib
- NumPy
- os

## Note

Please ensure that the image file and the output directory exist in the correct path as specified in the script.
