# Interpolation on Hand-Drawn Images of Functions

## Description

This project implements various interpolation techniques to analyze hand-drawn images of mathematical functions. The image is preprocessed using skeletonization and edge detection techniques, and points are extracted from the image for interpolation. The following interpolation methods are implemented:

1. **Linear Piecewise Interpolation**
2. **Natural Cubic Splines Interpolation**
3. **Lagrange Interpolation**
4. **Polynomial Least Squares Interpolation**

Each method is evaluated by measuring its execution time, and the results are visualized using `matplotlib`. The image processing is performed using `OpenCV` and `scikit-image` libraries.

## Features

- **Image Preprocessing**: The image is resized, thresholded, skeletonized, and processed to extract points.
- **Interpolation Methods**: Linear piecewise, natural cubic splines, Lagrange, and polynomial least squares interpolations.
- **Visualization**: The results of each interpolation method are plotted in a 2x2 grid.
- **Execution Time Measurement**: The time taken by each interpolation method is printed for performance analysis.

## Requirements

- Python 3.x
- Required Python libraries:
  - `matplotlib`
  - `numpy`
  - `opencv-python` (`cv2`)
  - `scikit-image`

You can install these dependencies using `pip`:

```bash
pip install matplotlib numpy opencv-python scikit-image
```

## Usage

1. **Prepare the image**: Ensure the hand-drawn image of the function is placed in the `./images/` folder. The image should be in a `.png` format.

2. **Run the script**:

```bash
python main.py
```

3. **View the Results**: The script will process the image, apply the interpolation methods, and display the resulting plots.

## Code Overview

- `main.py`: The main script that handles image preprocessing, point extraction, interpolation, and visualization.
- `convolution.py`: Contains the functions for point extraction from the image (`get_points_from_image`) and image preprocessing (`image_preprocess`).
- `interpolation.py`: Contains the implementations for the various interpolation methods and their respective plotting functions:
  - `linear_piecewise_interpolation`
  - `natural_cubic_splines`
  - `lagrange`
  - `polynomial_least_squares`

## How It Works

1. **Image Processing**:
   - The image is resized and thresholded to create a binary image.
   - The image is skeletonized to thin out the lines, then the contours of the function are extracted.
   
2. **Point Extraction**:
   - Points are extracted from the skeletonized image, which represent the underlying function.

3. **Interpolation**:
   - The extracted points are used for interpolation using four different techniques:
     - **Linear Piecewise Interpolation**: Straight lines between consecutive points.
     - **Natural Cubic Splines**: Smooth curves that pass through all the points.
     - **Lagrange Interpolation**: Polynomial interpolation based on the Lagrange formula.
     - **Polynomial Least Squares**: A polynomial fit that minimizes the error.

4. **Visualization**:
   - The results of each interpolation method are plotted in a 2x2 grid.

5. **Performance**:
   - The execution time for each interpolation method is measured and printed.

## Example Output

After running the script, you will see a 2x2 grid with plots corresponding to the four interpolation methods. Each plot will show the points extracted from the image and the resulting interpolation curve.

## Contributing

Feel free to fork this project, make improvements, or create pull requests. If you find any bugs or have suggestions for new features, please open an issue on GitHub.


# NOTE
This project is not completed, needs further improvement
