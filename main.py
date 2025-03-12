import time

import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.morphology import skeletonize

from convolution import get_points_from_image, image_preprocess
from interpolation import linear_piecewise_interpolation, plot_piecewise_linear_interpolation, natural_cubic_splines, \
    plot_natural_cubic_splines, lagrange, plot_lagrange, polynomial_least_squares, plot_polynomial_least_squares


def image_specs_with_cv(img):
    img = cv.resize(img, (720, 560))
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 83, 13)
    mask = skeletonize(img)
    tmp = np.full_like(img, dtype=np.uint8, fill_value=255)
    tmp[mask] = 0
    cv.imshow('image', tmp)
    cv.waitKey(0)
    cv.destroyAllWindows()
    imgEdge = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    countours, hierarchy = cv.findContours(imgEdge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    imgEdge = cv.drawContours(img, countours[:-1], -1, (0, 255, 0), 3)
    cv.imshow('image', imgEdge)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread(f'./images/test (2).png', 0)
image_specs_with_cv(img)
tmp = image_preprocess(img)
cv.imshow('image', tmp)
cv.waitKey(0)
cv.destroyAllWindows()

# -----------------------------------------------
tmp = cv.flip(tmp, 0)
points = get_points_from_image(tmp)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), layout='tight')  # type:figure.Figure, axes.Axes

# # linear piecewise interpolation
interpolations = [
    ("Linear Piecewise Interpolation", linear_piecewise_interpolation, plot_piecewise_linear_interpolation),
    ("Natural Cubic Spline Interpolation", natural_cubic_splines, plot_natural_cubic_splines),
    ("Lagrange Interpolation", lagrange, plot_lagrange),
    ("Polynomial Least Squares", polynomial_least_squares, plot_polynomial_least_squares)
]

# Perform and time each interpolation method
for i, (title, interp_func, plot_func) in enumerate(interpolations):
    k = bin(i)[2:].zfill(2)  # Shortened binary representation
    start_time = time.time()  # Start timing
    degree = 6  # Default degree for methods not requiring it
    try:
        if title == "Polynomial Least Squares":
            result = interp_func(points, degree)
        else:
            result = interp_func(points)
        exec_time = time.time() - start_time  # Calculate execution time
        print(f"{title} Execution Time: {exec_time:.6f} seconds")

        # Plot the results if successful
        plot_func(points, result, ax[int(k[0]), int(k[1])])
        ax[int(k[0]), int(k[1])].set_title(title)
        ax[int(k[0]), int(k[1])].set_ylim(min(points[:, 1]) - 10, max(points[:, 1]) + 10)
    except Exception as e:
        print(f"Error in {title}: {e}")

# # Show the final plot with all interpolations
plt.tight_layout()
plt.show()
#
