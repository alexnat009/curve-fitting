from skimage.morphology import skeletonize
import cv2 as cv
import numpy as np


def softening_curve_column(img, inp=None):
    result = inp if inp is not None else np.full_like(img, dtype=np.uint8, fill_value=255)

    # choose unique non-zero indices from each column
    for i in range(img.shape[1]):
        non_zero_indices = np.where(img[:, i] != 0)
        if len(non_zero_indices[0]) != 0:
            mid = non_zero_indices[0][len(non_zero_indices[0]) // 2]
            result[mid, i] = 0
    return result


def softening_curve_row(img, inp=None):
    result = inp if inp is not None else np.full_like(img, dtype=np.uint8, fill_value=255)
    for i in range(img.shape[0]):
        non_zero_indices = np.where(img[i, :] != 0)
        if len(non_zero_indices[0]) != 0:
            mid = non_zero_indices[0][len(non_zero_indices[0]) // 2]
            result[i, mid] = 0
    return result


def discrete_image(img):
    result = np.full_like(img, dtype=np.uint8, fill_value=255)
    # result = softening_curve_row(img, result)
    result = softening_curve_column(img, result)

    # using skeletonize from skimage which is a thinning algorithm

    mask = skeletonize(img)
    result[mask] = 0

    # # equidistant nodes
    # tmp = np.full_like(result, dtype=np.uint8, fill_value=255)
    # for i in range(0, result.shape[1], 2):
    #     tmp[:, i] = result[:, i]

    # finding the first and last non-zero column, to determine the range of chebyshev nodes
    # a = next((i for i in range(0, result.shape[0], 2) if not np.all(result[:, i])), None)
    # b = next((i for i in range(result.shape[0] - 1, 0, -2) if np.any(result[:, i])), 0)
    #
    # nodes = np.unique(np.floor(chebyshev_nodes(a, b, 100)).astype(np.int_))
    # tmp = np.full_like(result, dtype=np.uint8, fill_value=255)
    # for i in nodes:
    #     tmp[:, i] = result[:, i]

    return result


def chebyshev_nodes(a, b, k):
    x = np.zeros(k)
    for i in range(k):
        x[i] = (a + b) / 2 + (b - a) / 2 * np.cos((2 * (k - i) + 1) * np.pi / (2 * k))

    return x


def convolution(img: np.ndarray) -> np.ndarray:
    """
    :param img: represents the input image, expected as a NumPy array.
    :return: the convolved image.
    """

    if len(np.unique(img)) != 2:
        img = cv.bilateralFilter(img, 8, 108, 43)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 83, 13)
    return img


def get_points_from_image(img):
    ind = np.where(img == 0)
    points = np.array([ind[1], ind[0]]).T
    points = np.array(sorted(points, key=lambda x: x[0]))
    return points


def image_preprocess(img):
    img = cv.resize(img, (720, 560))
    img = convolution(img)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img = discrete_image(img)
    return img
