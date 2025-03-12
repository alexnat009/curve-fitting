import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure


def natural_cubic_splines(points):
    """
        The function constructs a coefficient matrix (mat) and a vector (y) used to solve
        for the coefficients of natural cubic splines. It ensures that the resulting
        cubic splines pass smoothly through the given points. The matrix (mat) holds the
        coefficient equations, and the vector (y) contains values for spline interpolation
        equations. This setup allows for solving the cubic spline equations to perform
        interpolation between the provided points.

        :param points :(numpy.ndarray) An array of shape (n, 2) containing (x, y) pairs representing the points for spline interpolation.
        :return:
            - mat (numpy.ndarray): Matrix representing the coefficient equations for cubic splines.
            - y (numpy.ndarray): Vector containing values for spline interpolation equations.
    """
    points = np.array(sorted(points, key=lambda x: x[0]))
    n = points.shape[0]
    mat = np.zeros((4 * (n - 1), 4 * (n - 1)))

    for i in range(1, n):
        x1 = [points[i - 1][0] ** j for j in range(3, -1, -1)]
        x2 = [points[i][0] ** j for j in range(3, -1, -1)]
        mat[(i - 1), 4 * (i - 1):(4 * (i - 1) + 4)] = x1
        mat[(n - 2) + i, 4 * (i - 1):(4 * (i - 1) + 4)] = x2

    for i in range(n - 2):
        x1 = np.multiply([points[i + 1][0] ** j if j >= 0 else 0 for j in range(2, -2, -1)], [3, 2, 1, 0])
        x2 = np.multiply([points[i + 1][0] ** j if j >= 0 else 0 for j in range(1, -3, -1)], [6, 2, 0, 0])
        mat[2 * (n - 1) + i, 4 * i:(4 * i + 8)] = np.append(x1, -x1)
        mat[2 * (n - 1) + (n - 2) + i, 4 * i:(4 * i + 8)] = np.append(x2, -x2)

    for i in range(2):
        x = np.multiply([points[i][0] ** j if j >= 0 else 0 for j in range(1, -3, -1)], [6, 2, 0, 0])
        mat[2 * (n - 1) + 2 * (n - 2) + i, 4 * (n - 2) * i:4 * (n - 2) * i + 4] = x

    y = np.append(points[:n - 1][:, 1], points[1:][:, 1])
    y = np.append(y, np.zeros(mat.shape[0] // 2))
    y[-2] = 3
    y[-1] = 0
    y = y[:, np.newaxis]

    # solve for coefficients
    coeffs = np.linalg.solve(mat, y)

    return coeffs


def plot_natural_cubic_splines(points, coefficients, ax):
    """
     Plot natural cubic splines based on the given points and coefficients.

    :param points: (numpy.ndarray) An array of shape (n, 2) containing (x, y) pairs
                             representing the original points.
    :param coefficients: Tuple containing arrays representing coefficients for cubic splines.

    Displays a plot showing the natural cubic spline interpolation curves
    passing through the given points.
    """
    points = np.array(sorted(points, key=lambda x: x[0]))

    def cubic(coefs):
        return lambda x: np.sum(np.multiply(coefs, [x ** 3, x ** 2, x ** 1, x ** 0]), axis=0)

    for i in range(points.shape[0]):
        ax.scatter(points[i][0], points[i][1])

    for i in range(points.shape[0] - 1):
        x = np.linspace(points[i, 0], points[i + 1, 0], 100)
        y = coefficients[4 * i:4 * i + 4]
        ax.plot(x, cubic(y)(x))

    plt.title('Cubic Spline Interpolation')
    ax.grid(True)
    return ax


def lagrange(points):
    points = np.array(sorted(points, key=lambda x: x[0]))
    x = points[:, 0]  # Extract x values
    y = points[:, 1]  # Extract y values

    xplt = np.linspace(x[0], x[-1], points.shape[0])
    yplt = np.array([], dtype=float)
    for xp in xplt:
        yp = 0
        for xi, yi in zip(x, y):
            yp += yi * np.prod((xp - x[x != xi]) / (xi - x[x != xi]))
        yplt = np.append(yplt, yp)

    return yplt


def plot_lagrange(points, yplt, ax):
    """
    Plot Lagrange polynomial based on the given points and coefficients.

    :param points: (numpy.ndarray) An array of shape (n, 2) containing (x, y) pairs
                             representing the original points.
    :param coefficients: Tuple containing arrays representing coefficients for cubic splines.

    Displays a plot showing the natural cubic spline interpolation curves
    passing through the given points.
    """
    points = np.array(sorted(points, key=lambda x: x[0]))

    for i in range(points.shape[0]):
        ax.scatter(points[i][0], points[i][1], color='red')

    ax.plot(np.linspace(points[0][0], points[-1][0], points.shape[0]), yplt)

    plt.title('Lagrange Interpolation')
    ax.grid(True)
    return ax


def linear_piecewise_interpolation(points):
    """
        This function calculates piecewise linear interpolation for a set of given points.
        It sorts the points based on their x-values and then computes the slope (a) and
        intercept (b) of each segment between adjacent points to perform linear
        interpolation. It plots the interpolated segments along with the original points
        and returns arrays of slopes (a) and intercepts (b) for each segment.
        Perform piecewise linear interpolation based on given points.

        :param points:(numpy.ndarray)  An array of shape (n, 2) containing (x, y) pairs representing the points to be interpolated.

        :return:    - a (numpy.ndarray): Array containing slopes of the linear segments.
                    - b (numpy.ndarray): Array containing intercepts of the linear segments.
    """

    points = np.array(sorted(points, key=lambda x: x[0]))
    x = points[:, 0]
    y = points[:, 1]
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    for i in range(len(x) - 1):
        if x[i + 1] - x[i] == 0:
            a[i] = 0
        else:
            a[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        b[i] = y[i]

    return [a, b]


def plot_piecewise_linear_interpolation(points, coefficients, ax):
    """
        Plot piecewise linear interpolation segments along with original points.

        :param points:(numpy.ndarray) An array of shape (n, 2) containing (x, y) pairs representing the original points.
        :param coefficients:(tuple((numpy.ndarray))) Tuple containing arrays of slopes (a) and intercepts (b) for each segment.

        Displays a plot showing the piecewise linear interpolation segments along
        with the original points.
    """
    points = np.array(sorted(points, key=lambda x: x[0]))
    x = points[:, 0]
    y = points[:, 1]
    a, b = coefficients

    for i in range(len(x) - 1):
        xplt = np.linspace(x[i], x[i + 1], 100)
        yplt = a[i] * (xplt - x[i]) + b[i]
        ax.plot(xplt, yplt)

    ax.plot(x, y, 'ro')
    plt.title('Piecewise Linear Interpolation')
    ax.grid(True)
    return ax


def polynomial_least_squares(points, degree):
    """
        This function calculates the coefficients of a polynomial of given degree
        that fits the given points in the least squares sense. It plots the polynomial
        along with the original points and returns the coefficients of the polynomial.

        :param points:(numpy.ndarray) An array of shape (n, 2) containing (x, y) pairs representing the points to be fitted.
        :param degree:(int) Degree of the polynomial to be fitted.

        :return:    - coefficients (numpy.ndarray): Array containing coefficients of the polynomial.
    """

    points = np.array(sorted(points, key=lambda x: x[0]))
    x = points[:, 0]
    y = points[:, 1]
    mat = np.zeros((len(x), degree + 1))
    for i in range(degree + 1):
        mat[:, i] = x ** i
    coefficient = np.linalg.lstsq(mat, y, rcond=None)[0]
    return coefficient


def plot_polynomial_least_squares(points, coefficients, ax):
    """
        Plot polynomial of given degree that fits the given points in the least
        squares sense.

        :param points:(numpy.ndarray) An array of shape (n, 2) containing (x, y) pairs representing the original points.
        :param coefficients:(numpy.ndarray) Array containing coefficients of the polynomial.

        Displays a plot showing the polynomial along with the original points.
    """
    points = np.array(sorted(points, key=lambda x: x[0]))
    x = points[:, 0]
    y = points[:, 1]

    xplt = np.linspace(x[0], x[-1], 100)
    yplt = np.zeros_like(xplt)
    for i in range(len(coefficients)):
        yplt += coefficients[i] * (xplt ** i)

    ax.plot(xplt, yplt)
    ax.plot(x, y, 'ro')
    plt.title(f'Polynomial Least Squares Degree {len(coefficients) - 1}')
    plt.ylim(min(y) - 10, max(y) + 10)
    ax.grid(True)
    return ax
