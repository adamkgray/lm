from typing import Tuple
from numpy import ndarray


def _d_mse_d_slope(x: ndarray, y: ndarray, predictions: ndarray) -> float:
    """Partial Derivative of MSE with respect to Slope (m)

    Args:
        x (ndarray): given x values
        y (ndarray): expected y values
        predictions (ndarray): predicted y values

    Returns:
        float: partial derivative of MSE evaluated at x, y and predictions
    """
    n: float = float(len(x))
    return (-2 / n) * sum(x * (y - predictions))


def _d_mse_d_intercept(x: ndarray, y: ndarray, predictions: ndarray) -> float:
    """Partial Derivative of MSE with respect to Intercept (b)

    Args:
        x (ndarray): given x values
        y (ndarray): given y values
        predictions (ndarray): predicted y values

    Returns:
        float: partial derivative of MSE evaluated at x, y and predictions
    """
    n: float = float(len(x))
    return (-2 / n) * sum(y - predictions)


def lm(x: ndarray, y: ndarray, learning_rate: float = 0.001, epochs: int = 1000) -> Tuple[float, float]:
    """linear regression with gradient descent

    Args:
        x (ndarray): given x values
        y (ndarray): known y values

    Returns:
        Tuple[float, float, List[float]]:
            - slope (m)
            - intercept (b)
    """
    slope: float = 0
    intercept: float = 0

    for _ in range(epochs):
        predictions: ndarray = (slope * x) + intercept
        slope -= (learning_rate * _d_mse_d_slope(x, y, predictions))
        intercept -= (learning_rate * _d_mse_d_intercept(x, y, predictions))

    return (slope, intercept)
