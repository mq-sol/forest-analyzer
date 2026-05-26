import cv2
import numpy as np


def calc_texture(
    gray: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Calculate local texture using local variance.

    This function computes the local variance of a grayscale image
    within a sliding window. Forest regions tend to have high local
    variance because tree crowns create complex brightness patterns.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
        Shape: (H, W)
        dtype: uint8 or float32

    kernel_size : int
        Size of the local window used to calculate variance.

    Returns
    -------
    np.ndarray
        Texture image based on local variance.
        Shape: (H, W)
        dtype: float32
    """

    gray = gray.astype(np.float32)

    mean = cv2.blur(
        gray,
        (kernel_size, kernel_size),
    )

    mean_sq = cv2.blur(
        gray**2,
        (kernel_size, kernel_size),
    )

    variance = mean_sq - (mean**2)

    return variance
