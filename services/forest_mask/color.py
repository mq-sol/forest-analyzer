import cv2
import numpy as np


def calc_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB image to HSV channels.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image.
        Shape: (H, W, 3)
        dtype: uint8

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        H, S, V channels.
        Shape: (H, W)
        dtype: uint8
    """

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    return h, s, v


def calc_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB image to Lab channels.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image.
        Shape: (H, W, 3)
        dtype: uint8

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        L, a, b channels.
        Shape: (H, W)
        dtype: uint8

    Notes
    -----
    OpenCV stores Lab channels in uint8 format.

    L : 0-255
    a : 0-255 (128 is center)
    b : 0-255 (128 is center)

    a < 128 : green side
    a > 128 : red side

    b < 128 : blue side
    b > 128 : yellow side
    """

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB,
    )

    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    return l, a, b
