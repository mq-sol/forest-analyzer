# standard library
from pathlib import Path

# third party
import cv2
import numpy as np


def calc_exg(rgb: np.ndarray) -> np.ndarray:
    """
    Calculate Excess Green Index.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image array.
        Shape: (H, W, 3)
        dtype: uint8

    Returns
    -------
    np.ndarray
        ExG image.
        Shape: (H, W)
        dtype: float32
    """

    rgb = rgb.astype(np.float32)

    r = rgb[:, :, 0]  # Red channel
    g = rgb[:, :, 1]  # Green channel
    b = rgb[:, :, 2]  # Blue channel

    exg = 2 * g - (r + b)

    return exg
