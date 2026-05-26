# standard library
from pathlib import Path

# third party
import cv2
import numpy as np


def apply_closing(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Apply closing to a binary mask.

    Closing = dilation -> erosion.
    It fills small black holes inside white regions.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
        Shape: (H, W)
        dtype: bool

    kernel_size : int
        Kernel size for morphology operation.

    Returns
    -------
    np.ndarray
        Processed binary mask.
        Shape: (H, W)
        dtype: bool
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8,
    )

    mask_uint8 = mask.astype(np.uint8) * 255

    closed = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return closed > 0


def apply_opening(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Apply opening to a binary mask.

    Opening = erosion -> dilation.
    It removes small white noise.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
        Shape: (H, W)
        dtype: bool

    kernel_size : int
        Kernel size for morphology operation.

    Returns
    -------
    np.ndarray
        Processed binary mask.
        Shape: (H, W)
        dtype: bool
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8,
    )

    mask_uint8 = mask.astype(np.uint8) * 255

    opened = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_OPEN,
        kernel,
    )

    return opened > 0


def apply_morphology(
    mask: np.ndarray, closing: int = 5, opening: int = 3
) -> np.ndarray:
    mask = apply_closing(mask, closing)
    mask = apply_opening(mask, opening)
    return mask
