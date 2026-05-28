import cv2
import numpy as np


def create_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Create semi-transparent overlay image from binary mask.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image.
        Shape: (H, W, 3)
        dtype: uint8

    mask : np.ndarray
        Binary mask image.
        True pixels are overlay targets.
        Shape: (H, W)
        dtype: bool

    color : tuple[int, int, int], optional
        Overlay RGB color.
        Default is red: (255, 0, 0)

    alpha : float, optional
        Overlay transparency.
        Range: 0.0 - 1.0

        0.0:
            original image only

        1.0:
            overlay color only

        Default is 0.3.

    Returns
    -------
    np.ndarray
        Overlay RGB image.
        Shape: (H, W, 3)
        dtype: uint8

    Notes
    -----
    This function is intended for visualization only.

    Examples
    --------
    >>> overlay = create_mask_overlay(
    ...     rgb,
    ...     anomaly_mask,
    ...     color=(255, 0, 0),
    ...     alpha=0.4,
    ... )
    """

    if rgb.ndim != 3:
        raise ValueError("rgb must be 3-dimensional")

    if mask.ndim != 2:
        raise ValueError("mask must be 2-dimensional")

    if rgb.shape[:2] != mask.shape:
        raise ValueError("rgb and mask shape mismatch")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

    overlay = rgb.astype(np.float32).copy()
    overlay[mask] = color
    result = rgb.astype(np.float32) * (1 - alpha) + overlay * alpha
    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)