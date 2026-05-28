import cv2
import numpy as np


def draw_mask_contours(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw contours from binary mask on RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image.
        Shape: (H, W, 3)
        dtype: uint8

    mask : np.ndarray
        Binary mask image.
        True pixels are contour targets.
        Shape: (H, W)
        dtype: bool

    color : tuple[int, int, int], optional
        Contour RGB color.
        Default is red: (255, 0, 0)

    thickness : int, optional
        Contour line thickness.
        Default is 2.

    Returns
    -------
    np.ndarray
        RGB image with contours.
        Shape: (H, W, 3)
        dtype: uint8

    Notes
    -----
    This function is intended for visualization only.
    """

    result = rgb.copy()

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        result,
        contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
    )

    return result