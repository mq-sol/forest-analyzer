import cv2
import numpy as np


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """
    Remove small connected components from a binary mask.

    This function labels connected white regions in the mask
    and removes components smaller than `min_size`.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
        Shape: (H, W)
        dtype: bool

    min_size : int
        Minimum component size (pixel count)
        to keep.

    Returns
    -------
    np.ndarray
        Filtered binary mask.
        Shape: (H, W)
        dtype: bool
    """

    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    filtered = np.zeros_like(mask_uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_size:
            filtered[labels == label] = 1

    return filtered.astype(bool)