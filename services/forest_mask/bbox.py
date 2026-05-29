import cv2
import numpy as np
from typing import Tuple

def draw_mask_bboxes(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes for connected components in a binary mask.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image.
        Shape: (H, W, 3)
        dtype: uint8

    mask : np.ndarray
        Binary mask image.
        True pixels represent target objects.
        Shape: (H, W)
        dtype: bool

    color : tuple[int, int, int], optional
        Bounding box RGB color.
        Default is red: (255, 0, 0)

    thickness : int, optional
        Bounding box line thickness.
        Default is 2.

    Returns
    -------
    np.ndarray
        RGB image with bounding boxes.
        Shape: (H, W, 3)
        dtype: uint8

    Notes
    -----
    Bounding boxes are generated from connected components
    detected in the input mask.

    This function is intended for visualization only.
    """

    if rgb.ndim != 3:
        raise ValueError("rgb must be 3-dimensional")

    if mask.ndim != 2:
        raise ValueError("mask must be 2-dimensional")

    if rgb.shape[:2] != mask.shape:
        raise ValueError("rgb and mask shape mismatch")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr_color = (color[2], color[1], color[0])

    bboxes = extract_bboxes(mask)

    for x, y, w, h in bboxes:
        cv2.rectangle(
            bgr,
            (x, y),
            (x + w, y + h),
            color=bgr_color,
            thickness=thickness,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_bboxes(
    mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """
    Extract bounding boxes from connected components.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask image.
        Shape: (H, W)
        dtype: bool

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of bounding boxes.

        Format:
            (x, y, width, height)

    Notes
    -----
    Connected components are extracted using
    OpenCV connected component labeling.
    """

    mask_uint8 = mask.astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    bboxes: list[tuple[int, int, int, int]] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        shape_score, shape_area = calc_shape_score(h, w)
        if shape_area < 100:
            continue
        if shape_score > 0.8:
            continue
        bboxes.append((x, y, w, h))

    return bboxes


def calc_shape_score(h: int, w: int) -> Tuple[float, float]:
    shape_score = abs(w - h) / (w + h)
    shape_area = w * h
    return shape_score, shape_area
