import cv2
import numpy as np


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
    max_shape_score: float | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Extract bounding boxes from connected components.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask image.
        Shape: (H, W)
        dtype: bool

    max_shape_score : float or None, optional
        Maximum allowed bbox shape score.
        Components whose shape score exceeds this value
        are excluded as elongated artifacts
        (e.g. forest roads, branch noise, linear artifacts).

        If None, no shape score filtering is applied.
        Default is None.

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

    Shape score is computed by :func:`calc_bbox_shape_score`.
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

        shape_area = w * h
        if shape_area < 100:
            continue

        shape_score = calc_bbox_shape_score(
            width=w,
            height=h,
        )

        if (
            max_shape_score is not None
            and shape_score > max_shape_score
        ):
            continue

        bboxes.append((x, y, w, h))

    return bboxes


def calc_bbox_shape_score(
    width: int,
    height: int,
) -> float:
    """
    Calculate a shape score for a bounding box.

    The score quantifies how elongated a bounding box is.

    Parameters
    ----------
    width : int
        Bounding box width in pixels.
        Must be positive.

    height : int
        Bounding box height in pixels.
        Must be positive.

    Returns
    -------
    float
        Shape score in the range [0.0, 1.0).

        - 0.0 for a perfect square.
        - Values close to 0.0 for near-square shapes.
        - Values close to 1.0 for elongated shapes.

    Notes
    -----
    The score is defined as::

        shape_score = abs(width - height) / (width + height)

    This is intended to suppress linear artifacts
    such as forest roads or branch-shaped noise.
    """

    return abs(width - height) / (width + height)
