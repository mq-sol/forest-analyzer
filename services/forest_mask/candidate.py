import cv2
import numpy as np

from services.forest_mask.bbox import calc_bbox_shape_score


def extract_candidates(
    mask: np.ndarray,
) -> list[dict[str, int | float]]:
    """
    Extract inspection candidates from an anomaly mask.

    Each connected component in the input mask is treated as a single
    candidate. The returned records are intended as the front-stage of
    CSV / GeoJSON export and downstream tabular reporting.

    Parameters
    ----------
    mask : np.ndarray
        Binary anomaly mask.
        True pixels represent anomaly candidates.
        Shape: (H, W)
        dtype: bool

    Returns
    -------
    list[dict[str, int | float]]
        List of candidate records.

        Each record has the following keys:

        - ``id`` : int
            Connected component label (1-indexed).
        - ``x`` : int
            Bounding box top-left x in image pixels.
        - ``y`` : int
            Bounding box top-left y in image pixels.
        - ``width`` : int
            Bounding box width in pixels.
        - ``height`` : int
            Bounding box height in pixels.
        - ``area_px`` : int
            Connected component area in pixels.
        - ``center_x`` : float
            Centroid x in image pixels.
        - ``center_y`` : float
            Centroid y in image pixels.
        - ``shape_score`` : float
            Bounding box shape score from
            :func:`services.forest_mask.bbox.calc_bbox_shape_score`.

    Notes
    -----
    Connected components are extracted using
    OpenCV connected component labeling with 8-connectivity.

    Coordinates are in image pixel space.
    They are NOT geographic coordinates.
    Conversion to geographic coordinates is out of scope for this
    function and is expected to be handled by a downstream stage.

    This function is the front-stage of CSV / GeoJSON export.
    No filtering (area, shape, etc.) is applied here.
    """

    mask_uint8 = mask.astype(np.uint8)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    candidates: list[dict[str, int | float]] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        center_x = float(centroids[label][0])
        center_y = float(centroids[label][1])

        shape_score = calc_bbox_shape_score(
            width=width,
            height=height,
        )

        candidates.append(
            {
                "id": label,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area_px": area,
                "center_x": center_x,
                "center_y": center_y,
                "shape_score": shape_score,
            }
        )

    return candidates
