import cv2
import numpy as np


def filter_components_by_surrounding_healthy(
    anomaly_mask: np.ndarray,
    healthy_mask: np.ndarray,
    min_size: int = 30,
    max_size: int = 2000,
    buffer_size: int = 15,
    min_healthy_ratio: float = 0.4,
) -> np.ndarray:
    """
    Filter anomaly components by surrounding healthy forest.

    This function keeps anomaly components that are small enough
    and surrounded by healthy forest.

    The purpose is to reduce false positives such as roads,
    bare ground, and large open areas, while keeping small
    anomaly candidates inside forest regions.

    Parameters
    ----------
    anomaly_mask : np.ndarray
        Binary anomaly mask.
        True pixels represent anomaly candidates.
        Shape: (H, W)
        dtype: bool

    healthy_mask : np.ndarray
        Binary healthy forest mask.
        True pixels represent healthy forest areas.
        Shape: (H, W)
        dtype: bool

    min_size : int
        Minimum component size in pixels to keep.

    max_size : int
        Maximum component size in pixels to keep.
        Components larger than this are treated as roads,
        bare ground, or large non-forest areas.

    buffer_size : int
        Kernel size used to create the surrounding area around
        each anomaly component.

    min_healthy_ratio : float
        Minimum ratio of healthy forest pixels in the surrounding
        area required to keep the component.
        Range: 0.0 - 1.0

    Returns
    -------
    np.ndarray
        Filtered anomaly mask.
        Shape: (H, W)
        dtype: bool

    Notes
    -----
    This function is a context-based filter.

    It does not determine whether a tree is truly dead.
    It only keeps anomaly candidates that are spatially located
    inside or near healthy forest regions.
    """

    if anomaly_mask.shape != healthy_mask.shape:
        raise ValueError("anomaly_mask and healthy_mask must have the same shape")

    anomaly_uint8 = anomaly_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        anomaly_uint8,
        connectivity=8,
    )

    result = np.zeros_like(anomaly_uint8)

    kernel = np.ones(
        (buffer_size, buffer_size),
        np.uint8,
    )

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_size:
            continue

        if area > max_size:
            continue

        component = labels == label

        component_uint8 = component.astype(np.uint8)

        dilated = cv2.dilate(
            component_uint8,
            kernel,
            iterations=1,
        ).astype(bool)

        surrounding = dilated & ~component

        surrounding_area = np.count_nonzero(surrounding)

        if surrounding_area == 0:
            continue

        healthy_area = np.count_nonzero(
            healthy_mask & surrounding
        )

        healthy_ratio = healthy_area / surrounding_area

        if healthy_ratio >= min_healthy_ratio:
            result[component] = 1

    return result.astype(bool)