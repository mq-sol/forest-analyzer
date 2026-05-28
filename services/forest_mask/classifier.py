import numpy as np


def classify_forest_status(
    exg: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    lab_a: np.ndarray,
    texture: np.ndarray,
    exg_threshold: float = 10.0,
    saturation_threshold: float = 80.0,
    lab_a_threshold: float = 125.0,
    anomaly_saturation_threshold: float = 60.0,
    anomaly_value_threshold: float = 170.0,
    texture_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify forest status into healthy, anomaly, and unknown masks.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        healthy_mask, anomaly_mask, unknown_mask
    """

    healthy_mask = (
        (exg > exg_threshold)
        & (s > saturation_threshold)
        & (lab_a < lab_a_threshold)
    )

    anomaly_mask = (
        (s < anomaly_saturation_threshold)
        & (v > anomaly_value_threshold)
        & (texture > texture_threshold)
    )

    unknown_mask = ~(healthy_mask | anomaly_mask)

    return healthy_mask, anomaly_mask, unknown_mask