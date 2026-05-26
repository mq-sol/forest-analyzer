# standard library
from pathlib import Path

# third party
import cv2
import numpy as np


def calc_forest_mask(
    exg: np.ndarray,
    threshold: float,
) -> np.ndarray:

    forest = exg > threshold

    return forest
