# standard library
from pathlib import Path

# third party
import cv2
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    """
    Load image as RGB ndarray.

    Parameters
    ----------
    path : Path
        Input image path.

    Returns
    -------
    np.ndarray
        RGB image.
        Shape: (H, W, 3)
        dtype: uint8
    """

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Failed to load image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    """
    Save RGB ndarray as image.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image.
        Shape: (H, W, 3)

    path : Path
        Output image path.
    """

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(path), bgr)

    if not success:
        raise IOError(f"Failed to save image: {path}")


def save_gray(gray: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if gray.dtype != np.uint8:
        gray = normalize_uint8(gray)

    success = cv2.imwrite(str(path), gray)

    if not success:
        raise IOError(f"Failed to save image: {path}")


def normalize_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 (0-255).
    """

    img_min = img.min()
    img_max = img.max()

    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)

    norm = (img - img_min) / (img_max - img_min)

    return (norm * 255).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path) -> None:

    mask_uint8 = mask.astype(np.uint8) * 255

    success = cv2.imwrite(
        str(path),
        mask_uint8,
    )

    if not success:
        raise IOError(f"Failed to save mask: {path}")
