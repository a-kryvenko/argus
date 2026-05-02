from common.schema_raw import Image
from pathlib import Path

import cv2
import numpy as np

def build_bz(_img: Image, destination: Path) -> Image:
    img = cv2.imread(_img.path, cv2.IMREAD_GRAYSCALE)

    # normalize to [-1, 1]
    img = img.astype(np.float32)
    norm = (img.astype(np.float32) - 127.5) / 127.5

    B_MAX = 2500.0  # Gauss (tune later!)
    B_los = norm * B_MAX

    h, w = img.shape
    cx, cy = w // 2, h // 2
    r = 0.48 * min(h, w)

    Y, X = np.ogrid[:h, :w]
    dx = X - cx
    dy = Y - cy

    r_pix = np.sqrt(dx**2 + dy**2)

    mask = r_pix <= r
    B_los[~mask] = np.nan

    with np.errstate(invalid="ignore"):
        mu = np.sqrt(1.0 - (r_pix / r) ** 2)

    mu[mu < 0.1] = np.nan

    Bz = B_los / mu

    cv2.imwrite(destination, np.nan_to_num(((Bz + 2500) / 5000 * 255), nan=0).clip(0, 255).astype(np.uint8))

    return Image(path=destination)