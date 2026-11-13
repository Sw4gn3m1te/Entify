import numpy as np
from scipy.ndimage import gaussian_filter

RADIUS = 2.0


def sharpen(image, amount=1.0, radius=RADIUS):
    original = image.astype(np.float32)
    blurred = gaussian_filter(original, sigma=(radius, radius, 0))
    return np.clip(original + amount * (original - blurred), 0, 255).astype(np.uint8)
