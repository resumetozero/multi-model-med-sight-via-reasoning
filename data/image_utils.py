import numpy as np
from PIL import Image


def preprocess_medical_image(image: Image.Image) -> Image.Image:
    if image.mode == "I":
        arr = np.array(image, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        image = Image.fromarray(arr.astype(np.uint8), mode="L")
    return image
