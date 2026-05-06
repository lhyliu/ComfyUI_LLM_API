import base64
import io
import math

from PIL import Image


JPEG_QUALITY_STEPS = (92, 88, 84, 80, 76, 72, 68, 64, 60, 56)

_RESAMPLING = getattr(Image, "Resampling", Image)


def _encode_jpeg_bytes(image, quality):
    buf = io.BytesIO()
    image.save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
        subsampling=2,
    )
    return buf.getvalue()


def _normalized_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _limit_max_side(image, max_side):
    if max_side is None:
        return image

    longest_side = max(image.size)
    if longest_side <= max_side:
        return image

    scale = max_side / float(longest_side)
    target = (
        max(1, int(image.width * scale)),
        max(1, int(image.height * scale)),
    )
    return image.resize(target, _RESAMPLING.LANCZOS)


def encode_image_to_safe_jpeg_b64(
    image,
    max_bytes,
    max_side,
):
    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if max_side is not None and max_side <= 0:
        raise ValueError("max_side must be a positive integer or None")

    working = _limit_max_side(_normalized_rgb(image), max_side)
    min_quality = JPEG_QUALITY_STEPS[-1]
    best_effort = _encode_jpeg_bytes(working, min_quality)

    for _ in range(6):
        for quality in JPEG_QUALITY_STEPS:
            encoded = _encode_jpeg_bytes(working, quality)
            if len(encoded) <= max_bytes:
                return base64.b64encode(encoded).decode("utf-8")

            if quality == min_quality:
                best_effort = encoded

        current_size = max(1, len(best_effort))
        scale = math.sqrt(max_bytes / float(current_size)) * 0.95
        scale = min(scale, 0.95)
        new_width = max(1, int(working.width * scale))
        new_height = max(1, int(working.height * scale))
        if new_width == working.width and new_height == working.height:
            new_width = max(1, working.width - 1)
            new_height = max(1, working.height - 1)

        working = working.resize((new_width, new_height), _RESAMPLING.LANCZOS)

    return base64.b64encode(best_effort).decode("utf-8")


def _tensor_item_to_rgb_image(item):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to encode ComfyUI image tensors") from exc

    array = np.clip(item, 0.0, 1.0)
    array = (array * 255.0).round().astype("uint8")

    channels = array.shape[-1] if array.ndim >= 3 else 1
    if channels == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    return Image.fromarray(array[..., :3], mode="RGB")


def _iter_images(images):
    if hasattr(images, "cpu") and hasattr(images.cpu(), "numpy"):
        batch = images.cpu().numpy()
        for item in batch:
            yield _tensor_item_to_rgb_image(item)
        return

    for image in images:
        yield image


def encode_comfy_image_batch_to_data_urls(
    images,
    max_images,
    max_bytes,
    max_side,
):
    data_urls = []
    if images is None or max_images <= 0:
        return data_urls

    for image in _iter_images(images):
        if len(data_urls) >= max_images:
            break

        jpeg_b64 = encode_image_to_safe_jpeg_b64(
            image,
            max_bytes=max_bytes,
            max_side=max_side,
        )
        data_urls.append("data:image/jpeg;base64," + jpeg_b64)

    return data_urls
