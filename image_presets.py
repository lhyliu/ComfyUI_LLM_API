from dataclasses import dataclass


MB = 1_000_000


@dataclass(frozen=True)
class ImagePreset:
    name: str
    max_mb: float
    max_side: int


IMAGE_PRESET_NAMES = ("fast", "balanced", "detail", "ocr_high", "custom")

_IMAGE_PRESETS = {
    "fast": ImagePreset("fast", 0.5, 768),
    "balanced": ImagePreset("balanced", 1.5, 1280),
    "detail": ImagePreset("detail", 3.0, 1600),
    "ocr_high": ImagePreset("ocr_high", 5.0, 2048),
}


def _mb_to_bytes(value):
    return int(float(value) * MB)


def _as_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _as_int(value, field_name):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def resolve_image_settings(image_preset, custom_image_max_mb, custom_image_max_side):
    if image_preset == "custom":
        max_mb = _as_float(custom_image_max_mb, "custom_image_max_mb")
        max_side = _as_int(custom_image_max_side, "custom_image_max_side")
    elif image_preset in _IMAGE_PRESETS:
        preset = _IMAGE_PRESETS[image_preset]
        max_mb = preset.max_mb
        max_side = preset.max_side
    else:
        raise ValueError(f"Unknown image preset: {image_preset}")

    if max_mb <= 0:
        raise ValueError("custom_image_max_mb must be a positive number")
    if max_mb > 32.0:
        raise ValueError("custom_image_max_mb must be at most 32.0")
    if max_side <= 0:
        raise ValueError("custom_image_max_side must be a positive integer")
    if max_side > 8_192:
        raise ValueError("custom_image_max_side must be at most 8192")

    return {
        "max_bytes": _mb_to_bytes(max_mb),
        "max_side": max_side,
    }
