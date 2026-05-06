import unittest

try:
    from . import test_path
except ImportError:
    import test_path
from ComfyUI_LLM_API.image_presets import IMAGE_PRESET_NAMES, resolve_image_settings


class ImagePresetTests(unittest.TestCase):
    def test_exposes_presets_in_ui_order(self):
        self.assertEqual(
            IMAGE_PRESET_NAMES,
            ("fast", "balanced", "detail", "ocr_high", "custom"),
        )

    def test_balanced_preset_uses_recommended_defaults(self):
        settings = resolve_image_settings(
            image_preset="balanced",
            custom_image_max_mb=9.9,
            custom_image_max_side=4096,
        )

        self.assertEqual(settings["max_bytes"], 1_500_000)
        self.assertEqual(settings["max_side"], 1280)

    def test_custom_uses_mb_and_side_inputs(self):
        settings = resolve_image_settings(
            image_preset="custom",
            custom_image_max_mb=2.5,
            custom_image_max_side=1400,
        )

        self.assertEqual(settings["max_bytes"], 2_500_000)
        self.assertEqual(settings["max_side"], 1400)

    def test_unknown_preset_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_image_settings(
                image_preset="giant",
                custom_image_max_mb=1.0,
                custom_image_max_side=1280,
            )


if __name__ == "__main__":
    unittest.main()
