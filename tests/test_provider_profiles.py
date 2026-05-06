import unittest

try:
    from . import test_path
except ImportError:
    import test_path
from ComfyUI_LLM_API.provider_profiles import (
    PROVIDER_PROFILE_NAMES,
    resolve_provider_profile,
)


class ProviderProfileTests(unittest.TestCase):
    def test_exposes_profile_names_in_required_order(self):
        self.assertEqual(
            PROVIDER_PROFILE_NAMES,
            (
                "custom_text_image",
                "custom_text_only",
                "openai",
                "deepseek",
                "qwen_dashscope_cn",
                "doubao_ark_cn",
                "zhipu_glm",
                "minimax_global",
            ),
        )

    def test_resolves_preset_base_url(self):
        profile = resolve_provider_profile("openai", "")

        self.assertEqual(profile["profile"], "openai")
        self.assertEqual(profile["base_url"], "https://api.openai.com/v1")
        self.assertTrue(profile["supports_images"])

    def test_non_empty_api_baseurl_overrides_preset_base_url(self):
        profile = resolve_provider_profile(
            "deepseek",
            "https://example.test/compatible/v1",
        )

        self.assertEqual(profile["profile"], "deepseek")
        self.assertEqual(profile["base_url"], "https://example.test/compatible/v1")

    def test_text_only_profiles_do_not_support_images(self):
        for profile_name in (
            "custom_text_only",
            "deepseek",
            "minimax_global",
        ):
            with self.subTest(profile_name=profile_name):
                profile = resolve_provider_profile(profile_name, "")
                self.assertFalse(profile["supports_images"])

    def test_unknown_profile_raises_value_error_with_profile_name(self):
        with self.assertRaisesRegex(ValueError, "not_a_provider"):
            resolve_provider_profile("not_a_provider", "")


if __name__ == "__main__":
    unittest.main()
