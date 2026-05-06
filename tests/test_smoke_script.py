import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from PIL import Image

try:
    from . import test_path
except ImportError:
    import test_path


def _completion(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class SmokeProviderScriptTests(unittest.TestCase):
    def test_missing_env_key_exits_with_readable_message(self):
        from ComfyUI_LLM_API.scripts import smoke_provider

        output = io.StringIO()
        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_PROVIDER_PROFILE": "openai",
                "LLM_API_MODEL": "demo-model",
            },
            clear=True,
        ):
            with redirect_stdout(output):
                exit_code = smoke_provider.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("LLM_API_KEY", output.getvalue())

    def test_text_smoke_passes_fake_client_without_printing_key(self):
        from ComfyUI_LLM_API.scripts import smoke_provider

        output = io.StringIO()
        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_PROVIDER_PROFILE": "openai",
                "LLM_API_KEY": "sk-secret",
                "LLM_API_MODEL": "demo-model",
            },
            clear=True,
        ):
            with mock.patch(
                "ComfyUI_LLM_API.scripts.smoke_provider.create_openai_client",
                return_value=object(),
            ) as create_client:
                with mock.patch(
                    "ComfyUI_LLM_API.scripts.smoke_provider.create_chat_completion",
                    return_value=_completion("hello"),
                ) as create_completion:
                    with redirect_stdout(output):
                        exit_code = smoke_provider.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(create_client.call_args.kwargs["api_key"], "sk-secret")
        self.assertFalse(create_completion.call_args.kwargs["has_images"])
        self.assertIn("text: ok", output.getvalue())
        self.assertIn("image: skipped", output.getvalue())
        self.assertNotIn("sk-secret", output.getvalue())

    def test_image_smoke_uses_image_path_only_when_set(self):
        from ComfyUI_LLM_API.scripts import smoke_provider

        image = Image.new("RGB", (8, 8), (20, 40, 60))
        output = io.StringIO()

        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_PROVIDER_PROFILE": "openai",
                "LLM_API_KEY": "sk-secret",
                "LLM_API_MODEL": "demo-model",
                "LLM_API_IMAGE_PATH": "demo.png",
            },
            clear=True,
        ):
            with mock.patch(
                "ComfyUI_LLM_API.scripts.smoke_provider.create_openai_client",
                return_value=object(),
            ):
                with mock.patch(
                    "ComfyUI_LLM_API.scripts.smoke_provider.create_chat_completion",
                    return_value=_completion("hello"),
                ) as create_completion:
                    with mock.patch(
                        "ComfyUI_LLM_API.scripts.smoke_provider.Image.open",
                        return_value=image,
                    ):
                        with redirect_stdout(output):
                            exit_code = smoke_provider.main()

        self.assertEqual(exit_code, 0)
        self.assertTrue(create_completion.call_args_list[-1].kwargs["has_images"])
        self.assertIn("image: ok", output.getvalue())


if __name__ == "__main__":
    unittest.main()
