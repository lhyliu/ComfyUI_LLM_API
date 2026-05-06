import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from . import test_path
except ImportError:
    import test_path


def _completion(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _run_node(**overrides):
    from ComfyUI_LLM_API.node import LLMAPIChatNode

    kwargs = {
        "provider_profile": "custom_text_image",
        "api_baseurl": "https://example.com/v1",
        "api_key": "sk-secret",
        "api_key_env": "",
        "model": "demo",
        "system_prompt": "system",
        "prompt": "hello",
        "temperature": 0.6,
        "execution_seed": 123,
        "filter_thinking": True,
        "max_images": 4,
        "image_preset": "balanced",
        "custom_image_max_mb": 1.5,
        "custom_image_max_side": 1280,
    }
    kwargs.update(overrides)
    return LLMAPIChatNode().run_llmapi(**kwargs)


class NodeContractTests(unittest.TestCase):
    def test_package_exports_only_new_node_mapping(self):
        import ComfyUI_LLM_API

        self.assertEqual(
            list(ComfyUI_LLM_API.NODE_CLASS_MAPPINGS),
            ["LLM_API_CHAT_NODE"],
        )
        self.assertEqual(
            list(ComfyUI_LLM_API.NODE_DISPLAY_NAME_MAPPINGS),
            ["LLM_API_CHAT_NODE"],
        )
        self.assertEqual(
            ComfyUI_LLM_API.NODE_DISPLAY_NAME_MAPPINGS["LLM_API_CHAT_NODE"],
            "LLM API Chat",
        )

    def test_input_contract_matches_refactored_chat_node(self):
        from ComfyUI_LLM_API.image_presets import IMAGE_PRESET_NAMES
        from ComfyUI_LLM_API.node import LLMAPIChatNode
        from ComfyUI_LLM_API.provider_profiles import PROVIDER_PROFILE_NAMES

        inputs = LLMAPIChatNode.INPUT_TYPES()
        required = inputs["required"]

        self.assertEqual(
            list(required),
            [
                "provider_profile",
                "api_baseurl",
                "api_key",
                "api_key_env",
                "model",
                "system_prompt",
                "prompt",
                "temperature",
                "execution_seed",
                "filter_thinking",
                "max_images",
                "image_preset",
                "custom_image_max_mb",
                "custom_image_max_side",
            ],
        )
        self.assertEqual(required["provider_profile"][0], PROVIDER_PROFILE_NAMES)
        self.assertEqual(
            required["provider_profile"][1]["default"],
            PROVIDER_PROFILE_NAMES[0],
        )
        self.assertEqual(required["api_baseurl"][1]["default"], "")
        self.assertEqual(required["api_key_env"][1]["default"], "")
        self.assertTrue(required["api_key_env"][1]["advanced"])
        self.assertEqual(required["image_preset"][0], IMAGE_PRESET_NAMES)
        self.assertEqual(required["image_preset"][1]["default"], "balanced")
        self.assertEqual(required["custom_image_max_mb"][1]["default"], 1.5)
        self.assertEqual(required["custom_image_max_side"][1]["default"], 1280)
        self.assertIn("images", inputs["optional"])
        self.assertNotIn("role", inputs["required"])
        self.assertNotIn("seed", inputs["required"])
        self.assertNotIn("video", inputs["optional"])
        self.assertEqual(LLMAPIChatNode.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(LLMAPIChatNode.RETURN_NAMES, ("response", "error"))

    def test_package_import_does_not_import_openai(self):
        removed_modules = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "ComfyUI_LLM_API" or name.startswith("ComfyUI_LLM_API.")
        }
        real_import = __import__

        def fail_openai_import(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise ImportError("openai intentionally unavailable")
            return real_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", side_effect=fail_openai_import):
                package = importlib.import_module("ComfyUI_LLM_API")

            self.assertIn("LLM_API_CHAT_NODE", package.NODE_CLASS_MAPPINGS)
        finally:
            for name in list(sys.modules):
                if name == "ComfyUI_LLM_API" or name.startswith("ComfyUI_LLM_API."):
                    sys.modules.pop(name)
            sys.modules.update(removed_modules)

    def test_validate_inputs_rejects_missing_model(self):
        from ComfyUI_LLM_API.node import LLMAPIChatNode

        result = LLMAPIChatNode.VALIDATE_INPUTS(
            provider_profile="custom_text_image",
            api_baseurl="https://example.com/v1",
            model="",
            temperature=0.6,
            max_images=4,
            image_preset="balanced",
            custom_image_max_mb=1.5,
            custom_image_max_side=1280,
        )

        self.assertIsInstance(result, str)
        self.assertIn("model", result.lower())

    def test_validate_inputs_rejects_image_limits_above_declared_bounds(self):
        from ComfyUI_LLM_API.node import LLMAPIChatNode

        too_many_bytes = LLMAPIChatNode.VALIDATE_INPUTS(
            provider_profile="custom_text_image",
            api_baseurl="https://example.com/v1",
            model="demo",
            temperature=0.6,
            max_images=4,
            image_preset="custom",
            custom_image_max_mb=32.1,
            custom_image_max_side=1280,
        )
        too_large_side = LLMAPIChatNode.VALIDATE_INPUTS(
            provider_profile="custom_text_image",
            api_baseurl="https://example.com/v1",
            model="demo",
            temperature=0.6,
            max_images=4,
            image_preset="custom",
            custom_image_max_mb=1.5,
            custom_image_max_side=8_193,
        )

        self.assertIn("custom_image_max_mb", too_many_bytes)
        self.assertIn("custom_image_max_side", too_large_side)

    def test_validate_inputs_returns_errors_for_non_numeric_runtime_inputs(self):
        from ComfyUI_LLM_API.node import LLMAPIChatNode

        cases = (
            ("temperature", "warm", 4, "temperature"),
            ("max_images", 0.6, "many", "max_images"),
        )

        for field, temperature, max_images, expected in cases:
            with self.subTest(field=field):
                result = LLMAPIChatNode.VALIDATE_INPUTS(
                    provider_profile="custom_text_image",
                    api_baseurl="https://example.com/v1",
                    model="demo",
                    temperature=temperature,
                    max_images=max_images,
                    image_preset="balanced",
                    custom_image_max_mb=1.5,
                    custom_image_max_side=1280,
                )

            self.assertIsInstance(result, str)
            self.assertIn(expected, result)

    def test_validate_inputs_returns_errors_for_bad_custom_image_values(self):
        from ComfyUI_LLM_API.node import LLMAPIChatNode

        result = LLMAPIChatNode.VALIDATE_INPUTS(
            provider_profile="custom_text_image",
            api_baseurl="https://example.com/v1",
            model="demo",
            temperature=0.6,
            max_images=4,
            image_preset="custom",
            custom_image_max_mb=None,
            custom_image_max_side=1280,
        )

        self.assertIsInstance(result, str)
        self.assertIn("custom_image_max_mb", result)

    def test_type_errors_are_not_converted_to_llm_output(self):
        with mock.patch("ComfyUI_LLM_API.node.create_openai_client", return_value=object()):
            with mock.patch(
                "ComfyUI_LLM_API.node.build_chat_messages",
                side_effect=TypeError("programming bug"),
            ):
                with self.assertRaises(TypeError):
                    _run_node()

    def test_runtime_errors_are_not_converted_to_llm_output(self):
        with mock.patch("ComfyUI_LLM_API.node.create_openai_client", return_value=object()):
            with mock.patch(
                "ComfyUI_LLM_API.node.build_chat_messages",
                side_effect=RuntimeError("programming bug"),
            ):
                with self.assertRaises(RuntimeError):
                    _run_node()

    def test_successful_fake_completion_returns_response_and_empty_error(self):
        with mock.patch("ComfyUI_LLM_API.node.create_openai_client", return_value=object()):
            with mock.patch(
                "ComfyUI_LLM_API.node.create_chat_completion",
                return_value=_completion("hello"),
            ) as create_completion:
                response = _run_node()

        self.assertEqual(response, ("hello", ""))
        _, kwargs = create_completion.call_args
        self.assertNotIn("execution_seed", kwargs)

    def test_create_openai_client_user_error_returns_error_output(self):
        from ComfyUI_LLM_API.node import LLMAPIUserError

        with mock.patch(
            "ComfyUI_LLM_API.node.create_openai_client",
            side_effect=LLMAPIUserError(
                "OpenAI Python package is not installed. Run pip install -r requirements.txt."
            ),
        ):
            with mock.patch("ComfyUI_LLM_API.node.logger.warning"):
                response = _run_node()

        self.assertEqual(response[0], "")
        self.assertTrue(response[1].startswith("LLM API Error: "))
        self.assertIn("requirements.txt", response[1])

    def test_api_key_env_overrides_plain_api_key(self):
        with mock.patch.dict("os.environ", {"LLM_API_TEST_KEY": "env-secret"}, clear=False):
            with mock.patch(
                "ComfyUI_LLM_API.node.create_openai_client",
                return_value=object(),
            ) as create_client:
                with mock.patch(
                    "ComfyUI_LLM_API.node.create_chat_completion",
                    return_value=_completion("ok"),
                ):
                    response = _run_node(
                        api_key="plain-secret",
                        api_key_env="LLM_API_TEST_KEY",
                    )

        self.assertEqual(response, ("ok", ""))
        self.assertEqual(create_client.call_args.kwargs["api_key"], "env-secret")

    def test_missing_api_key_env_returns_error_before_client(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("ComfyUI_LLM_API.node.create_openai_client") as create_client:
                with mock.patch("ComfyUI_LLM_API.node.logger.warning"):
                    response = _run_node(api_key_env="LLM_API_MISSING_KEY")

        self.assertEqual(response[0], "")
        self.assertIn("Environment variable LLM_API_MISSING_KEY is not set", response[1])
        create_client.assert_not_called()

    def test_api_key_env_value_is_redacted_from_errors(self):
        with mock.patch.dict("os.environ", {"LLM_API_TEST_KEY": "env-secret"}, clear=False):
            with mock.patch("ComfyUI_LLM_API.node.logger.warning") as warning:
                with mock.patch(
                    "ComfyUI_LLM_API.node.create_openai_client",
                    side_effect=ValueError("bad env-secret"),
                ):
                    response = _run_node(api_key_env="LLM_API_TEST_KEY")

        self.assertEqual(response[0], "")
        self.assertNotIn("env-secret", response[1])
        self.assertIn("[redacted]", response[1])
        self.assertNotIn(
            "env-secret",
            " ".join(str(arg) for arg in warning.call_args.args),
        )

    def test_text_only_profile_with_images_returns_error_before_encoding_or_client(self):
        with mock.patch(
            "ComfyUI_LLM_API.node.encode_comfy_image_batch_to_data_urls"
        ) as encode_images:
            with mock.patch("ComfyUI_LLM_API.node.create_openai_client") as create_client:
                with mock.patch("ComfyUI_LLM_API.node.logger.warning"):
                    response = _run_node(
                        provider_profile="custom_text_only",
                        images=object(),
                    )

        self.assertEqual(response[0], "")
        self.assertIn("does not support images", response[1])
        encode_images.assert_not_called()
        create_client.assert_not_called()

    def test_blank_openai_base_url_uses_profile_default_and_override_wins(self):
        with mock.patch("ComfyUI_LLM_API.node.create_openai_client", return_value=object()) as create_client:
            with mock.patch(
                "ComfyUI_LLM_API.node.create_chat_completion",
                return_value=_completion("ok"),
            ):
                _run_node(provider_profile="openai", api_baseurl="")
                _run_node(
                    provider_profile="openai",
                    api_baseurl="https://override.example/v1",
                )

        self.assertEqual(
            create_client.call_args_list[0].kwargs["base_url"],
            "https://api.openai.com/v1",
        )
        self.assertEqual(
            create_client.call_args_list[1].kwargs["base_url"],
            "https://override.example/v1",
        )

    def test_image_success_encodes_once_through_retry_message_factory(self):
        def fake_create_chat_completion(**kwargs):
            factory = kwargs["image_retry_message_factory"]
            messages = factory(kwargs["image_retry_limits"][0])
            self.assertTrue(kwargs["has_images"])
            self.assertEqual(messages[-1]["content"][1]["image_url"]["url"], "data:image/png;base64,abc")
            return _completion("vision hello")

        with mock.patch(
            "ComfyUI_LLM_API.node.encode_comfy_image_batch_to_data_urls",
            return_value=["data:image/png;base64,abc"],
        ) as encode_images:
            with mock.patch("ComfyUI_LLM_API.node.create_openai_client", return_value=object()):
                with mock.patch(
                    "ComfyUI_LLM_API.node.create_chat_completion",
                    side_effect=fake_create_chat_completion,
                ):
                    response = _run_node(images=object())

        self.assertEqual(response, ("vision hello", ""))
        encode_images.assert_called_once()
        self.assertEqual(encode_images.call_args.kwargs["max_bytes"], 1_500_000)
        self.assertEqual(encode_images.call_args.kwargs["max_side"], 1280)

    def test_custom_image_preset_forwards_custom_mb_and_side(self):
        def fake_create_chat_completion(**kwargs):
            kwargs["image_retry_message_factory"](kwargs["image_retry_limits"][0])
            return _completion("vision hello")

        with mock.patch(
            "ComfyUI_LLM_API.node.encode_comfy_image_batch_to_data_urls",
            return_value=["data:image/png;base64,abc"],
        ) as encode_images:
            with mock.patch("ComfyUI_LLM_API.node.create_openai_client", return_value=object()):
                with mock.patch(
                    "ComfyUI_LLM_API.node.create_chat_completion",
                    side_effect=fake_create_chat_completion,
                ):
                    response = _run_node(
                        images=object(),
                        image_preset="custom",
                        custom_image_max_mb=2.5,
                        custom_image_max_side=1400,
                    )

        self.assertEqual(response, ("vision hello", ""))
        self.assertEqual(encode_images.call_args.kwargs["max_bytes"], 2_500_000)
        self.assertEqual(encode_images.call_args.kwargs["max_side"], 1400)

    def test_value_error_returns_sanitized_error_string(self):
        with mock.patch("ComfyUI_LLM_API.node.logger.warning") as warning:
            with mock.patch(
                "ComfyUI_LLM_API.node.create_openai_client",
                side_effect=ValueError("bad sk-secret"),
            ):
                response = _run_node()

        warning.assert_called_once()
        self.assertNotIn("sk-secret", " ".join(str(arg) for arg in warning.call_args.args))

        self.assertEqual(response[0], "")
        self.assertTrue(response[1].startswith("LLM API Error: "))
        self.assertNotIn("sk-secret", response[1])

    def test_empty_completion_returns_error_output(self):
        with mock.patch("ComfyUI_LLM_API.node.create_openai_client", return_value=object()):
            with mock.patch(
                "ComfyUI_LLM_API.node.create_chat_completion",
                return_value=SimpleNamespace(choices=[]),
            ):
                with mock.patch("ComfyUI_LLM_API.node.logger.warning"):
                    response = _run_node()

        self.assertEqual(response[0], "")
        self.assertTrue(response[1].startswith("LLM API Error: "))
        self.assertIn("no choices", response[1].lower())


if __name__ == "__main__":
    unittest.main()
