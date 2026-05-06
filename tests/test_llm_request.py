import base64
import builtins
import inspect
import types
import unittest
from unittest import mock

from PIL import Image

try:
    from . import test_path
except ImportError:
    import test_path
from ComfyUI_LLM_API import image_encoding
from ComfyUI_LLM_API.image_encoding import encode_comfy_image_batch_to_data_urls
from ComfyUI_LLM_API.llm_request import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    LLMAPIUserError,
    build_chat_messages,
    create_chat_completion,
    create_openai_client,
    extract_completion_text,
    format_llm_error,
    is_openai_error,
)


class FakeTensor:
    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self):
        self.calls = []
        self.fail_first_image_attempt = False

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first_image_attempt and len(self.calls) == 1:
            raise RuntimeError("failed to process image")
        return FakeCompletion("hello")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class LLMRequestTests(unittest.TestCase):
    def test_builds_text_only_chat_messages(self):
        messages = build_chat_messages("system", "prompt")

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "prompt"},
            ],
        )

    def test_builds_chat_messages_with_multiple_images(self):
        image_urls = [
            "data:image/jpeg;base64,aaa",
            "data:image/jpeg;base64,bbb",
        ]

        messages = build_chat_messages("system", "describe", image_urls=image_urls)

        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        user_content = messages[1]["content"]
        self.assertEqual(user_content[0], {"type": "text", "text": "describe"})
        self.assertEqual(
            user_content[1:],
            [
                {"type": "image_url", "image_url": {"url": image_urls[0]}},
                {"type": "image_url", "image_url": {"url": image_urls[1]}},
            ],
        )

    def test_create_chat_completion_does_not_forward_seed(self):
        client = FakeClient()

        result = create_chat_completion(
            client=client,
            model="demo-model",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            has_images=False,
        )

        self.assertEqual(extract_completion_text(result), "hello")
        self.assertEqual(client.chat.completions.calls[0]["model"], "demo-model")
        self.assertNotIn("seed", client.chat.completions.calls[0])
        self.assertNotIn("execution_seed", client.chat.completions.calls[0])

    def test_image_completion_retry_rebuilds_messages_with_lower_limits(self):
        client = FakeClient()
        client.chat.completions.fail_first_image_attempt = True
        limits = []

        def build_messages(limit):
            limits.append(limit)
            return [{"role": "user", "content": f"limit {limit}"}]

        result = create_chat_completion(
            client=client,
            model="demo-model",
            messages=build_messages(100),
            temperature=0.5,
            has_images=True,
            image_retry_message_factory=build_messages,
            image_retry_limits=(100, 50),
        )

        self.assertEqual(extract_completion_text(result), "hello")
        self.assertEqual(limits, [100, 100, 50])
        self.assertEqual(len(client.chat.completions.calls), 2)
        self.assertEqual(
            client.chat.completions.calls[1]["messages"],
            [{"role": "user", "content": "limit 50"}],
        )

    def test_image_completion_requires_explicit_retry_policy(self):
        client = FakeClient()

        with self.assertRaises(ValueError):
            create_chat_completion(
                client=client,
                model="demo-model",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
                has_images=True,
                image_retry_message_factory=None,
                image_retry_limits=(100,),
            )

        with self.assertRaises(ValueError):
            create_chat_completion(
                client=client,
                model="demo-model",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
                has_images=True,
                image_retry_message_factory=lambda _limit: [],
                image_retry_limits=(),
            )

    def test_extract_completion_text_rejects_missing_or_empty_content(self):
        for content in (None, "", "   "):
            with self.subTest(content=content):
                with self.assertRaises(LLMAPIUserError):
                    extract_completion_text(FakeCompletion(content))

    def test_format_llm_error_redacts_api_key(self):
        error = RuntimeError("bad key sk-secret-value failed")

        formatted = format_llm_error(error, api_key="sk-secret-value")

        self.assertTrue(formatted.startswith("LLM API Error: "))
        self.assertNotIn("sk-secret-value", formatted)
        self.assertIn("[redacted]", formatted)

    def test_format_llm_error_scrubs_data_urls_and_truncates_long_messages(self):
        data_url = "data:image/jpeg;base64," + ("a" * 500)
        error = RuntimeError("bad request " + data_url + " " + ("b" * 500))

        formatted = format_llm_error(error)

        self.assertIn("[image data redacted]", formatted)
        self.assertNotIn("aaaaa", formatted)
        self.assertLessEqual(len(formatted), 260)

    def test_create_openai_client_reports_missing_dependency(self):
        real_import = builtins.__import__

        def import_without_openai(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named openai")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_openai):
            with self.assertRaises(LLMAPIUserError) as context:
                create_openai_client("sk-test", "https://example.test/v1")

        self.assertIn("requirements.txt", str(context.exception))

    def test_create_openai_client_uses_lazy_openai_import(self):
        calls = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        real_import = builtins.__import__

        def import_with_fake_openai(name, *args, **kwargs):
            if name == "openai":
                return fake_openai_module
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_with_fake_openai):
            client = create_openai_client("sk-test", "https://example.test/v1")

        self.assertIsInstance(client, FakeOpenAI)
        self.assertEqual(
            calls,
            [
                {
                    "api_key": "sk-test",
                    "base_url": "https://example.test/v1",
                    "timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS,
                    "max_retries": DEFAULT_MAX_RETRIES,
                }
            ],
        )

    def test_create_openai_client_accepts_explicit_timeout_policy(self):
        calls = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        real_import = builtins.__import__

        def import_with_fake_openai(name, *args, **kwargs):
            if name == "openai":
                return fake_openai_module
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_with_fake_openai):
            create_openai_client(
                "sk-test",
                "https://example.test/v1",
                timeout=30.0,
                max_retries=0,
            )

        self.assertEqual(calls[0]["timeout"], 30.0)
        self.assertEqual(calls[0]["max_retries"], 0)

    def test_is_openai_error_detects_openai_errors_when_available(self):
        class FakeOpenAIError(Exception):
            pass

        fake_openai_module = types.SimpleNamespace(OpenAIError=FakeOpenAIError)
        real_import = builtins.__import__

        def import_with_fake_openai(name, *args, **kwargs):
            if name == "openai":
                return fake_openai_module
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_with_fake_openai):
            self.assertTrue(is_openai_error(FakeOpenAIError("boom")))
            self.assertFalse(is_openai_error(RuntimeError("boom")))

    def test_is_openai_error_returns_false_when_openai_is_missing(self):
        real_import = builtins.__import__

        def import_without_openai(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named openai")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_openai):
            self.assertFalse(is_openai_error(RuntimeError("boom")))


class ImageBatchEncodingTests(unittest.TestCase):
    def test_image_encoder_has_no_legacy_size_defaults(self):
        signature = inspect.signature(image_encoding.encode_image_to_safe_jpeg_b64)

        self.assertEqual(signature.parameters["max_bytes"].default, inspect._empty)
        self.assertEqual(signature.parameters["max_side"].default, inspect._empty)
        self.assertFalse(hasattr(image_encoding, "DEFAULT_MAX_IMAGE_BYTES"))
        self.assertFalse(hasattr(image_encoding, "DEFAULT_MAX_IMAGE_SIDE"))

    def test_encodes_single_image_to_jpeg_data_url(self):
        image = Image.new("RGB", (8, 8), (20, 40, 60))
        urls = encode_comfy_image_batch_to_data_urls(
            [image],
            max_images=4,
            max_bytes=8_000,
            max_side=64,
        )

        self.assertEqual(len(urls), 1)
        self.assertTrue(urls[0].startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(urls[0].split(",", 1)[1])
        self.assertLessEqual(len(raw), 8_000)

    def test_truncates_image_batch_to_max_images(self):
        images = [Image.new("RGB", (4, 4), (idx, idx, idx)) for idx in range(6)]

        urls = encode_comfy_image_batch_to_data_urls(
            images,
            max_images=4,
            max_bytes=8_000,
            max_side=64,
        )

        self.assertEqual(len(urls), 4)


if __name__ == "__main__":
    unittest.main()
