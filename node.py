import logging
import os

from .image_encoding import encode_comfy_image_batch_to_data_urls
from .image_presets import IMAGE_PRESET_NAMES, resolve_image_settings
from .llm_request import (
    LLMAPIUserError,
    build_chat_messages,
    create_chat_completion,
    create_openai_client,
    extract_completion_text,
    format_llm_error,
    is_openai_error,
)
from .provider_profiles import (
    PROVIDER_PROFILE_NAMES,
    resolve_provider_profile,
)
from .reasoning_filter import strip_reasoning_content


logger = logging.getLogger(__name__)


def _resolve_api_key(api_key, api_key_env):
    env_name = str(api_key_env or "").strip()
    if not env_name:
        return api_key

    value = os.environ.get(env_name, "")
    if not value:
        raise ValueError(f"Environment variable {env_name} is not set")
    return value


def _image_retry_limits(max_bytes):
    first = max(1, int(max_bytes))
    limits = (first, max(1, int(first * 0.72)), max(1, int(first * 0.51)))
    deduped = []
    for limit in limits:
        if limit not in deduped:
            deduped.append(limit)
    return tuple(deduped)


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


class LLMAPIChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider_profile": (
                    PROVIDER_PROFILE_NAMES,
                    {"default": PROVIDER_PROFILE_NAMES[0]},
                ),
                "api_baseurl": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "api_key_env": ("STRING", {"default": "", "advanced": True}),
                "model": ("STRING", {"default": ""}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "You are a helpful assistant"},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "execution_seed": ("INT", {"default": 100, "min": 0, "max": 0xFFFFFFFF}),
                "filter_thinking": ("BOOLEAN", {"default": True}),
                "max_images": (
                    "INT",
                    {"default": 4, "min": 1, "max": 16, "step": 1, "advanced": True},
                ),
                "image_preset": (
                    IMAGE_PRESET_NAMES,
                    {"default": "balanced"},
                ),
                "custom_image_max_mb": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.1,
                        "max": 32.0,
                        "step": 0.1,
                        "advanced": True,
                    },
                ),
                "custom_image_max_side": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 1,
                        "max": 8_192,
                        "step": 1,
                        "advanced": True,
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "error")
    FUNCTION = "run_llmapi"
    CATEGORY = "ComfyUI LLM API"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        provider_profile,
        api_baseurl,
        model,
        temperature,
        max_images,
        image_preset,
        custom_image_max_mb,
        custom_image_max_side,
    ):
        try:
            resolve_provider_profile(provider_profile, api_baseurl)
            resolve_image_settings(
                image_preset,
                custom_image_max_mb,
                custom_image_max_side,
            )
            temperature = _as_float(temperature, "temperature")
            max_images = _as_int(max_images, "max_images")
        except ValueError as exc:
            return str(exc)
        if not str(model).strip():
            return "model is required"
        if not 0.0 <= temperature <= 2.0:
            return "temperature must be between 0.0 and 2.0"
        if not 1 <= max_images <= 16:
            return "max_images must be between 1 and 16"
        return True

    def run_llmapi(
        self,
        provider_profile,
        api_baseurl,
        api_key,
        api_key_env,
        model,
        system_prompt,
        prompt,
        temperature,
        execution_seed,
        filter_thinking,
        max_images,
        image_preset,
        custom_image_max_mb,
        custom_image_max_side,
        images=None,
    ):
        try:
            resolved_provider = resolve_provider_profile(provider_profile, api_baseurl)
            base_url = resolved_provider["base_url"]
            if not base_url:
                raise ValueError(f"api_baseurl is required for {provider_profile}")

            has_images = images is not None
            if has_images and not resolved_provider["supports_images"]:
                raise ValueError(
                    f"Provider profile {provider_profile} does not support images"
                )
            image_settings = resolve_image_settings(
                image_preset,
                custom_image_max_mb,
                custom_image_max_side,
            )
            resolved_api_key = _resolve_api_key(api_key, api_key_env)

            def build_messages_for_image_limit(max_bytes):
                image_urls = encode_comfy_image_batch_to_data_urls(
                    images,
                    max_images=max_images,
                    max_bytes=max_bytes,
                    max_side=image_settings["max_side"],
                )
                return build_chat_messages(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    image_urls=image_urls,
                )

            client = create_openai_client(api_key=resolved_api_key, base_url=base_url)
            messages = build_chat_messages(system_prompt=system_prompt, prompt=prompt)
            completion = create_chat_completion(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                has_images=has_images,
                image_retry_message_factory=(
                    build_messages_for_image_limit if has_images else None
                ),
                image_retry_limits=(
                    _image_retry_limits(image_settings["max_bytes"]) if has_images else None
                ),
            )
            response = extract_completion_text(completion)
            if filter_thinking:
                response = strip_reasoning_content(response)
            return (response, "")
        except Exception as exc:
            if not isinstance(exc, (LLMAPIUserError, ValueError)) and not is_openai_error(exc):
                raise
            safe_error = format_llm_error(
                exc,
                api_key=locals().get("resolved_api_key", api_key),
            )
            logger.warning("%s", safe_error)
            return ("", safe_error)
