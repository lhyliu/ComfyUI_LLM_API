import os
import sys

from PIL import Image

if __package__:
    from ..image_encoding import encode_comfy_image_batch_to_data_urls
    from ..image_presets import resolve_image_settings
    from ..llm_request import (
        build_chat_messages,
        create_chat_completion,
        create_openai_client,
        extract_completion_text,
        format_llm_error,
    )
    from ..provider_profiles import resolve_provider_profile
else:
    PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CUSTOM_NODES_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
    if CUSTOM_NODES_DIR not in sys.path:
        sys.path.insert(0, CUSTOM_NODES_DIR)

    from ComfyUI_LLM_API.image_encoding import encode_comfy_image_batch_to_data_urls
    from ComfyUI_LLM_API.image_presets import resolve_image_settings
    from ComfyUI_LLM_API.llm_request import (
        build_chat_messages,
        create_chat_completion,
        create_openai_client,
        extract_completion_text,
        format_llm_error,
    )
    from ComfyUI_LLM_API.provider_profiles import resolve_provider_profile


TEXT_PROMPT = "Reply with a short provider smoke-test confirmation."
IMAGE_PROMPT = "Describe this image in one short sentence."


def _env(name, default=""):
    return os.environ.get(name, default).strip()


def _require_env(name):
    value = _env(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _run_completion(client, model, system_prompt, prompt, image_urls=None):
    messages = build_chat_messages(
        system_prompt=system_prompt,
        prompt=prompt,
        image_urls=image_urls,
    )
    completion = create_chat_completion(
        client=client,
        model=model,
        messages=messages,
        temperature=0.2,
        has_images=bool(image_urls),
        image_retry_message_factory=(
            (lambda _limit: messages) if image_urls else None
        ),
        image_retry_limits=((len(image_urls),) if image_urls else None),
    )
    return extract_completion_text(completion)


def _load_image_url(path):
    image_settings = resolve_image_settings("balanced", 1.5, 1280)
    image = Image.open(path).convert("RGB")
    return encode_comfy_image_batch_to_data_urls(
        [image],
        max_images=1,
        max_bytes=image_settings["max_bytes"],
        max_side=image_settings["max_side"],
    )[0]


def main():
    api_key = ""
    try:
        provider_profile = _env("LLM_API_PROVIDER_PROFILE", "openai")
        base_url_override = _env("LLM_API_BASE_URL")
        api_key = _require_env("LLM_API_KEY")
        model = _require_env("LLM_API_MODEL")
        image_path = _env("LLM_API_IMAGE_PATH")

        provider = resolve_provider_profile(provider_profile, base_url_override)
        if not provider["base_url"]:
            raise ValueError("LLM_API_BASE_URL is required for custom provider profiles")

        print(f"provider: {provider_profile}")
        print(f"model: {model}")

        client = create_openai_client(api_key=api_key, base_url=provider["base_url"])
        text_response = _run_completion(
            client=client,
            model=model,
            system_prompt="You are a concise smoke-test assistant.",
            prompt=TEXT_PROMPT,
        )
        print(f"text: ok ({text_response[:80]})")

        if not image_path:
            print("image: skipped")
            return 0
        if not provider["supports_images"]:
            raise ValueError(f"Provider profile {provider_profile} does not support images")

        image_url = _load_image_url(image_path)
        image_response = _run_completion(
            client=client,
            model=model,
            system_prompt="You are a concise image smoke-test assistant.",
            prompt=IMAGE_PROMPT,
            image_urls=[image_url],
        )
        print(f"image: ok ({image_response[:80]})")
        return 0
    except Exception as exc:
        print(format_llm_error(exc, api_key=api_key))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
