import re

from .api_retry import create_completion_with_image_retry


DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 1
MAX_ERROR_MESSAGE_LENGTH = 220
_DATA_URL_RE = re.compile(r"data:[^,;\s]+(?:;[^,\s]+)?,[A-Za-z0-9+/=_-]+")


class LLMAPIUserError(Exception):
    pass


def create_openai_client(
    api_key,
    base_url,
    timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_retries=DEFAULT_MAX_RETRIES,
):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMAPIUserError(
            "OpenAI Python package is not installed. "
            "Run pip install -r requirements.txt."
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


def is_openai_error(exc):
    try:
        from openai import OpenAIError
    except ImportError:
        return False

    return isinstance(exc, OpenAIError)


def build_chat_messages(system_prompt, prompt, image_urls=None):
    messages = [{"role": "system", "content": system_prompt}]
    image_urls = image_urls or []

    if not image_urls:
        messages.append({"role": "user", "content": prompt})
        return messages

    user_content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    messages.append({"role": "user", "content": user_content})
    return messages


def create_chat_completion(
    client,
    model,
    messages,
    temperature,
    has_images=False,
    image_retry_message_factory=None,
    image_retry_limits=None,
):
    create_fn = client.chat.completions.create

    if not has_images:
        return create_fn(model=model, messages=messages, temperature=temperature)

    if image_retry_message_factory is None:
        raise ValueError("image_retry_message_factory is required for image requests")
    if not image_retry_limits:
        raise ValueError("image_retry_limits must contain at least one limit")

    return create_completion_with_image_retry(
        create_fn=create_fn,
        model=model,
        temperature=temperature,
        build_messages_for_limit=image_retry_message_factory,
        image_byte_limits=image_retry_limits,
        retry_sleep_seconds=0,
    )


def extract_completion_text(completion):
    choices = getattr(completion, "choices", None)
    if not choices:
        raise LLMAPIUserError("Provider response contained no choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content is None or str(content).strip() == "":
        raise LLMAPIUserError("Provider response contained empty content")
    return str(content)


def format_llm_error(exc, api_key=""):
    message = str(exc).strip()
    if api_key:
        message = message.replace(api_key, "[redacted]")
    message = _DATA_URL_RE.sub("[image data redacted]", message)
    if not message:
        message = exc.__class__.__name__
    if len(message) > MAX_ERROR_MESSAGE_LENGTH:
        message = message[: MAX_ERROR_MESSAGE_LENGTH - 3].rstrip() + "..."
    return "LLM API Error: " + message
