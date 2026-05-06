import time


def _is_failed_to_process_image_error(exc):
    return "failed to process image" in str(exc).lower()


def create_completion_with_image_retry(
    create_fn,
    model,
    temperature,
    build_messages_for_limit,
    image_byte_limits,
    retry_sleep_seconds=0.4,
):
    if not image_byte_limits:
        raise ValueError("image_byte_limits must contain at least one limit")

    for idx, max_bytes in enumerate(image_byte_limits):
        messages = build_messages_for_limit(max_bytes)
        try:
            return create_fn(model=model, messages=messages, temperature=temperature)
        except Exception as exc:
            can_retry = idx < len(image_byte_limits) - 1
            if (not can_retry) or (not _is_failed_to_process_image_error(exc)):
                raise
            if retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
