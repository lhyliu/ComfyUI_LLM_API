import re


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n+")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def strip_reasoning_content(content):
    if not isinstance(content, str):
        return content

    filtered = _THINK_BLOCK_RE.sub("", content)
    filtered = _UNCLOSED_THINK_RE.sub("", filtered)
    filtered = _TRAILING_WS_RE.sub("\n", filtered)
    filtered = _BLANK_LINES_RE.sub("\n", filtered)
    return filtered.strip()
