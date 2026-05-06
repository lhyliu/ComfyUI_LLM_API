import unittest

try:
    from . import test_path
except ImportError:
    import test_path
from ComfyUI_LLM_API.reasoning_filter import strip_reasoning_content


class ReasoningFilterTests(unittest.TestCase):
    def test_removes_closed_think_blocks(self):
        self.assertEqual(
            strip_reasoning_content("answer <think>hidden</think> visible"),
            "answer  visible",
        )

    def test_removes_unclosed_think_block(self):
        self.assertEqual(
            strip_reasoning_content("visible\n<think>hidden"),
            "visible",
        )


if __name__ == "__main__":
    unittest.main()
