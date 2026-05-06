import json
import os
import unittest

try:
    from . import test_path
except ImportError:
    import test_path


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")


class ExampleWorkflowTests(unittest.TestCase):
    def test_example_workflows_are_clean_and_current(self):
        self.assertTrue(os.path.isdir(EXAMPLES_DIR), "examples directory is missing")
        paths = [
            os.path.join(EXAMPLES_DIR, name)
            for name in os.listdir(EXAMPLES_DIR)
            if name.endswith(".json")
        ]
        self.assertGreaterEqual(len(paths), 2)

        for path in paths:
            with self.subTest(path=path):
                with open(path, "r", encoding="utf-8") as handle:
                    workflow = json.load(handle)
                raw = json.dumps(workflow)

                self.assertIn("LLM_API_CHAT_NODE", raw)
                self.assertNotIn("COMFYUI_LLM_API_NODE", raw)
                self.assertNotIn("ref_image", raw)
                self.assertNotIn("video_url", raw)
                self.assertNotIn('"video"', raw)
                self.assertNotIn("sk-", raw)

                llm_nodes = [
                    node
                    for node in workflow.get("nodes", [])
                    if node.get("type") == "LLM_API_CHAT_NODE"
                ]
                self.assertGreaterEqual(len(llm_nodes), 1)
                for node in llm_nodes:
                    widgets = node.get("widgets_values", [])
                    self.assertEqual(widgets[2], "")
                    self.assertTrue(str(widgets[3]).endswith("_API_KEY"))
                    output_names = [output["name"] for output in node.get("outputs", [])]
                    self.assertEqual(output_names, ["response", "error"])


if __name__ == "__main__":
    unittest.main()
