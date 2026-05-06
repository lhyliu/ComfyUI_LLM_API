import json
import os
import tomllib
import unittest

try:
    from . import test_path
except ImportError:
    import test_path


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PYPROJECT_PATH = os.path.join(ROOT_DIR, "pyproject.toml")
LICENSE_PATH = os.path.join(ROOT_DIR, "LICENSE")


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

    def test_load_image_examples_reference_tracked_assets(self):
        paths = [
            os.path.join(EXAMPLES_DIR, name)
            for name in os.listdir(EXAMPLES_DIR)
            if name.endswith(".json")
        ]

        for path in paths:
            with self.subTest(path=path):
                with open(path, "r", encoding="utf-8") as handle:
                    workflow = json.load(handle)
                for node in workflow.get("nodes", []):
                    if node.get("type") != "LoadImage":
                        continue
                    image_name = node.get("widgets_values", [""])[0]
                    image_path = os.path.join(EXAMPLES_DIR, image_name)
                    self.assertTrue(
                        os.path.isfile(image_path),
                        f"{image_name} is missing for {path}",
                    )

    def test_registry_release_metadata_is_present(self):
        self.assertTrue(os.path.isfile(PYPROJECT_PATH), "pyproject.toml is missing")
        self.assertTrue(os.path.isfile(LICENSE_PATH), "LICENSE is missing")

        with open(PYPROJECT_PATH, "rb") as handle:
            metadata = tomllib.load(handle)

        project = metadata["project"]
        self.assertEqual(project["name"], "ComfyUI_LLM_API")
        self.assertRegex(project["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(project["license"]["file"], "LICENSE")
        self.assertIn("openai>=1.0.0", project["dependencies"])
        self.assertIn("Pillow", project["dependencies"])
        self.assertIn("numpy", project["dependencies"])
        self.assertIn("Repository", project["urls"])
        dev_dependencies = metadata["project"]["optional-dependencies"]["dev"]
        self.assertIn("pytest", dev_dependencies)
        self.assertIn("comfy-cli", dev_dependencies)

        comfy = metadata["tool"]["comfy"]
        self.assertEqual(comfy["PublisherId"], "lhyliu")
        self.assertEqual(comfy["DisplayName"], "ComfyUI LLM API")


if __name__ == "__main__":
    unittest.main()
