import ast
import unittest
from pathlib import Path

from llm_settings import LLM_MODEL_ID, format_llm_model_label


APP_SOURCE = Path(__file__).resolve().parents[1] / "aecreformauth.py"


def is_configured_model(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "LLM_MODEL"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    )


class LLMSettingsTests(unittest.TestCase):
    def test_configured_model_id(self):
        self.assertEqual(LLM_MODEL_ID, "meta/meta-llama-3-70b-instruct")

    def test_format_llm_model_label_shows_identifier(self):
        self.assertEqual(
            format_llm_model_label(LLM_MODEL_ID),
            "LLM model: meta/meta-llama-3-70b-instruct",
        )

    def test_format_llm_model_label_preserves_empty_value(self):
        self.assertEqual(format_llm_model_label(""), "LLM model: ")

    def test_app_uses_configured_model_for_api_and_sidebar(self):
        tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

        config = classes["Config"]
        config_assignment = next(
            statement
            for statement in config.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "LLM_MODEL"
                for target in statement.targets
            )
        )
        self.assertIsInstance(config_assignment.value, ast.Name)
        self.assertEqual(config_assignment.value.id, "LLM_MODEL_ID")

        llama_method = next(
            node for node in classes["LLaMAClient"].body
            if isinstance(node, ast.FunctionDef) and node.name == "generate_response"
        )
        model_calls = [
            node for node in ast.walk(llama_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stream"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "replicate"
        ]
        self.assertEqual(len(model_calls), 1)
        self.assertTrue(is_configured_model(model_calls[0].args[0]))

        app = classes["RAGApp"]
        sidebar_method = next(
            node for node in app.body
            if isinstance(node, ast.FunctionDef) and node.name == "render_sidebar"
        )
        sidebar_block = next(
            node for node in sidebar_method.body
            if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Attribute)
            and node.items[0].context_expr.attr == "sidebar"
        )
        captions = [
            statement.value
            for statement in sidebar_block.body
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "caption"
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "st"
        ]
        self.assertEqual(len(captions), 1)
        label_call = captions[0].args[0]
        self.assertIsInstance(label_call, ast.Call)
        self.assertIsInstance(label_call.func, ast.Name)
        self.assertEqual(label_call.func.id, "format_llm_model_label")
        self.assertTrue(is_configured_model(label_call.args[0]))


if __name__ == "__main__":
    unittest.main()
