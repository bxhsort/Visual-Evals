"""Offline regressions for AST import failure and unchanged mPLUG calls."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from modelscope_ast_compat import modelscope_ast_compat


SOURCE = """
from modelscope.models import MODELS
@MODELS.register_module('visual-question-answering', module_name='mplug')
class MPLUG:
    def forward(self, image, question):
        return image, question
    async def prepare(self):
        pass
"""


def old_gast_tree(source=SOURCE):
    """Reproduce gast's declared-but-missing type_params on Python <3.12."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if "type_params" not in node._fields:
                node._fields = node._fields + ("type_params",)
            if hasattr(node, "type_params"):
                del node.type_params
    return tree


def strict_scan(tree):
    """ModelScope-style getattr traversal must see every declared field."""
    for node in ast.walk(tree):
        for field in node._fields:
            getattr(node, field)


class ModelScopeASTTests(unittest.TestCase):
    def test_missing_fields_are_filled_without_changing_registration_or_ast(self):
        tree = old_gast_tree()
        with self.assertRaisesRegex(AttributeError, "type_params"):
            strict_scan(tree)
        expected = ast.parse(SOURCE)
        for node in ast.walk(expected):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if "type_params" not in node._fields:
                    node._fields = node._fields + ("type_params",)
                node.type_params = []
        gast = types.ModuleType("gast")
        gast.parse = Mock(return_value=tree)
        original = gast.parse
        with patch.dict(sys.modules, {"gast": gast}):
            with modelscope_ast_compat():
                self.assertIs(gast.parse(SOURCE, filename="official.py"), tree)
                strict_scan(tree)
                self.assertEqual(ast.dump(tree, include_attributes=True),
                                 ast.dump(expected, include_attributes=True))
            self.assertIs(gast.parse, original)
        original.assert_called_once_with(SOURCE, filename="official.py")

    def test_existing_type_parameters_and_native_parser_are_unchanged(self):
        # Use real Python 3.12+ generics when this test runs on those versions.
        source = "class A[T]:\n    def f[U](self, x: U) -> U: return x\n" if sys.version_info >= (3, 12) else SOURCE
        tree = ast.parse(source)
        before = ast.dump(tree, include_attributes=True)
        params = [node.type_params for node in ast.walk(tree)
                  if hasattr(node, "type_params")]
        gast = types.ModuleType("gast")
        gast.parse = Mock(return_value=tree)
        native_parse = ast.parse
        with patch.dict(sys.modules, {"gast": gast}):
            with modelscope_ast_compat():
                self.assertIs(gast.parse(source), tree)
                self.assertIs(ast.parse, native_parse)
                self.assertEqual(ast.dump(tree, include_attributes=True), before)
                after_params = [node.type_params for node in ast.walk(tree)
                                if hasattr(node, "type_params")]
                for before_list, after_list in zip(params, after_params):
                    self.assertIs(before_list, after_list)

    def test_parse_failure_propagates_and_parser_is_restored(self):
        gast = types.ModuleType("gast")
        gast.parse = Mock(side_effect=SyntaxError("invalid source"))
        original = gast.parse
        with patch.dict(sys.modules, {"gast": gast}):
            with self.assertRaisesRegex(SyntaxError, "invalid source"):
                with modelscope_ast_compat():
                    gast.parse("invalid")
            self.assertIs(gast.parse, original)

    def test_unrelated_missing_fields_are_not_hidden(self):
        tree = old_gast_tree()
        del tree.body[1].name
        gast = types.ModuleType("gast")
        gast.parse = Mock(return_value=tree)
        with patch.dict(sys.modules, {"gast": gast}):
            with modelscope_ast_compat():
                with self.assertRaisesRegex(AttributeError, "name"):
                    strict_scan(gast.parse(SOURCE))

    def test_modern_modelscope_without_gast_can_initialize(self):
        with patch.dict(sys.modules, {"gast": None}):
            with modelscope_ast_compat():
                self.assertEqual(ast.parse("1").body[0].value.value, 1)

    def test_mplug_initialization_and_vqa_inputs_remain_identical(self):
        # Load only MPLUG's class; no torch/model dependencies are installed.
        source = Path(__file__).with_name("compute_dpg_bench.py").read_text()
        syntax = ast.parse(source)
        node = next(item for item in syntax.body if isinstance(item, ast.ClassDef)
                    and item.name == "MPLUG")
        scope = {"torch": types.SimpleNamespace(nn=types.SimpleNamespace(Module=object)),
                 "modelscope_ast_compat": modelscope_ast_compat}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "compute_dpg_bench.py", "exec"), scope)
        gast = types.ModuleType("gast")
        gast.parse = Mock(side_effect=lambda *_args, **_kwargs: old_gast_tree())
        original = gast.parse
        scorer = Mock(return_value={"text": "yes"})

        def pipeline(task, model, device):
            self.assertIsNot(gast.parse, original)
            strict_scan(gast.parse(SOURCE))
            self.assertEqual((task, model, device), ("vqa", "/weights/mplug", "cuda:0"))
            return scorer

        pipelines = types.ModuleType("modelscope.pipelines")
        pipelines.pipeline = Mock(side_effect=pipeline)
        constant = types.ModuleType("modelscope.utils.constant")
        constant.Tasks = types.SimpleNamespace(visual_question_answering="vqa")
        with patch.dict(sys.modules, {"gast": gast, "modelscope.pipelines": pipelines,
                                      "modelscope.utils.constant": constant}):
            model = scope["MPLUG"](ckpt="/weights/mplug", device="cuda:0")
            self.assertIs(gast.parse, original)
            self.assertEqual(model.vqa("image.png", "Is there a red car?"), "yes")
            scorer.assert_called_once_with({"image": "image.png", "question": "Is there a red car?"})


if __name__ == "__main__":
    unittest.main()
