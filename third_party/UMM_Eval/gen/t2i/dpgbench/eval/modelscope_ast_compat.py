"""Handle optional Python 3.12 AST fields in older ModelScope/gast scanners.

Some gast versions declare ClassDef.type_params on Python 3.8 but leave it
unset when converting the native AST. ModelScope then reads every declared
field with getattr and fails before loading mPLUG. An absent type-parameter
list means [], while an existing list must be preserved, including generics
on Python 3.12. No model, image, question or scoring behavior is patched.
"""
import ast
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch


@contextmanager
def modelscope_ast_compat():
    """Normalize gast trees only while importing/initializing ModelScope."""
    try:
        import gast
    except ModuleNotFoundError as exc:
        # Newer ModelScope versions use the native AST without gast.
        if exc.name != "gast":
            raise
        yield
        return

    original_parse = gast.parse

    @wraps(original_parse)
    def parse(*args, **kwargs):
        tree = original_parse(*args, **kwargs)
        for node in ast.walk(tree):
            if (type(node).__name__ in ("ClassDef", "FunctionDef", "AsyncFunctionDef")
                    and "type_params" in node._fields
                    and not hasattr(node, "type_params")):
                node.type_params = []
        return tree

    with patch.object(gast, "parse", parse):
        yield
