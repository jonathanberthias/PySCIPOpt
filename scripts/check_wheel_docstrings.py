"""Check the installed wheel's docstrings and preservation of the source stub."""

import ast
import inspect
import sys
from pathlib import Path

from pyscipopt import scip


def without_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            node.body = [
                statement
                for statement in node.body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and (
                        isinstance(statement.value.value, str)
                        or statement.value.value is Ellipsis
                    )
                )
            ]
    return ast.dump(tree, include_attributes=False)


def main():
    project = Path(sys.argv[1]).resolve()
    source = project / "src/pyscipopt/scip.pyi"
    installed = Path(scip.__file__).with_name("scip.pyi")
    assert project not in installed.resolve().parents, installed
    source_content = source.read_bytes()
    installed_content = installed.read_bytes()

    if sys.version_info < (3, 10):
        assert source_content == installed_content, "Wheel stub differs from the source"
        print("Python <3.10: wheel contains the unchanged source stub")
        return

    source_tree = ast.parse(source_content)
    installed_tree = ast.parse(installed_content)

    for owner, name in [
        ("Model", "addVar"),
        ("Model", "addCons"),
        ("Eventhdlr", "eventexec"),
        (None, "exp"),
    ]:
        suite = installed_tree.body
        runtime_owner = scip
        if owner:
            suite = next(
                node
                for node in suite
                if isinstance(node, ast.ClassDef) and node.name == owner
            ).body
            runtime_owner = getattr(scip, owner)
        function = next(
            node
            for node in suite
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        docstring = ast.get_docstring(function)
        assert docstring, f"Missing docstring for {owner}.{name}"
        assert docstring == inspect.getdoc(getattr(runtime_owner, name)), name
        print(f"Docstring matches runtime: {owner or 'scip'}.{name}")

    assert without_docstrings(source_tree) == without_docstrings(installed_tree), (
        "Wheel stub annotations differ from the source"
    )
    assert not any(
        ast.get_docstring(node)
        for node in ast.walk(ast.parse(source_content))
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
    ), "Source stub gained docstrings"
    print("Stub annotations preserved; source stub still contains no docstrings")


if __name__ == "__main__":
    main()
