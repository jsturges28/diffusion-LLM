"""Tests that the DiffusionGemma worker carries its stopping rule.

Strategy: source inspection through ``ast``. ``dgemma_worker`` cannot
be imported in this environment at all, because its quantized loader
reaches ``bitsandbytes``, which lives only in ``.venv-dgemma``; the
sampler it calls is tested by behaviour in
``tests/inference/test_dgemma_stopping.py``. What is left to pin is
the worker's half: that a run passes the rule it was asked for, keeps
it, and hands the same rule to every edit made on that run.

Passing proves the request's two thresholds reach the sampler on a
generation, survive in the state a resume reads, and reach the
sampler again on a resume, so an edit stops its canvas by the rule
the page's readout is drawn against.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Union

WORKER = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "backends"
    / "dgemma_worker.py"
)

RULE = ("confidence_threshold", "stability_threshold")

Function = Union[ast.FunctionDef, ast.AsyncFunctionDef]
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _method(name: str) -> Function:
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODES):
            continue
        if node.name == name:
            return node
    raise AssertionError(
        f"{name} is gone from dgemma_worker.py; update this test"
    )


def _call_keywords(function: Function, callee: str) -> Dict[str, str]:
    """The keyword arguments of the one call to ``callee``, as
    source text."""
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == callee:
            return {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
    raise AssertionError(f"{function.name} no longer calls {callee}")


def _stored_state(function: Function) -> Dict[str, str]:
    """The keys and values of the dict ``_store_state`` keeps."""
    for node in ast.walk(function):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant)
        ]
        if "frame_history" in keys:
            return {
                str(key.value): ast.unparse(value)
                for key, value in zip(
                    node.keys, node.values, strict=True
                )
                if isinstance(key, ast.Constant)
            }
    raise AssertionError("_store_state keeps no run state")


def test_a_generation_passes_the_requested_rule() -> None:
    passed = _call_keywords(
        _method("handle_generate"), "streaming_generate"
    )

    for name in RULE:
        assert passed.get(name) == f"params['{name}']", name


def test_the_rule_is_kept_for_an_edit() -> None:
    state = _stored_state(_method("_store_state"))

    for name in RULE:
        assert state.get(name) == f"params['{name}']", name


def test_a_resume_passes_the_rule_its_run_used() -> None:
    """From the stored state, not the request: an edit is part of
    the run it branches from, whatever the controls now say."""
    passed = _call_keywords(
        _method("handle_resume"), "streaming_resume"
    )

    for name in RULE:
        assert passed.get(name) == f"state['{name}']", name
