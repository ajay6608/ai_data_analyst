"""
safe_executor.py
-----------------
Runs LLM-generated pandas code safely.

Why this matters (this is a great interview talking point): an LLM can
hallucinate or, worse, generate code that reads/writes files or accesses
the system. Before running anything, we:
  1. Parse the code into an AST and reject imports, dunder access, and a
     blocklist of dangerous names (open, exec, eval, os, sys, etc.)
  2. Execute with a restricted set of builtins - only safe functions like
     len(), sum(), range() are available, nothing that touches the OS.
  3. Only expose df, pd, np, plt, sns to the executed code - nothing else.
"""

import ast

FORBIDDEN_NAMES = {
    "exec", "eval", "open", "__import__", "compile", "input",
    "os", "sys", "subprocess", "shutil", "socket", "globals", "locals",
}

SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "round": round,
    "zip": zip, "enumerate": enumerate, "abs": abs, "print": print,
}


class UnsafeCodeError(Exception):
    """Raised when generated code fails the safety check."""
    pass


def is_code_safe(code: str) -> bool:
    """Walk the parsed code's AST and reject anything dangerous."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Generated code has a syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Generated code tries to import modules - not allowed.")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Generated code uses a forbidden name: '{node.id}'.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError("Generated code accesses dunder attributes - not allowed.")

    return True


def run_safe_code(code: str, df, pd, np, plt, sns):
    """
    Validate then execute the generated code in a restricted namespace.
    Returns (result, fig) - either may be None if the code didn't set them.
    """
    is_code_safe(code)

    local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)

    result = local_vars.get("result", None)
    fig = local_vars.get("fig", None)
    return result, fig
