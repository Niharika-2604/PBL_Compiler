# Manipulate and inspect AST
"""
ast_utils.py

This module provides AST-based feature extraction functions for code.
It parses Python code into an AST and extracts structural features
useful for machine learning tasks.

Author: Your Name
"""

import ast

def get_ast_features(code: str) -> dict:
    """
    Extract basic features from the AST of the code.

    Args:
        code (str): Python source code string

    Returns:
        dict: A dictionary of code-level features
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print("❌ SyntaxError while parsing code:", e)
        return {}

    features = {
        "num_functions": 0,
        "num_loops": 0,
        "num_ifs": 0,
        "num_returns": 0,
        "num_calls": 0,
        "num_assignments": 0
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            features["num_functions"] += 1
        elif isinstance(node, (ast.For, ast.While)):
            features["num_loops"] += 1
        elif isinstance(node, ast.If):
            features["num_ifs"] += 1
        elif isinstance(node, ast.Return):
            features["num_returns"] += 1
        elif isinstance(node, ast.Call):
            features["num_calls"] += 1
        elif isinstance(node, ast.Assign):
            features["num_assignments"] += 1

    return features


def dump_ast(code: str) -> None:
    """
    Print a tree-like AST dump for visualization.

    Args:
        code (str): Source code input
    """
    try:
        tree = ast.parse(code)
        print(ast.dump(tree, indent=2))
    except SyntaxError as e:
        print("❌ Could not parse code:", e)


if __name__ == "__main__":
    # Demo
    example_code = """
def test_loop(n):
    result = 0
    for i in range(n):
        if i % 2 == 0:
            result += i * i
    return result
"""
    print("📋 Example Code:\n", example_code)
    print("\n🔍 AST Feature Extraction")
    print(get_ast_features(example_code))

    print("\n🌳 Raw AST Dump")
    dump_ast(example_code)