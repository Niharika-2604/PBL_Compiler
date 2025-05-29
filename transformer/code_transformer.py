

import ast
import astor

# ===============================
# Transformation: Loop Unrolling
# ===============================

class LoopUnroller(ast.NodeTransformer):
    """
    A basic unroller that doubles loop iterations.
    Only works on simple for-loops over a range.
    """
    def visit_For(self, node):
        self.generic_visit(node)
        if isinstance(node.iter, ast.Call) and hasattr(node.iter.func, "id") and node.iter.func.id == "range":
            try:
                # Create a new while-loop that unrolls in steps of 2
                loop_var = node.target.id
                new_body = []
                loop_range_var = ast.Name(id=loop_var, ctx=ast.Load())
                loop_iter = node.body

                # Unroll by applying body twice per iteration
                loop_body = []
                for stmt in loop_iter:
                    loop_body.append(ast.fix_missing_locations(ast.copy_location(stmt, stmt)))
                for stmt in loop_iter:
                    loop_body.append(ast.fix_missing_locations(ast.copy_location(stmt, stmt)))

                while_node = ast.While(
                    test=ast.Compare(
                        left=ast.Name(id=loop_var, ctx=ast.Load()),
                        ops=[ast.Lt()],
                        comparators=[node.iter.args[0]]
                    ),
                    body=loop_body + [
                        ast.AugAssign(
                            target=ast.Name(id=loop_var, ctx=ast.Store()),
                            op=ast.Add(),
                            value=ast.Constant(value=2)
                        )
                    ],
                    orelse=[]
                )

                return [ast.Assign(
                            targets=[ast.Name(id=loop_var, ctx=ast.Store())],
                            value=ast.Constant(value=0)
                        ),
                        while_node]
            except Exception as e:
                print("Loop unroll failed:", e)
                return node
        return node

# ===============================
# Transformation: Dead Code Elimination
# ===============================

class DeadCodeEliminator(ast.NodeTransformer):
    """
    Removes unreachable branches like `if False:`.
    """
    def visit_If(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and node.test.value == False:
            return node.orelse or []
        return node

# ===============================
# Transformation: Function Inlining (Simple)
# ===============================

def inline_functions(code: str) -> str:
    """
    Simple string-based function inliner:
    Assumes basic use case: one-liner functions used once.
    """
    lines = code.split("\n")
    func_defs = {}
    new_lines = []

    # Phase 1: Collect simple one-line defs
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            name = stripped.split()[1].split("(")[0]
            i = lines.index(line)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("return "):
                body = lines[i + 1].strip().replace("return", "").strip()
                func_defs[name] = body
                new_lines.append("# Inlined function removed: " + name)
                continue
        if any(func_name + "(" in line for func_name in func_defs):
            for fname, fbody in func_defs.items():
                line = line.replace(f"{fname}(", f"({fbody}) if True else None # inlined ")
        new_lines.append(line)
    
    return "\n".join(new_lines)

# ===============================
# Transformation Entry Point
# ===============================

def apply_transformation(code: str, transformation_type: str) -> str:
    """
    Apply the given transformation to the source code.

    Args:
        code (str): Original source code
        transformation_type (str): e.g., 'loop_unrolling'

    Returns:
        str: Optimized/transformed code
    """

    try:
        tree = ast.parse(code)

        if transformation_type == "loop_unrolling":
            transformer = LoopUnroller()
            transformed_tree = transformer.visit(tree)
            ast.fix_missing_locations(transformed_tree)
            return astor.to_source(transformed_tree)

        elif transformation_type == "dead_code_elimination":
            transformer = DeadCodeEliminator()
            transformed_tree = transformer.visit(tree)
            ast.fix_missing_locations(transformed_tree)
            return astor.to_source(transformed_tree)

        elif transformation_type == "function_inlining":
            return inline_functions(code)

        else:
            print(f"⚠️ Unsupported transformation: {transformation_type}")
            return code

    except Exception as e:
        print("❌ Transformation failed:", e)
        return code

# ===============================
# Test Mode
# ===============================

if __name__ == "__main__":
    print("🧪 Code Transformer Test")

    sample_code = """
def square_sum(n):
    result = 0
    for i in range(n):
        result += i * i
    return result
"""

    print("Original Code:\n", sample_code)

    optimized = apply_transformation(sample_code, "loop_unrolling")
    print("\nTransformed Code (loop_unrolling):\n", optimized)

    dead_code = '''
def compute(x):
    if False:
        return x * x
    return x + x
'''
    optimized_dc = apply_transformation(dead_code, "dead_code_elimination")
    print("\nTransformed Code (dead_code_elimination):\n", optimized_dc)