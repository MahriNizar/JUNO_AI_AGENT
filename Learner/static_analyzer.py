import ast
import json
import os
from typing import Dict, Any, Optional, List, Set


# ---------------------------------------------------------------------------
# Helper: collect all function names called inside an AST node
# ---------------------------------------------------------------------------
class _CallCollector(ast.NodeVisitor):
    """Walk an AST subtree and collect every function-call name."""

    def __init__(self):
        self.called_names: Set[str] = set()

    def visit_Call(self, node: ast.Call):
        # Simple call: foo(...)
        if isinstance(node.func, ast.Name):
            self.called_names.add(node.func.id)
        # Method call on a module/object: mod.foo(...) — skip, not a local helper
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Build a map  { function_name -> ast.FunctionDef }  for the whole file
# ---------------------------------------------------------------------------
def _build_function_map(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
    """Return a dict of all top-level function definitions in the file."""
    func_map: Dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_map[node.name] = node
    return func_map


def _build_constant_map(tree: ast.Module) -> Dict[str, Any]:
    """
    Extract top-level constant assignments of the form  NAME = <literal>.
    Returns { 'CONSTANT_NAME': resolved_value }.
    Only resolves simple literals (numbers, strings, booleans, None).
    """
    constants: Dict[str, Any] = {}
    for node in ast.iter_child_nodes(tree):  # top-level only
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    value = ast.literal_eval(node.value)
                    constants[target.id] = value
                except (ValueError, TypeError):
                    pass  # not a simple literal — skip
    return constants


# ---------------------------------------------------------------------------
# Recursively resolve helpers called by the target (same-file only)
# ---------------------------------------------------------------------------
def _resolve_helpers(
    target_node: ast.FunctionDef,
    func_map: Dict[str, ast.FunctionDef],
    target_name: str,
) -> List[ast.FunctionDef]:
    """
    Walk the target function, find every function it calls that is defined
    in the same file, then recurse into those helpers as well.
    Returns a deduplicated, ordered list of helper AST nodes.
    """
    resolved_order: List[str] = []
    visited: Set[str] = {target_name}  # don't re-include the target itself

    def _walk(node: ast.FunctionDef):
        collector = _CallCollector()
        collector.visit(node)
        for name in sorted(collector.called_names):  # sorted for determinism
            if name not in visited and name in func_map:
                visited.add(name)
                resolved_order.append(name)
                _walk(func_map[name])  # recurse into the helper

    _walk(target_node)
    return [func_map[n] for n in resolved_order]


class FunctionMetadataExtractor(ast.NodeVisitor):
    """
    Parses a Python file source code into an AST and extracts 
    metadata for a specific target function.
    """
    def __init__(self, target_function: str, constant_map: Dict[str, Any] = None):
        self.target_function = target_function
        self.found_function = False
        self.metadata = {}
        self.constant_map = constant_map or {}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Automatic callback when the AST walker finds a function definition.
        """
        if node.name != self.target_function:
            return

        self.found_function = True
        
        # 1. Extract Docstring
        docstring = ast.get_docstring(node)
        
        # 2. Extract Source Code (for the Neural Analyzer later)
        # unparse converts the AST node back to string code
        source_code = ast.unparse(node)

        # 3. Extract Return Type Hint
        return_type = "None"
        if node.returns:
            return_type = ast.unparse(node.returns)

        # 4. Extract Arguments
        args_data = self._parse_arguments(node.args)

        self.metadata = {
            "function_name": node.name,
            "docstring": docstring,
            "args": args_data,
            "return_type_hint": return_type,
            "source_code": source_code,
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }

    def _parse_arguments(self, args_node: ast.arguments) -> List[Dict[str, Any]]:
        """
        Maps AST argument nodes to a clean dictionary structure.
        Handles the complexity of mapping default values to arguments.
        """
        extracted_args = []
        
        # Combine positional only, standard args, and keyword only args
        # Note: We simplify for the LLM, treating pos-only as standard
        all_args = args_node.posonlyargs + args_node.args
        
        # Logic to map defaults to arguments
        # Defaults correspond to the last N arguments
        num_args = len(all_args)
        num_defaults = len(args_node.defaults)
        # Calculate the index where defaults start
        default_start_index = num_args - num_defaults

        for i, arg in enumerate(all_args):
            arg_name = arg.arg
            
            # Get Type Hint (Annotation)
            type_hint = "Any"
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)

            # Determine if it has a default value
            default_value = None
            is_required = True
            
            if i >= default_start_index:
                # This argument has a default
                default_node = args_node.defaults[i - default_start_index]
                
                # Try to resolve the actual value:
                # 1. If it's a literal (number, string, etc.) -> use it directly
                # 2. If it's a Name ref to a module constant -> resolve it
                # 3. Otherwise -> fall back to the unparsed string
                try:
                    default_value = ast.literal_eval(default_node)
                except (ValueError, TypeError):
                    if isinstance(default_node, ast.Name) and default_node.id in self.constant_map:
                        default_value = self.constant_map[default_node.id]
                    else:
                        default_value = ast.unparse(default_node)
                is_required = False

            extracted_args.append({
                "name": arg_name,
                "type": type_hint,
                "default": default_value,
                "required": is_required
            })

        # Handle **kwargs and *args if necessary, 
        # but for Skill Cards we usually focus on named args.
        if args_node.vararg:
             extracted_args.append({
                "name": f"*{args_node.vararg.arg}",
                "type": "tuple",
                "default": None,
                "required": False
            })
            
        if args_node.kwarg:
             extracted_args.append({
                "name": f"**{args_node.kwarg.arg}",
                "type": "dict",
                "default": None,
                "required": False
            })

        return extracted_args

def extract_function_info(file_path: str, function_name: str) -> Dict[str, Any]:
    """
    Main entry point to run the static analysis.
    Extracts metadata for the target function AND any same-file helpers it calls.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Parse the source into an AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"Syntax Error in target file: {e}"}

    # Build a map of top-level constants for resolving defaults
    constant_map = _build_constant_map(tree)

    # Initialize the visitor and walk the tree
    extractor = FunctionMetadataExtractor(function_name, constant_map)
    extractor.visit(tree)

    if not extractor.found_function:
        return {"error": f"Function '{function_name}' not found in {file_path}"}

    metadata = extractor.metadata

    # ---- Resolve same-file helper functions ----
    func_map = _build_function_map(tree)
    target_node = func_map.get(function_name)

    if target_node:
        helper_nodes = _resolve_helpers(target_node, func_map, function_name)

        if helper_nodes:
            # Build combined source: main function clearly labeled, then helpers
            parts = [f"# === MAIN FUNCTION ===\n{metadata['source_code']}"]
            helper_names = []
            for h_node in helper_nodes:
                parts.append(f"# === HELPER: {h_node.name}() ===\n{ast.unparse(h_node)}")
                helper_names.append(h_node.name)

            metadata["source_code"] = "\n\n".join(parts)
            metadata["helper_functions"] = helper_names
            print(f"   -> Resolved {len(helper_names)} same-file helper(s): {helper_names}")
        else:
            # No helpers — still label the main function clearly
            metadata["source_code"] = f"# === MAIN FUNCTION ===\n{metadata['source_code']}"
            metadata["helper_functions"] = []

    return metadata
