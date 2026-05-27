"""Tool definitions and execution for the Frontier assistant."""

import ast
from datetime import datetime
from google.genai import types


def get_current_time() -> str:
    """Returns the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


def simple_calculator(expression: str) -> str:
    """Safely evaluates a math expression using AST parsing with restricted builtins."""
    try:
        result = ast.literal_eval(expression)
        return str(result)
    except (ValueError, SyntaxError):
        pass

    try:
        # Restricted builtins to prevent code injection via eval
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


TOOL_DEFINITIONS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_current_time",
            description="Returns the current date and time. Use when the user asks about the current time or date.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="simple_calculator",
            description="Evaluates a mathematical expression. Use for calculations like addition, multiplication, etc.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="The math expression to evaluate, e.g. '2 + 3 * 4'",
                    ),
                },
                required=["expression"],
            ),
        ),
    ],
)


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatches tool calls to their implementations."""
    if tool_name == "get_current_time":
        return get_current_time()
    elif tool_name == "simple_calculator":
        return simple_calculator(tool_input.get("expression", ""))
    else:
        return f"Unknown tool: {tool_name}"
