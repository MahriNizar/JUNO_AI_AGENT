from typing import TypedDict, List, Union, Dict, Any, Literal, Optional, Annotated
from langgraph.graph.message import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
import importlib
from skill_registry import SkillRegistry
import json
import numpy as np
import pandas as pd
import tempfile
import os

try:
    from matplotlib.figure import Figure  # type: ignore
except Exception:
    Figure = None  # matplotlib may not be installed


# 1. Long-term memory: findings + compressed summary
class FoldedMemory(BaseModel):
    """
    Persistent memory that survives message trimming.
    - findings: append-only key facts (file paths, PMT IDs, anomalies)
    - summary: compressed narrative of all steps so far
    """
    findings: List[str] = Field(
        default_factory=list,
        description="Append-only list of key facts. Extracted by MemoryFolderNode. Never overwritten."
    )
    summary: str = Field(
        default="No history yet.",
        description="Compressed narrative of all steps executed so far."
    )



# 2.5 New Strict Plan Structure
class PlanStep(BaseModel):
    """
    A single step in the agent's plan.
    """
    id: str = Field(description="Unique identifier for the step (e.g., '1', '2', 'step_a').")
    description: str = Field(description="What to do in this step.")
    tool: Optional[str] = Field(default=None, description="The specific tool/skill to use, if known.")
    status: Literal["pending", "active", "success", "failed"] = Field(default="pending", description="Current status of the step.")
    error: Optional[str] = Field(default=None, description="Error message if the step failed.")
    output: Optional[str] = Field(default=None, description="Short summary of the output.")
    retries: int = Field(default=0, description="Number of times this step has been retried after failure.")

class AgentDecision(BaseModel):
    type: Literal["select", "final"]
    plan: List[PlanStep] = Field(description="The full, updated multi-step plan. This should include all original, new, and completed steps.")
    skill_to_use: Optional[str] = Field(description="The unique skill_id to be used, e.g., 'basic_stats.skill.v1' or 'juno_hardware_lookup'")
    final_answer: Optional[str] = Field(description="The complete, final answer to be shown to the user. Only if it was found, else leave it empty")
    thought: str = Field(description="A brief justification for why this skill was selected based on the memory and history.")


# Schema for the MemoryFolderNode's structured LLM output
class MemoryFoldResult(BaseModel):
    """Output of the MemoryFolderNode LLM call."""
    new_findings: List[str] = Field(
        default_factory=list,
        description="Key facts to permanently record from the recent history. "
                    "Examples: file paths, PMT IDs, anomaly counts, error patterns."
    )
    summary: str = Field(
        description="Compressed narrative of ALL steps executed so far, "
                    "integrating previous summary with new events."
    )

# 3. The main state for our entire graph
class AgentState(TypedDict):
    """
    The complete state of our agent, passed between nodes.
    """
    # The 'messages' key is special for LangGraph's MessagesState
    messages : Annotated[list[BaseMessage], add_messages]
    
    # Our custom long-term memory
    memory: FoldedMemory

    plan: List[PlanStep]
    tool_call_count: int


class ToolExecutor:
    """
    Dynamically imports and executes Python functions 
    based on Skill Card instructions.
    """
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Looks up the skill, imports the module, and runs the function.
        """
        print(f"--- TOOL EXECUTOR: Running '{tool_name}' ---")
        try:
            # 1. Get the "recipe" (Skill Card) from the registry
            skill_card = self.registry.get_skill_by_name(tool_name)
            
            exec_details = skill_card.execution_details
            
            if exec_details.type == "python_function":
                
                # 2. Get the module and function names from the YAML
                module_name = exec_details.call.module
                function_name = exec_details.call.function
                
                # 3. Dynamically import the module
                #    This is when the code in 'juno_tools/hardware_map.py'
                #    (like loading the CSV) is run for the first time.
                print(f"   -> Importing {module_name}.{function_name}")
                module = importlib.import_module(module_name)
                
                # 4. Get the specific function from the module
                tool_function = getattr(module, function_name)
                
                # 5. Execute the function with the provided arguments
                print(f"   -> Executing with args: {tool_args}")

                result = tool_function(**tool_args)
                
                print(f"   -> Raw result: {result}")
                return result
            
            else:
                raise NotImplementedError(f"Execution type '{exec_details.type}' not supported.")

        except Exception as e:
            print(f"--- TOOL EXECUTOR FAILED: {e} ---")
            # Return a clear error message to the agent for reflection
            return {"error": f"Tool execution failed: {e}"}
        

class ValidationAndArgs(BaseModel):
    """The structured output for the ArgumentFormatterNode."""
    validation_passed: bool = Field(description="True if the skill is appropriate, False if it is not.")
    args_or_reason: Union[Dict[str, Any], str] = Field(description="If validation_passed is True, this is the argument dict. If False, this is a string explaining *why* it failed.")






class NumpySafeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to safely handle NumPy and Pandas types
    that are not serializable by default.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            # Convert NumPy integers (int64, int32, etc.) to standard Python int
            return int(obj)
        elif isinstance(obj, np.floating):
            # Convert NumPy floats (float64, etc.) to standard Python float
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # Convert NumPy arrays to standard Python lists
            return obj.tolist()
        elif pd.isna(obj):
            # Convert Pandas NaT/NaN to None (which is JSON 'null')
            return None
        
        # This is the answer to "what if there are other types?"
        # Add more types here, e.g.:
        # elif isinstance(obj, datetime.date):
        #     return obj.isoformat()
        
        # Let the base class default method raise the TypeError for unknown types
        return super().default(obj)
    
MAX_OUTPUT_CHARS = 3000  # max serialized chars before truncation kicks in
MAX_LIST_ITEMS   = 30   # max list items to keep when truncating

def _truncate_large_output(normalized: Any) -> Any:
    """
    Post-normalization guard: if the serialized JSON of `normalized`
    exceeds MAX_OUTPUT_CHARS, truncate large lists and long string values
    to keep context window usage under control.
    """
    try:
        serialized = json.dumps(normalized, cls=NumpySafeEncoder)
    except Exception:
        serialized = str(normalized)

    if len(serialized) <= MAX_OUTPUT_CHARS:
        return normalized  # fits — no change

    # --- Truncate lists (most common source of blowup) ---
    if isinstance(normalized, list) and len(normalized) > MAX_LIST_ITEMS:
        return (
            normalized[:MAX_LIST_ITEMS]
            + [f"... ({len(normalized) - MAX_LIST_ITEMS} more items truncated, "
               f"{len(normalized)} total)"]
        )

    if isinstance(normalized, dict):
        truncated = {}
        for k, v in normalized.items():
            # Recurse into nested lists/dicts
            if isinstance(v, list) and len(v) > MAX_LIST_ITEMS:
                truncated[k] = (
                    v[:MAX_LIST_ITEMS]
                    + [f"... ({len(v) - MAX_LIST_ITEMS} more items truncated, "
                       f"{len(v)} total)"]
                )
            elif isinstance(v, str) and len(v) > MAX_OUTPUT_CHARS // 3:
                truncated[k] = v[:MAX_OUTPUT_CHARS // 3] + f"... (truncated, {len(v)} chars total)"
            else:
                truncated[k] = v
        return truncated

    # String fallback
    if isinstance(normalized, str) and len(normalized) > MAX_OUTPUT_CHARS:
        return normalized[:MAX_OUTPUT_CHARS] + f"... (truncated, {len(normalized)} chars total)"

    return normalized


def normalize_tool_output(result: Any) -> Any:
    """
    Normalize arbitrary tool outputs into something:
    - JSON-serializable (with NumpySafeEncoder),
    - and human-usable for the Agent.

    Rules:
    - Primitive types (None, bool, int, float, str) → return as-is.
    - dict / list / tuple → recurse into elements, then truncate if too large.
    - pandas.DataFrame → small preview as markdown (or string fallback).
    - pandas.Series → dict/preview.
    - matplotlib Figure → save to temp file, return a short message.
    - Unknown objects → str(...) fallback.
    """

    # 1) Primitives → just return as-is
    if result is None or isinstance(result, (bool, int, float, str)):
        return result

    # 2) Dict → recurse on values, then truncate
    if isinstance(result, dict):
        normalized = {k: normalize_tool_output(v) for k, v in result.items()}
        return _truncate_large_output(normalized)

    # 3) List / Tuple → recurse on items, then truncate
    if isinstance(result, (list, tuple)):
        normalized = [normalize_tool_output(v) for v in result]
        return _truncate_large_output(normalized)

    # 4) Pandas DataFrame → preview
    if isinstance(result, pd.DataFrame):
        try:
            preview_md = result.head().to_markdown(index=False)
        except Exception:
            # Fallback if tabulate/markdown is not available
            preview_md = result.head().to_string(index=False)

        return {
            "type": "dataframe_preview",
            "shape": [int(result.shape[0]), int(result.shape[1])],
            "columns": list(map(str, result.columns)),
            "preview": preview_md,
        }

    # 5) Pandas Series → small dict
    if isinstance(result, pd.Series):
        try:
            preview = result.head().to_dict()
        except Exception:
            preview = str(result.head())
        return {
            "type": "series_preview",
            "index": list(map(str, result.head().index)),
            "values": preview,
        }

    # 6) Matplotlib Figure (if matplotlib is available)
    if Figure is not None and isinstance(result, Figure):
        try:
            tmp_dir = tempfile.gettempdir()
            filename = f"juno_plot_{os.getpid()}_{id(result)}.png"
            path = os.path.join(tmp_dir, filename)
            # Save the figure
            result.savefig(path)
            msg = f"Image saved to {path}"
        except Exception as e:
            msg = f"Matplotlib figure could not be saved: {e}"
        return {
            "type": "image_file",
            "message": msg,
        }

    # 7) Objects that *have* a useful conversion method
    if hasattr(result, "to_dict"):
        try:
            return normalize_tool_output(result.to_dict())
        except Exception:
            pass

    if hasattr(result, "to_json"):
        try:
            # Keep it as a string so the Agent can read it
            return {
                "type": "json_text",
                "content": str(result.to_json()),
            }
        except Exception:
            pass

    # 8) Fallback: string representation
    try:
        return str(result)
    except Exception:
        return repr(result)