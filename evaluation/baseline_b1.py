"""
B1 Baseline: Standard LangChain ReAct Agent

This module implements a standard ReAct agent using LangGraph's prebuilt agent.
It uses the same task-specific skills as the main JUNO agent but without the multi-step planning,
validation, and memory folding architectures.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Callable, Union
import time
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import ast
import inspect
from langchain_core.tools import StructuredTool, Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent,AgentExecutor
from typing import Dict, List, Optional, Any, Tuple
# Import JUNO components
from skill_registry import SkillRegistry
from state_and_class import ToolExecutor, normalize_tool_output

SYSTEM_PROMPT_B1 = """You are an expert AI assistant for the JUNO (Jiangmen Underground Neutrino Observatory) detector.
Your purpose is to help operators monitor the detector, analyze data, and identify anomalies.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: a single JSON object string, for example {{"file_path": "data.csv"}}
Observation: the result of the action
... this Thought/Action/Action Input/Observation can repeat
Thought: I now know the final answer
Final Answer: the final answer to the user

Do not invent results. Valid information must come from tool observations.

Question: {input}
Thought: {agent_scratchpad}

IMPORTANT — CSV Inspection Protocol:
When you receive a request involving a CSV data file, your FIRST action MUST be to call `inspect_csv` on that file.
This tells you the file size, event count, and whether batching is needed.
Only AFTER inspecting the file should you proceed with analysis tools.
"""

SYSTEM_PROMPT_T = """You are an expert AI assistant for the JUNO (Jiangmen Underground Neutrino Observatory) detector.
Your purpose is to help operators monitor the detector, analyze data, and identify anomalies.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: a single JSON object string, for example {{"file_path": "data.csv"}}
Observation: the result of the action
... this Thought/Action/Action Input/Observation can repeat
Thought: I now know the final answer
Final Answer: the final answer to the user

Do not invent results. Valid information must come from tool observations.

Question: {input}
Thought: {agent_scratchpad}
"""






def _parse_untyped_input(raw: str) -> Dict[str, Any]:
    """
    Classic ReAct gives each tool one raw string input.
    We only parse it into kwargs; we do not validate names, types, enums, or required fields.
    """
    if isinstance(raw, dict):
        return raw

    raw = str(raw).strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    except Exception:
        pass

    return {"input": raw}

def _create_untyped_tool_function(executor: ToolExecutor, skill_name: str) -> Callable:
    def tool_func(raw_input: str) -> str:
        try:
            kwargs = _parse_untyped_input(raw_input)
            raw_result = executor.execute_tool(skill_name, kwargs)
            return normalize_tool_output(raw_result)
        except Exception as e:
            return f"Error executing {skill_name}: {str(e)}"

    return tool_func

def create_langchain_tools(skill_folder: str = "skills") -> List[Tool]:
    """
    Classic ReAct baseline: one untyped string input per tool.
    No Pydantic args_schema. No type-safe Skill Card argument model.
    """
    registry = SkillRegistry(skill_folder)
    executor = ToolExecutor(registry)

    tools = []

    for skill_name, skill in registry.skills.items():
        func = _create_untyped_tool_function(executor, skill_name)

        # Minimal prose description. You may keep this from the Skill Card,
        # but do not expose structured args_schema.
        description = (
            f"{skill.summary.description}\n"
            "Input must be a single JSON object string with argument names and values. "
            "Arguments are not validated before execution."
        )

        tools.append(
            Tool(
                name=skill_name,
                description=description,
                func=func,
            )
        )

    return tools


def create_react_agent_b1(config: Dict[str, Any] = None):
    """
    Create a standard ReAct agent with the benchmark configuration.
    """
    config = config or {}
    skill_folder = config.get("skill_folder", "skills")

    
    # 1. Setup Tools
    tools = create_langchain_tools(skill_folder)
    
    # 2. Setup LLM
    # Use the same model setup as the main agent
    llm = ChatOllama(
        model="gpt-oss:20b", #"gpt-oss:20b"
        base_url="http://127.0.0.1:13444",
    )
    
    
    



    agent_b1 = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT_T)



    return agent_b1 


