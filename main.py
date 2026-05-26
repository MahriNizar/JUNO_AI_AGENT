# In main_agent.py
import os
import functools
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# Import all our components
from state_and_class import AgentState, FoldedMemory, ToolExecutor
from skill_registry import SkillRegistry
from nodes import AgentNode, ArgumentFormatterNode, MemoryFolderNode, tool_node

FOLDING_NUMBER =5
USER_PROMPT = "Run a full BEC audit for test_data/scenario8_one.csv"
# This function defines the graph's conditional logic
def should_continue(state: AgentState) -> str:
    """
    The main conditional edge.
    Reads the last message from the agent_node and decides where to go.
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    
    if isinstance(last_message, tuple):
        _, last_message = last_message

    name = getattr(last_message, "name", None)
    
    if name == "SelectSkill":
        return "format_args"
    elif name == "FinalAnswer":
        return "__end__"
    return "__end__"

def route_after_tool(state: AgentState) -> str:
    """
    Decide where to go after a tool call:
    - 'fold' every folding number call
    - 'agent' otherwise
    """
    count = state.get("tool_call_count", 0)

    if count > 0 and count % FOLDING_NUMBER == 0:
        return "fold"
    return "agent"

def formatter_should_continue(state: AgentState) -> str:
    """
    Reads the last message from the formatter_node and decides
    whether to execute the tool or loop back to the agent.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Validation passed, go to executor
        return "executor"

    # Validation failed, go reflect in memory
    return "agent"

# --- Main Application ---

def build_graph(skill_folder : str):
    registry = SkillRegistry(skill_folder)
    executor = ToolExecutor(registry=registry)
    
   
    agent_llm = ChatOllama(model="gpt-oss:20b", #"gpt-oss:20b"
     base_url="http://127.0.0.1:13444" )

    # 2. Instantiate our nodes
    agent_node = AgentNode(registry, agent_llm)
    formatter_node = ArgumentFormatterNode(registry, agent_llm)
    memory_folder_node = MemoryFolderNode(agent_llm)
    
    # We use functools.partial to "bind" the executor to our tool_node
    # This makes it a clean function for LangGraph
    bound_tool_node = functools.partial(tool_node, executor=executor)

    # 3. Define the StateGraph
    workflow = StateGraph(AgentState)

    # 4. Add all the nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("executor", bound_tool_node)
    workflow.add_node("memory_folder", memory_folder_node)

    # 5. Wire the graph with edges
    
    # The entrypoint is the agent
    workflow.set_entry_point("agent")


    # The agent's output is conditional
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "format_args": "formatter",  # If it selected a skill, go format it
            "__end__": "__end__"           # If it gave a final answer, end
        }
    )
    
    workflow.add_conditional_edges(
    "formatter",
    formatter_should_continue,
    {
        "executor": "executor",          # If valid, run the tool
        "agent": "agent" # If invalid, go to memory
    }
    )
    
    workflow.add_conditional_edges(
    "executor",
    route_after_tool,
    {
        "fold": "memory_folder",
        "agent": "agent",
    }
    )
    workflow.add_edge("memory_folder", "agent")

    # 6. Compile the graph
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def main():
    app = build_graph("skills")

    # 7. Run the agent!
    print("--- RUNNING AGENT ---")
    
    
    inputs = {
        "messages": [("user", f"{USER_PROMPT}")],
        "memory": FoldedMemory(), # Start with an empty memory
        "plan": [],
        "tool_call_count": 0
    }
    
    config = {"configurable": {"thread_id": "juno-thread-1"},"recursion_limit": 150}

    for event in app.stream(inputs, config=config, stream_mode="values"):
        # stream_mode="values" gives us the full state at each step
        print("\n--- AGENT STEP ---")
        
        #event["messages"][-1].pretty_print()

if __name__ == "__main__":

    main()