from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.exceptions import OutputParserException


from typing import Dict
import json
from pydantic import ValidationError

from state_and_class import AgentState, FoldedMemory, AgentDecision, MemoryFoldResult, ToolExecutor, NumpySafeEncoder, ValidationAndArgs, normalize_tool_output, PlanStep
from skill_registry import SkillRegistry, create_pydantic_model_from_skill
from prompts import SYSTEM_PROMPT_2, FORMATTER_SYSTEM_PROMPT_2, MEMORY_FOLDER_PROMPT_2, SYSTEM_PROMPT_T


def _extract_user_query(messages: list[BaseMessage]) -> str:
    """Return the original user query from the message list. Always available."""
    user_msg = next((m for m in messages if isinstance(m, HumanMessage)), None)
    return user_msg.content if user_msg else "No user query found."


def _compact_plan_for_agent(plan: list) -> list[dict]:
    """
    Compact the plan for the AgentNode: completed steps get their output/error
    fields trimmed to save context tokens, while pending/active steps stay full.
    The agent still sees every step's id, description, tool, and status.
    """
    compacted = []
    for step in plan:
        d = step.model_dump()
        if d["status"] in ("success", "failed"):
            # Keep output summary short (first 120 chars)
            if d.get("output") and len(d["output"]) > 120:
                d["output"] = d["output"][:120] + "..."
            if d.get("error") and len(d["error"]) > 120:
                d["error"] = d["error"][:120] + "..."
        compacted.append(d)
    return compacted


def _window_plan_for_formatter(plan: list) -> list[dict]:
    """
    Window the plan for the FormatterNode: only the current pending step
    + last 2 completed steps. The formatter only needs immediate context
    to extract the right arguments (e.g., file paths from previous outputs).
    """
    # Find the first pending step
    pending_idx = None
    for i, step in enumerate(plan):
        if step.status == "pending":
            pending_idx = i
            break

    if pending_idx is None:
        # No pending steps — return last 3 steps for context
        return [s.model_dump() for s in plan[-3:]]

    # Take the 2 steps before the pending one + the pending step itself
    start = max(0, pending_idx - 2)
    window = plan[start : pending_idx + 1]
    return [s.model_dump() for s in window]


def _clean_content_for_history(msg: BaseMessage, max_len: int = 500) -> str:
    """Strip the 'plan' key from SelectSkill JSON to avoid duplicating it in recent history."""
    content = msg.content
    name = getattr(msg, 'name', None)
    if name == "SelectSkill":
        try:
            d = json.loads(content)
            d.pop("plan", None)
            content = json.dumps(d)
        except (json.JSONDecodeError, TypeError):
            pass
    if len(content) > max_len:
        content = content[:max_len] + "..."
    return content


class AgentNode:
    """
    The main reasoning node for the JUNO agent.
    It implements the "Select" and "Reflect" steps.
    """
    def __init__(self, registry: SkillRegistry, model: ChatOllama):
        self.registry = registry
        
        # 1. Get the "folded summaries" from all Skill Cards at startup
        self.skill_summaries = self.format_skill_summaries() 
        
        # 2. Bind the structured output to the model.
        # This forces the LLM to *always* output *only* a `SelectSkill` or `FinalAnswer` object.
        self.model = model.with_structured_output(AgentDecision)
        
        # 3. Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_T) # This is where the history goes
        ])
        
        # 4. Create the final runnable chain
        self.chain = self.prompt | self.model

    def _get_recent_history(self, messages: list[BaseMessage]) -> str:
        """Format the entire message history for the LLM."""
        return "\n".join(
            f"[{getattr(m, 'name', None) or m.type}] {_clean_content_for_history(m)}" for m in messages[-10:]
        )

    def format_skill_summaries(self) -> str:
        """Helper to format the skill summaries for the prompt."""
        summaries = []
        for skill in self.registry.get_all_skills():
            # We use the 'folded_summary' field from the Skill Card
            # as the high-level summary for tool selection.
            
            summary = skill.summary.description
            
            summaries.append(f"* **{skill.skill_id}**: {summary}")
        
        return "\n".join(summaries)

    def __call__(self, state: AgentState) -> dict:
        """
        This is the main entrypoint for the node in LangGraph.
        """
        print("--- AGENT NODE (Select/Reflect) ---")
        
        # 1. Get the current state
        messages = list(state["messages"])
        memory = state['memory']
        plan = state.get('plan', [])
        findings_list = memory.findings if memory.findings else []
        if findings_list:
            findings_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(findings_list))
        else:
            findings_str = "No findings recorded yet."
        # 2. Extract the original user query (always available)
        user_query = _extract_user_query(messages)

        # 3. Format the prompt with the latest info
        prompt_input = {
            "skill_summaries": self.skill_summaries,
            "recent_history": self._get_recent_history(state["messages"]),
            "folded_memory": memory.summary,
            "plan": json.dumps(_compact_plan_for_agent(plan), cls=NumpySafeEncoder, indent=2),
            "findings": findings_str,
            "user_query": user_query
        }
        
        # 3. Call the LLM with retry on parsing failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                decision = self.chain.invoke(prompt_input)
                break
            except OutputParserException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  -> Structured output failed (attempt {attempt+1}), retrying...")

        if decision.type == "select":
            print(f"  -> Decision: Select Skill ({decision.skill_to_use})")
            print(f"  -> Thought: {decision.thought}")
            return {"messages": [
                        AIMessage(content=decision.model_dump_json(), name="SelectSkill")],
                        "plan": decision.plan}
            
        elif decision.type == "final":

            print("\n=== FINAL ANSWER FOUND ===")
            print(decision.final_answer)
            print("==========================\n")
            return {"messages": 
                        AIMessage(content=decision.model_dump_json(indent=2), name="FinalAnswer"), "plan": decision.plan}
            
        else:
            raise ValueError(f"Unexpected decision type: {decision}")




class ArgumentFormatterNode:
    """
    Implements the "Deliberate" (Validate) step.
    Loads the full Skill Card for "Just-in-Time" context
    to generate tool arguments.
    """
    def __init__(self, registry: SkillRegistry, model: ChatOllama):
        self.registry = registry
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", FORMATTER_SYSTEM_PROMPT_2),
        ])
        self.chain = self.prompt | self.model.with_structured_output(ValidationAndArgs)




    def _get_recent_history(self, messages: list[BaseMessage]) -> str:
        """Format recent message history for the LLM (plan stripped)."""
        lines = []
        for m in messages[-5:]:
            name = getattr(m, 'name', None) or m.type
            content = _clean_content_for_history(m, max_len=500)
            lines.append(f"[{name}] {content}")
        return "\n".join(lines)

    def __call__(self, state: AgentState) -> dict:
        print("--- ARGUMENT FORMATTER NODE (Validate/Deliberate) ---")
        
        # 1. Find the 'SelectSkill' message (unchanged)
        select_skill_msg = next((m for m in reversed(state['messages']) if m.name == "SelectSkill"), None)
        
        try:
            skill_choice = json.loads(select_skill_msg.content)
            skill_id = skill_choice['skill_to_use']
            print(f"  -> Validating/Formatting arguments for: {skill_id}")
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error parsing skill choice: {e}")]}

        # 2. Get the full Skill Card (unchanged)
        try:
            skill_card = self.registry.get_skill_by_name(skill_id)
        except Exception as e:
             return {"messages": [AIMessage(content=f"Error loading skill card: {e}")]}

        # 3. Build findings string from memory
        memory = state["memory"]
        plan =state.get('plan', [])
        findings_list = memory.findings if memory.findings else []
        if findings_list:
            findings_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(findings_list))
        else:
            findings_str = "No findings recorded yet."

        # 4. Get JIT context and schema for the prompt
        param_schema_json = json.dumps(skill_card.summary.parameters.model_dump(), indent=2)
        
        # 5. Extract the original user query
        user_query = _extract_user_query(state["messages"])

        # 6. Build inputs for the prompt
        inputs = {
            "folded_memory": memory.summary,
            "recent_history": self._get_recent_history(state["messages"]),
            "findings": findings_str,
            "plan": json.dumps(_window_plan_for_formatter(plan), cls=NumpySafeEncoder, indent=2),
            "skill_id": skill_card.skill_id,
            "skill_name": skill_card.summary.name,
            "skill_description": skill_card.summary.description,
            "param_schema_json": param_schema_json,
            "usage_policy": skill_card.usage_policy.model_dump_json(indent=2),
            "output_schema": skill_card.output_schema.model_dump_json(indent=2),
            "user_query": user_query
        }

        try:
            # self.chain is defined in __init__ and returns ValidationAndArgs
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    validation_result = self.chain.invoke(inputs)
                    break
                except OutputParserException as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"  -> Formatter structured output failed (attempt {attempt+1}), retrying...")
        except Exception as e:
            return {"messages": [AIMessage(content=f"Validation or Formatting LLM call failed: {e}", name="ValidationFailed")]}

        # --- START OF NEW TYPE-SAFE LOGIC ---

        # 6. Check the high-level validation from the LLM
        if not validation_result.validation_passed:
            # FAILURE (Reason): LLM decided the skill was invalid.
            reason = str(validation_result.args_or_reason)
            validation_failed_message = AIMessage(
                content=reason, # This is the reason string
                name="ValidationFailed"
            )
            return {"messages": [validation_failed_message]}

        # 7. SUCCESS: Now, perform the critical type-safe validation
        try:
            # 7a. Get the raw argument dict returned by the LLM
            raw_args_dict = validation_result.args_or_reason
            if not isinstance(raw_args_dict, dict):
                 raise ValueError(f"LLM returned invalid argument format. Expected a dict, got {type(raw_args_dict)}")

            # 7b. Create the dynamic, type-safe Pydantic model
            tool_model = create_pydantic_model_from_skill(skill_card)
            
            # 7c. Validate and coerce the raw dict using the model.
            # This is where "31" -> 31 happens.
            validated_args = tool_model.model_validate(raw_args_dict)
            
            # 7d. Convert the validated Pydantic model back to a clean dict
            final_args_dict = validated_args.model_dump()

        except (ValidationError, ValueError, TypeError) as e:
            # FAILURE (Type Error): LLM said the skill was valid,
            # but provided bad/missing arguments that failed Pydantic validation.
            error_str = f"Argument validation failed: {e}."
            validation_failed_message = AIMessage(
                content=error_str,
                name="ValidationFailed"
            )
            return {"messages": [validation_failed_message]}

        # 8. FINAL SUCCESS: Build the ToolCall message with clean, validated args
        tool_call_message = AIMessage(
            content="", 
            tool_calls=[{
                "id": f"call_{skill_card.summary.name}",
                "name": skill_card.summary.name,
                "args": final_args_dict, # <-- Send the type-safe, validated dict
            }]
        )

        return {
            "messages": [tool_call_message]
        }




MAX_TOOL_RETRIES = 3   # max retries per plan step (covers both transient + logical)
TRANSIENT_RETRY_DELAY = 1.0  # seconds between transient retries

# Keys in tool output dicts that indicate file paths worth persisting
_PATH_KEYS = {"output_path", "file_path", "path", "results_saved_to", "full_report_saved_to"}

def _extract_paths_from_output(result, tool_name: str, findings: list):
    """Scan a tool result dict for file paths and append them as findings."""
    if not isinstance(result, dict):
        return
    for key, value in result.items():
        if key in _PATH_KEYS and isinstance(value, str) and value:
            findings.append(f"[{tool_name}] {key}: {value}")

def tool_node(state: AgentState, executor: ToolExecutor) -> dict:
    """
    The custom tool node.
    It reads the last AIMessage, extracts the tool calls,
    runs them through our ToolExecutor, and returns the results.
    
    RETRY STRATEGY (two layers):
    1. Transient exceptions (crashes, OOM, timeouts) are retried immediately
       up to MAX_TOOL_RETRIES times with a brief delay — no LLM round-trip.
    2. Logical errors (tool returns {"error": ...}) set the plan step back to
       "pending" so the Agent can revise arguments and retry on the next loop,
       as long as retries < MAX_TOOL_RETRIES.  Once exhausted, the step is
       permanently marked "failed".
    """
    import time

    print("--- TOOL NODE (Execute & Update Plan) ---")
    
    # 1. Get the last message
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    current_plan = state.get('plan', [])
    tool_results = []
    
    new_plan = [p.copy() for p in current_plan]
    new_path_findings = []  # collect file paths deterministically

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        print(f"  -> Executing tool call: {tool_name}")
        
        # --- LAYER 1: Transient retry loop (same arguments) ---
        result = None
        last_exception = None
        
        for attempt in range(1, MAX_TOOL_RETRIES + 1):
            try:
                result = executor.execute_tool(
                    tool_name=tool_name,
                    tool_args=tool_call['args']
                )
                last_exception = None
                break  # success — exit retry loop
            except Exception as e:
                last_exception = e
                if attempt < MAX_TOOL_RETRIES:
                    print(f"  -> [RETRY {attempt}/{MAX_TOOL_RETRIES}] "
                          f"Transient failure: {e}. Retrying in {TRANSIENT_RETRY_DELAY}s...")
                    time.sleep(TRANSIENT_RETRY_DELAY)
                else:
                    print(f"  -> [FAILED] All {MAX_TOOL_RETRIES} attempts exhausted: {e}")

        # --- Find the matching plan step ---
        matched_step = None
        for step in new_plan:
            if step.status == "pending":
                if not step.tool or step.tool == tool_name:
                    matched_step = step
                    break

        # --- CASE A: Unrecoverable crash (all transient retries exhausted) ---
        if last_exception is not None:
            error_msg = f"Tool crashed after {MAX_TOOL_RETRIES} attempts: {last_exception}"
            print(f"  -> {error_msg}")
            
            if matched_step:
                matched_step.status = "failed"
                matched_step.error = str(last_exception)
                matched_step.retries = matched_step.retries + MAX_TOOL_RETRIES
            
            tool_results.append(
                ToolMessage(
                    content=json.dumps({"error": error_msg}, cls=NumpySafeEncoder),
                    tool_call_id=tool_call['id']
                )
            )
            continue

        # --- CASE B: Tool returned a logical error dict ---
        is_logical_error = isinstance(result, dict) and "error" in result
        
        if is_logical_error and matched_step:
            matched_step.retries += 1
            
            if matched_step.retries < MAX_TOOL_RETRIES:
                # Keep step retryable — agent can adjust args
                matched_step.status = "pending"
                matched_step.error = str(result["error"])
                print(f"  -> [RETRYABLE] Step '{matched_step.id}' failed "
                      f"(attempt {matched_step.retries}/{MAX_TOOL_RETRIES}). "
                      f"Returning to agent for replanning.")
            else:
                # Budget exhausted — permanent failure
                matched_step.status = "failed"
                matched_step.error = str(result["error"])
                print(f"  -> [FAILED] Step '{matched_step.id}' permanently failed "
                      f"after {MAX_TOOL_RETRIES} attempts.")
        
        # --- CASE C: Success ---
        elif matched_step:
            matched_step.status = "success"
            # Store a clean JSON summary in the plan step output
            if isinstance(result, dict):
                matched_step.output = json.dumps(result, cls=NumpySafeEncoder)[:300]
            else:
                str_res = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                matched_step.output = str_res
            matched_step.error = None
            print(f"  -> Marking Step '{matched_step.id}' ({matched_step.description}) as SUCCESS.")
        
        if not matched_step:
            print("  -> [WARN] No pending step found to update for this tool call.")

        # --- Deterministically extract file paths from result ---
        _extract_paths_from_output(result, tool_name, new_path_findings)

        # --- Build the ToolMessage ---
        normalized = normalize_tool_output(result)
        tool_results.append(
            ToolMessage(
                content=json.dumps(normalized, cls=NumpySafeEncoder),
                tool_call_id=tool_call['id']
            )
        )
        
    print(f"  -> All tools executed.")
    prev_count = state.get("tool_call_count", 0)
    new_count = prev_count + 1
    
    # --- Persist extracted file paths as findings ---
    update = {"messages": tool_results, "tool_call_count": new_count, "plan": new_plan}
    if new_path_findings:
        memory = state.get("memory", FoldedMemory())
        merged_findings = list(memory.findings)
        for f in new_path_findings:
            if f not in merged_findings:
                merged_findings.append(f)
                print(f"  -> Auto-finding: {f}")
        update["memory"] = FoldedMemory(
            findings=merged_findings,
            summary=memory.summary
        )
    
    return update


class MemoryFolderNode:
    """
    Reads the FULL message history every 5 tool calls and produces:
    1. new_findings: key facts to permanently append (file paths, PMT IDs, etc.)
    2. summary: compressed narrative of ALL steps so far
    After folding, trims old messages to save context window space.
    """
    KEEP_LAST_N = 4  # messages to preserve after trimming

    def __init__(self, model: ChatOllama):
        self.model = model.with_structured_output(MemoryFoldResult)
        self.prompt = ChatPromptTemplate.from_template(MEMORY_FOLDER_PROMPT_2)
        self.chain = self.prompt | self.model

    def _format_full_history(self, messages: list[BaseMessage]) -> str:
        """Format the entire message history for the LLM."""
        lines = []
        for m in messages:
            name = getattr(m, 'name', None) or m.type
            content = m.content if len(m.content) < 500 else m.content[:500] + "..."
            lines.append(f"[{name}] {content}")
        return "\n".join(lines)

    def __call__(self, state: AgentState) -> dict:
        print("--- MEMORY FOLDER NODE (Extract & Compress) ---")
        
        previous_memory = state['memory']
        plan = state['plan']
        all_messages = state['messages']

        # Format inputs for the LLM
        full_history = self._format_full_history(all_messages)
        existing_findings_str = "\n".join(
            f"  {i+1}. {f}" for i, f in enumerate(previous_memory.findings)
        ) if previous_memory.findings else "None yet."

        plan_str = json.dumps([p.model_dump() for p in plan], cls=NumpySafeEncoder, indent=2)

        # Call the LLM
        result = self.chain.invoke({
            "previous_summary": previous_memory.summary,
            "existing_findings": existing_findings_str,
            "full_history": full_history,
            "plan": plan_str
        })

        # Merge findings: old + new (deduplicated)
        old_findings = list(previous_memory.findings)
        new_findings = [f for f in result.new_findings if f not in old_findings]
        merged_findings = old_findings + new_findings

        if new_findings:
            print(f"  -> New findings extracted: {new_findings}")
        print(f"  -> Summary updated ({len(result.summary)} chars)")

        # Build new memory
        new_memory = FoldedMemory(
            findings=merged_findings,
            summary=result.summary
        )

        # Trim old messages — keep only the last N
        if len(all_messages) > self.KEEP_LAST_N:
            trimmed = all_messages[-self.KEEP_LAST_N:]
            print(f"  -> Trimmed messages: {len(all_messages)} -> {len(trimmed)}")
        else:
            trimmed = all_messages

        return {"memory": new_memory, "messages": trimmed}