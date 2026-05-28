from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from Learner.learner_states_class import LearnerState, NeuralAnalysisResult, SimulationData, GateResult, ScenarioResult, ValidationReport
from Learner.static_analyzer import extract_function_info
from typing import Dict,Any
import json
import yaml
import tempfile, shutil
import os
from Learner.learner_prompts import ANALYZER_SYSTEM_PROMPT, DRAFTER_SYSTEM_PROMPT, SIMULATOR_SYSTEM_PROMPT
from skill_models import SkillCard,ParameterDetails, ParametersSchema 
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from skill_registry import SkillRegistry
from state_and_class import AgentState, FoldedMemory, ToolExecutor
from nodes import AgentNode, ArgumentFormatterNode
from main import build_graph

SKILL_DRAFT_DIRECTORY = "Learner/skills_drafts"
MOCK_DATA_DIRECTORY = "Learner/mock_data"
SIMULATION_LOGS_DIRECTORY = "Learner/simulation_runs"
SKILL_DIRECTORY ="skills/"




def setup_mixed_skill_environment(base_skills_dir: str, draft_skill_data: Dict[str, Any]) -> str:
    """
    Creates a temporary directory containing both the existing production skills 
    and the new draft skill.
    
    Args:
        base_skills_dir (str): Path to the folder containing existing .yaml skills (e.g., "skills").
        draft_skill_data (dict): The dictionary representation of the Draft Skill Card.
        
    Returns:
        str: The path to the new temporary directory to be passed to SkillRegistry.
    """
    # 1. Create a secure temporary directory
    # prefix makes it easy to identify in /tmp if you need to debug
    temp_dir = tempfile.mkdtemp(prefix="juno_agent_integration_")
    print(f"   -> Created temporary integration environment at: {temp_dir}")

    # 2. Copy all existing production skills
    if os.path.exists(base_skills_dir):
        files_copied = 0
        for filename in os.listdir(base_skills_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                src_path = os.path.join(base_skills_dir, filename)
                dst_path = os.path.join(temp_dir, filename)
                shutil.copy2(src_path, dst_path)
                files_copied += 1
        print(f"   -> Copied {files_copied} base skills from '{base_skills_dir}'.")
    else:
        print(f"   -> [WARN] Base skill directory '{base_skills_dir}' not found. Starting empty.")

    # 3. Inject the Draft Skill
    # We use the skill_id for the filename, defaulting to 'draft' if missing
    skill_id = draft_skill_data.get('skill_id', 'draft_skill')
    draft_filename = f"{skill_id}.yaml"
    draft_path = os.path.join(temp_dir, draft_filename)
    
    with open(draft_path, 'w') as f:
        # yaml.dump is standard, as used in your file_writer
        yaml.dump(draft_skill_data, f, sort_keys=False)
    
    print(f"   -> Injected draft skill '{skill_id}' into test environment.")
    
    return temp_dir






class StaticExtractionNode:
    """
    Wraps Step 1.1 (The pure Python script) into a Graph Node.
    """
    def __call__(self, state: LearnerState) -> Dict:
        print(f"--- STEP 1.1: Standard Static Analysis on {state['function_name']} ---")
        
        # Run the ast-based extractor we wrote previously
        metadata = extract_function_info(state['file_path'], state['function_name'])
        
        if "error" in metadata:
            raise ValueError(f"Extraction failed: {metadata['error']}")
            
        return {"static_metadata": metadata}

class NeuralAnalysisNode:
    """
    Implements Step 1.2: LLM-based Code Tracing.
    """
    def __init__(self, model: ChatOllama):
        self.model = model.with_structured_output(NeuralAnalysisResult)
        self.prompt = ChatPromptTemplate.from_template(ANALYZER_SYSTEM_PROMPT)
        self.chain = self.prompt | self.model

    def __call__(self, state: LearnerState) -> Dict:
        print(f"--- STEP 1.2: Neural Static Analysis (Safety & Schema) ---")
        
        metadata = state['static_metadata']
        source_code = metadata.get('source_code', '')
        
        # Invoke the LLM
        result = self.chain.invoke({
            "function_name": state['function_name'],
            "source_code": source_code
        })
        
        print(f"   -> Safety Check: {'SAFE' if result.is_safe else 'UNSAFE'}")
        # print(f"   -> Inferred Schema Keys: {list(result.predicted_output_schema.keys()) if isinstance(result.predicted_output_schema, dict) else 'Complex Object'}")

        return {"neural_analysis": result}
    
class DrafterNode:
    """
    Step 2: Synthesizes the Skill Card using the gathered intelligence.
    """
    def __init__(self, model: ChatOllama):
        # We force the LLM to output the exact SkillCard Pydantic structure
        self.model = model.with_structured_output(SkillCard)
        self.prompt = ChatPromptTemplate.from_template(DRAFTER_SYSTEM_PROMPT)
        self.chain = self.prompt | self.model

    def _infer_module_path(self, file_path: str) -> str:
        """
        Converts 'juno_tools/physics/analysis.py' -> 'juno_tools.physics.analysis'
        """
        # Remove extension
        base = os.path.splitext(file_path)[0]
        # Replace separators with dots
        return base.replace("/", ".")
    def _repair_parameters(self, skill_card: SkillCard, ast_args: list):
        """
        Ensures that every argument found in the AST is present in the Skill Card.
        If the LLM missed one (or returned empty properties), we inject it using static data.
        """
        props = skill_card.summary.parameters.properties
        
        print(f"   -> Validating parameters against AST...")
        for arg in ast_args:
            arg_name = arg['name']
            
            # Skip *args and **kwargs for now as they are hard to schema-tize
            if arg_name.startswith("*"): 
                continue

            if arg_name not in props:
                print(f"      [FIX] LLM missed argument '{arg_name}'. Injecting from AST.")
                
                # Convert Python type hint to JSON type
                py_type = arg.get('type', 'Any')
                json_type = "string" # Default fallback
                if "int" in py_type: json_type = "integer"
                elif "float" in py_type: json_type = "float"
                elif "bool" in py_type: json_type = "boolean"
                elif "list" in py_type or "List" in py_type: json_type = "array"
                
                # Create the missing parameter
                # We use a generic description since the LLM failed to provide one
                new_param = ParameterDetails(
                    type=json_type,
                    description=f"Argument '{arg_name}' (Type: {py_type}). inferred from source code.",
                    required=arg.get('required', True),
                    default=arg.get('default', None)
                )
                
                props[arg_name] = new_param
            else:
                # Parameter exists — check if its default is sane
                existing = props[arg_name]
                ast_default = arg.get('default', None)
                
                # Fix hallucinated dict defaults ($ref, {value: X}, etc.)
                if isinstance(existing.default, dict):
                    print(f"      [FIX] Repairing bad default for '{arg_name}': {existing.default} -> {ast_default}")
                    existing.default = ast_default
                
                # If LLM left default as None but AST has one, use the AST value
                if existing.default is None and ast_default is not None:
                    existing.default = ast_default
        
        return skill_card
    def __call__(self, state: LearnerState) -> Dict:
        print(f"--- STEP 2: The Drafter (Skill Synthesis) ---")
        
        static = state['static_metadata']
        neural = state['neural_analysis']
        module_path = self._infer_module_path(state['file_path'])
        safety_status = "SAFE" if neural.is_safe else f"UNSAFE ({neural.safety_reason})"
        
        feedback = state.get("refinement_feedback")
        iteration = state.get("iteration", 0)
        if feedback and iteration > 0:
            print(f"   -> REFINEMENT MODE (Iteration {iteration})")
            # Inject feedback into the prompt
            draft = state['draft_skill_card']
            draft_str = json.dumps(draft, indent=2)
            draft_escaped = draft_str.replace("{", "{{").replace("}", "}}")
            feedback_escaped = feedback.replace("{", "{{").replace("}", "}}")
            system_msg = DRAFTER_SYSTEM_PROMPT + f"\n\n### CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT\n{feedback_escaped}\n\nRefine the previous draft to fix these specific errors.\n Previous Draft :\n {draft_escaped}"
            
            # Re-compile chain with new prompt (temporary for this run)
            # Or simpler: create a dynamic prompt template
            prompt = ChatPromptTemplate.from_template(system_msg)
            chain = prompt | self.model
        else:
            # Normal First Run
            chain = self.chain
        # Invoke LLM
        skill_card_obj = chain.invoke({
            "function_name": state['function_name'],
            "docstring": static.get('docstring', ''),
            "args_json": json.dumps(static.get('args', []), indent=2),
            "safety_status": safety_status,
            "output_schema_json": json.dumps(neural.predicted_output_schema, indent=2),
            "functional_summary": neural.functional_summary,
            "module_path": module_path
        })

        # Post-Processing: Ensure execution details match the real file path
        # The LLM might hallucinate the module path, so we force overwrite it with the real one
        skill_card_obj.execution_details.call.module = module_path
        skill_card_obj.execution_details.call.function = state['function_name']
        skill_card_obj.execution_details.type = "python_function"

    
        skill_card_obj = self._repair_parameters(skill_card_obj, static.get('args', []))
        return {"draft_skill_card": skill_card_obj.model_dump()}
    

class SimulatorNode:
    """Step 3: Scenario & Data Generation (Hybrid)"""
    def __init__(self, model: ChatOllama):
        self.model = model.with_structured_output(SimulationData)
        self.prompt = ChatPromptTemplate.from_template(SIMULATOR_SYSTEM_PROMPT)
        self.chain = self.prompt | self.model

    def __call__(self, state: LearnerState) -> Dict:
        print(f"--- STEP 3: The Simulator ---")
        
        draft = state['draft_skill_card']
        neural = state['neural_analysis']
        manual_file = state.get('manual_test_files')
        
        # Branching Logic for the Prompt
        if manual_file:
            print(f"   -> Mode: REAL DATA PROVIDED ({manual_file})")
            data_instruction = (
                f"**Real Data Provided**: The user has provided valid tests file at: '{manual_file}'.\n"
                f"1. You MUST set `use_provided_data` to true.\n"
                f"2. You MUST set `provided_data_path` to '{manual_file}'.\n"
                f"3. You MUST set Leave `mock_files` to null.\n"
                f"4. Write test scenarios that explicitly use '{manual_file}' in the query."
            )
        else:
            print(f"   -> Mode: MOCK DATA GENERATION")
            data_instruction = (
                "**No Data Provided**: You must generate a MOCK data.\n"
                "1. You MUST set `use_provided_data` to false.\n"
                "2.Populate `mock_files` with ONE or MORE files as required by the tool input.\n"
                "   - Example: If the tool compares two files, generate 'file_A.csv' and 'file_B.csv'.\n"
                "3. Generate minimal content for `mock_file_content`.\n"
                "4. Write test scenarios that use your generated filename."
            )

        # Invoke LLM
        simulation_result = self.chain.invoke({
            "skill_name": draft['summary']['name'],
            "usage_policy": json.dumps(draft.get('usage_policy', {}), indent=2),
            "input_requirements": neural.input_requirements,
            "data_instruction": data_instruction
        })
        
        return {"simulation_data": simulation_result}
    

class FileWriterNode:
    """
    Step 3.5: Materializes the in-memory simulation data to physical files
    so the validation tools can actually read them.
    
    After writing mock files, patches all scenario queries to replace bare
    filenames (e.g. 'mock_run.csv') with the actual written paths
    (e.g. 'Learner/mock_data/mock_run.csv') so downstream nodes find them.
    """
    def __call__(self, state: LearnerState) -> dict:
        print("--- STEP 3.5: File Writer (Materializing Artifacts) ---")
        sim_data = state['simulation_data']
        draft = state['draft_skill_card']
        
        # 1. Write Mock Data and build filename -> real_path mapping
        path_mapping = {}  # bare_name -> actual_path
        
        if not sim_data.use_provided_data and sim_data.mock_files:
            print(f"   -> Materializing {len(sim_data.mock_files)} mock files...")
            os.makedirs(MOCK_DATA_DIRECTORY, exist_ok=True)

            for mock_file in sim_data.mock_files:
                file_path = os.path.join(MOCK_DATA_DIRECTORY, mock_file.name)
                print(f"      - Writing: {file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(mock_file.content or "")
                # Track: bare name -> real path
                path_mapping[mock_file.name] = file_path
        
        # 2. Patch scenario queries to use real paths
        if path_mapping:
            print(f"   -> Patching {len(sim_data.scenarios)} scenario queries with real file paths...")
            for scenario in sim_data.scenarios:
                original_query = scenario.user_query
                for bare_name, real_path in path_mapping.items():
                    scenario.user_query = scenario.user_query.replace(bare_name, real_path)
                if scenario.user_query != original_query:
                    print(f"      - Patched: '{original_query[:60]}...' -> uses '{real_path}'")

        # 3. Persist the Draft Skill
        os.makedirs(f"{SKILL_DRAFT_DIRECTORY}", exist_ok=True)
        skill_filename = f"{SKILL_DRAFT_DIRECTORY}/{draft['skill_id']}.yaml"
        print(f"   -> Persisting draft skill to disk: {skill_filename}")
        with open(skill_filename, "w", encoding="utf-8") as f:
            yaml.dump(draft, f, sort_keys=False)
            

        os.makedirs(SIMULATION_LOGS_DIRECTORY, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_dump_filename = f"{SIMULATION_LOGS_DIRECTORY}/{draft['skill_id']}_{timestamp}_sim_data.json"
        print(f"   -> Archiving simulation data: {sim_dump_filename}")
        
        with open(sim_dump_filename, "w", encoding="utf-8") as f:
            f.write(sim_data.model_dump_json(indent=2))
        
        # Return updated simulation_data so patched queries propagate to validation nodes
        return {"simulation_data": sim_data}


class ValidationUnitNode:
    """
    PHASE 1: Unit Testing (Isolation)
    Focus: Schema alignment, Runtime safety, Output validity.
    """
    def __init__(self, model: ChatOllama):
        self.llm = model

    def _setup_registry_iso(self, draft_skill_data: dict):
        registry = SkillRegistry(skill_directory="Learner/skills_drafts")
        registry.skills = {}  # Clear everything
        # Inject Draft Only
        skill_card = SkillCard.model_validate(draft_skill_data)
        registry.skills[skill_card.summary.name] = skill_card
        return registry

    def __call__(self, state: LearnerState) -> dict:
        print("--- PHASE 1: Gauntlet Unit Test (Isolation) ---")
        draft = state['draft_skill_card']
        sim_data = state['simulation_data']
        
        registry = self._setup_registry_iso(draft)
        
        # We need the full agent stack for isolation testing
        agent_node = AgentNode(registry, self.llm)
        formatter_node = ArgumentFormatterNode(registry, self.llm)
        executor = ToolExecutor(registry)
        
        scenario_results = []
        
        
        scenario = sim_data.scenarios[0] #tests only happy path scenario
        print(f"   -> Unit Test Scenario: {scenario.type}")
        gates = []
        all_passed = True
        
        # Prepare State
        agent_state = {
            "messages": [HumanMessage(content=scenario.user_query)],
            "memory": FoldedMemory(),
            "plan": [],
            "tool_call_count": 0
        }

        # GATE A (Isolation Sanity): Does it pick the tool when it's the ONLY option?
        # If this fails, the description is broken or the query is irrelevant.
        try:
            result_a = agent_node(agent_state)
            last_msg = result_a['messages'][0]
            if last_msg.name == "SelectSkill":
                decision = json.loads(last_msg.content)
                if decision.get("skill_to_use") == draft['summary']['name']:
                        gates.append(GateResult(gate_name="A_Selection_Iso", passed=True))
                        agent_state['messages'].append(last_msg)
                else:
                    gates.append(GateResult(gate_name="A_Selection_Iso", passed=False, error_message="Agent refused to select tool even in isolation."))
                    all_passed = False
            else:
                gates.append(GateResult(gate_name="A_Selection_Iso", passed=False, error_message="Agent provided Final Answer instead of selecting tool."))
                all_passed = False
        except Exception as e:
            gates.append(GateResult(gate_name="A_Selection_Iso", passed=False, error_message=f"Crash A: {e}"))
            all_passed = False

        # GATE B (Formatting): Argument Schema Check
        tool_call_msg = None
        if all_passed:
            try:
                result_b = formatter_node(agent_state)
                last_msg = result_b['messages'][0]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    gates.append(GateResult(gate_name="B_Formatting", passed=True))
                    tool_call_msg = last_msg
                else:
                    gates.append(GateResult(gate_name="B_Formatting", passed=False, error_message=f"Formatting Failed: {last_msg.content}"))
                    all_passed = False
            except Exception as e:
                gates.append(GateResult(gate_name="B_Formatting", passed=False, error_message=f"Crash B: {e}"))
                all_passed = False

        # GATE C (Execution) & D (Semantic)
        if all_passed:
            try:
                tool_call = tool_call_msg.tool_calls[0]
                # Execute
                result = executor.execute_tool(tool_call['name'], tool_call['args'])
                
                if isinstance(result, dict) and "error" in result:
                        gates.append(GateResult(gate_name="C_Execution", passed=False, error_message=result['error']))
                        all_passed = False
                else:
                    gates.append(GateResult(gate_name="C_Execution", passed=True))
                    # Semantic Check (Non-Empty)
                    if not result or (isinstance(result, (str, list, dict)) and len(result) == 0):
                            gates.append(GateResult(gate_name="D_Semantic", passed=False, error_message="Tool returned empty result."))
                            all_passed = False
                    else:
                            gates.append(GateResult(gate_name="D_Semantic", passed=True))
            except Exception as e:
                gates.append(GateResult(gate_name="C_Execution", passed=False, error_message=f"Crash C: {e}"))
                all_passed = False
        
        scenario_results.append(ScenarioResult(
            scenario_type=scenario.type,
            user_query=scenario.user_query,
            gates=gates,
            all_passed=all_passed
        ))

        # Calculate Success
        pass_count = sum(1 for s in scenario_results if s.all_passed)
        success_rate = pass_count / len(scenario_results)
        
        return {"unit_report": ValidationReport(scenario_results=scenario_results, overall_success_rate=success_rate)}


class ValidationIntegrationNode:
    """
    PHASE 2: Integration Testing (Black Box Style)
    
    Instead of manually calling nodes, we spin up the REAL Agent Graph.
    We runs the query and then perform a "forensic analysis" on the 
    message history to decide if the gates passed.
    """
    def __call__(self, state: LearnerState) -> dict:
        print("--- PHASE 2: Gauntlet Integration Test (Production Graph) ---")
        
        # 1. Check Prerequisites
        unit_report = state.get('unit_report')
        if not unit_report or unit_report.overall_success_rate < 1.0:
            print("   -> Skipping Integration (Unit Test Failed)")
            return {"integration_report": None}

        draft = state['draft_skill_card']
        sim_data = state['simulation_data']
        draft_name = draft['summary']['name']
        temp_registry_path = setup_mixed_skill_environment(SKILL_DIRECTORY, draft)
        # 2. Build the Graph (This loads the Registry from disk, including the Draft)

        app = build_graph(temp_registry_path) 
        
        scenario_results = []

        for scenario in sim_data.scenarios:
            print(f"   -> Integration Scenario: {scenario.type}")
            
            # 3. Prepare Standard Agent Inputs
            #    We emulate a fresh user session
            inputs = {
                "messages": [HumanMessage(content=scenario.user_query)],
                "memory": FoldedMemory(), # Start with clean memory
                "plan": [],
                "tool_call_count": 0,

            }
            
            gates = []
            all_passed = True
            
            # 4. RUN THE FULL AGENT
            try:
                # The graph runs autonomously until it hits "end" or needs user input
                result_state = app.invoke(inputs)
                final_history = result_state['messages']
                
                # --- FORENSIC ANALYSIS OF THE TRACE ---
                
                # Search for the "SelectSkill" decision
                # We iterate backwards to find the relevant decision
                selection_msg = next((m for m in reversed(final_history) if isinstance(m, AIMessage) and m.name=="SelectSkill"), None)
                
                # GATE E: SELECTION
                if not selection_msg:
                    gates.append(GateResult(gate_name="E_Integration_Selection", passed=False, error_message="Agent never made a selection decision."))
                    all_passed = False
                else:
                    decision = json.loads(selection_msg.content)
                    selected_skill = decision.get("skill_to_use")
                    
                    if selected_skill == draft_name:
                        gates.append(GateResult(gate_name="E_Integration_Selection", passed=True))

                    else:
                        # WRONG SELECTION (Distraction)
                        gates.append(GateResult(
                            gate_name="E_Integration_Selection", 
                            passed=False, 
                            error_message=f"Distracted by '{selected_skill}'",
                            details={"conflicting_skill_id": selected_skill}
                        ))
                        all_passed = False

            except Exception as e:
                # Catch unexpected graph crashes
                gates.append(GateResult(gate_name="E_Integration_Selection", passed=False, error_message=f"Graph Crash: {e}"))
                all_passed = False

            # Finalize Scenario
            scenario_results.append(ScenarioResult(
                scenario_type=scenario.type, 
                user_query=scenario.user_query, 
                gates=gates, 
                all_passed=all_passed
            ))

        # Calculate Statistics
        pass_count = sum(1 for s in scenario_results if s.all_passed)
        success_rate = pass_count / len(scenario_results) if scenario_results else 0.0

        return {
            "integration_report": ValidationReport(
                scenario_results=scenario_results, 
                overall_success_rate=success_rate
            )
        }


class RefinerNode:
    def __init__(self):
        # We need the registry to look up conflicting skills
        # model argument is kept for compatibility if your graph builder passes it, 
        # though not strictly used here.
        
        self.registry = SkillRegistry(skill_directory=SKILL_DIRECTORY)

    def __call__(self, state: LearnerState) -> dict:
        print("--- STEP 5: Refiner (Routed Feedback) ---")
        unit_report = state.get('unit_report')
        int_report = state.get('integration_report')
        
        feedback_lines = []
        
        # --- PATH 1: UNIT FAILURE (Code/Schema/Data) ---
        if unit_report and unit_report.overall_success_rate < 1.0:
            print("   -> Mode: FIXING CORRECTNESS")
            feedback_lines.append("Your skill FAILED Unit Testing. The tool is technically broken or the arguments are wrong.")
            
            for sc in unit_report.scenario_results:
                if not sc.all_passed:
                    feedback_lines.append(f"\n[Scenario: {sc.scenario_type}]")
                    for gate in sc.gates:
                        if not gate.passed:
                            if gate.gate_name == "B_Formatting":
                                feedback_lines.append(f"- SCHEMA ERROR: Argument mismatch. {gate.error_message}. FIX: Check 'parameters' types/required fields.")
                            elif gate.gate_name == "C_Execution":
                                feedback_lines.append(f"- RUNTIME CRASH: {gate.error_message}. FIX: Check python types and mock data compatibility.")
                            elif gate.gate_name == "D_Semantic":
                                feedback_lines.append(f"- LOGIC ERROR: Tool ran but returned empty. FIX: Check default values.")

        # --- PATH 2: INTEGRATION FAILURE (Ambiguity & Alignment) ---
        elif int_report and int_report.overall_success_rate < 1.0:
            print("   -> Mode: FIXING AMBIGUITY & ALIGNMENT")
            feedback_lines.append("Your skill PASSED Unit Tests, but FAILED Integration in the production graph.")
            
            for sc in int_report.scenario_results:
                if not sc.all_passed:
                    feedback_lines.append(f"\n[Scenario: {sc.scenario_type}]")
                    for gate in sc.gates:
                        if not gate.passed:
                            
                            # 1. DISTRACTION (Agent picked the wrong tool)
                            if gate.gate_name == "E_Integration_Selection":
                                rival_id = gate.details.get("conflicting_skill_id")
                                rival_policy = "Unknown"
                                if rival_id:
                                    try:
                                        rival_card = self.registry.get_skill_by_name(rival_id)
                                        rival_policy = rival_card.usage_policy
                                    except: pass
                                
                                feedback_lines.append(
                                    f"- DISTRACTION: Agent chose '{rival_id}' instead of yours."
                                    f"\n  RIVAL POLICY: {rival_policy}"
                                    f"\n  FIX: Update 'usage_policy.do_not_use_when' to explicitly exclude scenarios covered by '{rival_id}'."
                                    
                                )

        # SUCCESS
        if not feedback_lines:
            return {"refinement_feedback": "SUCCESS"}

        print(f"   -> Generated {len(feedback_lines)} lines of feedback.")
        return {
            "refinement_feedback": "\n".join(feedback_lines),
            "iteration": state.get("iteration", 0) + 1
        }







