import os
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
import json
import yaml
# Import our components
from Learner.learner_states_class import LearnerState
from Learner.learner_nodes import StaticExtractionNode, NeuralAnalysisNode, DrafterNode,SimulatorNode, ValidationUnitNode,ValidationIntegrationNode, FileWriterNode, RefinerNode
MAX_RETRIES = 6
def should_refine(state: LearnerState) -> str:
    """
    Decides if we loop back or end.
    """
    unit_report = state.get('unit_report')
    integration_report = state.get('integration_report')
    iter_count = state.get('iteration', 0)
    
    
    # 1. Check Success
    if integration_report and integration_report.overall_success_rate >= 1.0:
        print(">>> SUCCESS: Validation Passed! <<<")
        return "__end__"
        

    
    # 2. Check Max Retries
    if iter_count >= MAX_RETRIES:
        print(f">>> STOP: Max retries ({MAX_RETRIES}) reached. Generation Failed. <<<")
        return "__end__"
    

    
    # 3. Loop Back
    print(f">>> LOOP: Retrying (Attempt {iter_count+1}/{MAX_RETRIES}) <<<")
    return "drafter"

def should_integration_test(state: LearnerState)-> str:
    """
    Decides if we go to integration testing or not.
    """
    report = state.get('unit_report')
    if report and report.overall_success_rate >= 1.0:
        print(">>> SUCCESS: Unit testing passed! <<<")
        return "validation_integration"
    print("Unit testing failed -> Loop Again")
    return "refiner"




def build_learner_graph():
    # 1. Setup the LLM (Using the same configuration as your main agent)
    # Note: We use a high-context model if available, but 20b is fine for analysis
    llm = ChatOllama(model="gpt-oss:20b", base_url="http://127.0.0.1:13444")#gpt-oss:20b
    
    # 2. Initialize Nodes
    static_node = StaticExtractionNode()
    neural_node = NeuralAnalysisNode(llm)
    drafter_node = DrafterNode(llm)
    simulator_node = SimulatorNode(llm)
    file_writer_node = FileWriterNode()
    validation_unit_node = ValidationUnitNode(llm)
    validation_integration_node = ValidationIntegrationNode()
    refiner_node = RefinerNode()
    # 3. Define Graph
    workflow = StateGraph(LearnerState)
    
    workflow.add_node("static_extraction", static_node)
    workflow.add_node("neural_analysis", neural_node)
    workflow.add_node("drafter", drafter_node)
    workflow.add_node("simulator", simulator_node)
    workflow.add_node("file_writer", file_writer_node) 
    workflow.add_node("validation_unit",validation_unit_node)
    workflow.add_node("validation_integration",validation_integration_node)
    workflow.add_node("refiner", refiner_node)
    # 4. Define Edges (Linear flow for Phase 1)
    workflow.set_entry_point("static_extraction")
    workflow.add_edge("static_extraction", "neural_analysis")
    workflow.add_edge("neural_analysis", "drafter")
    workflow.add_edge("drafter", "simulator")
    workflow.add_edge("simulator", "file_writer") 
    workflow.add_edge("file_writer", "validation_unit") 

    workflow.add_conditional_edges(
        "validation_unit",
        should_integration_test,
        {
            "validation_integration": "validation_integration",  # The Loop Back
            "refiner": "refiner"
        }
    )

    workflow.add_edge("validation_integration", "refiner")
    
    workflow.add_conditional_edges(
        "refiner",
        should_refine,
        {
            "drafter": "drafter",  # The Loop Back
            "__end__": "__end__"
        }
    )
    return workflow.compile()


# --- DEMONSTRATION RUN ---
if __name__ == "__main__":

    # 2. Initialize Graph
    app = build_learner_graph()

    # 3. Run Pipeline
    print(">>> STARTING LEARNING PIPELINE >>>")
    inputs = {
        # Replace with your actual target file
        "file_path": "juno_tools/physics_analysis.py", 
        "function_name": "compute_run_kpis",
    }
    
    result = app.invoke(inputs)
    
    # 4. Inspect Results
    unit_rep = result.get('unit_report')
    int_rep = result.get('integration_report')
    
    print("\n" + "="*50)
    print("          LEARNER MODE EXECUTION REPORT")
    print("="*50)

    # --- PHASE 1: UNIT TESTING REPORT ---
    if unit_rep:
        print("\n>>> PHASE 1: UNIT CORRECTNESS (Isolation)")
        print(f"Overall Success: {unit_rep.overall_success_rate * 100:.1f}%")
        
        for sc in unit_rep.scenario_results:
            status = "PASS" if sc.all_passed else "FAIL"
            # Print Scenario Header
            print(f"\n  [{status}] Scenario: {sc.scenario_type}")
            print(f"  Query: \"{sc.user_query}\"")
            
            # Print Gates
            for gate in sc.gates:
                icon = "✅" if gate.passed else "❌"
                print(f"    {icon} Gate {gate.gate_name}")
                
                # IF FAILED: Print the explicit reason
                if not gate.passed:
                    print(f"       🔻 ERROR: {gate.error_message}")

    # --- PHASE 2: INTEGRATION TESTING REPORT ---
    if int_rep:
        print("\n" + "-"*30)
        print("\n>>> PHASE 2: INTEGRATION (Competitive Selection)")
        print(f"Overall Success: {int_rep.overall_success_rate * 100:.1f}%")
        
        for sc in int_rep.scenario_results:
            status = "PASS" if sc.all_passed else "FAIL"
            print(f"\n  [{status}] Scenario: {sc.scenario_type}")
            
            for gate in sc.gates:
                icon = "✅" if gate.passed else "❌"
                print(f"    {icon} Gate {gate.gate_name}")
                
                # IF FAILED: Print why (likely distraction)
                if not gate.passed:
                    print(f"       🔻 FAILURE: {gate.error_message}")
                    # If we captured the specific rival tool, show it
                    if gate.details and "conflicting_skill_id" in gate.details:
                        print(f"       ⚔️  RIVAL TOOL: {gate.details['conflicting_skill_id']}")

    elif unit_rep and unit_rep.overall_success_rate < 1.0:
        print("\n>>> PHASE 2: SKIPPED (Fix Unit Errors First)")

    # --- FINAL VERDICT ---
    print("\n" + "="*50)
    if int_rep and int_rep.overall_success_rate == 1.0:
        print("✅ SUCCESS: Skill Card is Valid and Distinct.")
        print(f"   Draft Saved to: Learner/skills_drafts/{result['draft_skill_card']['skill_id']}.yaml")
    else:
        print("❌ FAILED: Skill requires refinement.")
        if "refinement_feedback" in result:
             print("   Feedback for Drafter:")
             print(f"   {result['refinement_feedback']}")