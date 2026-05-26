from typing import TypedDict, Dict, Any, Optional, List
from pydantic import BaseModel, Field

# --- 1. The Output Model for the Neural Analyzer ---
class NeuralAnalysisResult(BaseModel):
    """
    The structured output we expect from the LLM when acting as a Static Analyzer.
    """
    predicted_output_schema: Dict[str, Any] = Field(
        description="A JSON Schema describing the return value. If it's a DataFrame, specify columns."
    )
    is_safe: bool = Field(
        description="True if the function has no side effects (disk write, DB calls). False otherwise."
    )
    safety_reason: Optional[str] = Field(
        description="If unsafe, explain why (e.g., 'Writes PDF to disk')."
    )
    input_requirements: str = Field(
        description="Description of specific data requirements (e.g., 'Needs a CSV with [time, charge] columns')."
    )
    functional_summary: str = Field(
        description="A concise description of what the function computes, "
                    "its core algorithm or approach, and any important behavioral "
                    "distinctions from similar functions."
    )

class MockFile(BaseModel):
    name: str = Field(description="The filename (e.g., 'run_data_A.csv').")
    content: str = Field(description="The string content of the file.")

class TestScenario(BaseModel):
    type: str = Field(description="The type of scenario: 'happy_path', 'ambiguous', or 'negative_constraint'")
    user_query: str = Field(description="The synthetic user query testing this scenario.")
    expected_behavior: str = Field(description="Brief description of what the agent should do.")

class SimulationData(BaseModel):
    """
    Holds the test plan. Can be either Mock-based or Real-Data-based.
    """
    scenarios: List[TestScenario]
    
    # Dual Mode Flag
    use_provided_data: bool = Field(
        description="True if we are using an existing file provided by the user. False if we generated mock data."
    )
    
    # Option A: Real Data (Preferred)
    provided_data_path: Optional[str] = Field(
        description="The path to the existing test file used in the scenarios (if use_provided_data is True)."
    )
    mock_files: Optional[List[MockFile]] = Field(
        default=[],
        description="A list of mock files to generate. Use this if the tool needs one OR MORE files."
    )


class GateResult(BaseModel):
    gate_name: str  
    passed: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ScenarioResult(BaseModel):
    scenario_type: str
    user_query: str
    gates: List[GateResult]
    all_passed: bool

class ValidationReport(BaseModel):
    scenario_results: List[ScenarioResult]
    overall_success_rate: float
    critical_failure: bool = False # e.g. Safety violation



# --- 2. The Graph State ---
class LearnerState(TypedDict):
    """
    The state of the Offline Skill Synthesizer pipeline.
    """
    # Inputs
    file_path: str
    function_name: str
    manual_test_files: Optional[List[str]]
    # Step 1.1 Output (Standard Static Analysis)
    static_metadata: Dict[str, Any] 
    
    # Step 1.2 Output (Neural Static Analysis)
    neural_analysis: NeuralAnalysisResult
    
    # Future Steps
    draft_skill_card: Optional[Dict[str, Any]]
    simulation_data: Optional[SimulationData]
    #validation_report: list[str]
    unit_report: Optional[ValidationReport] 
    integration_report: Optional[ValidationReport]

    iteration: int                
    refinement_feedback: str      