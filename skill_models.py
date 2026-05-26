from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field

# --- 1. PARAMETERS SECTION (The Input Schema) ---

class ParameterDetails(BaseModel):
    """
    Defines a single argument for the function.
    """
    type: str = Field(
        ..., 
        description="The data type of the parameter (e.g., 'string', 'integer', 'float', 'boolean', 'array')."
    )
    description: str = Field(
        ..., 
        description="A clear, human-readable explanation of what this parameter represents and valid values."
    )
    required: bool = Field(
        ..., 
        description="True if this parameter must be provided, False if it is optional."
    )
    default: Optional[Any] = Field(
        None, 
        description="The default value if the parameter is not provided. Use null/None if no default exists."
    )

class ParametersSchema(BaseModel):
    """
    The 'parameters' block in the YAML.
    """
    type: Literal["object"] = Field(
        "object", 
        description="The root type for arguments, almost always 'object'."
    )
    properties: Dict[str, ParameterDetails] = Field(
        ..., 
        description="A dictionary mapping argument names to their detailed definitions."
    )

# --- 2. SUMMARY SECTION ---

class SkillSummary(BaseModel):
    """
    High-level description of the skill.
    """
    name: str = Field(
        ..., 
        description="The callable name of the tool (must be snake_case, e.g., 'get_pmt_profile')."
    )
    description: str = Field(
        ..., 
        description="A concise, high-level summary (<50 words) of what the tool does."
    )
    parameters: ParametersSchema = Field(
        ..., 
        description="The strict schema defining the function's arguments."
    )

# --- 3. EXECUTION DETAILS SECTION ---

class FunctionCallDetails(BaseModel):
    """
    Specifics for Python dynamic loading.
    """
    module: str = Field(
        ..., 
        description="The full Python import path (e.g., 'juno_tools.physics.analysis')."
    )
    function: str = Field(
        ..., 
        description="The specific function name to execute."
    )

class SkillExecution(BaseModel):
    """
    The 'execution_details' block.
    """
    type: Literal['python_function'] = Field(
        "python_function", 
        description="The type of execution mechanism. Currently only supports 'python_function'."
    )
    call: FunctionCallDetails = Field(
        ..., 
        description="Details required to import and run the Python function."
    )

# --- 4. USAGE POLICY SECTION ---

class UsagePolicy(BaseModel):
    """
    The 'usage_policy' block: The Agent's manual for this tool.
    """
    use_when: List[str] = Field(
        ..., 
        description="A list of specific scenarios, user intents, or data conditions where this tool is the BEST choice."
    )
    do_not_use_when: List[str] = Field(
        ..., 
        description="A list of scenarios where this tool is inefficient, incorrect, unsafe, or better alternatives exist."
    )

# --- 5. OUTPUT SCHEMA SECTION ---

class OutputSchema(BaseModel):
    """
    The 'output_schema' block: Defines what the tool returns.
    """
    description: Optional[str] = Field(
        None, 
        description="A text description of the returned object or file artifact."
    )
    type: Optional[str] = Field(
        ..., 
        description="The JSON type of the output (e.g., 'object', 'array', 'string', 'number')."
    )
    # We use Dict[str, Any] for nested properties to allow flexibility 
    # and prevent infinite recursion issues during LLM generation.
    properties: Optional[Dict[str, Any]] = Field(
        None, 
        description="If type is 'object', this map defines the keys and their sub-schemas."
    )
    items: Optional[Dict[str, Any]] = Field(
        None, 
        description="If type is 'array', this defines the schema of the items in the list."
    )

# --- 6. ROOT SKILL CARD ---

class SkillCard(BaseModel):
    """
    The Master Schema for a JUNO Skill Card.
    """
    skill_id: str = Field(
        ..., 
        description="Unique identifier for the skill (usually matches the function name)."
    )
    summary: SkillSummary = Field(
        ..., 
        description="General information about the skill, including its name, description, and input parameters."
    )
    execution_details: SkillExecution = Field(
        ..., 
        description="Technical details on how to load and execute the underlying Python code."
    )
    usage_policy: UsagePolicy = Field(
        ..., 
        description="Guidelines for the AI Agent on when to select (or avoid) this tool."
    )
    output_schema: OutputSchema = Field(
        ..., 
        description="The expected structure of the data returned by the tool (used for planning)."
    )