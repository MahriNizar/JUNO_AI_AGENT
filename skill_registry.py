import os
import yaml
from skill_models import SkillCard # Assumes 'skill_models.py' exists as defined before
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
class SkillRegistry:
    def __init__(self, skill_directory: str = "skills"):
        """
        Initializes the registry by loading and validating all skills
        from the specified directory.
        """
        print(f"--- Initializing SkillRegistry from '{skill_directory}'... ---")
        self.skills: Dict[str, SkillCard] = {} # Stores skills by their "call name"
        
        for filename in os.listdir(skill_directory):
            if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                continue
                
            file_path = os.path.join(skill_directory, filename)
            try:
                with open(file_path, 'r') as f:
                    skill_data = yaml.safe_load(f)
                    
                    # Validate the YAML against our Pydantic model
                    skill_card = SkillCard.model_validate(skill_data)
                    
                    # Store the skill, keyed by its callable name
                    skill_name = skill_card.summary.name
                    if skill_name in self.skills:
                        print(f"  [WARN] Duplicate skill name '{skill_name}' found. Overwriting.")
                    
                    self.skills[skill_name] = skill_card
                    print(f"  [OK] Loaded and validated skill: {skill_name}")

            except Exception as e:
                print(f"  [FAIL] Error loading {filename}: {e}")
        
        print(f"--- SkillRegistry initialized with {len(self.skills)} skills. ---")

    def get_skill_by_name(self, name: str) -> SkillCard:
        """Finds a skill card by its callable name."""
        skill = self.skills.get(name)
        if not skill:
            raise KeyError(f"Skill '{name}' not found in registry.")
        return skill

    def get_all_skills(self) -> List[SkillCard]:
        """Returns a list of all loaded SkillCard objects."""
        return list(self.skills.values())

    def get_agent_summarys_for_tools(self) -> List[BaseModel]:
        """
        Dynamically creates and returns a list of Pydantic models 
        representing the 'summary' for use with LangChain's .bind_tools()
        """
        pydantic_tools = []
        for skill in self.skills.values():
            # Dynamically create a Pydantic model from the skill's summary
            # This is what LangChain's .bind_tools() expects
            tool_model = create_pydantic_model_from_skill(skill)
            pydantic_tools.append(tool_model)
        
        return pydantic_tools

# We need a helper function to convert our 'summary' into 
# the format LangChain's tool-binding expects (a Pydantic BaseModel).


from pydantic import create_model, Field

def create_pydantic_model_from_skill(skill: SkillCard) -> BaseModel:
    """
    Dynamically creates a Pydantic model for a specific tool
    based on the Skill Card's summary.
    """
    summary = skill.summary
    fields = {}
    
    # Check if parameters exist (it's now a Pydantic object, not a dict)
    if summary.parameters and summary.parameters.properties:
        
        # Iterate over the 'properties' dictionary of ParameterDetails objects
        for param_name, details in summary.parameters.properties.items():
            
            # 1. Get the type (Access via dot notation now)
            param_type_str = details.type
            
            if param_type_str == 'string':
                py_type = str
            elif param_type_str == 'integer':
                py_type = int
            elif param_type_str == "float" or param_type_str == "number":
                py_type = float
            elif param_type_str == "boolean":
                py_type = bool
            elif param_type_str == "array":
                py_type = list
            else:
                py_type = str # Default to string for unknown types

            # 2. Get description and default
            param_desc = details.description
            default_val = details.default
            is_required = details.required

            # 3. Create the Pydantic Field definition
            if is_required:
                
                fields[param_name] = (py_type, Field(..., description=param_desc))
            elif default_val is not None:
                # Optional with a specific default
                fields[param_name] = (py_type, Field(default=default_val, description=param_desc))
            else:
                # Optional but no specific default provided (default to None)
                fields[param_name] = (Optional[py_type], Field(default=None, description=param_desc))

    # Create the dynamic Pydantic model
    DynamicToolModel = create_model(
        summary.name, # The name of the tool
        **fields
    )
    # Attach the description as the docstring
    DynamicToolModel.__doc__ = summary.description
    
    return DynamicToolModel

