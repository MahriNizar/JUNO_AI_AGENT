"""
Evaluation Data Models

Pure data schemas for the JUNO Agent evaluation pipeline.
No classification or analysis logic — that happens post-hoc on the output JSON.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union


# =============================================================================
# STRUCTURED EVIDENCE SCHEMA
# =============================================================================

@dataclass
class ToolCallRecord:
    """Record of a single tool execution."""
    skill_name: str
    args: Dict[str, Any]     # Actual arguments used
    args_hash: str           # SHA256 of JSON-serialized args
    output_hash: str         # SHA256 of JSON-serialized output
    output_summary: str      # First 200 chars of output for debugging
    success: bool
    error_message: Optional[str] = None
    
    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute SHA256 hash of JSON-serialized data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@dataclass
class StructuredEvidence:
    """Machine-parseable evidence from agent execution."""
    
    # Tool execution trace
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    
    # Final answer text
    final_answer: str = ""
    
    # Crash indicator
    crashed: bool = False
    crash_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['tool_calls'] = [asdict(tc) for tc in self.tool_calls]
        return result


@dataclass
class QuerySpec:
    """Specification for an evaluation query (mirrors query_benchmark.json schema)."""
    id: str
    tier: int
    query: str
    description: str = ""
    expected_tools: Optional[List[str]] = None
    forbidden_tools: Optional[List[str]] = None
    alternative_tools: Optional[List[str]] = None
    allowed_refusals: Optional[List[str]] = None
    accepted_toolchains: Optional[List[List[str]]] = None
    tool_call_budget: int = 5
    expected_args: Union[Dict[str, Any], List[Dict[str, Any]]] = field(default_factory=dict)
    ground_truth: Optional[Dict[str, Any]] = None


# =============================================================================
# LOADER UTILITY
# =============================================================================

def load_benchmark(path: str) -> List[QuerySpec]:
    """Load benchmark queries from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        queries.append(QuerySpec(
            id=item['id'],
            tier=item['tier'],
            query=item['query'],
            description=item.get('description', ''),
            expected_tools=item.get('expected_tools'),
            forbidden_tools=item.get('forbidden_tools'),
            alternative_tools=item.get('alternative_tools'),
            allowed_refusals=item.get('allowed_refusals'),
            accepted_toolchains=item.get('accepted_toolchains'),
            tool_call_budget=item.get('tool_call_budget', 5),
            expected_args=item.get('expected_args', {}),
            ground_truth=item.get('ground_truth')
        ))
    
    return queries


if __name__ == "__main__":
    # Quick test
    benchmark = load_benchmark("evaluation/query_benchmark.json")
    print(f"Loaded {len(benchmark)} queries:")
    for q in benchmark:
        print(f"  [{q.tier}] {q.id}: {q.description}")
