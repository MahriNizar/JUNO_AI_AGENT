"""
Evaluation Runner for JUNO Agent Benchmark

Pure runner — executes agents on benchmark queries and records raw traces.
No outcome classification at run-time; analysis happens post-hoc on the output JSON.

Modes:
  --mode juno      Run only the JUNO agent
  --mode baseline  Run only the B1 baseline agent
  --mode compare   Run both agents side-by-side
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from baseline_b1 import create_react_agent_b1
from langchain_core.messages import AIMessage, BaseMessage
from langchain_ollama import ChatOllama

from main import build_graph
from state_and_class import FoldedMemory
from models import (
    QuerySpec,
    StructuredEvidence,
    ToolCallRecord,
    load_benchmark,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "skill_folder": "skills",
    "timeout_seconds": 1200,
    "recursion_limit": 50,
    "random_seed": 42
}
TEST_QUERY = 1

# =============================================================================
# EVIDENCE EXTRACTION
# =============================================================================

def extract_evidence_from_state(final_state: Dict[str, Any],
                                 crashed: bool = False,
                                 crash_message: Optional[str] = None) -> StructuredEvidence:
    """
    Extract structured evidence from the agent's final state.
    Pure extraction — no classification or scoring.
    """
    evidence = StructuredEvidence(
        crashed=crashed,
        crash_message=crash_message
    )
    
    if crashed:
        return evidence
    
    messages = final_state.get("messages", [])
    
    # Extract tool calls from the message history
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_output = _find_tool_output(messages, tc.get("id", ""))
                tool_args = tc.get("args", {})
                
                record = ToolCallRecord(
                    skill_name=tc.get("name", "unknown"),
                    args=tool_args,
                    args_hash=ToolCallRecord.compute_hash(tool_args),
                    output_hash=ToolCallRecord.compute_hash(tool_output) if tool_output else "",
                    output_summary=str(tool_output)[:200] if tool_output else "",
                    success=tool_output is not None and not _is_error_output(tool_output),
                    error_message=str(tool_output) if _is_error_output(tool_output) else None
                )
                evidence.tool_calls.append(record)
    
    # Extract final answer from the last AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            evidence.final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    
    return evidence


def _find_tool_output(messages: List[BaseMessage], tool_call_id: str) -> Optional[Any]:
    """Find the ToolMessage response for a given tool call ID."""
    for msg in messages:
        if hasattr(msg, "tool_call_id") and msg.tool_call_id == tool_call_id:
            return msg.content
    return None


def _is_error_output(output: Any) -> bool:
    """Check if the tool output represents an error."""
    if output is None:
        return True
    output_str = str(output).lower()
    error_patterns = ["error", "exception", "failed", "traceback", "could not"]
    return any(p in output_str for p in error_patterns)


# =============================================================================
# QUERY SELECTION
# =============================================================================

def select_queries(benchmark: List[QuerySpec],
                   queries_per_tier: Dict[int, int] = None,
                   random_seed: int = 42) -> List[QuerySpec]:
    """
    Select queries from the benchmark based on tier quotas.
    
    Args:
        benchmark: Full list of benchmark queries
        queries_per_tier: Dict mapping tier -> number of queries to select.
                         Use -1 to select all queries for a tier.
                         Use -2 to select only the TEST_QUERY index.
                         Default: select all queries.
        random_seed: Seed for reproducible random selection
    
    Returns:
        Selected subset of queries
    """
    random.seed(random_seed)
    
    # Group queries by tier
    by_tier: Dict[int, List[QuerySpec]] = defaultdict(list)
    for q in benchmark:
        by_tier[q.tier].append(q)
    
    # Default: select all if not specified
    if queries_per_tier is None:
        return benchmark
    
    selected = []
    for tier, count in queries_per_tier.items():
        tier_queries = by_tier.get(tier, [])
        
        if count == -1 or count >= len(tier_queries):
            # Select all
            selected.extend(tier_queries)
        elif count == -2:
            selected.extend([tier_queries[TEST_QUERY-1]])
        elif count > 0:
            # First N
            selected.extend(tier_queries[:count])
    
    # Sort by ID for consistent ordering
    selected.sort(key=lambda q: q.id)
    return selected


def get_tier_distribution(benchmark: List[QuerySpec]) -> Dict[int, int]:
    """Get the distribution of queries by tier."""
    distribution = defaultdict(int)
    for q in benchmark:
        distribution[q.tier] += 1
    return dict(sorted(distribution.items()))


# =============================================================================
# RESULT FORMATTING HELPERS
# =============================================================================

def _evidence_to_result(query: QuerySpec, run_id: int, agent_name: str,
                        evidence: StructuredEvidence, latency: float) -> dict:
    """Format a single agent run into the output JSON structure."""
    return {
        "query_id": query.id,
        "tier": query.tier,
        "query_text": query.query,
        "description": query.description,
        "run_id": run_id,
        "agent_name": agent_name,
        "latency_seconds": latency,
        "crashed": evidence.crashed,
        "crash_message": evidence.crash_message,
        "tool_calls": [
            {
                "skill_name": tc.skill_name,
                "args": tc.args,
                "output_summary": tc.output_summary,
                "success": tc.success,
                "error_message": tc.error_message,
            }
            for tc in evidence.tool_calls
        ],
        "tool_call_count": len(evidence.tool_calls),
        "final_answer": evidence.final_answer,
    }


# =============================================================================
# JUNO AGENT RUNNER
# =============================================================================

def run_single_query_juno(app, query: QuerySpec,
                          config: Dict[str, Any] = None,
                          verbose: bool = False) -> Tuple[StructuredEvidence, float]:
    """Run the JUNO agent on a single query and extract evidence."""
    config = config or DEFAULT_CONFIG.copy()
    thread_id = f"eval-{query.id}-{int(time.time())}"
    
    inputs = {
        "messages": [("user", query.query)],
        "memory": FoldedMemory(),
        "plan": [],
        "tool_call_count": 0,

    }
    
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.get("recursion_limit", 50)
    }
    
    if verbose:
        print(f"\n  Running query {query.id}: {query.query[:60]}...")
    
    start_time = time.time()
    final_state = None
    crashed = False
    crash_message = None
    
    try:
        for event in app.stream(inputs, config=run_config, stream_mode="values"):
            final_state = event
            
    except Exception as e:
        crashed = True
        crash_message = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"    ❌ Crashed: {crash_message}")
            import traceback
            traceback.print_exc()
    
    latency = time.time() - start_time
    
    if final_state is None:
        final_state = inputs
        crashed = True
        crash_message = crash_message or "No state returned"
    
    evidence = extract_evidence_from_state(final_state, crashed, crash_message)
    
    if verbose:
        status = "CRASH" if crashed else f"{len(evidence.tool_calls)} tool calls"
        print(f"    → {status} ({latency:.1f}s)")
    
    return evidence, latency


# =============================================================================
# BASELINE AGENT RUNNER
# =============================================================================

def run_single_query_b1(app, query: QuerySpec,
                        config: Dict[str, Any] = None,
                        verbose: bool = False) -> Tuple[StructuredEvidence, float]:
    """Run the baseline B1 agent on a single query and extract evidence."""
    config = config or DEFAULT_CONFIG.copy()
    
    inputs = {
        "messages": [("user", query.query)],
    }
    
    if verbose:
        print(f"\n  Running query {query.id}: {query.query[:60]}...")
    
    start_time = time.time()
    final_state = None
    crashed = False
    crash_message = None
    
    try:
        for chunk in app.stream(inputs, stream_mode="values"):
            final_state = chunk
            messages = chunk.get("messages", [])
            if messages and verbose:
                last_msg = messages[-1]
                print(f"    [{last_msg.type}]: {str(last_msg.content)[:100]}...")
                
    except Exception as e:
        crashed = True
        crash_message = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"    ❌ Crashed: {crash_message}")
            import traceback
            traceback.print_exc()
    
    latency = time.time() - start_time
    
    if final_state is None:
        final_state = inputs
        crashed = True
        crash_message = crash_message or "No state returned"
    
    evidence = extract_evidence_from_state(final_state, crashed, crash_message)
    
    if verbose:
        status = "CRASH" if crashed else f"{len(evidence.tool_calls)} tool calls"
        print(f"    → {status} ({latency:.1f}s)")
    
    return evidence, latency


# =============================================================================
# EVALUATION MODES
# =============================================================================

def run_single_agent(mode: str,
                     benchmark_path: str = "evaluation/query_benchmark.json",
                     queries_per_tier: Dict[int, int] = None,
                     runs_per_query: int = 1,
                     config: Dict[str, Any] = None,
                     output_dir: str = "evaluation/results",
                     verbose: bool = True) -> Dict[str, Any]:
    """
    Run a single agent (juno or baseline) on the benchmark and save raw traces.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    random.seed(config.get("random_seed", 42))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and select queries
    benchmark = load_benchmark(benchmark_path)
    selected = select_queries(benchmark, queries_per_tier, config["random_seed"])
    
    if verbose:
        agent_label = "JUNO" if mode == "juno" else "B1 Baseline (ReAct)"
        print(f"=== {agent_label} Agent Evaluation ===")
        print(f"Selected queries: {len(selected)}")
        print(f"Runs per query: {runs_per_query}")
        print(f"Tier distribution: {get_tier_distribution(selected)}")
        print()
    
    # Build agent
    if mode == "juno":
        if verbose: print("Building JUNO agent graph...")
        app = build_graph(config["skill_folder"])
        run_fn = run_single_query_juno
        agent_name = "juno"
    else:
        from baseline_b1 import create_react_agent_b1
        if verbose: print("Building Baseline B1 agent...")
        app = create_react_agent_b1(config)
        run_fn = run_single_query_b1
        agent_name = "baseline_b1"
    
    # Run evaluation
    all_results = []
    
    for i, query in enumerate(selected, 1):
        if verbose:
            print(f"\n[{i}/{len(selected)}] Query {query.id} (Tier {query.tier})")
        
        for run_idx in range(runs_per_query):
            if runs_per_query > 1 and verbose:
                print(f"  Run {run_idx + 1}/{runs_per_query}")
            
            evidence, latency = run_fn(app, query, config, verbose)
            result = _evidence_to_result(query, run_idx, agent_name, evidence, latency)
            all_results.append(result)
    
    # Save results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "config": config,
        "queries_per_tier": queries_per_tier,
        "runs_per_query": runs_per_query,
        "total_queries": len(selected),
        "total_runs": len(all_results),
        "results": all_results,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"eval_{mode}_{timestamp}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    if verbose:
        crashes = sum(1 for r in all_results if r["crashed"])
        print(f"\n=== Results Summary ===")
        print(f"Total runs: {len(all_results)}")
        print(f"Crashes: {crashes}")
        print(f"Results saved to: {output_path}")
    
    return summary


def run_comparison(benchmark_path: str = "evaluation/query_benchmark.json",
                   queries_per_tier: Dict[int, int] = None,
                   runs_per_query: int = 1,
                   config: Dict[str, Any] = None,
                   output_dir: str = "evaluation/results",
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Run both JUNO and B1 Baseline agents side-by-side on the benchmark.
    """
    
    
    config = {**DEFAULT_CONFIG, **(config or {})}
    random.seed(config.get("random_seed", 42))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and select queries
    benchmark = load_benchmark(benchmark_path)
    selected = select_queries(benchmark, queries_per_tier, config["random_seed"])
    
    if verbose:
        print(f"=== Agent Comparison: JUNO vs B1 Baseline (ReAct) ===")
        print(f"Selected queries: {len(selected)}")
        print(f"Runs per query: {runs_per_query}")
        print(f"Tier distribution: {get_tier_distribution(selected)}")
        print()
    
    # Build agents
    print("Building JUNO agent...")
    main_app = build_graph(config["skill_folder"])
    
    print("Building Baseline B1 agent...")
    baseline_app = create_react_agent_b1(config)
    
    comparison_results = []
    
    for i, query in enumerate(selected, 1):
        if verbose:
            print(f"\n[{i}/{len(selected)}] Query {query.id} (Tier {query.tier})")
            
        for run_idx in range(runs_per_query):
            if runs_per_query > 1 and verbose:
                print(f"  Run {run_idx + 1}/{runs_per_query}")
            
            # Run Main Agent
            if verbose: print("  Running JUNO Agent...")
            main_evidence, main_latency = run_single_query_juno(main_app, query, config, verbose=False)
            
            # Run Baseline Agent
            if verbose: print("  Running Baseline Agent...")
            baseline_evidence, baseline_latency = run_single_query_b1(baseline_app, query, config, verbose=False)
            
            if verbose:
                m_status = "CRASH" if main_evidence.crashed else f"{len(main_evidence.tool_calls)} tools"
                b_status = "CRASH" if baseline_evidence.crashed else f"{len(baseline_evidence.tool_calls)} tools"
                print(f"    JUNO: {m_status} ({main_latency:.1f}s)")
                print(f"    B1:   {b_status} ({baseline_latency:.1f}s)")
            
            comparison_results.append({
                "query_id": query.id,
                "tier": query.tier,
                "query_text": query.query,
                "description": query.description,
                "run_id": run_idx,
                "juno": _evidence_to_result(query, run_idx, "juno", main_evidence, main_latency),
                "baseline_b1": _evidence_to_result(query, run_idx, "baseline_b1", baseline_evidence, baseline_latency),
            })
    
    # Save results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "compare",
        "config": config,
        "queries_per_tier": queries_per_tier,
        "runs_per_query": runs_per_query,
        "total_queries": len(selected),
        "total_runs": len(comparison_results),
        "results": comparison_results,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
        
    if verbose:
        print(f"\n=== Comparison Complete ===")
        print(f"Total queries: {len(comparison_results)}")
        print(f"Results saved to: {output_path}")
    
    return summary


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Run evaluation with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run JUNO Agent Evaluation")
    parser.add_argument("--mode", choices=["juno", "baseline", "compare"],
                        default="compare", help="Which agent(s) to run")
    parser.add_argument("--benchmark", default="evaluation/query_benchmark.json",
                        help="Path to benchmark JSON file")
    parser.add_argument("--output", default="evaluation/results",
                        help="Output directory for results")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per query (for robustness)")
    parser.add_argument("--tier1", type=int, default=-1,
                        help="Queries from tier 1 (-1 for all)")
    parser.add_argument("--tier2", type=int, default=-1,
                        help="Queries from tier 2 (-1 for all)")
    parser.add_argument("--tier3", type=int, default=-1,
                        help="Queries from tier 3 (-1 for all)")
    parser.add_argument("--tier4", type=int, default=0,
                        help="Queries from tier 4 (-1 for all)")
    parser.add_argument("--tier5", type=int, default=-1,
                        help="Queries from tier 5 (-1 for all)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    queries_per_tier = {
        1: args.tier1,
        2: args.tier2,
        3: args.tier3,
        4: args.tier4,
        5: args.tier5
    }
    # Remove tiers with 0 queries; keep -1 (all) and -2 (test)
    queries_per_tier = {k: v for k, v in queries_per_tier.items() if v != 0} or None
    
    if args.mode == "compare":
        results = run_comparison(
            benchmark_path=args.benchmark,
            queries_per_tier=queries_per_tier,
            runs_per_query=args.runs,
            output_dir=args.output,
            verbose=not args.quiet
        )
    else:
        results = run_single_agent(
            mode=args.mode,
            benchmark_path=args.benchmark,
            queries_per_tier=queries_per_tier,
            runs_per_query=args.runs,
            output_dir=args.output,
            verbose=not args.quiet
        )
    
    return results


if __name__ == "__main__":
    main()
