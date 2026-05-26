"""
Long Scenario Evaluation Runner
================================
Runs JUNO agent (and optionally the B1 baseline) on the long-running benchmark
scenarios defined in `evaluation/long_scenario_queries.json`.

Unlike the regular benchmark, long scenarios:
  - Have no tier structure (all are tier 3+)
  - Require many sequential tool calls (inspect → slice × N → analyse × N)
  - Are evaluated on tool coverage and final answer quality, not speed

Output: a timestamped JSON in evaluation/results/long_eval_<mode>_<ts>.json

Usage:
    python evaluation/run_long_evaluation.py --mode juno
    python evaluation/run_long_evaluation.py --mode baseline
    python evaluation/run_long_evaluation.py --mode compare
    python evaluation/run_long_evaluation.py --mode juno --scenario long_A_flashing_pmt
"""

import os
import sys
import json
import time
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — make sure the project root is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, BaseMessage
from langchain_ollama import ChatOllama

from main import build_graph
from state_and_class import FoldedMemory


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    "skill_folder": "skills",
    "timeout_seconds": 3600,   # long scenarios can take a long time
    "recursion_limit": 150,    # ~40-45 tool calls
}

LONG_BENCH_PATH = str(ROOT / "evaluation" / "long_scenario_queries.json")
RESULTS_DIR     = str(ROOT / "evaluation" / "results")


# =============================================================================
# SCENARIO LOADING
# =============================================================================

def load_long_scenarios(path: str = LONG_BENCH_PATH) -> List[Dict[str, Any]]:
    """Load and validate the long scenario query list."""
    with open(path, encoding="utf-8") as f:
        scenarios = json.load(f)

    required = {"id", "query", "expected_tools", "ground_truth_facts", "min_tool_calls"}
    for s in scenarios:
        missing = required - s.keys()
        if missing:
            raise ValueError(f"Scenario '{s.get('id', '?')}' missing fields: {missing}")

    print(f"  Loaded {len(scenarios)} long scenarios from {path}")
    return scenarios


# =============================================================================
# EVIDENCE EXTRACTION  (shared by both agents)
# =============================================================================

def _find_tool_output(messages: List[BaseMessage], tool_call_id: str) -> Optional[str]:
    for msg in messages:
        if hasattr(msg, "tool_call_id") and msg.tool_call_id == tool_call_id:
            return msg.content
    return None


def _is_error_output(output: Any) -> bool:
    if output is None:
        return True
    s = str(output).lower()
    return any(p in s for p in ("error", "exception", "failed", "traceback", "could not"))


def extract_evidence(final_state: Dict[str, Any],
                     crashed: bool = False,
                     crash_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Pull tool call records and final answer out of the agent's final state.
    Returns a plain dict (no Pydantic dependency).
    """
    if crashed:
        return {
            "crashed": True,
            "crash_message": crash_message,
            "tool_calls": [],
            "tool_call_sequence": [],
            "final_answer": None,
        }

    messages = final_state.get("messages", [])
    tool_calls = []
    tool_sequence = []

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_output = _find_tool_output(messages, tc.get("id", ""))
                success = tool_output is not None and not _is_error_output(tool_output)
                record = {
                    "skill_name":     tc.get("name", "unknown"),
                    "args":           tc.get("args", {}),
                    "output_summary": str(tool_output)[:300] if tool_output else "",
                    "success":        success,
                    "error_message":  str(tool_output) if not success else None,
                }
                tool_calls.append(record)
                tool_sequence.append(tc.get("name", "unknown"))

    # Find final answer — last AIMessage named "FinalAnswer", or last AIMessage
    final_answer = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Try to parse the JSON blob to extract just the answer text
            try:
                parsed = json.loads(content)
                final_answer = parsed.get("final_answer") or content
            except (json.JSONDecodeError, TypeError):
                final_answer = content
            break

    return {
        "crashed":           False,
        "crash_message":     None,
        "tool_calls":        tool_calls,
        "tool_call_sequence": tool_sequence,
        "final_answer":      final_answer,
    }


# =============================================================================
# FORMAT SINGLE RESULT
# =============================================================================

def format_result(scenario: Dict[str, Any],
                  agent_name: str,
                  evidence: Dict[str, Any],
                  latency: float,
                  run_id: int = 0) -> Dict[str, Any]:
    """Build the output dict for one scenario × agent run."""
    return {
        "scenario_id":         scenario["id"],
        "tier":                scenario.get("tier", 3),
        "query_text":          scenario["query"],
        "agent_name":          agent_name,
        "run_id":              run_id,
        "latency_seconds":     round(latency, 2),
        # --- crash info ---
        "crashed":             evidence["crashed"],
        "crash_message":       evidence["crash_message"],
        # --- tool trace ---
        "tool_call_sequence":  evidence["tool_call_sequence"],
        "tool_call_count":     len(evidence["tool_calls"]),
        "tool_calls":          evidence["tool_calls"],
        # --- final answer ---
        "final_answer":        evidence["final_answer"],
        # --- ground truth (for manual / LLM-judge scoring) ---
        "ground_truth_facts":  scenario["ground_truth_facts"],
    }


# =============================================================================
# JUNO AGENT RUNNER
# =============================================================================

def run_juno(app, scenario: Dict[str, Any],
             config: Dict[str, Any],
             verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    thread_id = f"long-eval-{scenario['id']}-{int(time.time())}"

    inputs = {
        "messages": [("user", scenario["query"])],
        "memory": FoldedMemory(),
        "plan": [],
        "tool_call_count": 0,
    }
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.get("recursion_limit", 150),
    }

    if verbose:
        print(f"\n  [JUNO] {scenario['id']}: {scenario['query'][:80]}...")

    start = time.time()
    final_state = None
    crashed = False
    crash_msg = None

    try:
        for event in app.stream(inputs, config=run_config, stream_mode="values"):
            final_state = event
    except Exception as e:
        crashed = True
        crash_msg = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"    ❌ JUNO crashed: {crash_msg}")
            traceback.print_exc()

    latency = time.time() - start

    if final_state is None:
        final_state = inputs
        crashed = True
        crash_msg = crash_msg or "No state returned"

    evidence = extract_evidence(final_state, crashed, crash_msg)

    if verbose:
        status = "CRASH" if crashed else f"{len(evidence['tool_calls'])} tool calls"
        print(f"    → JUNO: {status} ({latency:.1f}s)")

    return evidence, latency


# =============================================================================
# BASELINE B1 RUNNER
# =============================================================================

def run_baseline(app, scenario: Dict[str, Any],
                 config: Dict[str, Any],
                 verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    inputs = {"messages": [("user", scenario["query"])]}

    if verbose:
        print(f"\n  [B1] {scenario['id']}: {scenario['query'][:80]}...")

    start = time.time()
    final_state = None
    crashed = False
    crash_msg = None

    try:
        for chunk in app.stream(inputs, stream_mode="values"):
            final_state = chunk
    except Exception as e:
        crashed = True
        crash_msg = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"    ❌ B1 crashed: {crash_msg}")
            traceback.print_exc()

    latency = time.time() - start

    if final_state is None:
        final_state = inputs
        crashed = True
        crash_msg = crash_msg or "No state returned"

    evidence = extract_evidence(final_state, crashed, crash_msg)

    if verbose:
        status = "CRASH" if crashed else f"{len(evidence['tool_calls'])} tool calls"
        print(f"    → B1:   {status} ({latency:.1f}s)")

    return evidence, latency


# =============================================================================
# MAIN EVALUATION MODES
# =============================================================================

def run_evaluation(mode: str,
                   scenarios: List[Dict[str, Any]],
                   config: Dict[str, Any],
                   runs_per_scenario: int = 1,
                   verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run the selected mode (juno | baseline | compare) over all scenarios.
    Each scenario is executed `runs_per_scenario` times.
    Returns a list of result dicts.
    """
    # Build agent(s)
    juno_app = baseline_app = None

    if mode in ("juno", "compare"):
        if verbose:
            print("Building JUNO agent graph...")
        juno_app = build_graph(config["skill_folder"])

    if mode in ("baseline", "compare"):
        if verbose:
            print("Building Baseline B1 agent...")
        from baseline_b1 import create_react_agent_b1
        baseline_app = create_react_agent_b1(config)

    results = []
    n = len(scenarios)

    for i, scenario in enumerate(scenarios, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i}/{n}] Scenario: {scenario['id']}")
            print(f"  Expected tools : {scenario['expected_tools']}")
            print(f"  Min tool calls : {scenario['min_tool_calls']}")

        for run_idx in range(runs_per_scenario):
            if runs_per_scenario > 1 and verbose:
                print(f"  Run {run_idx + 1}/{runs_per_scenario}")

            if mode == "compare":
                j_evidence, j_latency = run_juno(juno_app, scenario, config, verbose)
                b_evidence, b_latency = run_baseline(baseline_app, scenario, config, verbose)

                results.append({
                    "scenario_id":        scenario["id"],
                    "tier":               scenario.get("tier", 3),
                    "query_text":         scenario["query"],
                    "ground_truth_facts": scenario["ground_truth_facts"],
                    "run_id":             run_idx,
                    "juno":     format_result(scenario, "juno",        j_evidence, j_latency, run_idx),
                    "baseline": format_result(scenario, "baseline_b1", b_evidence, b_latency, run_idx),
                })

            elif mode == "juno":
                evidence, latency = run_juno(juno_app, scenario, config, verbose)
                results.append(format_result(scenario, "juno", evidence, latency, run_idx))

            else:  # baseline
                evidence, latency = run_baseline(baseline_app, scenario, config, verbose)
                results.append(format_result(scenario, "baseline_b1", evidence, latency, run_idx))

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run JUNO long-scenario evaluation"
    )
    parser.add_argument(
        "--mode", choices=["juno", "baseline", "compare"], default="juno",
        help="Which agent(s) to run (default: compare)"
    )
    parser.add_argument(
        "--benchmark", default=LONG_BENCH_PATH,
        help="Path to long_scenario_queries.json"
    )
    parser.add_argument(
        "--output", default=RESULTS_DIR,
        help="Output directory for result JSONs"
    )
    parser.add_argument(
        "--scenario", default=None,
        help="Run only a specific scenario ID (e.g. long_A_flashing_pmt)"
    )
    parser.add_argument(
        "--recursion-limit", type=int, default=150,
        help="LangGraph recursion limit (default 150)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs per scenario (default 1)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, "recursion_limit": args.recursion_limit}
    verbose = not args.quiet
    os.makedirs(args.output, exist_ok=True)

    # Load scenarios
    scenarios = load_long_scenarios(args.benchmark)

    # Optional filter by scenario ID
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            print(f"ERROR: Scenario '{args.scenario}' not found in benchmark.")
            sys.exit(1)
        print(f"  Running single scenario: {args.scenario}")

    if verbose:
        print(f"\n=== Long Scenario Evaluation ===")
        print(f"  Mode      : {args.mode}")
        print(f"  Scenarios : {len(scenarios)}")
        print(f"  Runs each : {args.runs}")
        print(f"  Recursion : {config['recursion_limit']}")

    # Run
    results = run_evaluation(args.mode, scenarios, config, args.runs, verbose)

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"long_eval_{args.mode}_{timestamp}.json")

    summary = {
        "timestamp":        datetime.now().isoformat(),
        "mode":             args.mode,
        "config":           config,
        "benchmark_path":   args.benchmark,
        "total_scenarios":  len(scenarios),
        "runs_per_scenario": args.runs,
        "total_runs":       len(results),
        "results":          results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"\n{'='*60}")
        print(f"=== Evaluation Complete ===")
        print(f"  Total scenarios : {len(results)}")
        crashes = sum(
            1 for r in results
            if (r.get("crashed") or
                (r.get("juno", {}).get("crashed") and r.get("baseline", {}).get("crashed")))
        )
        print(f"  Crashes         : {crashes}")
        print(f"  Results saved → {output_path}")

    return summary


if __name__ == "__main__":
    main()
