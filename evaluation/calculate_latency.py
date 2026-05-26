import sys
import json
import argparse
from collections import defaultdict
import statistics

def calculate_latency(data, file_name):
    results = data.get("results", [])
    if not results:
        print("No results found in the JSON file.")
        return

    # Autodetect format by looking at the first result object
    first_run = results[0]
    is_comparison = isinstance(first_run.get("juno"), dict) or isinstance(first_run.get("baseline_b1"), dict) or isinstance(first_run.get("baseline"), dict)

    print(f"=== Latency Analysis for {file_name} ===\n")

    if is_comparison:
        # Group by agent, then by scenario
        latencies = {
            "juno": defaultdict(list),
            "baseline": defaultdict(list)
        }
        for run in results:
            scenario_id = run.get("scenario_id", run.get("query_id", "Unknown"))
            
            juno_data = run.get("juno", {})
            if "latency_seconds" in juno_data and juno_data["latency_seconds"] is not None:
                latencies["juno"][scenario_id].append(juno_data["latency_seconds"])
                
            base_key = "baseline" if "baseline" in run else "baseline_b1"
            base_data = run.get(base_key, {})
            if "latency_seconds" in base_data and base_data["latency_seconds"] is not None:
                latencies["baseline"][scenario_id].append(base_data["latency_seconds"])
                
        for agent in ["juno", "baseline"]:
            agent_lats = latencies[agent]
            if not agent_lats:
                continue
            print(f"--- AGENT: {agent.upper()} ---")
            for sc_id, lats in sorted(agent_lats.items()):
                avg_lat = statistics.mean(lats)
                print(f"Scenario: {sc_id:<35} | Average Latency: {avg_lat:7.2f}s (over {len(lats)} runs)")
            print()
            
    else:
        # Standard single-agent JUNO run
        latencies = defaultdict(list)
        for run in results:
            scenario_id = run.get("scenario_id", run.get("query_id", "Unknown"))
            lat = run.get("latency_seconds")
            if lat is not None:
                latencies[scenario_id].append(lat)
                
        if not latencies:
            print("No latency data found.")
            return
            
        print(f"--- AGENT: JUNO (Standard Format) ---")
        for sc_id, lats in sorted(latencies.items()):
            avg_lat = statistics.mean(lats)
            print(f"Scenario: {sc_id:<35} | Average Latency: {avg_lat:7.2f}s (over {len(lats)} runs)")
        print()

def main():
    parser = argparse.ArgumentParser(description="Calculate average latency per scenario from evaluation JSON")
    parser.add_argument("input_file", help="Path to the JSON evaluation file")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {args.input_file}: {e}")
        sys.exit(1)

    calculate_latency(data, args.input_file)

if __name__ == "__main__":
    main()
