# Reproducibility Record

This file records the execution environment, artifacts, parameters, and commands needed
to reproduce the thesis experiments. 


## Known Reproducibility Limitation

| Limitation | Consequence | Mitigation / interpretation |
| --- | --- | --- |
| The exact specifications of the server used for the benchmark runs are no longer recoverable because that server stopped working around mid-March. | Exact performance reproducibility, especially latency and wall-clock runtime, is not possible from hardware metadata alone. | Treat the archived benchmark result files as the primary evidence for the reported benchmark outcomes. New reruns can test qualitative behavior, but should not be expected to reproduce identical timing. |
| The agent relies on structured LLM outputs for tool selection, argument formatting, and final-answer control. | Smaller models can fail by producing invalid JSON, malformed structured outputs, or incomplete tool-call decisions. | For meaningful reruns, use a tool-capable local model of at least approximately 14B parameters. The thesis code path used `gpt-oss:20b`. Models below this size should be reported as stress tests, not direct reproductions. |

## Hardware And Operating System (Unrelated to the LLM hosted)

| Item | Value observed locally | Notes |
| --- | --- | --- |
| Operating system | Microsoft Windows NT 10.0.19045.0 | Report exact OS/build used for final runs |
| Shell | Windows PowerShell 5.1.19041.6456 | Command examples below use PowerShell |
| Python interpreter | Python 3.13.7 in `.venv` | From `.venv\Scripts\python.exe --version` |
| CPU | AMD64 Family 23 Model 8, 16 logical processors | Replace with full CPU model if available |
| RAM | TO FILL | Include installed RAM, e.g. 32 GB |
| GPU | NVIDIA GeForce RTX 2060 SUPER, 8192 MiB VRAM | From `nvidia-smi` |
| NVIDIA driver | 581.80 | From `nvidia-smi` |


## LLM Backend

| Item | Value used / to be reported | Notes |
| --- | --- | --- |
| Backend | Ollama | Local LLM server |
| Ollama client version | 0.24.0 | `ollama --version`; server was not running during this table check |
| Ollama base URL in code | `http://127.0.0.1:13444` | Used in `main.py`, `learner_mode_run.py`, and evaluation runners |
| Chat model in code | `gpt-oss:20b` | Used by both JUNO and B1 code paths |
| Original benchmark server | Exact CPU/GPU/RAM unavailable; server stopped working around mid-March | Prevents exact hardware-level performance reproduction |
| Minimum practical model size for reruns | Approximately 14B parameters or larger | Below this size, structured-output failures were observed and benchmark behavior is not comparable |


## Python Dependencies

| Package | Version observed locally |
| --- | --- |
| `langchain` | 1.2.18 |
| `langgraph` | 1.1.10 |
| `langchain-ollama` | 0.3.8 |
| `langchain-core` | 1.4.0 |
| `pydantic` | 2.11.7 |
| `pandas` | 2.3.2 |
| `numpy` | 2.3.3 |
| `matplotlib` | 3.10.6 |
| `PyYAML` | 6.0.2 |
| `langsmith` | 0.8.5 |
| `seaborn` | 0.13.2 |

## Core System Parameters

| Parameter | Value | Source / note |
| --- | --- | --- |
| Skill directory | `skills/` | Main registry directory |
| Number of YAML Skill Cards | 14 | Counted from `skills/*.yaml` |
| Main analysis model | `gpt-oss:20b` | `main.py` |
| Agent graph nodes | `agent`, `formatter`, `executor`, `memory_folder` | JUNO Analysis Agent |
| Memory checkpointing | LangGraph `MemorySaver` | In-memory checkpointing |
| Memory folding frequency | Every 5 tool calls | `FOLDING_NUMBER = 5` |
| Learning pipeline max retries | 6 | `MAX_RETRIES = 6` |
| Long-scenario recursion limit | 150 by default | CLI option in `run_long_evaluation_b1.py` |
| Query repetitions | 5 runs per query | Thesis benchmark design and runner default |
| Long-scenario repetitions | 5 runs per scenario | Thesis benchmark design and runner default |

## Benchmark And Data Artifacts

| Artifact | Role | Count / content | SHA256 |
| --- | --- | --- | --- |
| `evaluation/query_benchmark.json` | Short-horizon tiered benchmark | 33 queries | `22846F844F84F986F27A6B0ED51F0C9192BE8AC10DF3FBD1B7361A97C02D25F9` |
| `evaluation/long_scenario_queries.json` | Long-horizon benchmark | 4 scenarios | `634F5F90065B0C0CD99465982404497E7306F978A91301FC069EA4BEE10C6098` |
| `pmt_bec_rmu_map.csv` | PMT/GCU/BEC/RMU hardware mapping used by tools | Hardware map CSV | `D2020E92295EE5B9D960E82FECD687A4983B229945016DC89791FDB0932BE1A9` |
| `test_data/` | Synthetic / extracted benchmark CSV files | Scenario inputs |
| `skills/` | Hand-written Skill Cards | 14 YAML cards |

## Result Artifacts Used By The Draft

| Artifact | Role | Count / content | SHA256 | Reproducibility status |
| --- | --- | --- | --- | --- |
| `evaluation/results/valid_tests/t1_t2_t5_5times.json` | Saved comparison results for tiers 1, 2, and 5 | 25 selected queries, 5 runs each | 
| `evaluation/results/valid_tests/t3_comp_5times.json` | Saved comparison results for tier 3 | 8 selected queries, 5 runs each | 
| `evaluation/results/valid_tests/long_eval_comparison_5times.json` | Saved long-horizon comparison traces | 4 scenarios, 5 runs each | 
| `evaluation/results/valid_tests/long_eval_0_tool_memory.json` | Saved Memory folding comparison traces | 4 scenarios, 5 runs each | 
| `evaluation/results/valid_tests/long_eval_3_tool_memory.json` | Saved Memory folding comparison traces | 4 scenarios, 5 runs each | 
| `evaluation/results/valid_tests/long_eval_6_tool_memory.json` | Saved Memory folding comparison traces | 4 scenarios, 5 runs each | 
| `evaluation/results/valid_tests/long_eval_9_tool_memory.json` | Saved Memory folding comparison traces | 4 scenarios, 5 runs each | 
| `evaluation/results/valid_tests/long_eval_12_tool_memory.json` | Saved Memory folding comparison traces | 4 scenarios, 5 runs each | 

## Commands

| Purpose | Command |
| --- | --- |
| Create environment | `python -m venv .venv` |
| Activate environment | `.venv\Scripts\Activate.ps1` |
| Install dependencies | `pip install -r requirements.txt` |
| Check Ollama model | `ollama list` and `ollama show gpt-oss:20b` |
| Run short-horizon comparison | `python evaluation\run_evaluation_b1.py --mode compare --benchmark evaluation\query_benchmark.json --output evaluation\results\valid_tests --runs 5 --tier1 -1 --tier2 -1 --tier3 -1 --tier5 -1` |
| Run long-horizon comparison | `python evaluation\run_long_evaluation_b1.py --mode compare --benchmark evaluation\long_scenario_queries.json --output evaluation\results\valid_tests --runs 5 --recursion-limit 150` |
| Analyze short-horizon results | `python evaluation\analyze_benchmark_combined.py` |
| Compute latency summary | `python evaluation\calculate_latency.py evaluation\results\valid_tests\long_eval_comparison_5times.json` |
| Run learning pipeline | `python learner_mode_run.py` |


