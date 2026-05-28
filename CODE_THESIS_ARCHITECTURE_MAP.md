# Code to Thesis Architecture Map

This document maps the architecturally important parts of the implementation to the corresponding sections of the thesis PDF. It intentionally does not link to extracted thesis text; the reader is expected to have the thesis PDF open alongside this repository.

## Reading Order

If you are trying to understand the system from code and thesis together, read in this order:

1. Thesis Section 3.1, "System Overview and Architecture"
2. Analysis Agent graph construction: [`main.py`](main.py#L67)
3. Thesis Section 3.2, "The Skill Card Abstraction"
4. Skill Card schema and registry: [`skill_models.py`](skill_models.py#L129), [`skill_registry.py`](skill_registry.py#L6)
5. Thesis Section 3.3, "The Analysis Agent"
6. Runtime nodes: [`nodes.py`](nodes.py#L80)
7. Thesis Section 3.4, "The Learning Pipeline"
8. Learning Pipeline graph: [`learner_mode_run.py`](learner_mode_run.py#L51)
9. Thesis Chapter 4, "Experiments and Results"
10. Evaluation code and artifacts: [`evaluation/`](evaluation)

## Top-Level System

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 3.1, "System Overview and Architecture" | [`main.py`](main.py#L67), [`learner_mode_run.py`](learner_mode_run.py#L51) | Implements the two subsystems described in the thesis: the online Analysis Agent and the offline Learning Pipeline. |
| Section 3.1, Table 3.1, "Main design problems addressed by the architecture" | [`main.py`](main.py#L67), [`nodes.py`](nodes.py#L80), [`skill_registry.py`](skill_registry.py#L6), [`state_and_class.py`](state_and_class.py#L20) | Provides the concrete implementation of the design-problem mapping: Skill Cards describe tools, graph nodes orchestrate execution, and folded memory stores compressed context. |
| Section 3.1, connection between Analysis Agent, Learning Pipeline, and Skill Cards | [`skill_models.py`](skill_models.py#L129), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L280) | Uses the same Skill Card structure for both runtime execution and learner-generated drafts. |

## Analysis Agent Graph

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.1, "Graph Architecture" | [`main.py`](main.py#L67) | `build_graph` constructs the LangGraph state graph. |
| Section 3.3.1, four-node graph description | [`main.py`](main.py#L88) | Registers the `agent`, `formatter`, `executor`, and `memory_folder` nodes. |
| Section 3.3.1, workflow routing | [`main.py`](main.py#L17) | `should_continue` decides whether the state should go to argument formatting, memory folding, or termination. |
| Section 4.1.1, "Models and Infrastructure" | [`main.py`](main.py#L72) | Uses `ChatOllama(model="gpt-oss:20b", base_url="http://127.0.0.1:13444")`. |
| Section 3.3.5, "Memory Management: Folded Memory" | [`main.py`](main.py#L14), [`main.py`](main.py#L47) | `FOLDING_NUMBER` controls how often the graph routes to memory compression. |
| Section 3.3.2, "Agent State" | [`main.py`](main.py#L129) | Uses LangGraph `MemorySaver` for in-memory checkpointing. |

## Agent State and Structured Outputs

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.2, "Agent State" | [`state_and_class.py`](state_and_class.py#L72) | `AgentState` contains the state shared by all graph nodes: messages, plan, memory, findings, errors, selected skill, formatted arguments, and counters. |
| Section 3.3.3, "The Select-then-Validate Workflow" | [`state_and_class.py`](state_and_class.py#L50) | `AgentDecision` is the structured Pydantic output returned by the manager LLM. |
| Section 3.3.3, explicit planning state | [`state_and_class.py`](state_and_class.py#L38) | `PlanStep` represents each structured plan entry used during execution and final-answer construction. |
| Section 3.3.5, "Memory Management: Folded Memory" | [`state_and_class.py`](state_and_class.py#L20), [`state_and_class.py`](state_and_class.py#L59) | `FoldedMemory` stores persistent summaries and findings; `MemoryFoldResult` is the structured output used to update it. |
| Section 3.3.4, "Tool Execution" | [`state_and_class.py`](state_and_class.py#L222) | `normalize_tool_output` turns figures, arrays, and structured objects into serializable state entries. |

## Skill Card Abstraction

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 3.2, "The Skill Card Abstraction" | [`skill_models.py`](skill_models.py#L129), [`skills/`](skills) | Represents each YAML file in `skills/` as a `SkillCard`. |
| Section 3.2.1, "Schema Design" | [`skill_models.py`](skill_models.py#L42), [`skill_models.py`](skill_models.py#L74), [`skill_models.py`](skill_models.py#L89), [`skill_models.py`](skill_models.py#L104), [`skill_models.py`](skill_models.py#L129) | Implements the main Skill Card blocks: summary, execution details, usage policy, output schema, and top-level card metadata. |
| Section 3.2.2, "Validation and Runtime Use" | [`skill_registry.py`](skill_registry.py#L71), [`state_and_class.py`](state_and_class.py#L137) | Builds runtime Pydantic models from Skill Card parameter schemas and stores formatter output in `ValidationAndArgs`. |
| Section 3.2.3, "Registry Loading" | [`skill_registry.py`](skill_registry.py#L6), [`skill_registry.py`](skill_registry.py#L40) | Loads YAML cards, validates card shape, imports target Python functions, and exposes lookup by skill name. |
| Section 3.2.1, Listing 3.1, `get_hardware_connections` example | [`skills/get_hardware_connections.yaml`](skills/get_hardware_connections.yaml), [`juno_tools/hardware_check.py`](juno_tools/hardware_check.py) | Shows the complete path from Skill Card metadata to the executable hardware-lookup function. |
| Appendix B, "Skill Library" | [`skills/`](skills), [`juno_tools/`](juno_tools) | Contains the exposed skill cards and the corresponding Python analysis functions. |

## Select-then-Validate Runtime

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.3, "The Select-then-Validate Workflow" | [`nodes.py`](nodes.py#L80), [`nodes.py`](nodes.py#L182) | `AgentNode` selects a skill from summaries; `ArgumentFormatterNode` loads the full card and validates the selected call. |
| Section 3.3.3, compact Skill Card summaries | [`nodes.py`](nodes.py#L80), [`skill_registry.py`](skill_registry.py#L6) | Supplies short Skill Card summaries to the manager prompt. |
| Section 3.3.3, full-card validation | [`nodes.py`](nodes.py#L182), [`state_and_class.py`](state_and_class.py#L137) | Uses the selected Skill Card's usage policy and parameter schema to produce executable arguments. |
| Section 3.3.4, "Tool Execution" | [`state_and_class.py`](state_and_class.py#L86), [`nodes.py`](nodes.py#L341) | `ToolExecutor` calls the Python function; `tool_node` records outputs, errors, paths, and plan updates. |
| Section 4.2.3, "Latency Analysis" | [`nodes.py`](nodes.py#L80), [`nodes.py`](nodes.py#L182) | The extra LLM call in the formatter node explains much of the agent's latency overhead relative to the baseline. |

## Folded Memory

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.5, "Memory Management: Folded Memory" | [`nodes.py`](nodes.py#L489), [`state_and_class.py`](state_and_class.py#L20) | `MemoryFolderNode` compresses accumulated messages and findings into a `FoldedMemory` object. |
| Section 3.3.5, folding frequency | [`main.py`](main.py#L14), [`main.py`](main.py#L47) | Routes to the memory folder after a configurable number of tool calls. |
| Appendix C, "Memory Compression Ablation" | [`evaluation/results/valid_tests/long_eval_0_tool_memory.json`](evaluation/results/valid_tests/long_eval_0_tool_memory.json), [`evaluation/results/valid_tests/long_eval_3_tool_memory.json`](evaluation/results/valid_tests/long_eval_3_tool_memory.json), [`evaluation/results/valid_tests/long_eval_6_tool_memory.json`](evaluation/results/valid_tests/long_eval_6_tool_memory.json), [`evaluation/results/valid_tests/long_eval_9_tool_memory.json`](evaluation/results/valid_tests/long_eval_9_tool_memory.json), [`evaluation/results/valid_tests/long_eval_12_tool_memory.json`](evaluation/results/valid_tests/long_eval_12_tool_memory.json) | Saved traces used to compare different memory-compression frequencies. |

## Prompts

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Appendix D, "Prompt Design" | [`prompts.py`](prompts.py) | Stores the runtime prompt templates for the graph nodes. |
| Appendix D, AgentNode prompt | [`prompts.py`](prompts.py#L1), [`nodes.py`](nodes.py#L97) | `SYSTEM_PROMPT_T` is the manager prompt currently passed to `AgentNode`. |
| Appendix D, Formatter Node prompt | [`prompts.py`](prompts.py#L197), [`nodes.py`](nodes.py#L192) | `FORMATTER_SYSTEM_PROMPT_2` is used by `ArgumentFormatterNode`. |
| Appendix D, Memory Folder prompt | [`prompts.py`](prompts.py#L251), [`nodes.py`](nodes.py#L489) | `MEMORY_FOLDER_PROMPT_2` controls folded-memory generation. |
| Appendix D, complete prompts in repository | [`prompts.py`](prompts.py) | The full operational prompts live here. |

## Tool Suite and Detector Diagnostics

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 2.1.4, "Operational Diagnostics: Motivation and Scope" | [`juno_tools/`](juno_tools), [`skills/`](skills) | Implements occupancy analysis, component activity checks, hardware mapping, plotting, filtering, and run summaries. |
| Section 4.1.3, "Tool Suite" | [`skills/`](skills) | Each YAML file is one exposed analysis skill. |
| Section 2.1, "The JUNO Detector: Technical Background" | [`pmt_bec_rmu_map.csv`](pmt_bec_rmu_map.csv), [`juno_tools/hardware_check.py`](juno_tools/hardware_check.py) | Hardware lookup and component-activity tools use the PMT/GCU/BEC/RMU mapping CSV. |
| Section 2.1.3, "Run Structure and the Diagnostic Data Format" | [`test_data/`](test_data), [`juno_tools/csv_tooling.py`](juno_tools/csv_tooling.py), [`juno_tools/physics_analysis.py`](juno_tools/physics_analysis.py) | Tools expect per-hit CSV files with event, PMT, charge, and time-like columns. |

## Learning Pipeline

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 3.4, "The Learning Pipeline" | [`learner_mode_run.py`](learner_mode_run.py#L51) | `build_learner_graph` creates the offline Skill Card synthesis graph. |
| Section 3.4.1, "Pipeline State" | [`Learner/learner_states_class.py`](Learner/learner_states_class.py) | Defines static analysis output, neural analysis, simulation data, gate results, validation reports, and iteration state. |
| Section 3.4.2, "Phase 1: Analysis" | [`Learner/static_analyzer.py`](Learner/static_analyzer.py), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L76) | `StaticExtractionNode` extracts signatures, docstrings, imports, and code facts. |
| Section 3.4.2, LLM-based semantic interpretation | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L91), [`Learner/learner_prompts.py`](Learner/learner_prompts.py) | `NeuralAnalysisNode` infers tool purpose, parameter meaning, and likely usage. |
| Section 3.4.3, "Phase 2: Synthesis" | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L117), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L234) | `DrafterNode` drafts the Skill Card; `SimulatorNode` produces test scenarios and mock data. |
| Section 3.4.4, "Phase 3: The Gauntlet Validation Protocol" | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L340), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L456) | `ValidationUnitNode` tests isolated selection, formatting, execution, and semantics; `ValidationIntegrationNode` tests competitive selection. |
| Section 3.4.5, "The Refiner" | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L557), [`learner_mode_run.py`](learner_mode_run.py#L9) | `RefinerNode` builds feedback from validation failures; `MAX_RETRIES` controls the retry loop. |
| Section 4.4, "Learning Pipeline Evaluation" | [`Learner/skills_drafts`](Learner/skills_drafts), [`Learner/simulation_runs`](Learner/simulation_runs) | Drafted cards and simulation logs are intended to support the hand-written versus auto-synthesized card comparison. |

## Evaluation and Results

| Thesis section | Code location | What the code does |
| --- | --- | --- |
| Section 4.1.4, "Benchmark Design" | [`evaluation/query_benchmark.json`](evaluation/query_benchmark.json), [`evaluation/models.py`](evaluation/models.py#L78) | Defines benchmark queries, tiers, expected tools, accepted toolchains, expected arguments, and allowed refusals. |
| Section 4.1.6, "Metrics" | [`evaluation/analyze_benchmark_combined.py`](evaluation/analyze_benchmark_combined.py#L19), [`evaluation/analyze_benchmark_combined.py`](evaluation/analyze_benchmark_combined.py#L56) | Implements Tool Selection Rate and Argument Correctness Rate scoring. |
| Section 4.2, "Tiered Benchmark Results" | [`evaluation/results/valid_tests/t1_t2_t5_5times.json`](evaluation/results/valid_tests/t1_t2_t5_5times.json), [`evaluation/results/valid_tests/t3_comp_5times.json`](evaluation/results/valid_tests/t3_comp_5times.json), [`evaluation/analyze_benchmark_combined.py`](evaluation/analyze_benchmark_combined.py#L115) | Stores benchmark traces and computes aggregate metrics and plots. |
| Section 4.1.2, "Baseline Agent (B1)", and Section 3.3.6, "Comparison to the ReAct Baseline" | [`evaluation/baseline_b1.py`](evaluation/baseline_b1.py#L121) | Builds the ReAct-style baseline over the same Python tools. |
| Section 4.1.5, "Long-Horizon Scenarios" | [`evaluation/long_scenario_queries.json`](evaluation/long_scenario_queries.json), [`evaluation/run_long_evaluation_b1.py`](evaluation/run_long_evaluation_b1.py#L54) | Stores scenario prompts, expected tools, ground-truth facts, and minimum tool-call counts. |
| Section 4.3, "Long-Horizon Scenario Results" | [`evaluation/results/valid_tests/long_eval_comparison_5times.json`](evaluation/results/valid_tests/long_eval_comparison_5times.json), [`evaluation/run_long_evaluation_b1.py`](evaluation/run_long_evaluation_b1.py#L156) | Stores long-horizon traces and formats outputs with ground-truth facts for scoring. |

