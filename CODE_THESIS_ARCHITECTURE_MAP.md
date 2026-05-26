# Code to Thesis Architecture Map

This document maps the architecturally important parts of the implementation to the corresponding explanations in the thesis draft. Thesis links point to [`thesis_draft.txt`](thesis_draft.txt), the text extracted from `main (19).pdf`, because the PDF itself does not expose stable section anchors.

## Reading Order

If you are trying to understand the system from code and thesis together, read in this order:

1. Thesis Section 3.1, System Overview and Architecture: [`thesis_draft.txt` lines 1159-1229](thesis_draft.txt#L1159)
2. Analysis Agent graph construction: [`main.py`](main.py#L67)
3. Skill Card schema and registry: [`skill_models.py`](skill_models.py#L129), [`skill_registry.py`](skill_registry.py#L6)
4. Runtime nodes: [`nodes.py`](nodes.py#L80)
5. Learning Pipeline: thesis Section 3.4 and [`learner_mode_run.py`](learner_mode_run.py#L51)
6. Evaluation: thesis Chapter 4 and [`evaluation/`](evaluation)

## Top-Level System

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 3.1 introduces two subsystems: the Analysis Agent and the Learning Pipeline ([`thesis_draft.txt` lines 1159-1172](thesis_draft.txt#L1159)). | [`main.py`](main.py#L67), [`learner_mode_run.py`](learner_mode_run.py#L51) | `main.py` builds the online diagnostic agent. `learner_mode_run.py` builds the offline graph that drafts and validates new Skill Cards. |
| Table 3.1 maps design problems to architectural components ([`thesis_draft.txt` lines 1200-1229](thesis_draft.txt#L1200)). | [`main.py`](main.py#L67), [`nodes.py`](nodes.py#L80), [`skill_registry.py`](skill_registry.py#L6), [`state_and_class.py`](state_and_class.py#L20) | The implementation matches this division: Skill Cards describe tools, the graph runs selection/validation/execution, and folded memory stores compressed task context. |
| The thesis states that both modes are connected by the Skill Card object ([`thesis_draft.txt` lines 1166-1172](thesis_draft.txt#L1166)). | [`skill_models.py`](skill_models.py#L129), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L280) | The same YAML Skill Card shape is used by the runtime registry and by the learner's draft writer. |

## Analysis Agent Graph

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.1 describes the four-node graph architecture ([`thesis_draft.txt` lines 1461-1491](thesis_draft.txt#L1461)). | [`main.py`](main.py#L67) | `build_graph` constructs the LangGraph state graph. |
| The thesis names the AgentNode, FormatterNode, ToolNode, and MemoryFolderNode ([`thesis_draft.txt` lines 1466-1491](thesis_draft.txt#L1466)). | [`main.py`](main.py#L88) | The graph registers `agent`, `formatter`, `executor`, and `memory_folder` nodes. |
| The workflow alternates between planning, validation, execution, memory folding, and final answer generation ([`thesis_draft.txt` lines 1491-1498](thesis_draft.txt#L1491)). | [`main.py`](main.py#L17) | `should_continue` routes the state after each agent step. It decides whether to validate a selected skill, fold memory, or end. |
| Section 4.1.1 describes the local model setup ([`thesis_draft.txt` lines 1970-1985](thesis_draft.txt#L1970)). | [`main.py`](main.py#L72) | The runtime uses `ChatOllama(model="gpt-oss:20b", base_url="http://127.0.0.1:13444")`. |
| Section 3.3.5 describes periodic memory folding ([`thesis_draft.txt` lines 1623-1660](thesis_draft.txt#L1623)). | [`main.py`](main.py#L14), [`main.py`](main.py#L47) | `FOLDING_NUMBER` controls how often the graph routes to memory compression. |
| The thesis describes persistent graph memory/checkpointing as part of the runtime context. | [`main.py`](main.py#L129) | `MemorySaver` is used as the LangGraph checkpointer. |

## Agent State and Structured Outputs

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.2 describes the shared agent state ([`thesis_draft.txt` lines 1501-1519](thesis_draft.txt#L1501)). | [`state_and_class.py`](state_and_class.py#L72) | `AgentState` defines the state passed between nodes: messages, plan, folded memory, findings, errors, selected skill, formatted arguments, and counters. |
| Section 3.3.3 explains the structured decision produced by the AgentNode ([`thesis_draft.txt` lines 1525-1564](thesis_draft.txt#L1525)). | [`state_and_class.py`](state_and_class.py#L50) | `AgentDecision` is the Pydantic object returned by the manager LLM. It contains the decision type, plan, optional selected skill, final answer, and thought field. |
| The thesis discusses explicit plan state for multi-step investigations ([`thesis_draft.txt` lines 1549-1558](thesis_draft.txt#L1549)). | [`state_and_class.py`](state_and_class.py#L38) | `PlanStep` is the structured plan entry used by the agent and final answer formatting. |
| Section 3.3.5 describes Folded Memory as a summary plus confirmed findings ([`thesis_draft.txt` lines 1623-1648](thesis_draft.txt#L1623)). | [`state_and_class.py`](state_and_class.py#L20), [`state_and_class.py`](state_and_class.py#L59) | `FoldedMemory` stores the persistent summary and findings. `MemoryFoldResult` is the LLM output used to update it. |
| Section 3.3.4 discusses normalized tool results ([`thesis_draft.txt` lines 1601-1615](thesis_draft.txt#L1601)). | [`state_and_class.py`](state_and_class.py#L222) | `normalize_tool_output` converts figures, arrays, and structured objects into serializable outputs that can be stored in agent state. |

## Skill Card Abstraction

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 3.2 introduces Skill Cards as structured descriptors for Python tools ([`thesis_draft.txt` lines 1247-1261](thesis_draft.txt#L1247)). | [`skill_models.py`](skill_models.py#L129), [`skills/`](skills) | Each YAML file in `skills/` is parsed into a `SkillCard` object. |
| Section 3.2.1 describes the five main blocks of a Skill Card ([`thesis_draft.txt` lines 1261-1282](thesis_draft.txt#L1261)). | [`skill_models.py`](skill_models.py#L42), [`skill_models.py`](skill_models.py#L74), [`skill_models.py`](skill_models.py#L89), [`skill_models.py`](skill_models.py#L104), [`skill_models.py`](skill_models.py#L129) | The five blocks are represented by `SkillSummary`, `SkillExecution`, `UsagePolicy`, `OutputSchema`, and `SkillCard`. |
| Section 3.2.2 explains Pydantic validation and runtime argument use ([`thesis_draft.txt` lines 1308-1319](thesis_draft.txt#L1308)). | [`skill_registry.py`](skill_registry.py#L71), [`state_and_class.py`](state_and_class.py#L137) | `create_pydantic_model_from_skill` turns the Skill Card parameter schema into a runtime model. `ValidationAndArgs` is the formatter's validated output. |
| Section 3.2.3 explains registry loading ([`thesis_draft.txt` lines 1324-1331](thesis_draft.txt#L1324)). | [`skill_registry.py`](skill_registry.py#L6), [`skill_registry.py`](skill_registry.py#L40) | `SkillRegistry` loads YAML files, validates the card shape, dynamically imports the Python function, and exposes lookup by skill name. |
| Listing 3.1 shows `get_hardware_connections` as the example Skill Card ([`thesis_draft.txt` lines 1299-1381](thesis_draft.txt#L1299)). | [`skills/get_hardware_connections.yaml`](skills/get_hardware_connections.yaml), [`juno_tools/hardware_check.py`](juno_tools/hardware_check.py) | The YAML card names the function and usage policy; the Python file contains the executable implementation. |
| Appendix B lists the current skill library ([`thesis_draft.txt` lines 2751-2755](thesis_draft.txt#L2751)). | [`skills/`](skills), [`juno_tools/`](juno_tools) | The active tool suite is the union of YAML cards in `skills/` and Python functions in `juno_tools/`. |

## Select-then-Validate Runtime

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.3 defines Select-then-Validate ([`thesis_draft.txt` lines 1525-1533](thesis_draft.txt#L1525)). | [`nodes.py`](nodes.py#L80), [`nodes.py`](nodes.py#L182) | `AgentNode` selects a skill from summaries. `ArgumentFormatterNode` loads the full card and validates whether the selection and arguments are appropriate. |
| The thesis says the selector receives compact Skill Card summaries ([`thesis_draft.txt` lines 1542-1549](thesis_draft.txt#L1542)). | [`nodes.py`](nodes.py#L80), [`skill_registry.py`](skill_registry.py#L6) | `AgentNode` builds its prompt from compact summaries supplied by the registry. |
| The thesis says the validator uses the full Skill Card, including usage policy and parameter schema ([`thesis_draft.txt` lines 1581-1593](thesis_draft.txt#L1581)). | [`nodes.py`](nodes.py#L182), [`state_and_class.py`](state_and_class.py#L137) | `ArgumentFormatterNode` checks the selected skill against the full card and produces `ValidationAndArgs`. |
| Section 3.3.4 describes execution after formatting ([`thesis_draft.txt` lines 1601-1615](thesis_draft.txt#L1601)). | [`state_and_class.py`](state_and_class.py#L86), [`nodes.py`](nodes.py#L341) | `ToolExecutor` calls the selected Python function. `tool_node` stores the output, errors, paths, and plan updates in the state. |
| Section 4.2.3 attributes latency to the extra validation call and card loading ([`thesis_draft.txt` lines 2292-2313](thesis_draft.txt#L2292)). | [`nodes.py`](nodes.py#L80), [`nodes.py`](nodes.py#L182) | The latency overhead comes from two LLM-backed nodes: selector and formatter. |

## Folded Memory

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 3.3.5 motivates Folded Memory for long diagnostic chains ([`thesis_draft.txt` lines 1623-1648](thesis_draft.txt#L1623)). | [`nodes.py`](nodes.py#L489), [`state_and_class.py`](state_and_class.py#L20) | `MemoryFolderNode` compresses accumulated messages and findings into the `FoldedMemory` object. |
| The thesis says folding is triggered periodically ([`thesis_draft.txt` lines 1660-1664](thesis_draft.txt#L1660)). | [`main.py`](main.py#L14), [`main.py`](main.py#L47) | The current implementation folds every `FOLDING_NUMBER` tool calls. |
| Appendix C evaluates memory compression frequency ([`thesis_draft.txt` lines 2828-2874](thesis_draft.txt#L2828)). | [`evaluation/results/valid_tests/long_eval_0_tool_memory.json`](evaluation/results/valid_tests/long_eval_0_tool_memory.json), [`evaluation/results/valid_tests/long_eval_3_tool_memory.json`](evaluation/results/valid_tests/long_eval_3_tool_memory.json), [`evaluation/results/valid_tests/long_eval_6_tool_memory.json`](evaluation/results/valid_tests/long_eval_6_tool_memory.json), [`evaluation/results/valid_tests/long_eval_9_tool_memory.json`](evaluation/results/valid_tests/long_eval_9_tool_memory.json), [`evaluation/results/valid_tests/long_eval_12_tool_memory.json`](evaluation/results/valid_tests/long_eval_12_tool_memory.json) | These result files are the saved runs used for the memory-frequency ablation. |

## Prompts

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Appendix D explains prompt design principles ([`thesis_draft.txt` lines 2916-3035](thesis_draft.txt#L2916)). | [`prompts.py`](prompts.py) | The runtime prompt templates are stored in one file. |
| The thesis describes the AgentNode prompt as manager/planner ([`thesis_draft.txt` lines 2933-2947](thesis_draft.txt#L2933)). | [`prompts.py`](prompts.py#L1), [`nodes.py`](nodes.py#L97) | `SYSTEM_PROMPT_T` is the prompt currently passed to `AgentNode`. |
| The thesis describes the Formatter Node prompt as a specialist validator ([`thesis_draft.txt` lines 2933-2935](thesis_draft.txt#L2933)). | [`prompts.py`](prompts.py#L197), [`nodes.py`](nodes.py#L192) | `FORMATTER_SYSTEM_PROMPT_2` is used by `ArgumentFormatterNode`. |
| The thesis describes the Memory Folder prompt as a summarizer ([`thesis_draft.txt` lines 2933-2935](thesis_draft.txt#L2933)). | [`prompts.py`](prompts.py#L251), [`nodes.py`](nodes.py#L489) | `MEMORY_FOLDER_PROMPT_2` controls how folded memory is generated. |
| Appendix D says full prompts are in the repository ([`thesis_draft.txt` lines 2957-3035](thesis_draft.txt#L2957)). | [`prompts.py`](prompts.py) | This file is the canonical place to inspect the full operational prompts. |

## Tool Suite and Detector Diagnostics

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 2.1.4 defines the operational diagnostic scope ([`thesis_draft.txt` lines 593-616](thesis_draft.txt#L593)). | [`juno_tools/`](juno_tools), [`skills/`](skills) | The diagnostic functions implement occupancy, component activity, hardware mapping, plotting, filtering, and run summaries. |
| Section 4.1.3 describes the tool suite used in experiments ([`thesis_draft.txt` lines 2014-2020](thesis_draft.txt#L2014)). | [`skills/`](skills) | Each YAML file is one exposed analysis skill. |
| The thesis uses JUNO hardware nomenclature from Section 2.1 ([`thesis_draft.txt` lines 413-524](thesis_draft.txt#L413)). | [`pmt_bec_rmu_map.csv`](pmt_bec_rmu_map.csv), [`juno_tools/hardware_check.py`](juno_tools/hardware_check.py) | Hardware lookup and component-activity tools use the PMT/BEC/RMU mapping CSV. |
| The thesis describes CSV filling-test data format in Section 2.1.3 ([`thesis_draft.txt` lines 524-559](thesis_draft.txt#L524)). | [`test_data/`](test_data), [`juno_tools/csv_tooling.py`](juno_tools/csv_tooling.py), [`juno_tools/physics_analysis.py`](juno_tools/physics_analysis.py) | The tools expect per-hit CSV files with event, PMT, charge, and time-like columns. |

## Learning Pipeline

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 3.4 introduces the offline Learning Pipeline ([`thesis_draft.txt` lines 1698-1712](thesis_draft.txt#L1698)). | [`learner_mode_run.py`](learner_mode_run.py#L51) | `build_learner_graph` creates the pipeline graph. |
| Section 3.4.1 describes the shared learner state ([`thesis_draft.txt` lines 1739-1749](thesis_draft.txt#L1739)). | [`Learner/learner_states_class.py`](Learner/learner_states_class.py) | The learner state classes define static analysis output, neural analysis, simulation data, gate results, validation reports, and iteration state. |
| Section 3.4.2 describes static extraction from Python source ([`thesis_draft.txt` lines 1761-1796](thesis_draft.txt#L1761)). | [`Learner/static_analyzer.py`](Learner/static_analyzer.py), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L76) | `StaticExtractionNode` calls the static analyzer to extract signatures, docstrings, imports, and code facts. |
| Section 3.4.2 also describes LLM-based interpretation of code ([`thesis_draft.txt` lines 1793-1796](thesis_draft.txt#L1793)). | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L91), [`Learner/learner_prompts.py`](Learner/learner_prompts.py) | `NeuralAnalysisNode` asks the model to infer semantic tool purpose and parameter meaning. |
| Section 3.4.3 describes Skill Card drafting and test-plan synthesis ([`thesis_draft.txt` lines 1817-1855](thesis_draft.txt#L1817)). | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L117), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L234) | `DrafterNode` drafts the Skill Card. `SimulatorNode` produces test scenarios and mock data. |
| Section 3.4.4 describes the Gauntlet validation protocol ([`thesis_draft.txt` lines 1859-1924](thesis_draft.txt#L1859)). | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L340), [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L456) | `ValidationUnitNode` tests isolated selection, formatting, execution, and semantics. `ValidationIntegrationNode` tests selection when other skills are present. |
| Section 3.4.5 describes refinement after validation failure ([`thesis_draft.txt` lines 1927-1944](thesis_draft.txt#L1927)). | [`Learner/learner_nodes.py`](Learner/learner_nodes.py#L557), [`learner_mode_run.py`](learner_mode_run.py#L9) | `RefinerNode` builds feedback from validation failures. `MAX_RETRIES` controls the retry loop. |
| Section 4.4 evaluates hand-written versus auto-synthesized Skill Cards ([`thesis_draft.txt` lines 2426-2486](thesis_draft.txt#L2426)). | [`Learner/skills_drafts`](Learner/skills_drafts), [`Learner/simulation_runs`](Learner/simulation_runs) | Drafted cards and simulation data are intended to be stored here. Use these folders to trace which generated cards correspond to the reported evaluation. |

## Evaluation and Results

| Thesis explanation | Code location | What the code does |
| --- | --- | --- |
| Section 4.1.4 describes the tiered benchmark design ([`thesis_draft.txt` lines 2023-2050](thesis_draft.txt#L2023)). | [`evaluation/query_benchmark.json`](evaluation/query_benchmark.json), [`evaluation/models.py`](evaluation/models.py#L78) | The JSON file defines benchmark queries, tiers, expected tools, accepted toolchains, expected arguments, and allowed refusals. |
| Section 4.1.6 defines TSR and ACR ([`thesis_draft.txt` lines 2068-2145](thesis_draft.txt#L2068)). | [`evaluation/analyze_benchmark_combined.py`](evaluation/analyze_benchmark_combined.py#L19), [`evaluation/analyze_benchmark_combined.py`](evaluation/analyze_benchmark_combined.py#L56) | `compute_tsr` and `compute_acr` implement the actual scoring. |
| Section 4.2 reports tiered benchmark results ([`thesis_draft.txt` lines 2174-2313](thesis_draft.txt#L2174)). | [`evaluation/results/valid_tests/t1_t2_t5_5times.json`](evaluation/results/valid_tests/t1_t2_t5_5times.json), [`evaluation/results/valid_tests/t3_comp_5times.json`](evaluation/results/valid_tests/t3_comp_5times.json), [`evaluation/analyze_benchmark_combined.py`](evaluation/analyze_benchmark_combined.py#L115) | The saved JSON files contain benchmark traces; the analysis script produces aggregate metrics and plots. |
| Section 4.1.2 and 3.3.6 define the B1 ReAct baseline ([`thesis_draft.txt` lines 1992-2002](thesis_draft.txt#L1992), [`thesis_draft.txt` lines 1673-1693](thesis_draft.txt#L1673)). | [`evaluation/baseline_b1.py`](evaluation/baseline_b1.py#L121) | The baseline wraps the same Python tools as LangChain `Tool` objects without Pydantic argument schemas. |
| Section 4.1.5 defines long-horizon scenarios ([`thesis_draft.txt` lines 2059-2068](thesis_draft.txt#L2059)). | [`evaluation/long_scenario_queries.json`](evaluation/long_scenario_queries.json), [`evaluation/run_long_evaluation_b1.py`](evaluation/run_long_evaluation_b1.py#L54) | The long-scenario JSON stores scenario prompts, expected tools, ground-truth facts, and minimum tool-call counts. |
| Section 4.3 reports long-horizon completion ([`thesis_draft.txt` lines 2320-2409](thesis_draft.txt#L2320)). | [`evaluation/results/valid_tests/long_eval_comparison_5times.json`](evaluation/results/valid_tests/long_eval_comparison_5times.json), [`evaluation/run_long_evaluation_b1.py`](evaluation/run_long_evaluation_b1.py#L156) | The saved comparison file stores the agent and baseline traces; the runner formats outputs and keeps ground-truth facts for scoring. |

## Known Traceability Notes

These are places where the thesis and code should be checked together before final submission.

| Topic | Code evidence | Thesis location | Note |
| --- | --- | --- | --- |
| Baseline description | [`evaluation/baseline_b1.py`](evaluation/baseline_b1.py#L121) | Section 3.3.6 and 4.1.2 | The baseline uses Skill Card summaries as prose descriptions, not raw Python docstrings. |
| Prompt-driven large-file behavior | [`prompts.py`](prompts.py#L151), [`nodes.py`](nodes.py#L97) | Appendix D and Section 3.3.3 | `SYSTEM_PROMPT_2` contains a large-file protocol, but `AgentNode` currently uses `SYSTEM_PROMPT_T`. |
| Validation strength | [`skill_models.py`](skill_models.py#L6), [`skill_registry.py`](skill_registry.py#L71) | Section 3.2.2 | Runtime validation is mainly basic Pydantic type validation. Enum and richer schema constraints are not fully enforced. |
| Fold frequency | [`main.py`](main.py#L14) | Section 3.3.5 and Appendix C | The code currently sets `FOLDING_NUMBER = 5`, while Appendix C discusses an optimum at another frequency. |
| Long-horizon scoring | [`evaluation/run_long_evaluation_b1.py`](evaluation/run_long_evaluation_b1.py#L156), [`evaluation/results/valid_tests/long_eval_comparison_5times.json`](evaluation/results/valid_tests/long_eval_comparison_5times.json) | Section 4.3 | The runner stores traces and ground truth, but completion labels should be made explicit in the saved artifacts if the table reports completion counts. |
| Benchmark/result synchronization | [`evaluation/query_benchmark.json`](evaluation/query_benchmark.json), [`evaluation/results/valid_tests/t1_t2_t5_5times.json`](evaluation/results/valid_tests/t1_t2_t5_5times.json) | Section 4.1.4 and 4.2 | The benchmark JSON and saved result IDs should be frozen together for reproducibility. |

