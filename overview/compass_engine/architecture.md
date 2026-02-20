# COMPASS Architecture Overview

## Clinical Ontology-driven Multi-modal Predictive Agentic Support System

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          COMPASS v1.0.0                                              │
│              Clinical Ontology-driven Multi-modal Predictive Agentic Support System                    │
│                                                                                                      │
│                          Binary Classification: CASE vs CONTROL                                      │
│                  Target: Neuropsychiatric / Neurologic Disorder Prediction                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘


                    ┌─────────────────────────────────────────────────────────────┐
                    │                    INPUT DATA LAYER                          │
                    │   (Always passed: data_overview + deviation_map + text)     │
                    └─────────────────────────────────────────────────────────────┘
                                               │
          ┌────────────────────────────────────┼────────────────────────────────────┐
          │                                    │                                    │
          ▼                                    ▼                                    ▼
┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐
│   data_overview.json │         │hierarchical_deviation│         │  non_numerical_data  │
│                      │         │     _map.json        │         │        .txt          │
│  • Domain coverage   │         │                      │         │                      │
│  • Token estimates   │         │  • Z-scores          │         │  • Medical history   │
│  • Available data    │         │  • Severity levels   │         │  • Clinical notes    │
│                      │         │  • Tree structure    │         │  • Demographics      │
└──────────────────────┘         └──────────────────────┘         └──────────────────────┘
          │                                    │                                    │
          └────────────────────────────────────┼────────────────────────────────────┘
                                               │
                                               ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      DATA LOADER                             │
                    │                  core/data_loader.py                         │
                    │              ParticipantData Container                       │
                    └─────────────────────────────────────────────────────────────┘
                                               │
                                               │ ParticipantData
                                               ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                     ACTOR-CRITIC LOOP                                               ┃
┃                                    (max 3 iterations)                                               ┃
┃                                                                                                    ┃
┃  ┌────────────────────────────────────────────────────────────────────────────────────────────┐   ┃
┃  │                                    ★ ACTOR PIPELINE ★                                      │   ┃
┃  │                                                                                            │   ┃
┃  │   ┌────────────────────────────────────────────────────────────────────────────────────┐   │   ┃
┃  │   │                         ORCHESTRATOR                                       │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  INPUT:  data_overview.json + critic_feedback (if iteration > 1)                   │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  ACTIONS:                                                                          │   │   ┃
┃  │   │    • Analyze available data domains                                                │   │   ┃
┃  │   │    • Plan optimal tool selection                                                   │   │   ┃
┃  │   │    • Determine execution order with dependencies                                   │   │   ┃
┃  │   │    • Respect token budget constraints                                              │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  OUTPUT: ExecutionPlan with stepwise tool calls                                    │   │   ┃
┃  │   └────────────────────────────────────────────────────────────────────────────────────┘  │   ┃
┃  │                                           │                                               │   ┃
┃  │                              ExecutionPlan│                                              │   ┃
┃  │                                           ▼                                               │   ┃
┃  │   ┌───────────────────────────────────────────────────────────────────────────────────┐   │   ┃
┃  │   │                            EXECUTOR                                               │   │   ┃
┃  │   │                                                                                   │   │   ┃
┃  │   │  INPUT: ExecutionPlan + ParticipantData                                           │   │   ┃
┃  │   │                                                                                   │   │   ┃
┃  │   │  ACTIONS:                                                                         │   │   ┃
┃  │   │    • Execute plan steps in dependency order                                       │   │   ┃
┃  │   │    • Call tools with deviation_map + non_numerical_data (ALWAYS)                  │   │   ┃
┃  │   │    • Handle failures with AutoRepair                                              │   │   ┃
┃  │   │    • Collect and manage tool outputs                                              │   │   ┃
┃  │   │                                                                                   │   │   ┃
┃  │   │  TOOLS             :                                                              │   │   ┃
┃  │   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐          │   │   ┃
┃  │   │  │Unimodal      ││Multimodal    ││Hypothesis    ││Code          │                 │   │   ┃
┃  │   │  │Compressor    ││Narrative     ││Generator     ││Executor      │                 │   │   ┃
┃  │   │  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘          │   │   ┃
┃  │   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                            │   │   ┃
┃  │   │  │Feature       ││Clinical      ││Anomaly       │                            │   │   ┃
┃  │   │  │Synthesizer   ││Ranker        ││Narrative     │                            │   │   ┃
┃  │   │  └───────────────┘ └───────────────┘ └───────────────┘                            │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  OUTPUT: Collection of ToolOutputs                                                │   │   ┃
┃  │   └────────────────────────────────────────────────────────────────────────────────────┘   │   ┃
┃  │                                           │                                               │   ┃
┃  │                               ToolOutputs │                                               │   ┃
┃  │                                           ▼                                               │   ┃
┃  │   ┌────────────────────────────────────────────────────────────────────────────────────┐   │   ┃
┃  │   │                         FUSION LAYER                                 │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  INPUT: All tool outputs + deviation_map + non_numerical_data                     │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  ACTIONS:                                                                         │   │   ┃
┃  │   │    • Integrate multi-tool findings                                                │   │   ┃
┃  │   │    • Identify convergent evidence patterns                                        │   │   ┃
┃  │   │    • Compress for predictor budget                                                │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  OUTPUT: FusedNarrative with key findings                                         │   │   ┃
┃  │   └────────────────────────────────────────────────────────────────────────────────────┘   │   ┃
┃  │                                           │                                               │   ┃
┃  │                           FusedNarrative  │                                               │   ┃
┃  │                                           ▼                                               │   ┃
┃  │   ┌────────────────────────────────────────────────────────────────────────────────────┐   │   ┃
┃  │   │                         PREDICTOR                                          │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  INPUT: FusedNarrative + deviation_map + non_numerical_data (ALWAYS)              │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  ACTIONS:                                                                         │   │   ┃
┃  │   │    • Evaluate all evidence domains                                                │   │   ┃
┃  │   │    • Apply clinical reasoning chain                                               │   │   ┃
┃  │   │    • Generate calibrated probability                                              │   │   ┃
┃  │   │                                                                                    │   │   ┃
┃  │   │  OUTPUT:                                                                          │   │   ┃
┃  │   │    • Binary: CASE or CONTROL                                                      │   │   ┃
┃  │   │    • Probability: 0.0 - 1.0                                                       │   │   ┃
┃  │   │    • Key findings + reasoning chain                                               │   │   ┃
┃  │   └────────────────────────────────────────────────────────────────────────────────────┘   │   ┃
┃  │                                                                                             │   ┃
┃  └─────────────────────────────────────────────────────────────────────────────────────────────┘   ┃
┃                                               │                                                    ┃
┃                               PredictionResult│                                                    ┃
┃                                               ▼                                                    ┃
┃  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐   ┃
┃  │                                    ★ CRITIC ★                                     │   ┃
┃  │                                                                                             │   ┃
┃  │  INPUT: PredictionResult + data_overview + fused_output                                    │   ┃
┃  │                                                                                             │   ┃
┃  │  EVALUATION CHECKLIST:                                                                     │   ┃
┃  │    ☑ Binary classification present (CASE/CONTROL)                                         │   ┃
┃  │    ☑ Probability score in valid range (0.0-1.0)                                           │   ┃
┃  │    ☑ All available data domains utilized                                                   │   ┃
┃  │    ☑ Reasoning chain is coherent                                                          │   ┃
┃  │    ☑ Evidence supports conclusion                                                          │   ┃
┃  │    ☑ Confidence appropriately calibrated                                                   │   ┃
┃  │                                                                                             │   ┃
┃  │  VERDICT:                                                                                  │   ┃
┃  │    ├─► SATISFACTORY ───────────────► Final Output                                         │   ┃
┃  │    │                                                                                        │   ┃
┃  │    └─► UNSATISFACTORY ─────────────► Feedback Loop ──────────┐                            │   ┃
┃  │                                        (with suggestions)     │                            │   ┃
┃  │                                                               │                            │   ┃
┃  └─────────────────────────────────────────────────────────────────────────────────────────────┘   ┃
┃                                                               │                                    ┃
┃                                    ┌──────────────────────────┘                                    ┃
┃                                    │                                                               ┃
┃                                    │ CriticFeedback                                                ┃
┃                                    │ (re-orchestrate with improvements)                            ┃
┃                                    │                                                               ┃
┃                                    └───────────────► Back to ORCHESTRATOR                          ┃
┃                                                                                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                               │
                                               │ Final PredictionResult
                                               ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    OUTPUT LAYER                              │
                    │                                                              │
                    │   ┌─────────────────┐  ┌─────────────────┐                  │
                    │   │ Patient Report  │  │ Decision Trace  │                  │
                    │   │   .json + .md   │  │     .json       │                  │
                    │   └─────────────────┘  └─────────────────┘                  │
                    │   ┌─────────────────┐  ┌─────────────────┐                  │
                    │   │ Execution Log   │  │ Token Usage     │                  │
                    │   │     .json       │  │   Summary       │                  │
                    │   └─────────────────┘  └─────────────────┘                  │
                    │                                                              │
                    └─────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
INPUT FILES                          PROCESSING STAGES                    OUTPUT
═══════════                          ════════════════                    ═══════

data_overview.json ──────────────────► Orchestrator (plan)
                                            │
hierarchical_deviation_map.json ─┬───► Executor + Tools                   
                                 │          │
non_numerical_data.txt ──────────┼───► Fusion Layer ─────────► report.json
                                 │          │                   report.md
multimodal_data.json ────────────┘     Predictor                decision_trace.json
                                            │
                                       Critic ───┐
                                            │    │
                                       [loop if unsatisfactory]
```

## Model Assignments

| Component | Model | Purpose |
|-----------|-------|---------|
| **ACTOR** | | |
| Orchestrator |  | Strategic planning, dependency resolution |
| Executor | - | Coordination and tool management |
| Tools (×7) |  | Specialized data processing |
| Fusion |  | Integration and compression |
| Predictor |  | Clinical reasoning, binary classification |
| **CRITIC** | | |
| Critic |  | Quality evaluation, feedback generation |

## Key Design Principles

1. **Actor-Critic Architecture**: Clear separation between action (prediction) and evaluation (criticism)
2. **Complete Data Coverage**: Orchestrator MUST plan to use ALL available domains
3. **Hierarchical Data Preserved**: Tree structure maintained when passing deviation data
4. **Critical Data Always Passed**: deviation_map + non_numerical_data → every component
5. **Iterative Refinement**: Up to 3 loops until Critic is satisfied
6. **Token Efficiency**: Compression at each stage to maximize information per token
7. **Self-Healing**: AutoRepair handles tool failures gracefully
8. **Full Transparency**: Detailed logging and decision traces
