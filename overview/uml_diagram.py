"""
COMPASS UML Diagram Generator

Generates Mermaid UML diagrams for the COMPASS system.
Exports to PNG using mermaid-cli if available.
"""

import subprocess
import tempfile
from pathlib import Path


MERMAID_DIAGRAM = '''
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1a365d', 'primaryTextColor': '#fff', 'primaryBorderColor': '#2c5282', 'lineColor': '#4299e1', 'secondaryColor': '#2d3748', 'tertiaryColor': '#edf2f7'}}}%%

flowchart TB
    subgraph INPUT["📥 INPUT DATA LAYER"]
        DO[data_overview.json<br/>Domain Coverage]
        HD[hierarchical_deviation_map.json<br/>Z-scores & Hierarchy]
        NN[non_numerical_data.txt<br/>Clinical Notes]
        MM[multimodal_data.json<br/>Raw Features]
    end

    subgraph DL["📂 DATA LOADER"]
        LOAD[DataLoader<br/>ParticipantData Container]
    end

    subgraph ACTOR["🎭 ACTOR PIPELINE"]
        direction TB
        
        subgraph ORCH["ORCHESTRATOR (GPT-5)"]
            O1[Analyze Data Coverage]
            O2[Plan Tool Selection]
            O3[Determine Dependencies]
            O4[Generate ExecutionPlan]
        end
        
        subgraph EXEC["EXECUTOR"]
            E1[Execute Plan Steps]
            E2[Manage Dependencies]
            E3[Handle AutoRepair]
            
            subgraph TOOLS["TOOLS (GPT-5-nano)"]
                T1[PhenotypeRepresentation]
                T2[FeatureSynthesizer]
                T3[AnomalyNarrative]
                T4[UnimodalCompressor]
                T5[ClinicalRanker]
                T6[MultimodalNarrative]
                T7[HypothesisGenerator]
                T8[DifferentialDiagnosis]
                T9[CodeExecutor]
            end
        end
        
        subgraph FUSION["INTEGRATOR AGENT (GPT-5)"]
            F1[Max Input Token Check]
            F2{"Is Input > 90% Limit?"}
            
            F3_PASS["Pass-Through Mode<br/>(Fit Raw Data)"]
            F3_RAG["Smart Fusion (RAG)<br/>(Fill Context to 90%)"]

            F1 --> F2
            F2 -->|No - Fits Budget| F3_PASS
            F2 -->|Yes - Overflow| F3_RAG
        end
        
        subgraph PRED["PREDICTOR (GPT-5)"]
            P1[Clinical Reasoning]
            P2[Phenotypic Prediction]
        end
    end

    subgraph CRITIC_BOX["🎯 CRITIC (GPT-5)"]
        C1[Evaluate Completeness]
        C2[Compute Multi-Composite Score]
        VERDICT{VERDICT}
    end

    subgraph OUTPUT["📤 OUTPUT LAYER"]
        REP[Patient Report<br/>JSON + Markdown]
        DT[Decision Trace]
        LOG[Execution Log]
    end

    DO --> LOAD
    HD --> LOAD
    NN --> LOAD
    MM --> LOAD
    
    LOAD -->|ParticipantData| ORCH
    
    O1 --> O2 --> O3 --> O4
    O4 -->|ExecutionPlan| EXEC
    
    E1 --> E2 --> E3
    E3 --> TOOLS
    T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8 & T9 -->|ToolOutputs| FUSION
    
    F3_PASS -->|FusedNarrative + Context| PRED
    F3_RAG -->|FusedNarrative + Context| PRED
    
    P1 --> P2
    P2 -->|PredictionResult| CRITIC_BOX
    
    C1 --> C2 --> VERDICT
    
    VERDICT -->|SATISFACTORY| OUTPUT
    VERDICT -->|UNSATISFACTORY<br/>+ Feedback| ORCH
    
    HD -.->|always passed| TOOLS
    NN -.->|always passed| TOOLS
    HD -.->|always passed| PRED
    NN -.->|always passed| PRED
'''


def generate_mermaid_file(diagram: str, output_path: Path) -> Path:
    """Write Mermaid diagram to file."""
    mmd_path = output_path.with_suffix('.mmd')
    with open(mmd_path, 'w') as f:
        f.write(diagram)
    print(f"[UML] Created Mermaid file: {mmd_path}")
    return mmd_path


def convert_to_png(mmd_path: Path) -> Path:
    """Convert Mermaid to PNG using mmdc if available."""
    png_path = mmd_path.with_suffix('.png')

    try:
        # Check for mmdc
        subprocess.run(['mmdc', '--version'], capture_output=True, check=True)

        result = subprocess.run(
            ['mmdc', '-i', str(mmd_path), '-o', str(png_path), '-b', 'white', '-w', '2000'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"[UML] Generated PNG: {png_path}")
            return png_path
        else:
            print(f"[UML] mmdc failed: {result.stderr}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[UML] mmdc not found. Skipping PNG generation. Install with: npm install -g @mermaid-js/mermaid-cli")
    except subprocess.TimeoutExpired:
        print("[UML] mmdc timed out")
    except Exception as e:
        print(f"[UML] Error: {e}")

    return None


def generate_flowchart(output_dir: Path) -> dict:
    """Generate only the Flowchart diagram."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Flowchart
    mmd = generate_mermaid_file(MERMAID_DIAGRAM, output_dir / 'compass_flowchart')
    png = convert_to_png(mmd)
    results['flowchart'] = {'mmd': mmd, 'png': png}

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path(__file__).parent

    print("=" * 60)
    print("COMPASS UML Diagram Generator")
    print("=" * 60)
    
    results = generate_flowchart(output_dir)

    print("\nGenerated files:")
    for name, paths in results.items():
        print(f"  {name}:")
        print(f"    Mermaid: {paths['mmd']}")
        if paths['png']:
            print(f"    PNG:     {paths['png']}")
