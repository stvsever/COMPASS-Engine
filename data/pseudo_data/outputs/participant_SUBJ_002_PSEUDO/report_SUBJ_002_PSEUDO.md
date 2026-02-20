# Patient Report: SUBJ_002_PSEUDO

**Generated**: 2026-02-16T20:53:33.448394

## Prediction
- **Classification**: CONTROL (Closer to: brain-implicated pathology, but NOT psychiatric)
- **Probability**: 5.0%
- **Confidence**: HIGH
- **Target Condition**: neuropsychiatric
- **Control Condition**: brain-implicated pathology, but NOT psychiatric

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 7/7 passed

## Key Findings
1. **[CLINICAL_NOTES]** No psychiatric or neurologic symptoms reported; healthy volunteer with normal exam
2. **[BRAIN_MRI]** Normal structural and diffusion metrics (structural z=0.0, diffusion z=0.1)
3. **[COGNITION]** Preserved cognition across domains (attention z=0.3, memory z=0.2, processing_speed z=0.1, executive z=0.1)
4. **[BIOLOGICAL_ASSAY]** Favorable biomarkers (inflammation z=-0.3, vitamins z=0.2, metabolic z=-0.1)
5. **[LIFESTYLE_ENVIRONMENT]** Protective factors (physical_activity z=0.6)

## Clinical Summary
This 42-year-old healthy volunteer presents with a comprehensively normal multimodal profile, including unremarkable clinical history (no psychiatric/neurologic symptoms, normal exam), negligible z-score deviations across all domains (max |z|=0.9 in physical activity, protective), normal brain MRI (incidental stable cyst z=0), preserved cognition, favorable biomarkers (low inflammation), and healthy lifestyle. No evidence supports neuropsychiatric phenotype matching; strongly aligns with healthy control, distinct from brain-implicated pathology.

## Reasoning Chain
1. Step 1: Prioritize non_numerical_data - explicit healthy status, no psychiatric history/symptoms, normal exam/labs/MRI (incidental cyst non-pathologic).
2. Step 2: Review hierarchical deviations - all |z|<0.6 (negligible), no domain means exceed threshold; uniform normality across BIOLOGICAL_ASSAY, BRAIN_MRI, COGNITION.
3. Step 3: Integrate tool outputs - MultimodalNarrative, AnomalyNarrative, HypothesisGenerator all converge on healthy/resilient profile, rule out neuropsychiatric (e.g., DifferentialDiagnosis UNLIKELY).
4. Step 4: Phenotype mismatch - no psychiatric symptoms, cognitive/mood clusters absent; biomarkers normal, not matching expected frontotemporal/inflammatory signature.
5. Step 5: Calibrate probability low (0.05) given zero convergent evidence, high confidence from consistent data; default CONTROL avoids false positive.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 91,275
- **Domains Processed**: BRAIN_MRI, COGNITION, BIOLOGICAL_ASSAY, LIFESTYLE_ENVIRONMENT, DEMOGRAPHICS