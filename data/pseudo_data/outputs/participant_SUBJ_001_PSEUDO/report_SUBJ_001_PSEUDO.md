# Patient Report: SUBJ_001_PSEUDO

**Generated**: 2026-02-19T16:21:11.877052

## Prediction
- **Classification**: CASE (Phenotype match found for: MAJOR DEPRESSIVE DISORDER)
- **Probability**: 90.0%
- **Confidence**: HIGH
- **Target Condition**: MAJOR DEPRESSIVE DISORDER
- **Control Condition**: HEALTHY

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 7/7 passed

## Key Findings
1. **[CLINICAL]** Persistent low mood, anhedonia, fatigue, early morning awakening, concentration impairment, weight loss (moderate mood scale 17/27)
2. **[BIOLOGICAL_ASSAY]** BDNF deficit (z=-2.0, 2nd percentile, MODERATE effect)
3. **[BIOLOGICAL_ASSAY]** CRP elevation (z=1.7, 95th percentile, MODERATE effect); inflammation markers mean z=1.3
4. **[BRAIN_MRI]** Left hippocampal volume reduction (z=-2.0, 2nd percentile, LARGE effect); mean subcortical z=-1.4
5. **[BRAIN_MRI]** DMN hyperconnectivity mPFC-PCC (z=2.5, 99th percentile, LARGE effect)

## Clinical Summary
This 45yo female presents with classic moderate melancholic MDD phenotype: 7-8 months persistent low mood, anhedonia, early awakening insomnia, fatigue, concentration/psychomotor slowing, appetite/weight loss (4-5kg), social withdrawal, occupational impairment, meeting DSM-5 criteria. Supported by convergent biomarkers (BDNF z=-2.0 2nd%, hippocampal atrophy left z=-2.0, CRP z=1.7, HPA cortisol z=1.2, DMN hyperconnectivity z=2.5, mood PRS z=1.8) across 4/5 domains, stress/sleep elevations (z=1.4-1.6), mild cognitive deficits. Prior responsive episode, family hx; distinguishes from healthy/stress via multi-hit neuroimmune-limbic signature. High confidence CASE match; recommend SSRI trial, sleep/psychotherapy.

## Reasoning Chain
1. Step 1: Prioritize clinical notes - explicit MDD symptoms (low mood, anhedonia, insomnia, concentration, weight loss, impairment) meet DSM-5 criteria; past episodes, family hx weight strongly to CASE.
2. Step 2: Multimodal convergence - BIOLOGICAL_ASSAY/BRAIN_MRI prioritized by FeatureSynthesizer/ClinicalRelevanceRanker show MDD-specific patterns (BDNF low z=-2.0, hippocampal atrophy z=-2.0, DMN high z=2.5, inflammation/HPA high); 4/5 domains abnormal.
3. Step 3: Tool consensus - DifferentialDiagnosis 85% MDD likelihood; multimodal narratives HIGH confidence TOWARD_CASE; no rule-outs (stable thyroid, no hypomania).
4. Step 4: Avoid FP - Convergent clinical + biomarkers override isolated risks; prevalence-calibrated prob 0.90 (strongly elevated) with HIGH confidence due to multi-domain coherence.
5. Step 5: No symptoms hallucination - All referenced from raw data/notes.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 178,698
- **Domains Processed**: BIOLOGICAL_ASSAY, BRAIN_MRI, COGNITION, LIFESTYLE_ENVIRONMENT, DEMOGRAPHICS