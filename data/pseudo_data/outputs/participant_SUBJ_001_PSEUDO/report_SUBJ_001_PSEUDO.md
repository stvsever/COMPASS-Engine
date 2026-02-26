# Patient Report: SUBJ_001_PSEUDO

**Generated**: 2026-02-26T10:45:48.077071

## Prediction
- **Prediction Type**: binary_classification
- **Primary Output**: DEPRESSION
- **Probability / Root Confidence**: 92.0%
- **Confidence**: HIGH
- **Target Label Context**: DEPRESSION
- **Comparator Label Context**: HEALTHY

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[CLINICAL_NOTES]** Persistent low mood, anhedonia, fatigue, early morning awakening, concentration impairment, appetite/weight loss (4-5kg) over 7-8 months with functional impairment
2. **[BIOLOGICAL_ASSAY]** Neurotrophic factors deficiency (BDNF z=-2.0, 2nd percentile, MODERATE effect)
3. **[BIOLOGICAL_ASSAY]** Elevated sphingolipids (z=1.7, 96th percentile, MODERATE effect); inflammation (CRP z=1.7); HPA axis (cortisol z=1.4)
4. **[BRAIN_MRI]** Bilateral hippocampal atrophy (left z=-2.0, 2nd percentile; right z=-1.7, 4th percentile, MODERATE effect)
5. **[BRAIN_MRI]** DMN hyperconnectivity (z=2.5, 99th percentile, LARGE effect); amygdala reactivity (z=1.8)

## Clinical Summary
This 45-year-old female exhibits a classic Major Depressive Disorder phenotype with persistent low mood, anhedonia, somatic symptoms (fatigue, insomnia, weight loss), cognitive slowing, and functional impairment following work/caregiving stress, corroborated by moderate mood scale scores and family history. Convergent multimodal evidence includes moderate-to-large deviations in key depression markers: reduced neurotrophics/hippocampal volume (z=-2.0), elevated inflammation/sphingolipids/HPA (z=1.4-1.7), DMN hyperconnectivity (z=2.5), and lifestyle stress/sleep perturbations. No medical mimics or bipolar features; high-confidence CASE designation over healthy control, warranting antidepressant trial and monitoring.

## Reasoning Chain
1. Step 1: Prioritize non_numerical_data - explicit DSM-5 MDD symptoms (low mood, anhedonia, sleep/appetite change, fatigue, concentration) with functional impact, prior episode, family history; mood scale moderate (17/27) - strongly favors CASE.
2. Step 2: Assess hierarchical deviations and synthesizers - multi-domain convergence (|z|>1.5 in neurotrophics z=-2.0, sphingolipids z=1.7, hippocampal z=-2.0, DMN z=2.5, HPA/inflammation); FeatureSynthesizer/ClinicalRelevanceRanker prioritize bioassay/MRI for discrimination.
3. Step 3: Integrate multimodal narratives/DifferentialDiagnosis - VERY_LIKELY MDD (85%), high confluence across clinical+biomarkers; no rule-out criteria met (e.g., stable thyroid, no mania).
4. Step 4: Calibrate probability - strong clinical primary evidence + convergent biomarkers (no isolated deficits) yield strongly elevated risk (0.92); HIGH confidence from data quality/consistency.
5. Step 5: False positive check - symptoms not biomarker-originated; convergent, not generic stress/sleep alone.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 113,770
- **Domains Processed**: BIOLOGICAL_ASSAY, BRAIN_MRI, COGNITION, LIFESTYLE_ENVIRONMENT, DEMOGRAPHICS