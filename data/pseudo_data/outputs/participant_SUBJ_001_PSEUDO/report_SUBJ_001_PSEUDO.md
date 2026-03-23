# Patient Report: SUBJ_001_PSEUDO

**Generated**: 2026-03-18T15:29:43.174039

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
1. **[CLINICAL]** Persistent low mood, anhedonia, fatigue, early morning awakening, weight loss (4-5kg), concentration impairment, functional decline over 7-8 months
2. **[BIOLOGICAL_ASSAY]** BDNF reduction (z=-2.0, 2nd percentile, LARGE effect)
3. **[BRAIN_MRI]** Left hippocampal volume reduction (z=-2.0, 2nd percentile, LARGE effect); right z=-1.7
4. **[BIOLOGICAL_ASSAY]** CRP elevation (z=1.7, 96th percentile, MODERATE effect); inflammation markers cluster
5. **[LIFESTYLE_ENVIRONMENT]** Sleep dysregulation (insomnia severity z=1.6, 95th percentile, MODERATE); stress z=1.4

## Clinical Summary
This 45-year-old female exhibits a classic Major Depressive Disorder phenotype with 7-8 months of persistent low mood, anhedonia, neurovegetative disturbances (early awakening, weight loss), cognitive slowing, and functional impairment, corroborated by moderate mood scale (17/27), family history, and convergent multimodal abnormalities including BDNF deficiency (z=-2.0), hippocampal atrophy (left z=-2.0), elevated CRP (z=1.7), HPA hyperactivity, sleep dysregulation (z=1.6), and stress burden (z=1.4). No contraindications (stable thyroid, no mania); profile strongly discriminates from healthy controls via depression-canonical biotype.

## Reasoning Chain
1. 1. Prioritize non_numerical_data: Explicit MDD symptoms (low mood, anhedonia, sleep/appetite changes, impairment) meet DSM criteria; no mania/psychosis; past response to SSRI supports unipolar depression.
2. 2. Multimodal convergence: 4/5 domains abnormal (|z|>1.2 in neurotrophic/HPA/inflammation/limbic/stress axes per FeatureSynthesizer/ClinicalRelevanceRanker; e.g., BDNF z=-2.0, hippocampal z=-2.0, CRP z=1.7).
3. 3. Tool consensus: DifferentialDiagnosis 85% MDD; narratives TOWARD_CASE HIGH confidence; no divergences.
4. 4. Rule out controls: Overwhelming symptom+biomarker match exceeds healthy/stress norms; hypothyroidism stable.
5. 5. Calibrate: Strong evidence elevates probability to 0.92 (strongly elevated risk); HIGH confidence from multi-domain consistency.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 131,397
- **Domains Processed**: BRAIN_MRI, BIOLOGICAL_ASSAY, COGNITION, LIFESTYLE_ENVIRONMENT, DEMOGRAPHICS