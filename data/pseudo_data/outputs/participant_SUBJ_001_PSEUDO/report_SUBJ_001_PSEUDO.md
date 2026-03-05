# Patient Report: SUBJ_001_PSEUDO

**Generated**: 2026-03-01T18:23:45.929921

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
1. **[CLINICAL]** Persistent low mood, anhedonia, fatigue, insomnia, weight loss, concentration impairment with functional impact; mood scale 17/27 (moderate)
2. **[BIOLOGICAL_ASSAY]** Neurotrophic factors deficit (z=-1.7; BDNF z=-2.0, 2nd percentile, LARGE effect)
3. **[BIOLOGICAL_ASSAY]** Sphingolipids elevation (z=1.7; ceramides up to z=2.4, MODERATE-LARGE effect)
4. **[BRAIN_MRI]** Subcortical volumes reduction (z=-1.4; hippocampus mean z=-1.85, left z=-2.0, LARGE effect)
5. **[BRAIN_MRI]** Resting-state connectivity elevation (z=1.3; DMN z=2.5, LARGE effect)

## Clinical Summary
This 45-year-old female exhibits a textbook Major Depressive Disorder phenotype, with 7-8 months of persistent low mood, anhedonia, neurovegetative symptoms (insomnia, fatigue, 4-5kg weight loss), cognitive slowing, and functional impairment across occupational/social domains, corroborated by MSE findings (restricted affect, psychomotor slowing) and moderate mood scale score (17/27). Multimodal data reinforces via severe neurotrophic deficit (BDNF z=-2.0), sphingolipid elevation (z=1.7-2.4), inflammation (CRP z=1.7), HPA activation (z=1.2), hippocampal atrophy (z=-1.85 mean), DMN hyperconnectivity (z=2.5), elevated stress (z=1.4), and mild executive slowing (z=-1.1), forming a coherent stress-inflammation-neuroplasticity axis absent in healthy controls. Prior responsive episode, family history, and absence of mania/psychosis support unipolar MDD; stable hypothyroidism ruled out as primary. Recommend SSRI/therapy with monitoring.

## Reasoning Chain
1. 1. Prioritize non_numerical_data: Explicit DSM-matching symptoms (low mood/anhedonia/insomnia/weight loss/concentration/fatigue >2 weeks, impairment) with MSE (restricted affect, slowing) and scales (moderate mood) confirm DEPRESSION phenotype; no mania/SI rules out alternatives.
2. 2. Convergent multimodal evidence: Tools (DifferentialDiagnosis 85% MDD, all MultimodalNarratives/Compressors HIGH/TOWARD_CASE) + hierarchical z-scores (neurotrophics -1.7, sphingolipids 1.7, subcortical -1.4, stress 1.4, sleep 1.2) show clustered moderate-large abnormalities (|z|>1.5 in 12+ features across 4/5 domains).
3. 3. Feature importance (FeatureSynthesizer/ClinicalRelevanceRanker): Heaviest weight to proteomics/lipidomics/brain + clinical anchors; directionality coherent (hypo-neurotrophics + hyper-stress/inflammation).
4. 4. False positive guards: Overwhelming clinical primacy + multi-domain convergence exceeds thresholds; stable thyroid/no substances rule out confounds.
5. 5. Calibration: Population prevalence low but data quality/consistency yields strongly elevated probability (0.92); HIGH confidence from no gaps in core signals.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 127,329
- **Domains Processed**: BIOLOGICAL_ASSAY, BRAIN_MRI, COGNITION, LIFESTYLE_ENVIRONMENT, DEMOGRAPHICS