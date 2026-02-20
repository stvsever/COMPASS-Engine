# Patient Report: SUBJ_003_PSEUDO

**Generated**: 2026-02-16T21:42:08.196740

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
1. **[LIFESTYLE_ENVIRONMENT]** Elevated stress (z=0.7, 76th percentile, NEGLIGIBLE effect)
2. **[LIFESTYLE_ENVIRONMENT]** Sleep issues (z=0.6, 73rd percentile, NEGLIGIBLE effect)
3. **[COGNITION]** Psychomotor speed deficit (z=-0.5, 31st percentile, NEGLIGIBLE effect)
4. **[BIOLOGICAL_ASSAY]** Iron status reduced (z=-0.6, 27th percentile, NEGLIGIBLE effect)
5. **[BIOLOGICAL_ASSAY]** Vitamins reduced (z=-0.5, 31st percentile, NEGLIGIBLE effect)

## Clinical Summary
This 33-year-old female, 6 months postpartum, presents with mild situational sleep disruption, stress, fatigue-related cognitive slowing, and subclinical iron/vitamin deficiencies (all |z|<1.0, negligible effects), fully attributable to physiological adaptation per clinical notes and tools. Absent psychiatric symptoms (no mood dysregulation, psychosis, anxiety) and normal neurologic exam rule out neuropsychiatric phenotype matching; aligns with control (non-psychiatric brain pathology absent). Strong evidence favors healthy control status with monitoring recommended.

## Reasoning Chain
1. Step 1: Review clinical notes (non_numerical_data): No psychiatric symptoms (no low mood, psychosis, anxiety); explicit clinician impression of healthy postpartum fatigue. Prioritizes CONTROL per NO SYMPTOMS rule.
2. Step 2: Assess hierarchical deviations and features: All |z| <1.0 (negligible effects, max z=1.1 in one sleep leaf); weak convergence in lifestyle/cognition/biology explained by postpartum context.
3. Step 3: Integrate tool outputs: Differential diagnosis rates neuropsychiatric UNLIKELY (5%); multimodal narrative TOWARD_CONTROL; phenotype requires psychiatric + neuro symptoms (absent).
4. Step 4: Calibrate probability: Very low risk (0.05) given prevalence, no symptoms, small effects; high confidence from consistent data and explicit healthy impression.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 100,912
- **Domains Processed**: COGNITION, BIOLOGICAL_ASSAY, LIFESTYLE_ENVIRONMENT, DEMOGRAPHICS