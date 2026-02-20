# Patient Report: SUBJ_001_PSEUDO

**Generated**: 2026-02-20T20:30:08.901791

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: expected age of death: 76.000
- **Probability / Root Confidence**: N/A
- **Confidence**: MEDIUM
- **Target Label Context**: MORTALITY

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[BIOLOGICAL_ASSAY]** Élévation marquée des sphingolipides (z=1.7, 95e percentile, effet MODÉRÉ)
2. **[BIOLOGICAL_ASSAY]** Déficit sévère en BDNF (z=-2.0, 2e percentile, effet MODÉRÉ)
3. **[BRAIN_MRI]** Atrophie hippocampique gauche sévère (z=-2.0, 2e percentile, effet MODÉRÉ)
4. **[BIOLOGICAL_ASSAY]** Inflammation systémique (CRP z=1.7, effet MODÉRÉ)
5. **[LIFESTYLE_ENVIRONMENT]** Stress chronique élevé (z=1.4, 92e percentile, effet MODÉRÉ)

## Clinical Summary
Cette patiente de 45 ans présente un profil multimodal à risque modéré-élevé de mortalité prématurée, dominé par une convergence d'anomalies inflammatoires (sphingolipides z=1.7, CRP z=1.7), neurotrophiques (BDNF z=-2.0), structurales (hippocampe z=-2.0) et psychosociales (stress z=1.4), cohérent avec une neurodégénérescence inflammatoire accélérée et inflammaging. Symptômes dépressifs chroniques et facteurs de mode de vie aggravants renforcent cette vulnérabilité, sans preuves d'imminence aiguë. Âge de décès attendu estimé à 76 ans (réduction de ~9 ans vs. norme), nécessitant surveillance longitudinale.

## Reasoning Chain
1. Évaluation des features prioritaires (BIOLOGICAL_ASSAY et BRAIN_MRI) montrant convergence d'anomalies modérées à sévères (|z|>1.5) indiquant inflammaging et neurodégénérescence.
2. Intégration des notes cliniques : dépression persistante, stress psychosocial, antécédents familiaux (suicide tante à 50 ans), sans signes aigus de mortalité imminente.
3. Estimation quantitative : espérance de vie standard ~85 ans pour femme de 45 ans ; soustraction conservatrice de 9 ans basée sur risques cumulés (inflammation + BDNF bas + atrophie + stress, cf. littérature sur réduction LE par dépression/inflammation).
4. Âge attendu de décès estimé à 76 ans, avec incertitude due à absence de données longitudinales et vitals.
5. Calibration : probabilité risque prématuré ~65% (cf. DifferentialDiagnosis), traduite en régression conservatrice.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 118,154
- **Domains Processed**: BIOLOGICAL_ASSAY, BRAIN_MRI, COGNITION, LIFESTYLE_ENVIRONMENT