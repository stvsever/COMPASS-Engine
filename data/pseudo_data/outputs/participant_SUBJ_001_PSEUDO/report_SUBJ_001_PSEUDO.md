# Patient Report: SUBJ_001_PSEUDO

**Generated**: 2026-02-24T18:32:24.898810

## Prediction
- **Prediction Type**: hierarchical
- **Primary Output**: VERBAL: -0.700, SPATIAL: -0.500, CREATIVITY: -0.600 | nodes: 3
- **Probability / Root Confidence**: 75.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Phenotype Profile

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 10/10 passed

## Key Findings
1. **[COGNITION]** Déficits légers en mémoire épisodique (rappel différé z=-1.0, 16e percentile, effet petit) et fonction exécutive (TMT_B z=-1.3, 10e percentile, effet petit)
2. **[BRAIN_MRI]** Atrophie hippocampique bilatérale modérée (gauche z=-2.0, droite z=-1.7, moyenne |z|=1.85, effet modéré)
3. **[BIOLOGICAL_ASSAY]** Déficit neurotrophique (BDNF z=-2.0, 2e percentile, effet modéré) et élévation inflammatoire (CRP z=1.7)
4. **[LIFESTYLE_ENVIRONMENT]** Stress perçu élevé (z=1.4) et insomnie (z=1.6), avec isolement social (z=1.3)

## Clinical Summary
Chez cette patiente de 45 ans présentant un épisode thymique persistant avec symptômes dépressifs (humeur basse, anhedonie, insomnie, fatigue), les données multimodales révèlent des altérations cognitives légères (déficits exécutifs et mnésiques z≈-1.0 à -1.3), une atrophie hippocampique modérée et des marqueurs neuroinflammatoires (BDNF bas z=-2.0, CRP élevé z=1.7), sans évidence d'un phénotype de déficience intellectuelle. Le profil suggère des impacts fonctionnels secondaires à un trouble thymique neuroinflammatoire, avec z-scores cognitifs spécifiques proches de la normale (VERBAL/SPATIAL/CREATIVITY ≈ -0.6). Risque phénotypique faible ; surveillance et interventions multimodales recommandées.

## Reasoning Chain
1. Évaluation prioritaire des notes cliniques : absence de signes de déficience intellectuelle (éducation 16 ans, profession enseignante, MMSE 28/30) ; symptômes dominés par humeur basse et fatigue dépressive.
2. Intégration hiérarchique : déviations modérées en cognition (moyenne |z|~1.0, petits effets) sans déficits verbaux/spatiaux/créatifs sévères ; priorisation BIOLOGICAL_ASSAY/BRAIN_MRI per FeatureSynthesizer.
3. Estimation régressive : z-scores verbaux/spatiaux/créatifs négatifs légers (-0.5 à -0.7) basés sur verbal_learning (-0.7), visual_memory (-0.4), fluency (-0.7).
4. Classification risque : faible risque (low_risk) car profils cognitifs subcliniques, convergence vers phénotype thymique (80% per DifferentialDiagnosis).
5. Incertitude modérée due à couverture incomplète (19-46% par domaine) et absence de tests IQ directs.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 141,413
- **Domains Processed**: COGNITION, BRAIN_MRI, BIOLOGICAL_ASSAY, LIFESTYLE_ENVIRONMENT, DEMOGRAPHICS