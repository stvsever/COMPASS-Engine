# deep_phenotype.md

## Participant: SUBJ_001_PSEUDO | Condition cible: DEPRESSION vs HEALTHY

### Résumé exécutif
Cette femme de 45 ans présente un phénotype classique de dépression majeure (MDD) avec 7-8 mois de symptômes persistants : humeur basse, anhedonie, fatigue, éveil matinal précoce, perte de poids (4-5 kg), troubles de concentration et altération fonctionnelle. Confirmé par une échelle d'humeur modérée (17/27), antécédents familiaux et convergence multimodale : déficit en BDNF (z=-2.0), atrophie hippocampique gauche (z=-2.0, droite z=-1.7), CRP élevé (z=1.7), hyperactivité HPA, dysrégulation du sommeil (z=1.6) et stress (z=1.4). Pas de contre-indications (thyroïde stable, absence de manie). **Prédiction : DEPRESSION (probabilité 0.92, confiance ÉLEVÉE)**. Forte discrimination vs contrôles sains via biotype dépressionnel canonique. Focus cerveau : atrophie limbique et hyperconnectivité DMN (z=2.5) convergent avec déficits mnésiques et exécutifs, soutenant un phénotype à haut risque de chronicité. Suivi longitudinal et traitement multimodal recommandé.

### Raisonnement de la prédiction
Tâche : Classification binaire (DEPRESSION vs HEALTHY).  
**Sortie principale** : DEPRESSION (probabilité 0.92, HEALTHY 0.08).  
**Confiance** : ÉLEVÉE (score 0.92).  
**Chaîne de raisonnement** :  
1. Priorité aux données cliniques non-numériques : Symptômes MDD DSM-5 explicites (humeur basse, anhedonie, changements sommeil/appétit, altération) avec durée/impairment ; absence manie/psychose ; réponse passée aux IRSN.  
2. Convergence multimodale : 4/5 domaines anormaux (|z|>1.2 axes neurotrophique/HPA/inflammation/limbique/stress ; ex. BDNF z=-2.0, hippocampe z=-2.0, CRP z=1.7).  
3. Consensus outils : Diagnostic différentiel 85% MDD ; narratifs multimodaux vers CAS à haute confiance ; sans divergences.  
4. Exclusion contrôles : Symptômes + biomarqueurs excèdent normes saines/stress ; hypothyroïdie stable.  
5. Calibration : Preuves fortes portent probabilité à 0.92 ; confiance ÉLEVÉE par cohérence multi-domaines.  
Verdict critique : SATISFACTORIE (score composite 1.0).  
**Phénotype matching** (non diagnostic) : Profil aligné MDD mélancolique (85% vraisemblance).

Table: Résumé des sorties prédictives

| Noeud | Mode | Label prédit | Probabilités | Confiance | Score confiance |
|-------|------|--------------|--------------|-----------|-----------------|
| root | Classification binaire | DEPRESSION | DEPRESSION: 0.92, HEALTHY: 0.08 | ÉLEVÉE | 0.92 |

Table: Preuves pour/contre la prédiction sélectionnée (DEPRESSION)

| Preuves POUR (cas) | Domaine | z-score | Pertinence |
|--------------------|---------|---------|------------|
| Symptômes DSM-5 explicites avec durée/impairment | CLINICAL | non fourni | Symptômes MDD canoniques ; driver principal |
| Réduction BDNF (2e percentile, effet LARGE) | BIOLOGICAL_ASSAY | -2.0 | Déficit neuroplasticité signature dépression |
| Atrophie volume hippocampique gauche (2e percentile, LARGE) ; droite z=-1.7 | BRAIN_MRI | -2.0 | Corrélat structurel classique ; converge avec mémoire |
| Élévation CRP (96e percentile, MODÉRÉ) ; cluster inflammation | BIOLOGICAL_ASSAY | 1.7 | État pro-inflammatoire commun en MDD |
| Dysrégulation sommeil (95e percentile, MODÉRÉ) ; stress z=1.4 | LIFESTYLE_ENVIRONMENT | 1.6 | Symptômes neurovégétatifs alignés |
| Déficit fonction exécutive (10e percentile, MODÉRÉ) ; apprentissage récompense z=-1.1 | COGNITION | -1.3 | Ralentissement cognitif aligné concentration |
| Convergence 4/5 domaines (|z|>1.2 ; 70%+ features anormales) ; outils 85% MDD | Multimodal | - | Cohérence forte vs sains |

| Preuves CONTRE (contrôle) | Domaine | z-score | Pertinence |
|----------------------------|---------|---------|------------|
| Aucune significative ; démographiques quasi-normaux, thyroïde stable | Tous | - | Signaux cognitifs légers chevauchent stress mais surpassés |

### Phénotypage profond par domaine (focus cerveau)
**CLINICAL** (non-numérique) : Humeur basse persistante, anhedonie, fatigue, éveil 3-4h, perte poids 4-5kg, concentration altérée, retrait social, productivité ↓ (jours maladie), échelle humeur 17/27 modérée. Antécédents : épisode passé SSRI/réponse partielle ; famille (mère humeur, tante suicide). Médical : hypothyroïdie stable. z-score : non fourni. Direction : ABNORMAL_LOW. Pertinence : Symptômes MDD DSM-5 primaires.

**BRAIN_MRI** (focus prioritaire) : Atrophie hippocampique sévère bilatérale (gauche z=-2.0/2e percentile LARGE ; droite z=-1.7 ; subcortical z=-1.4). Amygdale réduite (g/z=-1.0/-1.3), NAcc z=-1.1. Éclaircissement cortical préfrontal/cingulaire (dlPFC gauche z=-1.3, vmPFC z=-1.1, ACC z=-1.0). Hyperconnectivité DMN (mPFC-PCC z=2.5 LARGE), hypoconnectivité exécutive (dlPFC-PPC z=-1.0). Hyperréactivité amygdalienne (z=1.8 MODÉRÉ), connectivité limbique z=1.6. Déficits glutamate hippocampique z=-1.2, GABA préfrontal z=-0.9. WMH frontaux légers z=0.6. Couverture : 19/140 feuilles (13.6%). Direction : ABNORMAL_LOW/HIGH mixte. Pertinence : Signature structurelle/fonctionnelle dépression (atrophie limbique, hyperconnectivité rumination) converge cognition (mémoire/exécutif).

**BIOLOGICAL_ASSAY** : BDNF ↓ z=-2.0 LARGE, CRP ↑ z=1.7 MODÉRÉ, cluster inflammation (IL-6/MCP-1). Céramides ↑ (max z=2.4), HPA cortisol ↑ (AM z=1.4). PRS humeur/stress ↑ (z=1.8/1.5). Oméga-3 ↓ z=-0.8. Thyroïde stable z=0.1. Couverture : 42/220 (19.1%). Direction : ABNORMAL_LOW/HIGH. Pertinence : Biotype inflammatoire/neurotrophique.

**COGNITION** : Exécutif ↓ z=-1.1 (TMT-B z=-1.3), vitesse traitement z=-1.0, mémoire z=-0.8, biais affectif z=1.1 (Stroop émotionnel z=1.2, récompense z=-1.1). Couverture : 14/100 (14%). Direction : ABNORMAL_LOW. Pertinence : Pseudodémence dépressive.

**LIFESTYLE_ENVIRONMENT** : Sommeil ↑ z=1.2 (insomnie z=1.6), stress ↑ z=1.4, isolement social z=1.0. Sédentarité ↓ z=-0.5. Couverture : 21/80 (26.2%). Direction : ABNORMAL_HIGH. Pertinence : Amplificateurs.

**DEMOGRAPHICS** : Femme 45 ans, éducation 16 ans, BMI z=1.2 léger. Couverture : 7/15 (46.7%). Direction : Neutre.

Table: Anomalies hiérarchiques clés (focus cerveau)

| Domaine | Sous-domaine | z-score | Direction |
|---------|--------------|---------|-----------|
| BRAIN_MRI | structural/subcortical_volumes | -1.4 | ABNORMAL_LOW |
| BRAIN_MRI | structural/cortical_thickness | -1.0 | ABNORMAL_LOW |
| BRAIN_MRI | functional_connectivity/resting_state | 1.3 | ABNORMAL_HIGH |
| BRAIN_MRI | functional_connectivity/task_activation | 1.6 | ABNORMAL_HIGH |
| BRAIN_MRI | spectroscopy | -0.9 | ABNORMAL_LOW |
| BIOLOGICAL_ASSAY | proteomics/neurotrophic_factors | -1.7 | ABNORMAL_LOW |
| LIFESTYLE_ENVIRONMENT | sleep | 1.2 | ABNORMAL_HIGH |
| LIFESTYLE_ENVIRONMENT | stress | 1.4 | ABNORMAL_HIGH |

### Couverture des données, incertitudes et limitations
**Couverture** : 103 features total ; 103 représentées (0 manquantes). Domaines traités : 5/5 (BIOLOGICAL_ASSAY 19.1%, BRAIN_MRI 13.6%, COGNITION 14%, DEMOGRAPHICS 46.7%, LIFESTYLE_ENVIRONMENT 26.2%). Itération 1/1 sélectionnée (verdict SATISFACTORIE).  
**Incertitudes** : Pas de scores PHQ-9/HDRS formels (résumé humeur modéré) ; détails multimodaux partiels (ex. PRS exacts) ; absence données longitudinales ; comorbidité hypothyroïdie légère (exclue primaire).  
**Limitations** : Couverture incomplète feuilles (ex. BRAIN_MRI 121/140 manquantes) ; pas d'explainabilité activée ; focus phénotypage (non diagnostic formel). Evidence insuffisante pour claims forts sur sous-types sans suivi.

### Annexe : Extraits d'évidence traçables
- **Clinique** : "7-8 months of persistent low mood, reduced interest... early morning awakening... weight change over 3 months... concentration impaired."
- **BRAIN_MRI (focus)** : Unimodal : "Bilateral Hippocampal Atrophy... left z=-2.0... right z=-1.7" ; Multimodal : "hippocampal atrophy |z|=1.85 mapping to memory z=-0.7/-1.0".
- **BDNF** : "BDNF reduction (z=-2.0, 2nd percentile, LARGE effect)".
- **Multimodal cerveau-cognition** : "Strong convergence... hippocampal-memory, prefrontal-executive... mean |z|>1.1".
- **Diagnostic différentiel** : "Major Depressive Disorder (Melancholic subtype) 85%... low BDNF z=-2.0, hippocampal |z|=1.85".
- **Narratif fusion** : "4/5 domains converge... BDNF z=-2.0... CRP z=1.7... insomnia z=1.6".