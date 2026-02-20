# Annotation Templates

These templates define recommended annotation payload structures for validation.

## Files

- `examples/binary_targets_example.json`
  - Ground-truth labels for `--prediction_type binary`
- `examples/multiclass_annotations_example.json`
  - Ground-truth class labels for `--prediction_type multiclass`
- `examples/regression_univariate_annotations_example.json`
  - Exactly one numeric output per participant
- `examples/regression_multivariate_annotations_example.json`
  - Two or more numeric outputs per participant
- `examples/hierarchical_annotations_example.json`
  - Node-wise mixed labels/values for hierarchical validation

## Notes

- Participant IDs can be stored as `eid`, `participant_id`, or object keys.
- Optional grouping keys (`disorder`, `group`, `cohort`, `phenotype_group`) are used for per-group analysis.
- For non-binary modes, pass the JSON via `--annotations_json`.
- Hierarchical mode requires consistent node schema across participants (same node IDs, node modes, and regression output keys per node).

## Recommended JSON Envelope

Use either a dictionary keyed by participant ID or an `annotations` list.

Dictionary form:

```json
{
  "SUBJ_001": {"label": "CASE", "disorder": "GROUP_1"},
  "SUBJ_002": {"label": "CONTROL", "disorder": "GROUP_1"}
}
```

List form:

```json
{
  "annotations": [
    {"eid": "SUBJ_001", "label": "CASE", "disorder": "GROUP_1"},
    {"eid": "SUBJ_002", "regression": {"total_iq": 101.4}, "disorder": "GROUP_1"}
  ]
}
```

For non-binary modes, replace `label` with the appropriate payload:
- multiclass: `label`
- regression: `regression` (or `values`)
- hierarchical: `nodes`
