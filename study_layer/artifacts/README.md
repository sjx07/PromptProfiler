# Study Artifact Library

This directory stores study-level feature metadata. It is intentionally separate from `features/`, which remains the executable runtime inventory consumed by `FeatureRegistry`.

Boundary:

- `concept_id`: cross-task feature meaning used for analysis.
- `implementation_id`: task-specific realization of a concept.
- `canonical_id`: executable feature JSON id under `features/<task>/`.
- `variant_id`: matrix row handle used by a benchmark execution manifest.

Runtime code should continue to load from `features/<task>`. Matrix compilers and exporters can read this artifact library to recover cross-task feature vectors.
