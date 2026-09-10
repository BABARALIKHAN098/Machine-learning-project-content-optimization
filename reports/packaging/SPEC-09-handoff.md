# SPEC-08 completion and SPEC-09 handoff

**Verified:** 2026-09-10
**Status:** Research packaging complete; production_ready=false; recommended_model=null.

## Audit result

The initial checkout already contained most phase 1-4 code and 258 passing tests. Phase 0 was still a placeholder specification, packaging integration coverage was absent, the verifier failed lint, and no real packaging runs or handoff existed. A real build exposed a Windows wheel metadata line-ending bug. This completion fixes that parser, binds manifest provenance and comparison flags to the frozen decision/model metadata, protects additional artifact destinations, adds synthetic end-to-end coverage and an invented CSV example, adopts SPEC-08 and supplies the verification below.

## Verified artifacts

- [Reference run](spec08-final-reference/packaging_manifest.json)
- [Reproduction run](spec08-final-reproduction/packaging_manifest.json)
- [Verification: commands, return codes, hashes, counters and timings](spec08-final-reference/verification.json)

| Research family | Concrete package manifest | Manifest SHA-256 |
| --- | --- | --- |
| logistic_regression | [spec08-final-reference-logistic_regression](../../artifacts/packages/spec08-final-reference-logistic_regression/package_manifest.json) | `9bd1ae7043ea49339bbc21f8311e9b1e296043b83db8f90c25e220a0741e658a` |
| random_forest | [spec08-final-reference-random_forest](../../artifacts/packages/spec08-final-reference-random_forest/package_manifest.json) | `17d3d181eea90223de0f5ea59edd750d495b1efd32f0a8cafaae45c10d4279e6` |

Both reference packages and both reproduction packages are locally complete. Copied joblib files and runtime wheels remain local generated files excluded from Git; JSON manifests alone are insufficient to load a package. Moving the complete package directory is supported on the declared environment.

## Acceptance evidence

- Full suite, including all focused packaging/inference tests: **259 passed**, 28 existing sklearn warnings from small metric fixtures. Ruff passed. Actual subprocess output and return codes are in verification.json.
- Both synthetic builds and both real builds reproduce semantic contents. Original finalist model bytes are unchanged.
- Exact source/package labels on **5,857 verified validation rows** per family; probabilities agree within absolute 1e-9. Scoped prediction fingerprints match frozen evidence. No test rows scored.
- Real packaging/inference audit: **zero fit calls**. Dry-run/build phases perform no predictions. Label-only mode capability and global request validation are tested.
- Protected source/split/feature/baseline/training/evaluation/model hashes unchanged. Aggregate privacy scan passed; no source-row predictions are published.
- Both offline isolated runtime checks passed: local project wheel installation, existing dependency files, disabled site/editable hooks, upstream access denied, matching synthetic predictions and zero fitting.

## Observed synthetic CPU workload

30,000 invented rows per model, 5,000-row chunks and one numerical thread. Prediction timings include request validation and exclude loading/serialization. These are observations, not an SLA.

| Family | Cold load (s) | Label-only prediction (s) | With probabilities (s) |
| --- | ---: | ---: | ---: |
| logistic_regression | 0.093 | 3.712 | 5.892 |
| random_forest | 0.466 | 5.897 | 6.265 |

Supported observed runtime: CPython 3.12.0, Windows AMD64, exact dependencies and source digest in each environment.json. The isolated check reused installed dependency files; a clean-machine dependency installation and cross-platform portability are not certified. The bundled requirements pin the dependencies but do not bundle their wheels.

## SPEC-09 interface

Choose a concrete package; no latest/best/production pointer exists. Use PackagedPredictor.load(path, purpose="research", expected_manifest_sha256=...) for a trusted pinned package. input_schema.json is the authoritative ordered feature contract; output_schema.json describes v2 output. Preserve the envelope fields (package identity/hash, original model version, research status, row count and probability mode) and ordered six-field records.

Accept only content_id plus approved features; validate the entire request before scoring. IDs must be unique nonempty strings without surrounding whitespace. Preserve leading zeros and literal NA. Numeric Python strings, extra/outcome columns and nonfinite values are rejected. Missing cells use fitted imputation. Optional scores are uncalibrated and do not establish causal refresh benefit.

```bash
python scripts/predict_batch.py --package-dir artifacts/packages/spec08-final-reference-random_forest --purpose research --input-path data/examples/synthetic_inference.csv --output-path reports/private-inference/spec08-example.json
```

Add --include-probabilities only when requested. Outputs are private atomic JSON with no overwrite. See [README usage](../../readme.md) for Python loading, offline installation and fresh-build commands.

## Readiness boundaries

Both finalists still fail the unchanged project macro-F1 target. Research packaging is not model acceptance. Validation has informed development and is not independent evidence; historical holdout exposure and cutoff review remain unresolved. SPEC-09/10 own application controls, retention, deployment and operational acceptance. No production deployment or automated content changes are implied.
