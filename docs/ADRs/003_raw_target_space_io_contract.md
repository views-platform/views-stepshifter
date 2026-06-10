# ADR-003: Model I/O Operates in Raw Target Space (Untransformed In, Untransformed Out)

**Status:** Proposed
**Date:** 2026-06-08
**Proposed by:** Simon Polichinel von der Maase (cross-repo audit lead)
**Deciders:** Project maintainers
**Reviewed:** expert-code-review (2026-06-08) — authority-scope, enforcement, and ShurfModel/registry correctness fixes incorporated
**Consulted:** views-pipeline-core ADR-003 (Authority of Declarations), ADR-042 (PredictionFrame Adoption); views-hydranet ADR-046 / ADR-003 (prefix precedent); views-r2darts2 ADR-012

> **Normative summary.** A views-stepshifter model takes its target in **raw space** and returns
> predictions in **raw space**. Any target transform is applied **inside** the library, **declared**
> in config, and **inverted before output**. Numerical scale is governed by the config declaration —
> **never inferred from or encoded in a column name.**

> **⟨PENDING — views-pipeline-core ADR-XXX⟩** This ADR is the model-repo expression of a
> forthcoming **platform-wide** views-pipeline-core ADR that will ratify, for every model
> repository: (a) transforms are model-internal with raw I/O at the model boundary;
> (b) deprecation of the `lr_`/`ln_`/`lx_` column-name prefix scheme; (c) the config-declared
> transform as the **sole** source of truth for numerical scale. The number above is a placeholder
> and **must be filled in** once that platform ADR is ratified. This stepshifter ADR governs only
> stepshifter's own behavior; it does **not** itself deprecate any upstream construct (see clause 6).

---

## Context

views-stepshifter models consume a feature/target DataFrame and return a prediction DataFrame whose column is named `pred_<target>` (e.g. `pred_lr_ged_sb`). The platform convention is that **a model takes untransformed (raw target-space) input and returns untransformed (raw target-space) output**; any target-space transform (e.g. `log1p`, `asinh`) is the model library's own internal concern, applied during training and inverted before predictions leave the model, and is **declared in configuration** rather than hardcoded silently.

This convention is real but, for views-stepshifter, **not locally ratified**. An audit on 2026-06-08 established its provenance and its current governance gap:

- At the **platform level it exists only in planning documents**, not a ratified ADR: `views-pipeline-core/documentation/plans/2026-03-15_prediction_frame_two_track_status.md` (§Item 3) states *"transformations are now the model repo's responsibility — models must return predictions in the original scale… views-r2darts2, views-stepshifter, and views-baseline should follow,"* explicitly **blocked on verifying** those three repos undo their transforms; `plans/2026-06-01_pfe_production_roadmap.md` §7 (risk **C-140**) confirms only views-hydranet was verified.
- In **views-stepshifter it is only a Proposed story** — `docs/stories/target_transform_declarative.md`.
- The base-class contract `views-pipeline-core/CICs/ForecastingModelManager.md` governs prediction **return type, format, and step-window coverage** — it says nothing about numerical **scale**.

Two code-history facts make the gap concrete:

1. Commit `5fcfe43` ("log and unlog inside stepshifter", 2025-11-20) silently added `np.log1p` to `StepshifterModel._process_data` and `np.expm1` to `_predict_by_step`, forcing every stepshifter model into log-space regardless of config. Commit `08ee2eb` (2026-04-11) reverted both lines, citing the platform convention — but that convention was never ratified as binding governance in this repo.
2. views-pipeline-core has since **removed** the `# TEMPORARY` transform-undo blocks that previously inverted transforms at save time (the 2026-03-15 plan's Item 3). The current `views_pipeline_core/managers/model/model.py` contains no inversion. **Neither side of the boundary inverts transforms today**, and no declarative mechanism exists in stepshifter to make either side do so.

The motivating incident: the 2026-06-04 calibration report for ensemble `big_chungus` (`lr_ged_sb`, cm) scored MSLE 2.519 — **worse than an all-zeros baseline** and ~3× the fatalities002 retrospective gold standard (country-level MSLE 0.835) — while the four r2darts2 deep-learning constituents (which apply their own asinh transform internally) matched or beat the gold standard. The divergence is isolated to the stepshifter-family constituents. Whether a target-space mismatch is the proximate cause is **still under investigation** (it requires the per-model training space and the views-models configs); this ADR ratifies the *contract* whose absence permitted the ambiguity, independent of that finding.

### Why a column-name prefix is not a scale contract

Historically the "what scale is this column in?" question was answered by a **column-name prefix**: views-pipeline-core's `DatasetTransformationModule` (CIC) implements `ln_` (`ln(x+1)`), `lx_` (`ln(x+exp(offset))`), and `lr_` (identity / "linear-raw" marker), renaming the column as it applies or undoes the math; views-hydranet ADR-003 (Law 6, "Prefix-Purity") elevates this to an invariant that `lr_` signifies raw-count intent and must be preserved. **stepshifter does not rely on that scheme as a scale contract**, for four reasons:

1. **The boundary it guarded no longer exists.** Transformed target data no longer travels *outside* a model library — it is transformed and inverted entirely within. There is no cross-boundary handoff for a prefix to annotate.
2. **A column name carries intention, not enforced truth.** Renaming a column and actually applying (or correctly undoing) the transform are *separate tasks*; the name guarantees neither. `DatasetTransformationModule` itself documents the failure: an undo run with the wrong offset *"completes without error but produces incorrect values — no validation against history,"* and a duplicate transform is *"silently skipped."* A column labeled `ln_` is therefore not evidence that the values are in log space.
3. **The would-be source of truth is unreliable.** The only mechanism that could make a name trustworthy — a transform-history log (`DatasetTransformationModule.transformation_history`) — exists **in memory only, is never persisted, and is unvalidated**. stepshifter does not adopt it. Instead it makes the question unnecessary (see Decision).
4. **Framework-agnostic frames make name-based scale tracking structurally impossible.** As model output migrates to `PredictionFrame` (ADR-042), predictions are carried as a bare NumPy `(N×S)` array with only `time`/`unit` identifiers — **no column names and no scale metadata** (see `CICs/PredictionFrame.md`). There is nowhere to write `lr_`/`ln_`, and nowhere to record "this is log-space." Scale *must* be governed by the config, because the transport object cannot encode it.

This is the platform's existing **Authority of Declarations over Inference** principle (pipeline-core ADR-003) applied to numerical scale: ADR-003 already forbids inferring `level` "from index column names." Inferring *scale* from a column-name prefix is the same forbidden pattern. The platform-wide *deprecation* of the prefix scheme is the `⟨PENDING⟩` pipeline-core ADR's to declare; this ADR only commits stepshifter to not depending on it.

---

## Decision

**views-stepshifter models operate in raw target space at their public boundary, and numerical scale is governed by configuration — never by column names.**

1. **Input contract.** `StepshifterModel.fit(df)` (and all subclasses — `HurdleModel`, `ShurfModel`) consume the target column in **raw target space** (the native scale of the target as delivered by the queryset). The model library MUST NOT assume a pre-applied target transform.

2. **Output contract.** `StepshifterModel.predict(...)` (and subclasses) return predictions in **raw target space**.

3. **Transforms are model-internal and declared, never silent.** Any target-space transform applied during training MUST be (a) applied **inside** the model library, (b) **declared in configuration** (not hardcoded), and (c) **inverted before predictions leave the model**. The implementation mechanism is the declarative `TRANSFORMS` registry + `target_transform` config key specified in `docs/stories/target_transform_declarative.md`. Until that mechanism lands, the only compliant target transform is the identity — the post-`08ee2eb` behavior — and re-introducing a silent transform (as `5fcfe43` did) is a contract violation.

4. **The training-time declaration is the sole source of truth for scale.** The inverse step needs to know **only** which transform was declared *at training*. That declaration is serialized onto the pickled model instance (`_target_transform_name`, per the story) and is **authoritative over any ambient predict-time config** — a loaded artifact inverts with the transform it was trained under, not whatever the current config happens to say. There is no separate scale annotation to consult: not a column-name prefix, not a transform-history log. A declared forward transform *automatically implies* its inverse on output.

5. **Registry transforms must be monotonic and zero-preserving.** Every `(forward, inverse)` pair admitted to the `TRANSFORMS` registry MUST be strictly monotonic on the target domain and satisfy `forward(0) = 0` (and `inverse(0) = 0`). This is a correctness precondition for `HurdleModel`/`ShurfModel`, whose binary stage relies on `forward(x) > 0 ⇔ x > 0` (`hurdle_model.py:101-103`). `log1p`, `asinh`, and `identity` all satisfy it; a future non-monotonic or non-zero-preserving transform would silently corrupt the classification stage and MUST be rejected.

6. **stepshifter encodes no scale information in column names.** stepshifter neither reads nor writes scale claims in column names; the `lr_`/`ln_`/`lx_` prefix scheme plays no role in its scale logic. The only naming that remains is the target's identity name and the `pred_` prefix on outputs — vestigial under `PredictionFrame`, which carries no column names. (Platform-wide deprecation of the scheme is deferred to the `⟨PENDING⟩` pipeline-core ADR; this clause binds stepshifter only.)

7. **Sole ownership of inversion.** The model library is the **only** owner of target-space inversion. Neither views-pipeline-core nor views-evaluation will undo a stepshifter transform; a model that emits non-raw predictions and relies on a downstream consumer to invert them is non-compliant.

8. **Scope.** This contract governs the **target** only. Feature transforms are applied upstream at queryset level and are out of scope (consistent with the story and with `StepshifterModel`'s `len(targets) == 1` rule).

### Enforcement required at ratification (not deferred to the full mechanism)

The full `TRANSFORMS`/gate mechanism is future work, but two guards are cheap, require no registry, and MUST ship with this ADR so the contract is not unfalsifiable in CI in the interim:

- **E1 — Characterization test.** A test that fits a small `StepshifterModel` on a known-scale target and asserts predictions come back in raw space (e.g. not log-compressed into single digits). This pins the current post-`08ee2eb` behavior and turns a future silent `5fcfe43`-style regression red.
- **E2 — Load-time transform-name guard.** When unpickling an artifact, if `_target_transform_name` is absent (a pre-contract artifact), the model MUST fail loud or apply an explicit, logged default — never silently assume raw. This closes the frozen-artifact ambiguity (see Obligations).

### Relationship to ADR-001 (ReproducibilityGate)

When the declarative mechanism lands, `target_transform` becomes a `CORE_GENOME` key and `ReproducibilityGate.Config.audit_manifest()` validates it against the registry (including the clause-5 monotonic/zero-preserving check). At that point this contract is enforced at the configuration boundary, not merely documented. The gate change is separate from guards E1/E2, which ship now.

---

## Rationale

- **Single source of truth for scale.** With pipeline-core's undo blocks removed, the model library is the only place an inverse transform can live; the training-time config is the only place its identity is declared. Stating both explicitly closes the "who undoes the transform, and how do we know?" question that risk C-140 raised — and removes the need for any name- or log-based scale bookkeeping.
- **Ratifies the premise of `08ee2eb`.** The revert was correct in direction but rested on an unratified convention. This ADR supplies the local authority the revert assumed, making the repo's behavior defensible and auditable.
- **Kills a class of silent bug.** A hardcoded, undeclared transform (the `5fcfe43` pattern) is invisible at the config boundary and produces scale-mismatched output. Requiring declaration + internal inversion — and pinning it with guard E1 — makes such a change fail review and CI, not production.
- **Aligns with the platform direction.** Declarations over inference (ADR-003) and framework-agnostic frames (ADR-042 / `PredictionFrame`) both point the same way: scale is a declared property, not an inferred or name-encoded one.

---

## Considered Alternatives

1. **Keep the prefix scheme as the scale signal.** Rejected: a column name is intention, not truth (see Context); `DatasetTransformationModule`'s own failure modes (wrong-offset undo → silent wrong values; duplicate transform silently skipped) demonstrate that the prefix cannot be trusted, and `PredictionFrame` has no column names to carry it.
2. **Adopt a persisted transform-history log as the source of truth.** Rejected: it adds a second bookkeeping system that must be kept in sync with the actual math, re-introducing the same "did the rename match the operation?" drift. Making transforms model-internal with config-declared inverses removes the need for any external scale ledger.
3. **Leave inversion to pipeline-core (status quo ante).** Rejected: pipeline-core has already removed its `TEMPORARY` undo blocks and the plan designates inversion as the model repo's responsibility. Re-adding it centrally would re-couple every model's scale semantics to the core and re-open C-140.
4. **Defer the whole contract until the declarative `TRANSFORMS` mechanism is implemented.** Rejected: the contract is needed *now* to adjudicate the current incident and to bind any artifact retraining; guards E1/E2 give it teeth before the full mechanism. The mechanism is fuller enforcement, not the contract.

---

## Consequences

### Positive

- A locally binding, citable, CI-guarded rule for stepshifter model I/O scale, consistent with the platform's declarations-over-inference principle and frame migration.
- The silent-transform pattern (`5fcfe43`) is a documented contract violation that guard E1 turns red.
- Removes dependence on the `lr_`/`ln_`/`lx_` prefix relic, easing migration to `PredictionFrame` (and a future `FeatureFrame`), which cannot carry such prefixes.

### Obligations / Negative

- **Every views-models stepshifter config must declare its intended `target_transform`** once the gate key lands (default `"identity"`, matching pre-`5fcfe43` behavior). Until then, configs are assumed identity and any model relying on an undeclared transform is non-compliant.
- **Frozen artifacts predating this contract may silently violate it.** Artifacts trained in a non-raw space (e.g. during the `5fcfe43` window, 2025-11 → 2026-04) and now predicted under reverted/raw code emit mis-scaled output. Guard E2 makes loading them fail loud rather than silently mis-scale; non-compliant artifacts must be audited and retrained or assigned a declared inverse. This is the open thread of the 2026-06-08 investigation.
- A future `2.0.0` bump is implied when `target_transform` becomes a required `CORE_GENOME` key (breaking config contract), per the story's migration plan.

---

## Implementation Notes

- **Beyond guards E1/E2, no further code change is mandated by this ADR.** It ratifies the current (post-`08ee2eb`) raw-space behavior and the contract the declarative mechanism will fully enforce.
- The forward/inverse call sites are the same two locations `5fcfe43` touched and the story re-uses: `StepshifterModel._process_data` (forward, on the target column) and `StepshifterModel._predict_by_step` (inverse, on the prediction column). Per clause 4, the transform name is stored on the pickled instance so artifacts carry their own inverse.
- **ShurfModel inverse-composition (binding interim rule).** `ShurfModel.predict_sequence` applies its **own** `np.expm1`/`np.log1p` in sampling space (`shurf_model.py:156,180,212,214`), separate from the base-class `_predict_by_step` inverse. To avoid a double-inverse, the two transforms MUST NOT both fire. Until the story's `log_target → sample_in_log_space` split formally separates training-space inversion (registry) from sampling-space drawing (ShurfModel), **ShurfModel configs MUST set `target_transform="identity"`** (equivalently, the registry inverse is a no-op and ShurfModel's existing `log_target` logic is the only transform). A ShurfModel config combining a non-identity `target_transform` with `log_target=True` is non-compliant and MUST be gate-rejected when the key lands.

---

## Validation & Monitoring

- **Now (guards E1/E2):** the characterization test (E1) and the load-time transform-name guard (E2) ship with this ADR.
- **With the mechanism:** a round-trip test asserting `inverse(forward(x)) ≈ x` for each registered transform; an `identity` no-op test; a gate test rejecting an unknown/undeclared/non-monotonic `target_transform`; a ShurfModel test rejecting the `target_transform != identity` + `log_target=True` combination.
- **Failure signals:**
  - Predictions for a target compressed into single digits where raw counts are expected → a log-space leak (transform applied, not inverted).
  - Predictions astronomically large / `inf` → double-inversion blowup (inverse applied twice; see the ShurfModel rule above).
  - Ensemble or constituent MSLE **worse than the all-zeros baseline** on a zero-inflated target → strong indicator of a scale mismatch in a constituent.

---

## References

- **Commits:** `5fcfe43` ("log and unlog inside stepshifter", 2025-11-20); `08ee2eb` ("revert… target-space transforms must be declared explicitly", 2026-04-11).
- **Story:** `docs/stories/target_transform_declarative.md` (the mechanism this ADR's contract will be fully enforced by).
- **ADR-001:** `docs/ADRs/001_reproducibility_gate.md` (the gate that will validate `target_transform`).
- **views-pipeline-core ADR-003:** Authority of Declarations over Inference — semantics are declared in config, *not inferred from column names*; the governing principle for clauses 4 and 6.
- **views-pipeline-core ADR-042 + `CICs/PredictionFrame.md`:** PredictionFrame Adoption — framework-agnostic, column-name-free, scale-metadata-free transport; the structural reason the prefix scheme cannot survive.
- **views-pipeline-core `CICs/DatasetTransformationModule.md`:** the `lr_`/`ln_`/`lx_` prefix + in-memory transform-history machinery; documents the silent-corruption failure modes cited above.
- **views-pipeline-core plans:** `plans/2026-03-15_prediction_frame_two_track_status.md` §Item 3 ("models must return untransformed predictions"); `plans/2026-06-01_pfe_production_roadmap.md` §7 / risk **C-140**.
- **views-pipeline-core CIC:** `CICs/ForecastingModelManager.md` (governs return type/format, not scale).
- **views-hydranet ADR-046 / ADR-003 (Law 6):** the raw-count handoff and `lr_` prefix-purity invariant — cited as precedent; their platform-wide retirement is the `⟨PENDING⟩` pipeline-core ADR's to declare.
- **views-r2darts2 ADR-012:** Scaling Pipeline & Calibration Integrity (correctness of inverse transforms).
- **Investigation:** `reports/investigation_metrics_08062026/` (the 2026-06-04 `big_chungus` divergence that motivated this ADR).
