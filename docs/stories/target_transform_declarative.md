# Story: Declarative Target Transform for views-stepshifter

**Status:** Proposed
**Date:** 2026-04-11
**Origin:** review-diff audit on `fix/darts04` (revert of commit `5fcfe43` "log and unlog inside stepshifter")
**Reference pattern:** `views_hydranet.utils.config_initializer.TRANSFORMS` — pattern parity only, no code import

---

## Problem

views-stepshifter has no config-level declaration of target-space transformations. On 2025-11-20, commit `5fcfe43` ("log and unlog inside stepshifter", author xiaolongsun) silently added `np.log1p` inside `StepshifterModel._process_data` and `np.expm1` inside `StepshifterModel._predict_by_step`, forcing every stepshifter model to train on log-space targets regardless of its config. The commit message provides no motivation — no model name, no ticket, no experiment reference.

The review-diff audit of `fix/darts04` (2026-04-11) flagged and reverted the change because it violates the project convention that *all model-side transformations must be explicitly declared in a config*. It also silently broke `ShurfModel`: the base class now inverse-transforms predictions before `ShurfModel.predict_sequence` sees them, which makes the `_log_target == True` path double-`expm1` a raw value.

views-hydranet solved the same problem with a `TRANSFORMS` registry (`views_hydranet/utils/config_initializer.py:16-20`) mapping string keys (`"log1p"`, `"asinh"`, `"identity"`) to `(forward, inverse)` callable pairs, plus a `transformations: Dict[str, List[str]]` config field validated against the registry. Unknown names fail loudly at config validation. Each column maps to exactly one transform; classification channels are skipped at inversion.

views-stepshifter has no equivalent. This story proposes adapting the pattern.

### Why not reuse `views_pipeline_core.modules.transformations`?

`views_pipeline_core` already exposes a `DatasetTransformationModule` with `ln_transform()` / `undo_ln_transform()` methods, but it is a **stateful dataset lifecycle manager**, not a named-transform registry. Its API is imperative (method calls) rather than declarative (config keys), and it doesn't expose a `Dict[str, tuple[Callable, Callable]]` for lookup by config string. Stepshifter could layer on top of it, but the coupling would be awkward and the core abstraction (named forward/inverse pairs) would still need to live in stepshifter.

### Why not import from views-hydranet?

Model packages in views-platform (stepshifter, hydranet, r2darts2, baseline) must not depend on or import from each other. Stepshifter gets pattern parity by implementing a local copy of the registry idea — three lines of code — not a shared dependency.

---

## Scope

### `TRANSFORMS` registry (stepshifter-local)

New module: `views_stepshifter/infrastructure/transforms.py`

```python
import numpy as np
from typing import Callable

TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "log1p": (np.log1p, np.expm1),
    "asinh": (np.arcsinh, np.sinh),
    "identity": (lambda x: x, lambda x: x),
}
```

Closed registry. Extending it requires a stepshifter code change — intentional, matches hydranet's design.

### `CORE_GENOME` extension

Add `"target_transform"` to `ReproducibilityGate.Config.CORE_GENOME`:

```python
CORE_GENOME = ["steps", "time_steps", "parameters", "target_transform"]
```

Extend `ReproducibilityGate.Config.audit_manifest` with a registry-membership check:

```python
from views_stepshifter.infrastructure.transforms import TRANSFORMS

target_transform = config["target_transform"]
if target_transform not in TRANSFORMS:
    available = list(TRANSFORMS.keys())
    raise MissingHyperparameterError(
        f"REPRODUCIBILITY CONTRACT VIOLATED: "
        f"Unknown target_transform '{target_transform}'. Available: {available}"
    )
```

This runs after the existing presence / None checks.

### Forward / inverse call sites in `StepshifterModel`

Store the transform name on the model instance in `__init__`:

```python
self._target_transform_name = config["target_transform"]
self._forward, self._inverse = TRANSFORMS[self._target_transform_name]
```

Apply in `_process_data`:

```python
df[self._targets] = self._forward(df[self._targets])
```

Apply in `_predict_by_step`:

```python
df_preds[f"pred_{self._targets}"] = self._inverse(df_preds[f"pred_{self._targets}"])
```

Both sites were the intended homes of commit `5fcfe43`'s silent additions; this story re-lands that placement behind an explicit, per-config flag.

### Pickle / serialization

Storing `self._target_transform_name` on the instance means pickled `StepshifterModel` artifacts carry the transform name with them and can inverse-transform predictions at load time without re-reading the config. This is a deliberate divergence from hydranet, which re-reads the config at inference. Stepshifter's divergence is strictly safer — it protects against config drift between training and prediction.

### `HurdleModel` / `ShurfModel` inheritance

Both inherit `StepshifterModel._process_data` and `StepshifterModel._predict_by_step`, so both get the transform for free. The classification stage in `HurdleModel` works on binarized targets (`(x > 0).astype(float)`), which is binarization-after-log-transform: `log1p(x) > 0 ⇔ x > 0`, so the behavior is preserved across transform choices.

### `ShurfModel._log_target` reconciliation

Today `ShurfModel` reads `config["log_target"]` (a boolean) at 5 sites inside `predict_sequence` (`shurf_model.py:23, 142, 165, 210, 214`). The flag conflates two concerns: whether the base class trains on log-space and whether the lognormal sampler draws around `x` or `log1p(x)`.

After this story, the training-space concern is owned entirely by `target_transform`. The sampling-space concern is orthogonal and remains ShurfModel's responsibility.

**Proposed cleanup:**

1. Rename `log_target` → `sample_in_log_space` in `ShurfModel` and in the `ReproducibilityGate.ALGORITHM_GENOMES["ShurfModel"]["config_keys"]` list.
2. Update the 6 downstream views-models ShurfModel configs (see "Migration plan" below).
3. Optionally: collapse the flag into `draw_dist` by introducing `"Lognormal-from-raw"` vs `"Lognormal-from-log"` as distinct distribution names. Defer this refinement — rename-only is sufficient for this story.

The rename is non-semantic; the current behavior of each of the 6 configs is preserved bit-for-bit.

---

## Acceptance Criteria

1. `views_stepshifter/infrastructure/transforms.py` exists with the `TRANSFORMS` registry (`log1p`, `asinh`, `identity`) as `dict[str, tuple[Callable, Callable]]`.
2. `"target_transform"` added to `ReproducibilityGate.Config.CORE_GENOME`.
3. `ReproducibilityGate.Config.audit_manifest` rejects unknown transform names with `MissingHyperparameterError` and includes the available list in the error message.
4. `StepshifterModel.__init__` stores `self._target_transform_name` and resolves `self._forward` / `self._inverse` from the registry.
5. `StepshifterModel._process_data` applies `self._forward` to the target column; `StepshifterModel._predict_by_step` applies `self._inverse` to the prediction column.
6. `ShurfModel` has `log_target` renamed to `sample_in_log_space` (5 read sites + 1 gate entry).
7. Unit tests added to `tests/`:
   - `test_transforms.py`: registry contains expected keys; each pair satisfies `inverse(forward(x)) ≈ x` for positive floats; `identity` is a true no-op.
   - `test_reproducibility_gate.py`: new test rejecting unknown `target_transform`; existing tests updated to include the key in all valid configs.
   - `test_stepshifter.py`: fit-predict round-trip produces raw-space predictions for each transform.
   - `test_shurf_model.py` (or extension): sampler behavior for the renamed flag preserved across `target_transform` choices.
8. All 20+ views-stepshifter configs in views-models updated to declare `target_transform` explicitly. The 6 ShurfModel configs also migrate `log_target` → `sample_in_log_space`.
9. views-stepshifter bumped to `2.0.0` — the gate change is a breaking config contract.
10. ADR-001 (ReproducibilityGate) updated (amendment) to note the new core key; or a new ADR-002 on the transform mechanism, whichever reviewers prefer.
11. CIC for `ReproducibilityGate` updated with the new responsibility (registry membership validation).
12. A new CIC written for `TRANSFORMS` / the transforms module.

---

## Migration plan

### Stepshifter package

1. Add `transforms.py`, update the gate, update the model classes, rename `ShurfModel` flag — all in one PR on a new branch (`feature/target_transform`).
2. Bump to `2.0.0` in `pyproject.toml`.
3. Tests green. Ship.

### views-models configs

Every stepshifter model config must declare `target_transform` before the stepshifter 2.0.0 bump lands. Survey of existing configs informs the default:

- **HurdleModel configs** (`blank_space`, `little_lies`, `lavender_haze`, `fluorescent_adolescent`, and siblings — ~15 models): today these query raw counts via the `lr_*` alias with `ops.ln()` commented out. They currently train on raw targets. → declare `"target_transform": "identity"`.
- **ShurfModel configs** (6 models):
  - `purple_haze`, `wild_rose`, `cheap_thrills`, `wuthering_heights`, `lovely_creature`: currently `log_target: False`. → declare `"target_transform": "identity"` and `"sample_in_log_space": False`.
  - `fourtieth_symphony`: currently `log_target: True`. → declare `"target_transform": "identity"` and `"sample_in_log_space": True`. (Rename-only, same behavior.)
- **Flat XGB/LGBM configs** (XGBRegressor, XGBRFRegressor, LGBMRegressor variants): today trained on raw targets. → declare `"target_transform": "identity"`.

**Exception:** whichever model(s) motivated commit `5fcfe43` — if the original author confirms any specific model was intended to train on log-space, those configs declare `"target_transform": "log1p"` instead. Without that confirmation, `"identity"` is the safe default because it matches the behavior on `main` before `5fcfe43` was written.

### Coordination

The gate-change PR in stepshifter cannot merge until the views-models config migration PR is ready. Recommended order:

1. Open stepshifter PR (`feature/target_transform`) as draft.
2. Open views-models PR adding `target_transform` to every stepshifter config, pointing at the stepshifter draft.
3. Merge stepshifter PR. Publish 2.0.0.
4. Merge views-models PR. Bump views-models' stepshifter dependency to `^2.0.0`.

---

## Notes / Divergences from hydranet

| Aspect | Hydranet | Stepshifter (this story) |
|---|---|---|
| Config shape | `transformations: Dict[str, List[str]]` (multi-target) | `target_transform: str` (single-target, per stepshifter's `len(targets)==1` rule) |
| Registry location | `views_hydranet.utils.config_initializer` | `views_stepshifter.infrastructure.transforms` (local copy — pattern parity, not shared code) |
| Inverse application site | `InferenceOrchestrator` post-predict | Inside `_predict_by_step` (the natural model-output boundary) |
| Serialization of transform | Re-read from config at inference | Stored on pickled model instance |
| Validation point | Pydantic `ConfigInitializer` | `ReproducibilityGate.audit_manifest` |
| Binary-channel skipping | Automatic (`by_` prefix) | Not needed — HurdleModel binarizes after transform, and `log1p(x) > 0 ⇔ x > 0` |

These divergences are deliberate. The config shape reflects stepshifter's single-target constraint; the serialization choice protects pickled artifacts from config drift; the inverse-site choice keeps `model.predict()` a raw-space boundary.

---

## Out of scope

- **Sharing the `TRANSFORMS` registry across model packages.** Per the package-isolation rule, stepshifter/hydranet/r2darts2/baseline may not import from each other. If a shared `views-model-utils` library is ever created, that's the right time to factor; until then, each package keeps its own local copy.
- **Value-range validation.** `log1p` requires `x ≥ -1` and is typically used on non-negative counts; `asinh` accepts any real. The gate currently validates key presence and non-None, not value ranges. Range validation belongs in a future `ReproducibilityGate.Data` sub-gate and is not part of this story.
- **Multi-target stepshifter support.** `StepshifterModel.__init__` still enforces `len(targets) == 1`. If that ever changes, the config key shape grows to `Dict[str, List[str]]` (hydranet-style) in a later migration.
- **Transforms on features, not just targets.** Hydranet transforms feature columns too. Stepshifter's features are pre-processed at queryset level; feature transforms at the model layer are not needed and are explicitly out of scope.
- **Collapsing `draw_dist` and `sample_in_log_space` in ShurfModel.** Rename-only for this story; deeper cleanup is a separate ShurfModel refactor.
- **Backfilling the question of what `5fcfe43` intended.** Ideally the original author confirms before the migration PR lands; if not, `"identity"` is the default.
