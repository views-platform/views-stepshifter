"""
Declared target-space transform registry for views-stepshifter.

A **closed** registry of stateless ``(forward, inverse)`` callable pairs. The model
library applies the declared *forward* to the target **once** during ``fit`` (in
``StepshifterModel._process_data``) and the matching *inverse* **once** at the
``predict()`` output boundary, per ADR-003. Scale is governed by the config-declared
``target_transform`` key — never by a column name, never scattered in between.

Every transform here is **strictly monotonic** and **zero-preserving** (``forward(0) == 0``),
which keeps the missing-combination zero-fill and ``HurdleModel``'s ``(x > 0)`` binarization
consistent under the transform.

Pattern parity with ``views_hydranet.utils.config_initializer.TRANSFORMS`` — a local copy,
**not** an import (the model-package isolation rule forbids cross-package dependencies).

Adding a transform is a deliberate code change (the registry is closed); the
``ReproducibilityGate`` validates that a config's ``target_transform`` is a member.
"""
from typing import Callable

import numpy as np

TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "identity": (lambda x: x, lambda x: x),
    "log1p": (np.log1p, np.expm1),
    "asinh": (np.arcsinh, np.sinh),
}
