class ReproducibilityError(Exception):
    """Base class for all reproducibility gate failures."""

    pass


class MissingHyperparameterError(ReproducibilityError):
    """Raised when a mandatory hyperparameter is missing from the config."""

    pass


class PreContractArtifactError(ReproducibilityError):
    """Raised when a loaded model artifact predates the target-transform contract.

    Per ADR-003 E2 (raw-target-space I/O contract): an artifact missing the
    ``_target_transform_name`` stamp has an unknown prediction scale that cannot
    be safely inverted, so it must be rejected loudly at load time rather than
    silently assumed to be raw space.
    """

    pass
