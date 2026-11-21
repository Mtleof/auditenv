"""auditenv: a reproducible RL environment for an AI auditor of synthetic financial records."""

ENV_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"

FINDING_TYPES = (
    "duplicate_invoice",
    "missing_or_partial_allocation",
    "wrong_fx_date_or_rate",
    "period_misattribution",
)

__all__ = ["ENV_VERSION", "SCHEMA_VERSION", "GENERATOR_VERSION", "FINDING_TYPES"]
