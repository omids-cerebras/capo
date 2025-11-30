"""
Top-level package for CAPO + Empirical Bayes extensions for VERL.

This package is intentionally small. It exposes a single submodule
`capo.verl_integration` that contains all VERL-specific glue code.

To ensure your CAPO components are registered with VERL's internal
registries, import the integration module at program start:

    import capo.verl_integration  # noqa: F401

This triggers side-effects via registration decorators (for both the
CAPO reward manager and the Empirical Bayes advantage estimator).
"""

__all__ = ["verl_integration"]
