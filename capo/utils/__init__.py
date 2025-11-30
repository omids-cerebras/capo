"""
Utility subpackage for CAPO + VERL integration.

Currently contains:

- `parsing` – helpers for segmenting solutions into steps.
- `prompts` – helpers for building GenPRM prompts.

These modules are deliberately kept lightweight to avoid polluting the
core VERL integration with dataset- or prompt-specific logic.
"""

__all__ = ["parsing", "prompts"]
