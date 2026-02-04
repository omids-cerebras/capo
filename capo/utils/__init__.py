"""
CAPO Utility Subpackage.

This subpackage provides helper functions for text parsing and prompt
construction that support the CAPO reward system.

Modules
-------
parsing
    Functions for segmenting model solutions into discrete reasoning steps.
    The primary function `segment_solution_into_steps` provides a simple
    line-based segmentation strategy that can be replaced with more
    sophisticated parsers for specific domains.

prompts
    Templates and builders for GenPRM (Generative Process Reward Model)
    prompts. Currently provides placeholder functions that demonstrate
    the expected interface. Replace with your actual prompt templates
    for your specific LLM-as-judge setup.

Design Philosophy
-----------------
These utilities are intentionally kept separate from the core EB algorithms
and VERL integration to enable:

1. **Easy customization**: Replace parsing logic without touching core code.
2. **Independent testing**: Each utility can be tested in isolation.
3. **Domain adaptation**: Different datasets may need different parsers.

Example Usage
-------------
>>> from capo.utils.parsing import segment_solution_into_steps
>>> steps = segment_solution_into_steps("Step 1: x=2\\nStep 2: y=x+3")
>>> print(steps)
['Step 1: x=2', 'Step 2: y=x+3']

>>> from capo.utils.prompts import build_dummy_prompt
>>> messages = build_dummy_prompt("What is 2+2?", "4", "4")
>>> print(messages[0]["role"])
'system'
"""

from capo.utils.parsing import segment_solution_into_steps
from capo.utils.prompts import build_dummy_prompt

__all__ = ["parsing", "prompts", "segment_solution_into_steps", "build_dummy_prompt"]
