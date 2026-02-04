# CAPO (LaTeX)

This project is organized as a standard multi-file LaTeX manuscript.

## Entry point
- `capo.tex` — main file (preamble + document structure)

## Structure
- `sections/` — paper sections (included from `capo.tex`)
- `figures/` — figures referenced by the manuscript
- `reference/ref.bib` — bibliography database

## Custom style
- `cs.cls` — Cerebras Systems preprint class
- `iclr2026_conference.bst` — bibliography style

## Notes
- `math_commands.tex` contains project-wide macro definitions and is included
  from the main preamble.
