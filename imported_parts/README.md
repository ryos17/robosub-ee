# Robosub Shared PCB Parts Library

This folder is the shared custom parts library for the project.  
`Micro_Breakout` and future boards should reference these libraries instead of keeping per-project duplicates.

## Directory overview

- `robosub_symbols/`
  - Main project symbol library: `robosub_symbols.kicad_sym`
  - Contains consolidated custom symbols used by project schematics.

- `robosub_footprints.pretty/`
  - Main project footprint library.
  - KiCad nickname should be `robosub_footprints`.

- `robosub_3d_models/`
  - Shared 3D models used by custom footprints.
  - Footprints should reference this through `${ROBOSUB_3D_MODELS}`.

- `robosub_jlc_parts_log.csv`
  - Import log for JLC part ingestion.
  - Used by tooling to prevent accidental duplicate imports.

- `jlc_to_robosub.py`
  - Helper script for importing JLC/EasyEDA parts into this shared library structure.

## How Micro_Breakout should be configured

- In `gen3/PCBs/Micro_Breakout/sym-lib-table`, point to:
  - `imported_parts/robosub_symbols/robosub_symbols.kicad_sym`

- In `gen3/PCBs/Micro_Breakout/fp-lib-table`, include:
  - `robosub_footprints` -> `imported_parts/robosub_footprints.pretty`

- In KiCad path variables, set:
  - `ROBOSUB_3D_MODELS` -> absolute path to `imported_parts/robosub_3d_models`

## Workflow note

When adding a new custom part:
1. Add/merge symbol into `robosub_symbols.kicad_sym`.
2. Add footprint into `robosub_footprints.pretty`.
3. Add 3D model into `robosub_3d_models` and reference it via `${ROBOSUB_3D_MODELS}`.
4. Reopen or refresh libraries in KiCad.
