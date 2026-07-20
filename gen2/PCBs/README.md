# Gen 2 PCBs

KiCad board designs for the gen 2 vehicle. Not every board here is in active
use — some are older revisions, and some are experimental.

## Currently in use

- **`MLU_BreakoutV2/`** — Main Logic Unit breakout, version 2. Routes I/O
  between the main compute/Teensy and the vehicle's sensors, actuators, and
  peripherals. This is the breakout currently flying.
- **`Power_Distribution/`** — takes battery input and distributes it to the
  vehicle's power rails. Currently in use.

## Older / superseded

- **`MLU_Breakout/`** — version 1 of the Main Logic Unit breakout, superseded
  by `MLU_BreakoutV2/`.
- **`Power_Management/`** and **`Power_ManagementV2/`** — power-management
  boards. **Neither works well** — both have heat-dissipation problems, so they
  are not used. (The gen 3 power-management work in
  [`../../gen3/PCBs/Battery_Boards`](../../gen3/PCBs/Battery_Boards) is the
  follow-on attempt to fix this.)

## Experimental

- **`Battery_hot_swap/`** and **`Battery_hot_swap_top/`** — experimental
  battery hot-swap modules (swap a battery without powering the vehicle down).
  Not yet proven / not in regular use.
