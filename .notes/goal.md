# Project Goal: Pupper Leg-Raise Transition Animation

## Overview

Build an animation/action for **Pupper** (a quadruped research robot) that, for
each leg:

1. **Raises** the leg into the air.
2. **Holds** it raised for a **set, configurable amount of time**.
3. **Lowers** it back down.

Legs are lifted **one at a time** so the robot stays stable on its three
remaining legs. The sequence cycles through all four legs.

## Why (Research Context)

Our research explores **smart materials as robotic limbs**. Pupper's legs use a
**smart polymer** as the final link of the end effector, which lets the robot
function with both **legs and wheels**.

The smart polymer **changes shape when heated**. To apply heat and let the
material transition safely, the affected leg must be **off the ground**. This
leg-raise animation provides that window: raise the leg, keep it up long enough
for heat to be applied and the shape to change, then lower it.

> **Scope note:** This project is only responsible for the **mechanical motion**
> (raise / hold / lower). It does **not** apply heat, monitor the polymer, or
> detect when a transition has finished. The hold simply lasts for the
> configured duration.

## Functional Requirements

- Lift legs **one at a time** (sequential, not simultaneous) to keep the robot
  stable on its three remaining legs.
- **Hold duration is configurable** and can be changed easily — the leg stays
  raised for exactly that duration before lowering.
- Maintain **balance and stability** while any single leg is lifted.
- **Smooth raise/lower motion** to avoid stressing the smart-polymer link.

## Decisions

- **Leg order doesn't matter** — choose whatever order is easiest to implement
  or keeps the robot most stable.

## Open Questions / To Be Determined

- Target raise height / joint angles per leg.
- Whether to re-stabilize the body (shift weight) before lifting each leg.
