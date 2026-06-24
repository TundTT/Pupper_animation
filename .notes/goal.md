# Project Goal: Pupper Leg-Lift for Material Transition

## Overview

Give **Pupper** (a quadruped research robot) the ability to, for a given leg:

1. **Raise** the leg into the air.
2. **Hold** it raised for long enough to apply heat.
3. **Lower** it back down.

A leg is lifted **one at a time** so the robot stays stable on its three
remaining legs. Repeating the behavior across legs covers the whole robot.

This is realized as a **learned reinforcement-learning policy**, in the same
spirit as Pupper's existing locomotion policies — a trained skill the robot can
run, not a scripted or hand-keyframed animation.

## Why (Research Context)

Our research explores **smart materials as robotic limbs**. Pupper's legs use a
**smart polymer** as the final link of the end effector, which lets the robot
function with both **legs and wheels**.

The smart polymer **changes shape when heated**. To apply heat and let the
material transition safely, the affected leg must be **off the ground**. The
leg-lift provides that window: raise the leg, keep it up while heat is applied
and the shape changes, then lower it.

> **Scope note:** This project is only responsible for the **mechanical motion**
> (raise / hold / lower). It does **not** apply heat, monitor the polymer, or
> detect when a transition has finished — the hold simply lasts a set duration.

## Functional Requirements

- Lift legs **one at a time** so the robot stays balanced and stable on its
  three remaining legs throughout the motion.
- Keep the leg raised long enough for the material transition to happen.
- **Smooth raise/lower motion** to avoid stressing the smart-polymer link.

## Approach Decisions

- **A standalone learned policy**, separate from the robot's locomotion behavior.
- **Invoked on demand** by the operator when a leg needs a transition.
- **Trained for a fixed hold duration**; a different duration means retraining
  rather than runtime configuration.
