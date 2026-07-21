# PorkyPig

PorkyPig is a priority-based agent: each tick it updates its world model, then runs policies until one returns a safe action.

```text
update memory
for policy in [Escape, Split, Virus, Chase, Unstuck, Food]:
    if safe action exists: execute it
continue previous heading
```

## Escape
Builds direct and multi-split threat envelopes from visible and remembered enemies, walls, and virus shields, then selects the safest direction.

## Split
Applies cheap geometric filters before bounded engine-style rollouts simulating movement, decay, merging, virus contact, captures, and post-split risk.

## Virus
Stores virus coordinates beyond visibility, values nearby virus chains, and follows safe farming routes while preserving target continuity.

## Chase
Tracks individual enemy fragments, estimates velocity, predicts interceptions, uses walls to trap prey, avoids merge traps, and prevents oscillation around viruses.

## Unstuck and Food
Unstuck handles genuine low movement. Food targets dense clusters and otherwise roams smoothly.

Late-game rank awareness increases aggression only when trailing.
