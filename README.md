# PorkyPig

```text
observe visible state
update enemy, fragment, and virus memory

for policy in [escape, split, virus, chase, unstuck, food]:
    action = policy()
    if action is valid and safe:
        execute action
        stop

continue previous safe direction
```

**Escape** models direct and multi-split kill ranges, remembered threats, walls, and viruses that can shield us from pursuers.

**Split** first uses cheap geometric reach tests, then runs bounded engine-style simulations of movement, decay, merging, virus collisions, and captures.

**Virus** maintains a persistent coordinate map, allowing the bot to remember farming locations and estimate the value of nearby virus chains after they leave vision.

**Chase** tracks individual fragment velocity, predicts interception points, uses walls to trap prey, avoids opponents about to merge, and preserves one detour direction when a virus blocks the path.

**Unstuck** resolves genuine low-movement states, while **Food** targets dense clusters and otherwise roams smoothly.

Late-game rank awareness increases aggression only when trailing. Expensive calculations are capped, cached, and limited to promising targets.
