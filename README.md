# PorkyPig

PorkyPig makes a bounded decision each tick: build a cache from visible objects and memory, then try each override in priority order.

```text
cache = build_cache(game)
for name in OVERRIDE_ORDER:
    result = OVERRIDES[name](cache)
    if result:
        save direction and split flag
        return result
return previous direction
```

## Escape
Checks whether enemies can reach our blobs now or after splitting, including recent hidden enemies, then chooses the safest heading.

## Split
Finds valuable captures, rolls the split forward, and rejects attacks likely to hit viruses, miss the target, or get punished by another enemy.

## Virus
Remembers virus coordinates after they leave view, values nearby virus chains, and keeps farming the same safe target.

## Chase
Tracks enemy fragments, predicts prey movement, uses walls to trap them, avoids merging opponents, and holds one detour around blocking viruses.

## Food / Unstuck
Food collects clusters. Unstuck is a fallback.

Late-game rank logic takes more risk when trailing.
