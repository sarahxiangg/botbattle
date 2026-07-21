# PorkyPig

On each move, the bot builds a cache of visible blobs, food and viruses, plus remembered enemy/virus state.

```text
for override in [Escape, Split, Virus, Chase, Unstuck, Food]:
    if override finds a safe action:
        execute it
```

## Overrides

### Escape
Safety layer. It estimates whether visible or recently seen enemies can reach our pieces now or after splitting, then chooses space that survives.

### Split
The main snowball tool. It tests capture plans by rolling them forward and rejects attacks that miss, hit bad viruses, or feed another enemy.

### Virus
Remembers virus coordinates after they leave view, so the bot can revisit farms and chain nearby viruses instead of reacting only to vision.

### Chase
Predicts fragment movement, uses walls to finish targets, and avoids merging opponents.

### Unstuck / Food
Unstuck breaks stalls. Food collects dense clusters before smooth roaming.

Late-game rank logic takes more risk when behind.
