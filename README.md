# PorkyPig

On each move, the bot builds a cache of visible blobs, food, viruses, and previously remembered enemies/viruses.

```text
for override in [Escape, Split, Virus, Chase, Unstuck, Food]:
    if override finds a safe action:
        execute it
```

### Overrides

** Escape **
Estimates whether visible or recently seen enemies can reach our pieces now or after splitting, then chooses space that survives.

** Split ** 
The main snowball tool. It tests capture plans by rolling them forward and rejects attacks that miss, hit bad viruses, or feed another enemy.

** Virus ** 
Remembers virus coordinates after they leave view, so the bot can revisit farms and chain nearby viruses instead of reacting only to vision.

** Chase ** 
Predicts enemy blob movement by a few ticks and uses walls to corner targets.

** Unstuck / Food ** 
Unstuck breaks stalls. Food collects dense clusters before smooth roaming.

** Other ** 
Takes more risk when behind (7th/8th) on leader-board.
ML (optimise.py) used for parameter tuning.

