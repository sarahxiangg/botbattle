# PorkyPig (`bots/my_bot.py`)

On each move, the bot builds a cache of visible blobs, food, viruses, and previously seen enemies/viruses. Then executes main loop:

```text
for override in [Escape, Split, Virus, Chase, Unstuck, Food]:
    if override(cache) is safe:
        execute and return
```

### Overrides

**Escape**<br>
Estimates whether visible or recently seen enemies can reach our blob now or after splitting, then chooses safe direction.

**Split**<br>
Tests capture plans by rolling them forward and rejects attacks that miss, hit bad viruses, or feed another enemy.

**Virus**<br>
Remembers virus coordinates after they leave view, so the bot can revisit nearby viruses instead of reacting only to vision.

**Chase**<br>
Predicts enemy blob movement by a few ticks and uses walls to corner targets.

**Unstuck / Food**<br>
Unstuck stop glitching/stalling. Food favours dense clusters.

**Other**<br>
Bot takes more risk when behind (7th/8th) on leaderboard.
ML (`optimise.py`) used for parameter tuning (genetic algorithm).
