# PorkyPig (`bots/my_bot.py`)

On each move, the bot builds a cache of visible blobs, food, viruses, and previously remembered enemies and viruses.

```text
for override in [Escape, Split, Virus, Chase, Unstuck, Food]:
    if override finds a safe action:
        execute it
```

### Overrides

**Escape**

Estimates whether visible or recently seen enemies can reach our pieces now or after splitting, then chooses space that survives.

<br>

**Split**

The main snowball tool. It tests capture plans by rolling them forward and rejects attacks that miss, hit bad viruses, or feed another enemy.

<br>

**Virus**

Remembers virus coordinates after they leave view, so the bot can revisit farms and chain nearby viruses instead of reacting only to vision.

<br>

**Chase**

Predicts enemy blob movement by a few ticks and uses walls to corner targets.

<br>

**Unstuck / Food**

Unstuck breaks stalls. Food collects dense clusters before smooth roaming.

<br>

**Other**

Takes more risk when behind in 7th or 8th place on the leaderboard.

Machine learning through `optimise.py` was used for parameter tuning.
