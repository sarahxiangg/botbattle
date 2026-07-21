# PorkyPig (`bots/my_bot.py`)

On each move, the bot builds a cache of visible blobs, food, viruses, and previously remembered enemies and viruses.

```text
for override in [Escape, Split, Virus, Chase, Unstuck, Food]:
    if override finds a safe action:
        execute it
```

### Overrides

**Escape**

Estimates whether visible or recently seen enemies can reach our pieces immediately or after splitting, then chooses the space with the highest chance of survival.

**Split**

The main snowball tool. It tests capture plans by rolling them forward and rejects attacks that would miss, hit dangerous viruses, or feed another enemy.

**Virus**

Remembers virus coordinates after they leave view, allowing the bot to revisit farms and chain nearby viruses instead of reacting only to currently visible targets.

**Chase**

Predicts enemy blob movement several ticks ahead and uses walls to corner targets.

**Unstuck / Food**

Unstuck breaks stalls. Food targets dense clusters before transitioning to smooth roaming.

**Other**

Takes greater risks when ranked 7th or 8th on the leaderboard.

Machine learning through `optimise.py` was used for parameter tuning.
