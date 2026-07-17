from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np
import json
import os
from pathlib import Path


# =========================
# Constants
# =========================

# Number of evenly spaced movement directions evaluated each tick.
NUM_DIRECTIONS = 16
# Precomputed unit vectors for the full movement-direction set.
DIRECTIONS = np.array([
    [np.cos(2 * np.pi * i / NUM_DIRECTIONS),
     np.sin(2 * np.pi * i / NUM_DIRECTIONS)]
    for i in range(NUM_DIRECTIONS)
], dtype=float)

# Used only when already split. Chase is disabled then, so 8-way movement is
# enough and cuts repeated safety scoring roughly in half.
# Reduced 8-direction set used while split to lower CPU cost.
DIRECTIONS_FAST = DIRECTIONS[::2]

# Width and height of the square arena in game-coordinate units.
ARENA_SIZE = 60.0
# Engine maximum number of simultaneous blobs controlled by one player.
MAX_BLOB_COUNT = 16
# Effectively infinite negative score used to reject impossible or unsafe moves.
OFF_MAP_PENALTY = 1_000_000_000.0

# Performance caps
# Maximum nearby food items retained in each frame cache.
MAX_FOOD_CONSIDERED = 28
# Maximum nearby enemy blobs retained in each frame cache.
MAX_BLOBS_CONSIDERED = 22
# Maximum nearby viruses retained in each frame cache.
MAX_VIRUSES_CONSIDERED = 12
# Maximum own pieces used in repeated safety calculations.
MAX_OWN_BLOBS_CONSIDERED = 3


# Movement
# Prediction step length as a multiple of the current player radius.
STEP_DISTANCE_MULT = 1.5
# Bonus for continuing in the previous direction instead of turning sharply.
TURN_WEIGHT = 41.07
# Persistent unit vector storing the bot's most recently chosen direction.
LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)

# Food target locking / clustering
# Radius-scaled distance at which a food target is treated as reached.
FOOD_REACHED_DIST_MULT = 0.40
# Minimum absolute distance for declaring a food target reached.
FOOD_REACHED_DIST_MIN = 0.35
# Maximum ticks spent following one food target before abandoning it.
FOOD_TARGET_MAX_TICKS = 10
# Consecutive non-improving ticks allowed before abandoning a food target.
FOOD_NO_PROGRESS_LIMIT = 4
# Ticks for which an abandoned food target remains ignored.
FOOD_BLACKLIST_TICKS = 25
# Fixed minimum wall distance used to reject food in geometric corners.
FOOD_CORNER_MARGIN = 0.45
# Radius multiplier used to expand the corner exclusion zone for large blobs.
FOOD_CORNER_RADIUS_MULT = 1.6  # corner exclusion also scales with own size

# Corner/dead-end awareness. Danger/virus scoring alone doesn't discourage
# heading into a pocket where two walls meet, so most overrides that read
# report["score"] never learn to avoid it until they're already stuck there.
# Radius-scaled distance over which corner pressure begins to matter.
CORNER_AWARE_RADIUS_MULT = 6.0
# Penalty applied to candidate moves that enter boxed-in corner space.
CORNER_PRESSURE_WEIGHT = 2200.0
# Weight given to corner avoidance while selecting an emergency escape.
CORNER_ESCAPE_WEIGHT = 0.6
# Spatial scale used to decide whether nearby food belongs to one cluster.
FOOD_CLUSTER_RADIUS = 4.0
# Strength of the bonus for targeting dense food clusters.
FOOD_CLUSTER_WEIGHT = 0.90
# Exponent controlling how strongly food score favours nearby targets.
FOOD_DISTANCE_POWER = 1.53
# Minimum food distance considered valid as a target.
FOOD_MIN_TARGET_DIST = 0.15
# Overall scale used when scoring food around a predicted position.
FOOD_SCORE_WEIGHT = 900.0

# Roam / unstuck
# Ticks for which a chosen roaming direction remains locked.
ROAM_LOCK_TICKS = 6
# Bonus for roaming toward the arena centre.
ROAM_CENTER_WEIGHT = 350.0
# Bonus for candidate roam moves that achieve more actual displacement.
ROAM_MOVE_WEIGHT = 1000.0
# Bonus for preserving the previous direction during roaming.
ROAM_STICKINESS_WEIGHT = 250.0
# Extra clearance maintained between a blob edge and the arena wall.
WALL_BUFFER = 0.25
# Consecutive low-movement ticks required before unstuck behaviour activates.
STUCK_TICK_LIMIT = 4
# Radius-scaled movement threshold below which the bot counts as stationary.
STUCK_MOVE_EPS_MULT = 0.04
# Bonus for moving toward the centre while escaping a stuck state.
STUCK_CENTER_WEIGHT = 600.0
# Bonus for maximising displacement during unstuck movement.
STUCK_MOVE_WEIGHT = 2000.0

# Enemy danger
# Radius-scaled range within which larger enemies trigger escape mode.
DANGER_OVERRIDE_RANGE_MULT = 5.96
# Safety ratio for treating an enemy as directly large enough to threaten us.
DANGER_DIRECT_RATIO = 1.06
# Safety ratio for treating an enemy split piece as large enough to eat us.
DANGER_SPLIT_RATIO = 1.08
# Enemy-radius-scaled range used to detect possible enemy split attacks.
DANGER_SPLIT_RANGE_MULT = 5.0
# Overall penalty scale for proximity to dangerous enemies.
DANGER_WEIGHT = 120000.0
# Exponent controlling how rapidly danger increases at close range.
DANGER_DISTANCE_POWER = 1.2
# Own-radius-scaled distance that marks direct enemy contact as unsafe.
DANGER_HARD_CLOSE_MULT = 2.5
# Penalty for escape directions that produce little movement at a wall.
DANGER_WALL_PUSH_PENALTY = 50000.0
# Reward for actual displacement while escaping danger.
DANGER_MOVE_REWARD = 5000.0

# Enemy scoring / chasing
# Engine-style minimum eater-to-target mass ratio used for blob eligibility.
EAT_RATIO = 1.12
# Penalty scale for larger enemies in positional scoring.
ENEMY_DANGER_WEIGHT = 14000.0
# Reward scale for edible enemies in positional scoring.
ENEMY_HUNT_WEIGHT = 2500.0
# Own-radius-scaled maximum range for beginning a chase.
CHASE_RANGE_MULT = 6.50
# Larger range allowed after committing to an existing chase target.
CHASE_RETENTION_RANGE_MULT = 12.0
# Ticks for which the current enemy player remains the preferred chase target.
CHASE_LOCK_TICKS = 50
# Unproductive chase ticks allowed before dropping the target.
CHASE_LOST_LIMIT = 24
# Maximum number of velocity-prediction ticks used to lead a moving enemy.
CHASE_LEAD_TICKS = 3.48
# Minimum acceptable distance improvement before a chase counts as stalled.
CHASE_MIN_CLOSING_RATE = -0.10
# Maximum sibling edge gap for treating an enemy remerge as imminent.
CHASE_REMERGE_GAP_MULT = 1.5
# Distance from a wall at which direct finishing behaviour is preferred.
CHASE_WALL_DIST = 8.0
# Reward for reducing distance to the selected chase target.
CHASE_CLOSE_WEIGHT = 24.0
# Reward for choosing a larger edible target.
CHASE_SIZE_WEIGHT = 5.0
# Bonus for preserving movement direction during a chase.
CHASE_STICKINESS_WEIGHT = 8.0
# Penalty for remaining far from the calculated interception point.
CHASE_BLOCK_DIST_WEIGHT = 8.0
# Reward for reaching the centre-side of an enemy to cut off escape.
CHASE_CENTER_SIDE_WEIGHT = 42.0
# Own-radius-scaled edge distance considered close enough for a direct finish.
CHASE_DIRECT_EDGE_MULT = 2.8

# Virus safety / farming
# Penalty scale for movement paths that approach dangerous viruses.
VIRUS_DANGER_WEIGHT = 12000.0
# Own-radius-scaled clearance required from a dangerous virus path.
VIRUS_PATH_BUFFER_MULT = 0.65
# Size ratio required before the bot considers a virus farmable.
VIRUS_FARM_EAT_RATIO = 1.10

# Once we are already very large, virus farming has poor risk/reward: popping
# gives opponents many vulnerable pieces and can throw away a winning lead.
# Despite its name, the current code treats this as the maximum mass for virus
# farming. A high threshold deliberately allows late-game fragmentation and
# rare 80-100+ mass snowballs instead of protecting every existing lead.
VIRUS_FARM_MAX_RADIUS = 120.00

# Radius-scaled enemy scan range used before committing to virus farming.
VIRUS_FARM_ENEMY_RANGE_MULT = 5.5
# Estimated radius of vulnerable pieces created by a virus pop.
VIRUS_FARM_PIECE_RADIUS_MULT = 0.35
# Radius-scaled distance for expiring an unseen remembered virus.
VIRUS_MEMORY_REACHED_DIST_MULT = 1.4
# Decimal precision used to create stable coordinate keys for viruses.
VIRUS_MEMORY_KEY_DECIMALS = 1
# Ticks for which the current virus target remains preferred.
VIRUS_LOCK_TICKS = 18
# Required relative distance improvement before switching virus targets.
VIRUS_SWITCH_DIST_MULT = 0.65
# Maximum number of remembered virus locations.
VIRUS_MEMORY_MAX = 30
# Maximum remembered or visible viruses evaluated per tick.
VIRUS_MAX_CANDIDATES = 8

# Intentional virus split: when close enough, split first so only the launched
# half contacts the virus instead of driving the entire main blob into it.
# Size ratio required for a launched half to intentionally pop a virus.
VIRUS_SPLIT_EAT_RATIO = 1.10
# Conservative multiplier applied to intentional virus-split reach.
VIRUS_SPLIT_RANGE_SAFETY_MULT = 0.78
# Minimum extra radius-scaled reach required before virus splitting.
VIRUS_SPLIT_REACH_MARGIN_MULT = 0.08
# Clearance multiplier used to avoid hitting additional viruses during a split.
VIRUS_SPLIT_OTHER_VIRUS_BUFFER_MULT = 1.15

# Split kill
# Minimum launcher radius allowed to perform an offensive split.
SPLIT_MIN_RADIUS = 2.0
# Engine-style minimum launched-piece-to-target mass ratio for split kills.
SPLIT_EAT_RATIO = 1.10
# Distance travelled by a split piece as a multiple of launcher radius.
SPLIT_RANGE_MULT = 2.5
# Conservative multiplier applied to estimated split travel distance.
SPLIT_RANGE_SAFETY_MULT = 0.71
# Minimum target radius relative to launcher radius worth splitting for.
SPLIT_TARGET_MIN_RADIUS_MULT = 0.14
# Clearance multiplier required from viruses along a normal split path.
SPLIT_VIRUS_BUFFER_MULT = 1.3
# Launcher-radius-scaled scan range for threats near the split landing point.
SPLIT_POST_DANGER_RANGE_MULT = 5.5
# Reserved split-hit safety buffer retained for tuning compatibility.
SPLIT_HIT_BUFFER = 0.75
# Maximum own-piece count at which normal offensive splitting is allowed.
# A split command can split every eligible piece, so allowing a split at eight
# pieces can produce a full aggressive swarm when the opportunity is valuable.
SPLIT_MAX_OWN_BLOBS = 8
# Total-mass threshold at which stricter large-blob split limits apply.
SPLIT_HUGE_RADIUS = 40.0
# Maximum own-piece count allowed to split when already extremely large.
SPLIT_HUGE_MAX_OWN_BLOBS = 8
# Minimum radius-scaled reach margin required before approving a split.
SPLIT_REACH_MARGIN_MULT = 0.12
# Safety ratio used to identify enemies that threaten a launched piece.
SPLIT_LANDING_DANGER_RATIO = 1.08
# Maximum edible enemy targets considered during split planning.
SPLIT_MAX_TARGETS = 7
# Maximum own pieces considered as possible split launchers.
SPLIT_MAX_LAUNCHERS = 3
# Maximum expensive exact safety checks performed on top split candidates.
SPLIT_TOP_EXACT_CHECKS = 2
# Maximum cheap split candidates retained before exact validation.
SPLIT_PRECHECK_LIMIT = 10
# Minimum victim mass relative to its launcher before an offensive split is
# worth the extra fragmentation risk.
SPLIT_MIN_TARGET_MASS_FRACTION = 0.035
# Extra victim-mass fraction required for every piece we already control.
SPLIT_EXISTING_PIECE_MASS_PENALTY = 0.005
# Maximum fraction of eligible own mass allowed to be exposed by secondary
# pieces when one global split command is issued.
SPLIT_GLOBAL_RISK_FRACTION = 0.30

# Split-state hunting uses the largest own piece as the active predator.
SPLIT_CHASE_RANGE_MULT = 7.5
SPLIT_CHASE_RETENTION_RANGE_MULT = 12.0
# Only begin a planned two-command split while the resulting blob count can
# remain within the 16-blob engine cap.
DOUBLE_SPLIT_MAX_START_BLOBS = 4
# Ticks allowed to complete the second command after initiating a double split.
DOUBLE_SPLIT_PENDING_LIMIT = 3
# Slightly conservative multiplier for the combined two-launch travel estimate.
DOUBLE_SPLIT_RANGE_SAFETY_MULT = 0.74



# =========================
# Global memory
# =========================

# Coordinate key of the currently tracked food target.
FOOD_TARGET_KEY = None
# Ticks spent pursuing the current food target.
FOOD_TARGET_TICKS = 0
# Consecutive ticks without meaningful progress toward the food target.
FOOD_TARGET_NO_PROGRESS = 0
# Previous measured distance to the current food target.
FOOD_TARGET_LAST_DIST = None
# Map from abandoned food keys to remaining blacklist duration.
FOOD_BLACKLIST = {}

# Currently locked roaming direction, or None when no roam is locked.
ROAM_DIRECTION = None
# Ticks spent following the current roaming direction.
ROAM_TICKS = 0

# Player position recorded on the previous tick for stuck detection.
LAST_POSITION = None
# Consecutive ticks during which the player barely moved.
STUCK_TICKS = 0

# Previous enemy positions used to estimate per-blob velocity.
ENEMY_LAST_POS = {}
# Player ID of the currently locked chase target.
CHASE_TARGET_KEY = None
# Ticks spent chasing the currently locked enemy player.
CHASE_TARGET_TICKS = 0
# Consecutive ticks for which the chase failed its progress test.
CHASE_LOST_TICKS = 0

# Remembered virus locations and radii keyed by rounded coordinates.
VIRUS_MEMORY = {}
# Coordinate key of the currently locked virus target.
VIRUS_TARGET_KEY = None
# Ticks spent pursuing the currently locked virus target.
VIRUS_TARGET_TICKS = 0

# Player and last position of a target for which the first half of a planned
# two-command double split has already been launched.
DOUBLE_SPLIT_TARGET_PID = None
DOUBLE_SPLIT_TARGET_POS = None
DOUBLE_SPLIT_PENDING_TICKS = 0


#tunable constants
# Mapping of externally configurable parameter names to their built-in defaults.
TUNABLE_DEFAULTS = {
    "TURN_WEIGHT": TURN_WEIGHT,
    "FOOD_CLUSTER_WEIGHT": FOOD_CLUSTER_WEIGHT,
    "FOOD_DISTANCE_POWER": FOOD_DISTANCE_POWER,
    "CHASE_RANGE_MULT": CHASE_RANGE_MULT,
    "CHASE_LEAD_TICKS": CHASE_LEAD_TICKS,
    "CHASE_MIN_CLOSING_RATE": CHASE_MIN_CLOSING_RATE,
    "CHASE_CLOSE_WEIGHT": CHASE_CLOSE_WEIGHT,
    "CHASE_BLOCK_DIST_WEIGHT": CHASE_BLOCK_DIST_WEIGHT,
    "CHASE_CENTER_SIDE_WEIGHT": CHASE_CENTER_SIDE_WEIGHT,
    "SPLIT_RANGE_SAFETY_MULT": SPLIT_RANGE_SAFETY_MULT,
    "VIRUS_FARM_MAX_RADIUS": VIRUS_FARM_MAX_RADIUS,
    "DANGER_OVERRIDE_RANGE_MULT": DANGER_OVERRIDE_RANGE_MULT,
    "DANGER_DIRECT_RATIO": DANGER_DIRECT_RATIO,
    "SPLIT_TARGET_MIN_RADIUS_MULT": SPLIT_TARGET_MIN_RADIUS_MULT
}


def load_tuning_config() -> None:
    """Load optional tuning values from the JSON file named by BOT_CONFIG."""
    config_path = os.environ.get("BOT_CONFIG")

    if not config_path:
        return  # Use normal constants.

    values = json.loads(Path(config_path).read_text(encoding="utf-8"))

    unknown = set(values) - set(TUNABLE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown tuning parameters: {sorted(unknown)}")

    for name, value in values.items():
        default = TUNABLE_DEFAULTS[name]
        globals()[name] = type(default)(value)

# =========================
# Basic helpers
# =========================

def vec_norm(vec):
    """Return a unit-length copy of a vector, or None when its length is effectively zero."""
    norm = np.linalg.norm(vec)
    if norm <= 1e-9:
        return None
    return vec / norm

def can_eat_blob_by_radius(eater_radius, target_radius, eat_size_ratio):
    """Apply the engine mass-ratio test using blob radii.

    Blob mass is radius squared, so this is equivalent to:
    eater.mass >= target.mass * eat_size_ratio. Scalars and NumPy arrays
    are both supported.
    """
    return eater_radius ** 2 >= target_radius ** 2 * eat_size_ratio


def corner_pressure_batch(x_arr, y_arr, player_radius):
    """How boxed-in a set of positions is, in [0, 1].

    Only high when BOTH axes are close to a wall (an actual corner pocket);
    being near a single wall alone stays low, since one escape axis remains
    free. The ramp-in distance scales with player_radius so a bigger blob
    starts feeling squeezed further out from the literal corner point, which
    is what actually determines maneuverability.
    """
    scale = max(player_radius * CORNER_AWARE_RADIUS_MULT, 3.0)
    dx_wall = np.minimum(x_arr, ARENA_SIZE - x_arr)
    dy_wall = np.minimum(y_arr, ARENA_SIZE - y_arr)
    close_x = np.clip(1.0 - dx_wall / scale, 0.0, 1.0)
    close_y = np.clip(1.0 - dy_wall / scale, 0.0, 1.0)
    return close_x * close_y


def clamp_for_radius(x, y, radius):
    """Clamp one centre point so a blob of the given radius remains inside the arena."""
    min_pos = radius + WALL_BUFFER
    max_pos = ARENA_SIZE - radius - WALL_BUFFER

    if min_pos > max_pos:
        min_pos = radius
        max_pos = ARENA_SIZE - radius

    return (
        min(max(float(x), min_pos), max_pos),
        min(max(float(y), min_pos), max_pos),
    )


def clamp_many_for_radius(points, radii):
    """Vectorized clamp for many own blobs."""
    min_pos = radii + WALL_BUFFER
    max_pos = ARENA_SIZE - radii - WALL_BUFFER

    too_big = min_pos > max_pos
    if np.any(too_big):
        min_pos = np.where(too_big, radii, min_pos)
        max_pos = np.where(too_big, ARENA_SIZE - radii, max_pos)

    out = points.copy()
    out[:, 0] = np.minimum(np.maximum(out[:, 0], min_pos), max_pos)
    out[:, 1] = np.minimum(np.maximum(out[:, 1], min_pos), max_pos)
    return out


def clamp_player(cache, x, y):
    """Clamp a proposed player-centre position using the cached player radius."""
    return clamp_for_radius(x, y, cache["player_radius"])


def cap_nearest(locs, max_items, player_pos, *arrays):
    """Keep only the nearest objects and apply the same selection to aligned arrays."""
    if len(locs) <= max_items:
        return (locs, *arrays)

    dists = np.sum((locs - player_pos) ** 2, axis=1)
    keep = np.argpartition(dists, max_items - 1)[:max_items]

    result = [locs[keep]]
    for arr in arrays:
        result.append(arr[keep])

    return tuple(result)


def move_future(cache, dx, dy, step_distance):
    """Predict and clamp the player centre after moving in a direction for one step."""
    future_x = cache["player_x"] + dx * step_distance
    future_y = cache["player_y"] + dy * step_distance
    return clamp_player(cache, future_x, future_y)


def move_distance(cache, future_x, future_y):
    """Return the actual clamped distance between the current and proposed player positions."""
    return float(np.linalg.norm(
        np.array([
            future_x - cache["player_x"],
            future_y - cache["player_y"],
        ], dtype=float)
    ))


def report_to_result(report, split=False):
    """Convert a cached movement report into the direction-and-split tuple returned by an override."""
    return float(report["dx"]), float(report["dy"]), bool(split)


def report_towards(cache, target_vec, reports=None, stickiness=True, danger_scale=0.001):
    """Choose the safest cached report that most closely points toward a target vector."""
    target_dir = vec_norm(target_vec)
    if target_dir is None:
        return None

    if reports is None:
        reports = cache["safe_reports"]

    best_score = -float("inf")
    best_report = None

    for report in reports:
        direction = report["dir"]
        score = 1000.0 * np.dot(direction, target_dir)
        score += danger_scale * report["score"]
        if stickiness:
            score += TURN_WEIGHT * np.dot(direction, LAST_DIRECTION)

        if score > best_score:
            best_score = score
            best_report = report

    return best_report


def report_for_direction(cache, direction):
    """Return the cached movement report whose direction best matches a requested vector."""
    if direction is None:
        return None

    best_dot = -float("inf")
    best_report = None

    for report in cache["move_reports"]:
        dot = float(np.dot(report["dir"], direction))
        if dot > best_dot:
            best_dot = dot
            best_report = report

    return best_report


def exact_move_safe(cache, direction, step_distance, check_virus=True, enemy_only=False):
    """Score one exact direction when 16-way snapping is too coarse."""
    if direction is None:
        return False

    dx, dy = float(direction[0]), float(direction[1])

    if enemy_only:
        enemy = score_enemy_threat(cache, dx, dy, step_distance)
        return enemy["safe"]

    report = score_move(cache, dx, dy, step_distance, check_virus=check_virus)
    return report["safe"] and report["actual_move"] >= step_distance * 0.18


def exact_move_safe_to_virus(cache, direction, step_distance, target_key):
    """Validate exact travel toward one virus while avoiding every other virus."""
    if direction is None:
        return False

    dx, dy = float(direction[0]), float(direction[1])
    enemy = score_enemy_threat(cache, dx, dy, step_distance)
    if not enemy["safe"]:
        return False

    future_x, future_y = move_future(cache, dx, dy, step_distance)
    if move_distance(cache, future_x, future_y) < step_distance * 0.18:
        return False

    virus_score = score_virus_segment(
        cache,
        cache["player_pos"],
        np.array([future_x, future_y], dtype=float),
        cache["player_radius"],
        VIRUS_PATH_BUFFER_MULT,
        ignored_key=target_key,
    )
    return virus_score > -OFF_MAP_PENALTY / 2


# =========================
# Raw cache
# =========================

def cache_raw(game):
    """Read visible game objects into bounded NumPy arrays used by all later calculations."""
    me = game.state.me

    player_pos = np.array([me.x, me.y], dtype=float)
    player_radius = float(me.radius)
    player_id = me.player_id

    own_blobs = list(me.blobs.values())
    if own_blobs:
        own_locs_all = np.array([blob.pos for blob in own_blobs], dtype=float)
        own_rads_all = np.array([blob.radius for blob in own_blobs], dtype=float)
    else:
        own_locs_all = np.array([[me.x, me.y]], dtype=float)
        own_rads_all = np.array([me.radius], dtype=float)

    own_blob_count = len(own_rads_all)

    # Safety calculations are the main timeout risk when we are heavily split.
    # Keep the true blob count for decision logic, but only score the largest
    # pieces for repeated movement-safety reports.
    if own_blob_count > MAX_OWN_BLOBS_CONSIDERED:
        keep = np.argsort(own_rads_all)[-MAX_OWN_BLOBS_CONSIDERED:]
        own_locs = own_locs_all[keep]
        own_rads = own_rads_all[keep]
    else:
        own_locs = own_locs_all
        own_rads = own_rads_all

    foods = game.state.visible_food
    if foods:
        food_locs = np.array([food.pos for food in foods], dtype=float)
        if len(food_locs) > MAX_FOOD_CONSIDERED:
            food_locs, = cap_nearest(food_locs, MAX_FOOD_CONSIDERED, player_pos)
    else:
        food_locs = np.empty((0, 2), dtype=float)

    visible_blobs = game.state.visible_blobs
    enemy_blobs = [
        blob for blob in visible_blobs
        if getattr(blob, "player_id", None) != player_id
    ]

    if enemy_blobs:
        blob_locs = np.array([blob.pos for blob in enemy_blobs], dtype=float)
        blob_rads = np.array([blob.radius for blob in enemy_blobs], dtype=float)
        blob_player_ids = np.array([blob.player_id for blob in enemy_blobs], dtype=int)

        if len(blob_locs) > MAX_BLOBS_CONSIDERED:
            blob_locs, blob_rads, blob_player_ids = cap_nearest(
                blob_locs,
                MAX_BLOBS_CONSIDERED,
                player_pos,
                blob_rads,
                blob_player_ids,
            )

        merged_rads = np.zeros(len(blob_rads), dtype=float)
        for pid in np.unique(blob_player_ids):
            same_player = blob_player_ids == pid
            total_mass = np.sum(blob_rads[same_player] ** 2)
            merged_rads[same_player] = np.sqrt(total_mass)
    else:
        blob_locs = np.empty((0, 2), dtype=float)
        blob_rads = np.empty(0, dtype=float)
        blob_player_ids = np.empty(0, dtype=int)
        merged_rads = np.empty(0, dtype=float)

    viruses = game.state.visible_viruses
    if viruses:
        virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
        virus_rads = np.array([virus.radius for virus in viruses], dtype=float)
        if len(virus_locs) > MAX_VIRUSES_CONSIDERED:
            virus_locs, virus_rads = cap_nearest(
                virus_locs,
                MAX_VIRUSES_CONSIDERED,
                player_pos,
                virus_rads,
            )
    else:
        virus_locs = np.empty((0, 2), dtype=float)
        virus_rads = np.empty(0, dtype=float)

    return {
        "player_pos": player_pos,
        "player_x": float(player_pos[0]),
        "player_y": float(player_pos[1]),
        "player_radius": player_radius,
        "player_id": player_id,

        "own_locs": own_locs,
        "own_rads": own_rads,
        # Retain the complete set only for one-off split-command validation.
        # Repeated movement scoring still uses the capped arrays above.
        "own_locs_all": own_locs_all,
        "own_rads_all": own_rads_all,
        "own_blob_count": own_blob_count,

        "food_locs": food_locs,

        "blob_locs": blob_locs,
        "blob_rads": blob_rads,
        "blob_player_ids": blob_player_ids,
        "merged_rads": merged_rads,

        "virus_locs": virus_locs,
        "virus_rads": virus_rads,
    }


# =========================
# Score functions
# =========================

def score_virus_segment(cache, start_pos, end_pos, radius, buffer_mult, ignored_key=None):
    """Score whether one blob trajectory passes dangerously close to visible viruses.

    ignored_key is used only while deliberately approaching one chosen virus;
    every other visible virus remains a hard obstacle.
    """
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    # At the engine blob cap a virus cannot fragment us any further and is
    # therefore food rather than a movement hazard.
    if cache["own_blob_count"] >= MAX_BLOB_COUNT:
        return 0.0

    if len(virus_locs) == 0:
        return 0.0

    dangerous = radius > virus_rads * 1.1
    if ignored_key is not None:
        dangerous = dangerous.copy()
        for idx, pos in enumerate(virus_locs):
            if virus_key(pos) == ignored_key:
                dangerous[idx] = False

    if not np.any(dangerous):
        return 0.0

    segment = end_pos - start_pos
    segment_len_sq = np.dot(segment, segment)

    if segment_len_sq <= 1e-9:
        dists = np.linalg.norm(virus_locs - start_pos, axis=1)
    else:
        to_viruses = virus_locs - start_pos
        t = np.sum(to_viruses * segment, axis=1) / segment_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest_points = start_pos + t.reshape(-1, 1) * segment
        dists = np.linalg.norm(virus_locs - closest_points, axis=1)

    clearances = dists - radius - virus_rads

    if np.any(dangerous & (clearances < radius * buffer_mult)):
        return -OFF_MAP_PENALTY

    clearances[clearances < 1.0] = 1.0
    return -float(np.sum(VIRUS_DANGER_WEIGHT / (clearances[dangerous] ** 3)))


def score_own_blob_virus(cache, dx, dy, step_distance):
    """Score virus risk for all considered own blobs moving in one shared direction."""
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if cache["own_blob_count"] >= MAX_BLOB_COUNT:
        return 0.0

    if len(own_locs) == 0 or len(virus_locs) == 0:
        return 0.0

    dangerous = own_rads[:, None] > virus_rads[None, :] * 1.1
    if not np.any(dangerous):
        return 0.0

    starts = own_locs
    ends = own_locs + np.array([dx * step_distance, dy * step_distance], dtype=float)
    ends = clamp_many_for_radius(ends, own_rads)

    segments = ends - starts
    seg_len_sq = np.sum(segments * segments, axis=1)

    to_viruses = virus_locs[None, :, :] - starts[:, None, :]

    # Projection of each virus onto each own-blob movement segment.
    denom = np.where(seg_len_sq <= 1e-9, 1.0, seg_len_sq)
    t = np.sum(to_viruses * segments[:, None, :], axis=2) / denom[:, None]
    t = np.clip(t, 0.0, 1.0)
    t[seg_len_sq <= 1e-9, :] = 0.0

    closest = starts[:, None, :] + t[:, :, None] * segments[:, None, :]
    dists = np.linalg.norm(virus_locs[None, :, :] - closest, axis=2)
    clearances = dists - own_rads[:, None] - virus_rads[None, :]

    if np.any(dangerous & (clearances < own_rads[:, None] * VIRUS_PATH_BUFFER_MULT)):
        return -OFF_MAP_PENALTY

    safe_clearances = np.maximum(clearances, 1.0)
    scores = VIRUS_DANGER_WEIGHT / (safe_clearances ** 3)
    return -float(np.sum(scores[dangerous]))

def score_enemy_threat(cache, dx=0.0, dy=0.0, step_distance=0.0):
    """Evaluate enemy danger after moving all considered own blobs in one direction."""
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return {"active": False, "safe": True, "score": 0.0}

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    if len(own_locs) == 0:
        return {"active": False, "safe": True, "score": 0.0}

    player_radius = cache["player_radius"]
    # Immediate danger must use the actual blob that could collide with us.
    # A player's combined mass is handled separately as long-horizon remerge
    # risk; treating every fragment as the fully merged player causes false
    # emergency escapes from harmless small pieces.
    danger_size = blob_rads

    enemy_split_rads = cache.get("enemy_split_rads")
    if enemy_split_rads is None or len(enemy_split_rads) != len(blob_rads):
        enemy_split_rads = blob_rads / np.sqrt(2.0)

    futures = own_locs + np.array([dx * step_distance, dy * step_distance], dtype=float)
    futures = clamp_many_for_radius(futures, own_rads)

    dists = np.linalg.norm(blob_locs[None, :, :] - futures[:, None, :], axis=2)
    edge_dists = dists - own_rads[:, None] - blob_rads[None, :]

    direct_danger = danger_size[None, :] > own_rads[:, None] * DANGER_DIRECT_RATIO
    direct_active = direct_danger & (edge_dists < player_radius * DANGER_OVERRIDE_RANGE_MULT)
    direct_unsafe = direct_danger & (edge_dists < own_rads[:, None] * DANGER_HARD_CLOSE_MULT)

    split_danger = enemy_split_rads[None, :] > own_rads[:, None] * DANGER_SPLIT_RATIO
    split_near = edge_dists < blob_rads[None, :] * DANGER_SPLIT_RANGE_MULT
    split_active = split_danger & split_near

    active = bool(np.any(direct_active | split_active))
    unsafe = bool(np.any(direct_unsafe | split_active))

    score_dists = np.maximum(edge_dists, 1.0)
    scored = direct_danger | split_active
    danger_scores = (
        DANGER_WEIGHT
        * (danger_size[None, :] / own_rads[:, None])
        / (score_dists ** DANGER_DISTANCE_POWER)
    )

    return {
        "active": active,
        "safe": not unsafe,
        "score": -float(np.sum(danger_scores[scored])),
    }

def score_move(cache, dx, dy, step_distance, check_virus=True):
    """Combine enemy, virus, wall-clamping, and displacement information for one move."""
    enemy = score_enemy_threat(cache, dx, dy, step_distance)
    virus_score = score_own_blob_virus(cache, dx, dy, step_distance) if check_virus else 0.0
    virus_safe = virus_score > -OFF_MAP_PENALTY / 2

    future_x, future_y = move_future(cache, dx, dy, step_distance)
    actual_move = move_distance(cache, future_x, future_y)

    return {
        "enemy_active": enemy["active"],
        "enemy_safe": enemy["safe"],
        "virus_safe": bool(virus_safe),
        "safe": bool(enemy["safe"] and virus_safe),
        "score": float(enemy["score"] + virus_score),
        "enemy_score": float(enemy["score"]),
        "virus_score": float(virus_score),
        "actual_move": float(actual_move),
        "future_x": float(future_x),
        "future_y": float(future_y),
    }


def score_moves_batch(cache, directions, step_distance, check_virus=True):
    """Vectorized equivalent of calling score_move() once per direction.

    This computes the exact same per-direction quantities as score_move ->
    score_enemy_threat / score_own_blob_virus, but does it for every direction
    at once via broadcasting instead of looping in Python and re-entering NumPy
    16 (or 8) separate times per tick. The formulas are untouched; only the
    evaluation strategy changed. This is the main late-game CPU saver, since
    cache_move_reports runs unconditionally every tick regardless of which
    override ends up firing.
    """
    directions = np.asarray(directions, dtype=float)
    num_dirs = len(directions)
    deltas = directions * step_distance  # (D, 2)

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    num_own = len(own_locs)

    # --- future positions of every own blob, for every direction: (D, O, 2) ---
    futures_own = own_locs[None, :, :] + deltas[:, None, :]

    if num_own > 0:
        min_pos = own_rads + WALL_BUFFER
        max_pos = ARENA_SIZE - own_rads - WALL_BUFFER
        too_big = min_pos > max_pos
        if np.any(too_big):
            min_pos = np.where(too_big, own_rads, min_pos)
            max_pos = np.where(too_big, ARENA_SIZE - own_rads, max_pos)
        futures_own[:, :, 0] = np.clip(futures_own[:, :, 0], min_pos[None, :], max_pos[None, :])
        futures_own[:, :, 1] = np.clip(futures_own[:, :, 1], min_pos[None, :], max_pos[None, :])

    # --- enemy threat, batched over directions ---
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    player_radius = cache["player_radius"]

    if len(blob_locs) == 0 or num_own == 0:
        enemy_active = np.zeros(num_dirs, dtype=bool)
        enemy_safe = np.ones(num_dirs, dtype=bool)
        enemy_score = np.zeros(num_dirs, dtype=float)
    else:
        # Only an actual enemy blob can eat us on this movement step. Combined
        # owner mass is a strategic remerge signal, not an immediate eater.
        danger_size = blob_rads

        enemy_split_rads = cache.get("enemy_split_rads")
        if enemy_split_rads is None or len(enemy_split_rads) != len(blob_rads):
            enemy_split_rads = blob_rads / np.sqrt(2.0)

        # (D, O, E)
        dists = np.linalg.norm(
            blob_locs[None, None, :, :] - futures_own[:, :, None, :], axis=3
        )
        edge_dists = dists - own_rads[None, :, None] - blob_rads[None, None, :]

        direct_danger = danger_size[None, None, :] > own_rads[None, :, None] * DANGER_DIRECT_RATIO
        direct_active = direct_danger & (edge_dists < player_radius * DANGER_OVERRIDE_RANGE_MULT)
        direct_unsafe = direct_danger & (edge_dists < own_rads[None, :, None] * DANGER_HARD_CLOSE_MULT)

        split_danger = enemy_split_rads[None, None, :] > own_rads[None, :, None] * DANGER_SPLIT_RATIO
        split_near = edge_dists < blob_rads[None, None, :] * DANGER_SPLIT_RANGE_MULT
        split_active = split_danger & split_near

        active_mask = direct_active | split_active
        unsafe_mask = direct_unsafe | split_active

        enemy_active = np.any(active_mask, axis=(1, 2))
        enemy_safe = ~np.any(unsafe_mask, axis=(1, 2))

        score_dists = np.maximum(edge_dists, 1.0)
        scored = direct_danger | split_active
        danger_scores = (
            DANGER_WEIGHT
            * (danger_size[None, None, :] / own_rads[None, :, None])
            / (score_dists ** DANGER_DISTANCE_POWER)
        )
        enemy_score = -np.sum(np.where(scored, danger_scores, 0.0), axis=(1, 2))

    # --- own-blob virus safety, batched over directions ---
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if (
        not check_virus
        or cache["own_blob_count"] >= MAX_BLOB_COUNT
        or len(virus_locs) == 0
        or num_own == 0
    ):
        virus_score = np.zeros(num_dirs, dtype=float)
        virus_safe = np.ones(num_dirs, dtype=bool)
    else:
        dangerous = own_rads[:, None] > virus_rads[None, :] * 1.1  # (O, V)
        if not np.any(dangerous):
            virus_score = np.zeros(num_dirs, dtype=float)
            virus_safe = np.ones(num_dirs, dtype=bool)
        else:
            starts = own_locs  # (O, 2)
            segments = futures_own - starts[None, :, :]  # (D, O, 2)
            seg_len_sq = np.sum(segments * segments, axis=2)  # (D, O)

            to_viruses = virus_locs[None, None, :, :] - starts[None, :, None, :]  # (1, O, V, 2)
            denom = np.where(seg_len_sq <= 1e-9, 1.0, seg_len_sq)[:, :, None]  # (D, O, 1)
            t = np.sum(to_viruses * segments[:, :, None, :], axis=3) / denom  # (D, O, V)
            t = np.clip(t, 0.0, 1.0)
            zero_seg = seg_len_sq <= 1e-9  # (D, O)
            t = np.where(zero_seg[:, :, None], 0.0, t)

            closest = starts[None, :, None, :] + t[:, :, :, None] * segments[:, :, None, :]  # (D, O, V, 2)
            dists_v = np.linalg.norm(virus_locs[None, None, :, :] - closest, axis=3)  # (D, O, V)
            clearances = dists_v - own_rads[None, :, None] - virus_rads[None, None, :]

            danger_broadcast = dangerous[None, :, :]  # (1, O, V)
            unsafe_mask_v = danger_broadcast & (clearances < own_rads[None, :, None] * VIRUS_PATH_BUFFER_MULT)
            unsafe_any = np.any(unsafe_mask_v, axis=(1, 2))  # (D,)

            safe_clearances = np.maximum(clearances, 1.0)
            v_scores = VIRUS_DANGER_WEIGHT / (safe_clearances ** 3)
            virus_score = -np.sum(np.where(danger_broadcast, v_scores, 0.0), axis=(1, 2))
            virus_score = np.where(unsafe_any, -OFF_MAP_PENALTY, virus_score)
            virus_safe = ~unsafe_any

    # --- reported (single-point) future position / actual move distance ---
    min_p = player_radius + WALL_BUFFER
    max_p = ARENA_SIZE - player_radius - WALL_BUFFER
    if min_p > max_p:
        min_p = player_radius
        max_p = ARENA_SIZE - player_radius

    future_player = cache["player_pos"][None, :] + deltas  # (D, 2)
    future_player[:, 0] = np.clip(future_player[:, 0], min_p, max_p)
    future_player[:, 1] = np.clip(future_player[:, 1], min_p, max_p)
    actual_move = np.linalg.norm(future_player - cache["player_pos"][None, :], axis=1)

    # Dead-end awareness: penalize directions that land somewhere boxed in by
    # two walls at once, so roam/food/chase/virus scoring (all of which read
    # report["score"]) naturally steer around corner pockets instead of only
    # noticing a problem once already stuck in one.
    corner_pressure = corner_pressure_batch(future_player[:, 0], future_player[:, 1], player_radius)
    corner_score = -CORNER_PRESSURE_WEIGHT * corner_pressure

    safe = enemy_safe & (virus_score > -OFF_MAP_PENALTY / 2)
    total_score = enemy_score + virus_score + corner_score

    reports = []
    for i in range(num_dirs):
        reports.append({
            "idx": i,
            "dir": directions[i],
            "dx": float(directions[i, 0]),
            "dy": float(directions[i, 1]),
            "enemy_active": bool(enemy_active[i]),
            "enemy_safe": bool(enemy_safe[i]),
            "virus_safe": bool(virus_safe[i]),
            "safe": bool(safe[i]),
            "score": float(total_score[i]),
            "enemy_score": float(enemy_score[i]),
            "virus_score": float(virus_score[i]),
            "corner_score": float(corner_score[i]),
            "actual_move": float(actual_move[i]),
            "future_x": float(future_player[i, 0]),
            "future_y": float(future_player[i, 1]),
        })

    return reports


def score_food_position(cache, x, y):
    """Score a position by summing inverse-square attraction to visible food."""
    food_locs = cache["food_locs"]
    if len(food_locs) == 0:
        return 0.0

    pos = np.array([x, y], dtype=float)
    dists = np.linalg.norm(food_locs - pos, axis=1)
    dists[dists < 1.0] = 1.0
    return float(np.sum(FOOD_SCORE_WEIGHT / (dists ** 2)))


def score_enemy_position(cache, x, y):
    """Score a position using danger from larger enemies and reward from edible enemies."""
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    if len(blob_locs) == 0:
        return 0.0

    pos = np.array([x, y], dtype=float)
    player_radius = cache["player_radius"]

    dists = np.linalg.norm(blob_locs - pos, axis=1)
    edge_dists = dists - player_radius - blob_rads
    edge_dists[edge_dists < 1.0] = 1.0

    danger_size = blob_rads
    dangerous = danger_size > player_radius * 1.08
    edible = can_eat_blob_by_radius(player_radius, blob_rads, EAT_RATIO)

    danger_scores = (
        ENEMY_DANGER_WEIGHT
        * (danger_size / player_radius)
        / (edge_dists ** 1.3)
    )
    hunt_scores = (
        ENEMY_HUNT_WEIGHT
        * (player_radius / blob_rads)
        / (edge_dists ** 2)
    )

    score = -np.sum(danger_scores[dangerous])
    if cache["own_blob_count"] == 1:
        score += np.sum(hunt_scores[edible])

    return float(score)


# =========================
# Derived cache functions
# =========================

def cache_enemy_velocity(cache):
    """Estimate per-enemy velocity by matching blobs from the same player."""
    global ENEMY_LAST_POS

    blob_locs = cache["blob_locs"]
    blob_player_ids = cache["blob_player_ids"]
    blob_vels = np.zeros_like(blob_locs, dtype=float)

    if len(blob_locs) == 0:
        ENEMY_LAST_POS = {}
        cache["blob_vels"] = blob_vels
        return

    previous_by_pid = {}
    for key, pos in ENEMY_LAST_POS.items():
        previous_by_pid.setdefault(key[0], []).append((key, pos))

    new_last = {}

    for pid in np.unique(blob_player_ids):
        indices = np.where(blob_player_ids == pid)[0]
        previous = previous_by_pid.get(int(pid), [])
        used_previous = set()

        for local_num, idx in enumerate(indices):
            pos = blob_locs[idx].copy()
            best_key = None
            best_pos = None
            best_dist = float("inf")

            for prev_key, prev_pos in previous:
                if prev_key in used_previous:
                    continue

                dist = float(np.sum((pos - prev_pos) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best_key = prev_key
                    best_pos = prev_pos

            if best_pos is None:
                vel = np.array([0.0, 0.0], dtype=float)
            else:
                vel = pos - best_pos
                used_previous.add(best_key)

            speed = np.linalg.norm(vel)
            max_speed = cache["player_radius"] * 2.5
            if speed > max_speed and speed > 1e-9:
                vel = vel / speed * max_speed

            blob_vels[idx] = vel
            new_last[(int(pid), int(local_num))] = pos

    ENEMY_LAST_POS = new_last
    cache["blob_vels"] = blob_vels


def cache_enemy(cache):
    """Derive enemy distances, directions, predictions, size information, and edible masks."""
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    if len(blob_locs) == 0:
        cache["enemy_vectors"] = np.empty((0, 2), dtype=float)
        cache["enemy_dists"] = np.empty(0, dtype=float)
        cache["enemy_edge_dists"] = np.empty(0, dtype=float)
        cache["enemy_dirs"] = np.empty((0, 2), dtype=float)
        cache["enemy_edible"] = np.empty(0, dtype=bool)
        cache["enemy_pred_locs"] = np.empty((0, 2), dtype=float)
        cache["enemy_danger_size"] = np.empty(0, dtype=float)
        cache["enemy_split_rads"] = np.empty(0, dtype=float)
        return

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    blob_vels = cache.get("blob_vels")
    if blob_vels is None or len(blob_vels) != len(blob_locs):
        blob_vels = np.zeros_like(blob_locs, dtype=float)

    pred_locs = blob_locs + blob_vels * CHASE_LEAD_TICKS
    pred_locs[:, 0] = np.clip(pred_locs[:, 0], 0.0, ARENA_SIZE)
    pred_locs[:, 1] = np.clip(pred_locs[:, 1], 0.0, ARENA_SIZE)

    # Cache actual immediate eater size. merged_rads remains available for
    # chase/split/virus decisions that deliberately consider future remerging.
    danger_size = blob_rads.copy()

    cache["enemy_vectors"] = vectors
    cache["enemy_dists"] = dists
    cache["enemy_edge_dists"] = dists - player_radius - blob_rads
    cache["enemy_dirs"] = vectors / safe_dists.reshape(-1, 1)
    cache["enemy_edible"] = can_eat_blob_by_radius(player_radius, blob_rads, EAT_RATIO)
    cache["enemy_pred_locs"] = pred_locs
    cache["enemy_danger_size"] = danger_size
    cache["enemy_split_rads"] = blob_rads / np.sqrt(2.0)


def cache_food(cache):
    """Derive food directions and cluster-weighted target scores for the current frame."""
    food_locs = cache["food_locs"]
    player_pos = cache["player_pos"]

    if len(food_locs) == 0:
        cache["food_dists"] = np.empty(0, dtype=float)
        cache["food_dirs"] = np.empty((0, 2), dtype=float)
        cache["food_scores"] = np.empty(0, dtype=float)
        return

    vectors = food_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    pairwise = np.linalg.norm(
        food_locs[:, None, :] - food_locs[None, :, :],
        axis=2,
    )
    cluster_density = np.exp(-((pairwise / FOOD_CLUSTER_RADIUS) ** 2)).sum(axis=1) - 1.0

    scores = (
        (1.0 + FOOD_CLUSTER_WEIGHT * cluster_density)
        / (safe_dists ** FOOD_DISTANCE_POWER)
    )

    cache["food_dists"] = dists
    cache["food_dirs"] = vectors / safe_dists.reshape(-1, 1)
    cache["food_scores"] = scores


def virus_key(pos):
    """Create a stable rounded-coordinate key for a virus position."""
    return (
        round(float(pos[0]), VIRUS_MEMORY_KEY_DECIMALS),
        round(float(pos[1]), VIRUS_MEMORY_KEY_DECIMALS),
    )


def cache_virus_memory(cache):
    """Update remembered virus locations and expire stale or excessive entries."""
    global VIRUS_MEMORY

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    visible_keys = set()

    for pos, rad in zip(cache["virus_locs"], cache["virus_rads"]):
        key = virus_key(pos)
        visible_keys.add(key)
        VIRUS_MEMORY[key] = {
            "pos": np.array(pos, dtype=float),
            "radius": float(rad),
        }

    reached_dist = max(1.0, player_radius * VIRUS_MEMORY_REACHED_DIST_MULT)
    expired = []

    for key, info in VIRUS_MEMORY.items():
        if key in visible_keys:
            continue
        if np.linalg.norm(info["pos"] - player_pos) < reached_dist:
            expired.append(key)

    for key in expired:
        del VIRUS_MEMORY[key]

    # Long games can remember too many old viruses. Keep closest remembered
    # viruses only; far-away memory should not cost time every tick.
    if len(VIRUS_MEMORY) > VIRUS_MEMORY_MAX:
        ordered = sorted(
            VIRUS_MEMORY.items(),
            key=lambda item: float(np.sum((item[1]["pos"] - player_pos) ** 2)),
        )
        VIRUS_MEMORY = dict(ordered[:VIRUS_MEMORY_MAX])


def cache_move_reports(cache, step_distance):
    # When we are already split, chase/virus farming are disabled and the
    # expensive part is repeated safety checks over own pieces. 8 directions is
    # enough for split-state navigation and saves a lot of late-game CPU.
    """Precompute scored candidate movement reports for the current frame."""
    directions = DIRECTIONS_FAST if cache["own_blob_count"] > 1 else DIRECTIONS

    # v9 optimisation: score all directions in one batched/vectorized pass
    # instead of looping and calling score_move() (and therefore NumPy) once
    # per direction. Same formulas, same results, far fewer Python<->NumPy
    # round trips per tick. This is the function that runs unconditionally
    # every single tick, so its cost is what accumulates into the late-game
    # cumulative timeout.
    reports = score_moves_batch(cache, directions, step_distance, check_virus=True)

    cache["enemy_active"] = any(report["enemy_active"] for report in reports)
    cache["move_reports"] = reports
    cache["safe_reports"] = [
        report for report in reports
        if report["safe"] and report["actual_move"] >= step_distance * 0.20
    ]
    cache["enemy_safe_reports"] = [
        report for report in reports
        if report["enemy_safe"] and report["actual_move"] >= step_distance * 0.20
    ]


def cache_frame(game):
    """Build the complete per-tick cache consumed by all behaviour overrides."""
    cache = cache_raw(game)
    # Enemy velocity is only useful for chase, and chase is disabled while split.
    # Skipping matching in split states saves late-game CPU.
    if cache["own_blob_count"] == 1:
        cache_enemy_velocity(cache)
    else:
        cache["blob_vels"] = np.zeros_like(cache["blob_locs"], dtype=float)

    cache_enemy(cache)
    cache_virus_memory(cache)

    step_distance = STEP_DISTANCE_MULT * cache["player_radius"]
    cache["step_distance"] = step_distance
    cache_move_reports(cache, step_distance)

    return cache


# =========================
# Split helpers
# =========================

def enemy_owner_can_remerge_eat(cache, target_idx, eater_radius):
    """Return whether the target owner's visible combined mass could eat us.

    This is deliberately a strategic commitment guard, not immediate danger:
    fragments do not trigger escape, but we refuse a long chase that becomes
    losing as soon as their owner recombines.
    """
    if not (0 <= target_idx < len(cache["merged_rads"])):
        return False

    return bool(can_eat_blob_by_radius(
        cache["merged_rads"][target_idx],
        eater_radius,
        EAT_RATIO,
    ))


def enemy_owner_remerge_imminent(cache, target_idx, eater_radius):
    """Return whether dangerous sibling fragments are already close enough to regroup.

    Combined mass alone no longer forbids a chase. We only back off while the
    hunted fragment is physically close to a sibling and their combined mass
    could eat us after remerging.
    """
    if not enemy_owner_can_remerge_eat(cache, target_idx, eater_radius):
        return False

    player_ids = cache["blob_player_ids"]
    if not (0 <= target_idx < len(player_ids)):
        return False

    same_owner = np.where(player_ids == player_ids[target_idx])[0]
    same_owner = same_owner[same_owner != target_idx]
    if len(same_owner) == 0:
        return False

    target_pos = cache["blob_locs"][target_idx]
    target_radius = cache["blob_rads"][target_idx]
    sibling_locs = cache["blob_locs"][same_owner]
    sibling_rads = cache["blob_rads"][same_owner]
    sibling_edge_gaps = (
        np.linalg.norm(sibling_locs - target_pos, axis=1)
        - target_radius
        - sibling_rads
    )

    imminent_gap = max(eater_radius, target_radius) * CHASE_REMERGE_GAP_MULT
    return bool(np.min(sibling_edge_gaps) < imminent_gap)


def enemy_owner_can_eat_after_split_kill(cache, target_idx, split_radius):
    """Model the intended victim being eaten, then test the owner's remainder."""
    if not (0 <= target_idx < len(cache["merged_rads"])):
        return False

    target_radius = float(cache["blob_rads"][target_idx])
    owner_mass = float(cache["merged_rads"][target_idx] ** 2)
    remaining_enemy_mass = max(0.0, owner_mass - target_radius ** 2)
    fed_split_radius = np.sqrt(split_radius ** 2 + target_radius ** 2)
    remaining_enemy_radius = np.sqrt(remaining_enemy_mass)

    return bool(can_eat_blob_by_radius(
        remaining_enemy_radius,
        fed_split_radius,
        EAT_RATIO,
    ))

def split_path_safe(cache, start_pos, split_radius, split_landing):
    """Return whether a proposed offensive split trajectory avoids dangerous viruses."""
    score = score_virus_segment(
        cache,
        start_pos,
        split_landing,
        split_radius,
        SPLIT_VIRUS_BUFFER_MULT,
    )
    return score > -OFF_MAP_PENALTY / 2


def split_candidate(launcher_pos, launcher_radius, target_pos, target_radius):
    """Return exact split geometry if this launcher can reach target."""
    split_radius = launcher_radius / np.sqrt(2.0)
    split_range = launcher_radius * SPLIT_RANGE_MULT

    vector = target_pos - launcher_pos
    dist = np.linalg.norm(vector)
    if dist <= 1e-9:
        return None

    direction = vector / dist
    safe_range = split_range * SPLIT_RANGE_SAFETY_MULT
    # The engine eats a blob only when the TARGET CENTRE enters the eater radius,
    # so target_radius must not be added to the estimated hit distance.
    max_hit_dist = safe_range + split_radius
    reach_margin = max_hit_dist - dist

    if reach_margin <= launcher_radius * SPLIT_REACH_MARGIN_MULT:
        return None

    split_landing = launcher_pos + direction * split_range

    return {
        "dir": direction,
        "split_radius": split_radius,
        "split_landing": split_landing,
        "dist": float(dist),
        "reach_margin": float(reach_margin),
    }


def double_split_candidate(launcher_pos, launcher_radius, target_pos, target_radius):
    """Return geometry for a target reachable only after two split commands.

    The leading piece has radius launcher_radius / 2 after the second split,
    so it must independently satisfy the engine mass-ratio eating rule.
    """
    first_piece_radius = launcher_radius / np.sqrt(2.0)
    final_piece_radius = launcher_radius / 2.0

    if not can_eat_blob_by_radius(final_piece_radius, target_radius, SPLIT_EAT_RATIO):
        return None

    vector = target_pos - launcher_pos
    dist = float(np.linalg.norm(vector))
    if dist <= 1e-9:
        return None

    direction = vector / dist
    first_travel = launcher_radius * SPLIT_RANGE_MULT
    second_travel = first_piece_radius * SPLIT_RANGE_MULT

    single_hit_dist = (
        first_travel * SPLIT_RANGE_SAFETY_MULT
        + first_piece_radius
    )
    double_hit_dist = (
        (first_travel + second_travel) * DOUBLE_SPLIT_RANGE_SAFETY_MULT
        + final_piece_radius
    )

    # This planner is only for prey outside reliable single-split reach.
    if dist <= single_hit_dist + launcher_radius * SPLIT_REACH_MARGIN_MULT:
        return None

    reach_margin = double_hit_dist - dist
    if reach_margin <= launcher_radius * SPLIT_REACH_MARGIN_MULT:
        return None

    first_landing = launcher_pos + direction * first_travel
    first_landing = np.array(
        clamp_for_radius(
            first_landing[0],
            first_landing[1],
            first_piece_radius,
        ),
        dtype=float,
    )
    second_landing = first_landing + direction * second_travel
    second_landing = np.array(
        clamp_for_radius(
            second_landing[0],
            second_landing[1],
            final_piece_radius,
        ),
        dtype=float,
    )

    return {
        "dir": direction,
        "first_piece_radius": float(first_piece_radius),
        "final_piece_radius": float(final_piece_radius),
        "first_landing": first_landing,
        "second_landing": second_landing,
        "dist": dist,
        "reach_margin": float(reach_margin),
    }


def split_landing_enemy_safe(cache, landing_pos, split_radius, launcher_radius, target_idx):
    """Check that the new split piece is not landing next to something that eats it."""
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    if len(blob_locs) == 0:
        return True

    # Landing collision safety is immediate, so only actual blobs count here.
    # The intended victim owner's remaining combined mass is checked separately
    # after modelling the victim being consumed.
    danger_size = blob_rads
    dists = np.linalg.norm(blob_locs - landing_pos, axis=1)
    edge_dists = dists - split_radius - blob_rads

    dangerous = danger_size > split_radius * SPLIT_LANDING_DANGER_RATIO
    nearby = edge_dists < launcher_radius * SPLIT_POST_DANGER_RANGE_MULT

    # Ignore the intended victim, because the whole point is to land on it.
    if 0 <= target_idx < len(dangerous):
        dangerous[target_idx] = False

    return not bool(np.any(dangerous & nearby))


def all_split_launchers_safe(cache, direction, intended_launcher_pos, intended_launcher_radius, target_idx):
    """Validate a global split using a bounded secondary-piece mass-risk budget.

    Movement safety stays capped for performance, but this one-off finalist
    check uses all own pieces. The intended attacking launch must be safe;
    secondary launched/remaining halves may take limited risk so one imperfect
    piece does not suppress a high-upside late-game expansion.
    """
    own_locs = cache["own_locs_all"]
    own_rads = cache["own_rads_all"]
    intended_matched = False
    intended_launch_safe = False
    eligible_mass = 0.0
    exposed_mass = 0.0

    for launcher_pos, launcher_radius in zip(own_locs, own_rads):
        launcher_radius = float(launcher_radius)
        if launcher_radius < SPLIT_MIN_RADIUS:
            continue

        launcher_mass = launcher_radius ** 2
        half_mass = launcher_mass * 0.5
        eligible_mass += launcher_mass

        split_radius = launcher_radius / np.sqrt(2.0)
        split_landing = launcher_pos + direction * (launcher_radius * SPLIT_RANGE_MULT)
        split_landing[0] = np.clip(split_landing[0], split_radius, ARENA_SIZE - split_radius)
        split_landing[1] = np.clip(split_landing[1], split_radius, ARENA_SIZE - split_radius)

        is_intended = (
            not intended_matched
            and abs(launcher_radius - intended_launcher_radius) <= 1e-6
            and np.linalg.norm(launcher_pos - intended_launcher_pos) <= 1e-6
        )
        if is_intended:
            intended_matched = True

        launch_safe = split_path_safe(cache, launcher_pos, split_radius, split_landing)

        landing_target_idx = target_idx if is_intended else -1
        if launch_safe:
            launch_safe = split_landing_enemy_safe(
                cache,
                split_landing,
                split_radius,
                launcher_radius,
                landing_target_idx,
            )

        if is_intended:
            intended_launch_safe = launch_safe
        elif not launch_safe:
            exposed_mass += half_mass

        # The half left behind also becomes smaller and newly vulnerable.
        parent_safe = split_landing_enemy_safe(
            cache,
            launcher_pos,
            split_radius,
            launcher_radius,
            -1,
        )
        if not parent_safe:
            exposed_mass += half_mass

    if not intended_matched or not intended_launch_safe or eligible_mass <= 0.0:
        return False

    return exposed_mass <= eligible_mass * SPLIT_GLOBAL_RISK_FRACTION


def clear_double_split_plan():
    """Clear all state associated with a planned second split command."""
    global DOUBLE_SPLIT_TARGET_PID, DOUBLE_SPLIT_TARGET_POS
    global DOUBLE_SPLIT_PENDING_TICKS

    DOUBLE_SPLIT_TARGET_PID = None
    DOUBLE_SPLIT_TARGET_POS = None
    DOUBLE_SPLIT_PENDING_TICKS = 0


def continue_double_split(cache):
    """Complete or briefly pursue the target of an initiated double split."""
    global DOUBLE_SPLIT_TARGET_POS, DOUBLE_SPLIT_PENDING_TICKS

    if DOUBLE_SPLIT_TARGET_PID is None:
        return None

    if cache["own_blob_count"] >= MAX_BLOB_COUNT:
        clear_double_split_plan()
        return None

    DOUBLE_SPLIT_PENDING_TICKS += 1
    if DOUBLE_SPLIT_PENDING_TICKS > DOUBLE_SPLIT_PENDING_LIMIT:
        clear_double_split_plan()
        return None

    target_indices = np.where(
        cache["blob_player_ids"] == DOUBLE_SPLIT_TARGET_PID
    )[0]
    if len(target_indices) == 0:
        clear_double_split_plan()
        return None

    if DOUBLE_SPLIT_TARGET_POS is not None:
        target_indices = target_indices[np.argsort(
            np.sum(
                (cache["blob_locs"][target_indices] - DOUBLE_SPLIT_TARGET_POS) ** 2,
                axis=1,
            )
        )]

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    for target_idx in target_indices[:SPLIT_MAX_TARGETS]:
        target_pos = cache["blob_locs"][target_idx]
        target_radius = cache["blob_rads"][target_idx]
        launcher_order = np.argsort(np.sum((own_locs - target_pos) ** 2, axis=1))

        for launcher_i in launcher_order[:SPLIT_MAX_LAUNCHERS]:
            launcher_pos = own_locs[launcher_i]
            launcher_radius = float(own_rads[launcher_i])
            if launcher_radius < SPLIT_MIN_RADIUS:
                continue

            candidate = split_candidate(
                launcher_pos,
                launcher_radius,
                target_pos,
                target_radius,
            )
            if candidate is None:
                continue

            if not can_eat_blob_by_radius(
                candidate["split_radius"],
                target_radius,
                SPLIT_EAT_RATIO,
            ):
                continue

            if not all_split_launchers_safe(
                cache,
                candidate["dir"],
                launcher_pos,
                launcher_radius,
                int(target_idx),
            ):
                continue

            clear_double_split_plan()
            return (
                float(candidate["dir"][0]),
                float(candidate["dir"][1]),
                True,
            )

    # The target moved slightly beyond the second launch window. Keep closing
    # for a few ticks instead of abandoning the committed double split at once.
    target_idx = int(target_indices[0])
    DOUBLE_SPLIT_TARGET_POS = cache["blob_locs"][target_idx].copy()
    hunter_idx = int(np.argmax(own_rads))
    report = report_towards(
        cache,
        DOUBLE_SPLIT_TARGET_POS - own_locs[hunter_idx],
        reports=cache["safe_reports"],
        danger_scale=0.001,
    )
    if report is None:
        return None
    return report_to_result(report, split=False)


def start_double_split(cache):
    """Launch the first command for valuable prey inside reliable double-split reach."""
    global DOUBLE_SPLIT_TARGET_PID, DOUBLE_SPLIT_TARGET_POS
    global DOUBLE_SPLIT_PENDING_TICKS

    own_blob_count = cache["own_blob_count"]
    if (
        own_blob_count > DOUBLE_SPLIT_MAX_START_BLOBS
        or own_blob_count * 4 > MAX_BLOB_COUNT
    ):
        return None

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    if len(own_locs) == 0 or len(blob_locs) == 0:
        return None

    candidates = []
    for target_idx in np.argsort(blob_rads)[::-1][:SPLIT_MAX_TARGETS]:
        target_pos = blob_locs[target_idx]
        target_radius = blob_rads[target_idx]
        launcher_order = np.argsort(np.sum((own_locs - target_pos) ** 2, axis=1))

        for launcher_i in launcher_order[:SPLIT_MAX_LAUNCHERS]:
            launcher_pos = own_locs[launcher_i]
            launcher_radius = float(own_rads[launcher_i])
            if launcher_radius < SPLIT_MIN_RADIUS:
                continue

            target_mass_fraction = (target_radius / launcher_radius) ** 2
            required_mass_fraction = (
                SPLIT_MIN_TARGET_MASS_FRACTION
                + max(0, own_blob_count - 1) * SPLIT_EXISTING_PIECE_MASS_PENALTY
            )
            if target_mass_fraction < required_mass_fraction:
                continue

            candidate = double_split_candidate(
                launcher_pos,
                launcher_radius,
                target_pos,
                target_radius,
            )
            if candidate is None:
                continue

            # Validate the intended leading piece across both launch segments.
            if score_virus_segment(
                cache,
                candidate["first_landing"],
                candidate["second_landing"],
                candidate["final_piece_radius"],
                SPLIT_VIRUS_BUFFER_MULT,
            ) <= -OFF_MAP_PENALTY / 2:
                continue

            if not split_landing_enemy_safe(
                cache,
                candidate["second_landing"],
                candidate["final_piece_radius"],
                candidate["first_piece_radius"],
                int(target_idx),
            ):
                continue

            score = 6.0 * target_radius ** 2
            score -= 0.20 * candidate["dist"]
            score += 0.50 * candidate["reach_margin"]
            candidates.append((
                score,
                int(target_idx),
                int(launcher_i),
                candidate,
            ))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, target_idx, launcher_i, candidate in candidates:
        launcher_pos = own_locs[launcher_i]
        launcher_radius = float(own_rads[launcher_i])

        if not all_split_launchers_safe(
            cache,
            candidate["dir"],
            launcher_pos,
            launcher_radius,
            -1,
        ):
            continue

        DOUBLE_SPLIT_TARGET_PID = int(cache["blob_player_ids"][target_idx])
        DOUBLE_SPLIT_TARGET_POS = blob_locs[target_idx].copy()
        DOUBLE_SPLIT_PENDING_TICKS = 0
        return (
            float(candidate["dir"][0]),
            float(candidate["dir"][1]),
            True,
        )

    return None

def virus_enemy_safe(cache, virus_pos, virus_rad):
    """Return whether nearby enemies are unlikely to consume pieces created by a virus pop."""
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return True

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    estimated_piece_radius = player_radius * VIRUS_FARM_PIECE_RADIUS_MULT
    danger_size = np.maximum(blob_rads, merged_rads)

    can_eat_pieces = danger_size > estimated_piece_radius * 1.08

    dists_to_player = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_to_player = dists_to_player - player_radius - blob_rads

    dists_to_virus = np.linalg.norm(blob_locs - virus_pos, axis=1)
    edge_to_virus = dists_to_virus - virus_rad - blob_rads

    close_to_player = edge_to_player < player_radius * VIRUS_FARM_ENEMY_RANGE_MULT
    close_to_virus = edge_to_virus < player_radius * VIRUS_FARM_ENEMY_RANGE_MULT

    return not bool(np.any(can_eat_pieces & (close_to_player | close_to_virus)))


def virus_split_candidate(cache, virus_pos, virus_rad):
    """Return intentional split geometry only when the launched half can hit.

    The split is delayed until collision is comfortably inside reach. This
    prevents us from splitting early and then slowly travelling as two blobs.
    """
    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    split_radius = player_radius / np.sqrt(2.0)

    # The launched half itself must still be large enough to consume/pop the
    # virus; otherwise manual splitting only weakens us before contact.
    if split_radius <= virus_rad * VIRUS_SPLIT_EAT_RATIO:
        return None

    vector = virus_pos - player_pos
    dist = float(np.linalg.norm(vector))
    if dist <= 1e-9:
        return None

    direction = vector / dist
    split_range = player_radius * SPLIT_RANGE_MULT
    reliable_range = split_range * VIRUS_SPLIT_RANGE_SAFETY_MULT
    max_hit_dist = reliable_range + split_radius + virus_rad
    reach_margin = max_hit_dist - dist

    if reach_margin <= player_radius * VIRUS_SPLIT_REACH_MARGIN_MULT:
        return None

    split_landing = player_pos + direction * split_range
    split_landing[0] = np.clip(split_landing[0], split_radius, ARENA_SIZE - split_radius)
    split_landing[1] = np.clip(split_landing[1], split_radius, ARENA_SIZE - split_radius)

    return {
        "dir": direction,
        "split_radius": float(split_radius),
        "split_landing": split_landing,
    }


def virus_split_avoids_other_viruses(cache, start_pos, split_landing, split_radius, target_key):
    """Allow contact with the chosen virus, but reject accidental chain pops."""
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) <= 1:
        return True

    segment = split_landing - start_pos
    segment_len_sq = float(np.dot(segment, segment))

    if segment_len_sq <= 1e-9:
        dists = np.linalg.norm(virus_locs - start_pos, axis=1)
    else:
        to_viruses = virus_locs - start_pos
        t = np.sum(to_viruses * segment, axis=1) / segment_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = start_pos + t[:, None] * segment
        dists = np.linalg.norm(virus_locs - closest, axis=1)

    clearances = dists - split_radius - virus_rads
    dangerous = split_radius > virus_rads * VIRUS_SPLIT_EAT_RATIO

    for idx, pos in enumerate(virus_locs):
        if virus_key(pos) == target_key:
            dangerous[idx] = False

    unsafe = dangerous & (
        clearances < split_radius * VIRUS_SPLIT_OTHER_VIRUS_BUFFER_MULT
    )
    return not bool(np.any(unsafe))


def safe_intentional_virus_split(cache, virus_pos, virus_rad, target_key):
    """Choose a manual virus split only when both trajectory and aftermath are safe."""

    if cache["player_radius"] ** 2 >= VIRUS_FARM_MAX_RADIUS:
        return None

    candidate = virus_split_candidate(cache, virus_pos, virus_rad)
    if candidate is None:
        return None

    # Do not voluntarily fragment while a real enemy threat is already active.
    if cache["enemy_active"]:
        return None

    if not virus_enemy_safe(cache, virus_pos, virus_rad):
        return None

    if not virus_split_avoids_other_viruses(
        cache,
        cache["player_pos"],
        candidate["split_landing"],
        candidate["split_radius"],
        target_key,
    ):
        return None

    # Landing safety covers the launched half before/around virus contact.
    if not split_landing_enemy_safe(
        cache,
        candidate["split_landing"],
        candidate["split_radius"],
        cache["player_radius"],
        -1,
    ):
        return None

    direction = candidate["dir"]
    return float(direction[0]), float(direction[1]), True


def enemy_near_wall(pos):
    """Return whether an enemy position is close enough to a wall or corner for direct pursuit."""
    x, y = pos
    dist_left = x
    dist_right = ARENA_SIZE - x
    dist_bottom = y
    dist_top = ARENA_SIZE - y

    near_wall = min(dist_left, dist_right, dist_bottom, dist_top) < CHASE_WALL_DIST
    near_corner = (
        (dist_left < CHASE_WALL_DIST or dist_right < CHASE_WALL_DIST)
        and (dist_bottom < CHASE_WALL_DIST or dist_top < CHASE_WALL_DIST)
    )
    return near_wall or near_corner


def chase_block_point(enemy_pos, enemy_radius, player_radius, enemy_vel=None):
    """Point that cuts off the enemy's escape route.

    In open space, being directly behind the enemy just follows their trail.
    This point is biased to the centre-side of the enemy, with a smaller bias
    toward their current velocity, so we intercept/cut off instead of shadowing.
    """
    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    centre_dir = vec_norm(center - enemy_pos)

    if centre_dir is None:
        return enemy_pos.copy(), None

    block_dir = centre_dir.copy()

    if enemy_vel is not None:
        vel_dir = vec_norm(enemy_vel)
        if vel_dir is not None:
            mixed = 0.75 * centre_dir + 0.25 * vel_dir
            mixed_dir = vec_norm(mixed)
            if mixed_dir is not None:
                block_dir = mixed_dir

    offset = min(player_radius * 1.15, enemy_radius * 2.5 + player_radius * 0.45)
    block_point = enemy_pos + block_dir * offset

    block_point[0] = np.clip(block_point[0], 0.0, ARENA_SIZE)
    block_point[1] = np.clip(block_point[1], 0.0, ARENA_SIZE)

    return block_point, centre_dir


# =========================
# Food memory helpers
# =========================

def food_key(food_pos):
    """Create a stable rounded-coordinate key for a food position."""
    return (round(float(food_pos[0]), 1), round(float(food_pos[1]), 1))


def food_update_blacklist():
    """Decrease food blacklist timers and remove entries whose timers have expired."""
    global FOOD_BLACKLIST

    expired = []
    for key in FOOD_BLACKLIST:
        FOOD_BLACKLIST[key] -= 1
        if FOOD_BLACKLIST[key] <= 0:
            expired.append(key)

    for key in expired:
        del FOOD_BLACKLIST[key]


def food_blacklist(key):
    """Temporarily blacklist a food target and clear it if it is currently locked."""
    global FOOD_TARGET_KEY, FOOD_TARGET_TICKS
    global FOOD_TARGET_NO_PROGRESS, FOOD_TARGET_LAST_DIST

    if key is None:
        return

    FOOD_BLACKLIST[key] = FOOD_BLACKLIST_TICKS

    if FOOD_TARGET_KEY == key:
        FOOD_TARGET_KEY = None
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None


def food_unreachable_corner(food_pos, player_radius=0.0):
    # The fixed FOOD_CORNER_MARGIN only rules out food sitting right on the
    # geometric corner point. For any real-sized blob that's not the region
    # that actually restricts movement, so also scale the exclusion zone with
    # how big we are: a bigger blob needs more clearance from both walls at
    # once before a corner food item is worth detouring into a pocket for.
    """Return whether a food position lies in a radius-scaled inaccessible corner pocket."""
    margin = max(FOOD_CORNER_MARGIN, player_radius * FOOD_CORNER_RADIUS_MULT)
    x, y = food_pos
    near_left = x < margin
    near_right = ARENA_SIZE - x < margin
    near_bottom = y < margin
    near_top = ARENA_SIZE - y < margin
    return (near_left or near_right) and (near_bottom or near_top)


def food_ranked_targets(cache):
    """Maintain food-target memory and return valid food directions ordered by score."""
    global FOOD_TARGET_KEY, FOOD_TARGET_TICKS
    global FOOD_TARGET_NO_PROGRESS, FOOD_TARGET_LAST_DIST

    # Food clustering is only needed if we actually reach food/roam mode.
    # Higher-priority split/chase/virus ticks should not pay the pairwise cost.
    if "food_scores" not in cache:
        cache_food(cache)

    food_update_blacklist()

    food_locs = cache["food_locs"]
    if len(food_locs) == 0:
        FOOD_TARGET_KEY = None
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None
        return []

    food_keys = [food_key(food_pos) for food_pos in food_locs]
    dists = cache["food_dists"]
    dirs = cache["food_dirs"]
    scores = cache["food_scores"].copy()

    current_idx = None
    if FOOD_TARGET_KEY is not None:
        for i, key in enumerate(food_keys):
            if key == FOOD_TARGET_KEY:
                current_idx = i
                break
        if current_idx is None:
            FOOD_TARGET_KEY = None
            FOOD_TARGET_TICKS = 0
            FOOD_TARGET_NO_PROGRESS = 0
            FOOD_TARGET_LAST_DIST = None

    valid = dists > FOOD_MIN_TARGET_DIST
    for i, key in enumerate(food_keys):
        if key in FOOD_BLACKLIST or food_unreachable_corner(food_locs[i], cache["player_radius"]):
            valid[i] = False

    if not np.any(valid):
        return []

    reached_dist = max(
        FOOD_REACHED_DIST_MIN,
        cache["player_radius"] * FOOD_REACHED_DIST_MULT,
    )

    if current_idx is not None and valid[current_idx]:
        current_dist = dists[current_idx]
        FOOD_TARGET_TICKS += 1

        if FOOD_TARGET_LAST_DIST is not None:
            if current_dist >= FOOD_TARGET_LAST_DIST - 0.03:
                FOOD_TARGET_NO_PROGRESS += 1
            else:
                FOOD_TARGET_NO_PROGRESS = 0

        FOOD_TARGET_LAST_DIST = current_dist

        if (
            current_dist < reached_dist
            or FOOD_TARGET_TICKS >= FOOD_TARGET_MAX_TICKS
            or FOOD_TARGET_NO_PROGRESS >= FOOD_NO_PROGRESS_LIMIT
        ):
            food_blacklist(FOOD_TARGET_KEY)
            current_idx = None

    scores[~valid] = -1.0

    if current_idx is not None and scores[current_idx] > 0:
        scores[current_idx] *= 1.25

    candidates = []
    for idx in np.argsort(scores)[::-1]:
        if scores[idx] <= 0:
            break

        direction = dirs[idx]
        candidates.append((
            float(scores[idx]),
            float(direction[0]),
            float(direction[1]),
            food_keys[idx],
        ))

    if candidates and FOOD_TARGET_KEY is None:
        FOOD_TARGET_KEY = candidates[0][3]
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None

    return candidates


# =========================
# State helpers
# =========================

def stuck_now(cache):
    """Update movement history and return whether the bot has been nearly stationary for too long."""
    global LAST_POSITION, STUCK_TICKS

    pos = cache["player_pos"]
    r = cache["player_radius"]

    if LAST_POSITION is None:
        LAST_POSITION = pos.copy()
        return False

    moved = np.linalg.norm(pos - LAST_POSITION)
    if moved < r * STUCK_MOVE_EPS_MULT:
        STUCK_TICKS += 1
    else:
        STUCK_TICKS = 0

    LAST_POSITION = pos.copy()
    return STUCK_TICKS >= STUCK_TICK_LIMIT


# =========================
# Override functions
# =========================

def override_escape(cache, step_distance):
    """Choose an emergency direction when a threatening enemy is active."""
    if not cache["enemy_active"]:
        return None

    virus_safe_reports = [
        report for report in cache["move_reports"]
        if report["virus_safe"]
    ]
    if not virus_safe_reports:
        return None

    # Never trade a genuinely enemy-safe escape for an unsafe direction merely
    # because the latter moves farther. Only score unsafe directions when no
    # enemy-safe alternative exists.
    enemy_safe_reports = [
        report for report in virus_safe_reports
        if report["enemy_safe"]
    ]
    candidate_reports = enemy_safe_reports or virus_safe_reports

    best_score = -float("inf")
    best_report = None

    for report in candidate_reports:
        score = report["enemy_score"]
        # Tiebreaker, not an override of safety: if two directions are
        # similarly safe right now, prefer the one that doesn't run into a
        # dead end. Weighted low enough that genuine danger avoidance always
        # wins first.
        score += CORNER_ESCAPE_WEIGHT * report["corner_score"]
        if report["actual_move"] < step_distance * 0.20:
            score -= DANGER_WALL_PUSH_PENALTY
        else:
            score += DANGER_MOVE_REWARD * report["actual_move"]

        if score > best_score:
            best_score = score
            best_report = report

    if best_report is None:
        return None

    return report_to_result(best_report, split=False)


def override_unstuck(cache, step_distance):
    """Choose a high-displacement safe direction after repeated low-movement ticks."""
    if not stuck_now(cache):
        return None

    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    center_dir = vec_norm(center - cache["player_pos"])

    best_score = -float("inf")
    best_report = None

    for report in cache["safe_reports"]:
        if report["actual_move"] < step_distance * 0.25:
            continue

        score = 0.0
        score += STUCK_MOVE_WEIGHT * report["actual_move"]
        score += report["score"]
        if center_dir is not None:
            score += STUCK_CENTER_WEIGHT * np.dot(report["dir"], center_dir)

        if score > best_score:
            best_score = score
            best_report = report

    if best_report is None:
        return None

    return report_to_result(best_report, split=False)


def override_split(cache, step_distance):
    """Exact-aim split kill with bounded late-game work.

    v8 optimisation: first collect cheap geometry candidates, then run the
    expensive virus-path / landing-danger / exact-enemy checks only on the best
    few. This keeps the strong multi-split behaviour but avoids cumulative
    timeout when many enemies and own blobs are visible.
    """
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    own_blob_count = cache["own_blob_count"]
    player_radius = cache["player_radius"]

    pending_double = continue_double_split(cache)
    if pending_double is not None:
        return pending_double

    # player_radius squared is total mass, matching the virus-farming guard.
    max_blobs = (
        SPLIT_HUGE_MAX_OWN_BLOBS
        if player_radius ** 2 >= SPLIT_HUGE_RADIUS
        else SPLIT_MAX_OWN_BLOBS
    )
    if own_blob_count > max_blobs:
        return None

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    if len(blob_locs) == 0 or len(own_locs) == 0:
        return None

    late_fast = own_blob_count > 1 or player_radius ** 2 >= SPLIT_HUGE_RADIUS
    max_targets = 5 if late_fast else SPLIT_MAX_TARGETS
    max_launchers = 2 if late_fast else SPLIT_MAX_LAUNCHERS
    top_exact = 1 if late_fast else SPLIT_TOP_EXACT_CHECKS
    precheck_limit = 6 if late_fast else SPLIT_PRECHECK_LIMIT

    max_split_radius = float(np.max(own_rads)) / np.sqrt(2.0)
    target_ok = can_eat_blob_by_radius(max_split_radius, blob_rads, SPLIT_EAT_RATIO)
    if not np.any(target_ok):
        return None

    target_indices = np.where(target_ok)[0]
    target_order = target_indices[np.argsort(blob_rads[target_indices])[::-1]][:max_targets]

    cheap_candidates = []

    for target_idx in target_order:
        target_pos = blob_locs[target_idx]
        target_rad = blob_rads[target_idx]

        launcher_dists = np.sum((own_locs - target_pos) ** 2, axis=1)
        launcher_order = np.argsort(launcher_dists)[:max_launchers]

        for launcher_i in launcher_order:
            launcher_pos = own_locs[launcher_i]
            launcher_radius = own_rads[launcher_i]

            if launcher_radius < SPLIT_MIN_RADIUS:
                continue

            if target_rad <= launcher_radius * SPLIT_TARGET_MIN_RADIUS_MULT:
                continue

            target_mass_fraction = (target_rad / launcher_radius) ** 2
            required_mass_fraction = (
                SPLIT_MIN_TARGET_MASS_FRACTION
                + max(0, own_blob_count - 1) * SPLIT_EXISTING_PIECE_MASS_PENALTY
            )
            if target_mass_fraction < required_mass_fraction:
                continue

            candidate = split_candidate(
                launcher_pos,
                launcher_radius,
                target_pos,
                target_rad,
            )
            if candidate is None:
                continue

            split_radius = candidate["split_radius"]
            if not can_eat_blob_by_radius(split_radius, target_rad, SPLIT_EAT_RATIO):
                continue

            # Remaining combined enemy mass is now a preference penalty rather
            # than an absolute veto: fast fragment kills are a major source of
            # high-variance snowballs.
            remerge_aftermath_risk = enemy_owner_can_eat_after_split_kill(
                cache,
                target_idx,
                split_radius,
            )

            split_dir = candidate["dir"]
            score = 0.0
            score += target_rad * 4.0
            score -= candidate["dist"] * 0.25
            score += candidate["reach_margin"] * 0.35
            score -= own_blob_count * 0.75
            if remerge_aftermath_risk:
                score -= 10.0

            cheap_candidates.append((
                score,
                float(split_dir[0]),
                float(split_dir[1]),
                launcher_pos,
                float(launcher_radius),
                float(split_radius),
                candidate["split_landing"],
                int(target_idx),
            ))

            # Keep cheap list bounded throughout the loop.
            if len(cheap_candidates) > precheck_limit * 2:
                cheap_candidates.sort(key=lambda item: item[0], reverse=True)
                cheap_candidates = cheap_candidates[:precheck_limit]

    if not cheap_candidates:
        return start_double_split(cache)

    cheap_candidates.sort(key=lambda item: item[0], reverse=True)

    checked_exact = 0
    for _, dx, dy, launcher_pos, launcher_radius, split_radius, split_landing, target_idx in cheap_candidates[:precheck_limit]:
        split_direction = np.array([dx, dy], dtype=float)
        if not all_split_launchers_safe(
            cache,
            split_direction,
            launcher_pos,
            launcher_radius,
            target_idx,
        ):
            continue

        # Exact enemy safety is the most expensive finalist check; bound it.
        enemy = score_enemy_threat(cache, dx, dy, step_distance)
        checked_exact += 1
        if enemy["safe"]:
            return dx, dy, True

        if checked_exact >= top_exact:
            break

    return start_double_split(cache)

def override_capped_virus_farm(cache):
    """Consume remembered/visible viruses once the 16-blob cap removes pop risk."""
    global VIRUS_TARGET_KEY, VIRUS_TARGET_TICKS

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    if len(own_locs) == 0 or not cache["safe_reports"]:
        return None

    hunter_idx = int(np.argmax(own_rads))
    hunter_pos = own_locs[hunter_idx]
    hunter_radius = float(own_rads[hunter_idx])

    candidates = []
    visible_keys = {virus_key(pos) for pos in cache["virus_locs"]}
    for key, info in VIRUS_MEMORY.items():
        if hunter_radius <= info["radius"] * VIRUS_FARM_EAT_RATIO:
            continue
        dist = float(np.linalg.norm(info["pos"] - hunter_pos))
        visible_bonus = 0 if key in visible_keys else 1
        candidates.append((visible_bonus, dist, key, info["pos"], info["radius"]))

    if not candidates:
        VIRUS_TARGET_KEY = None
        VIRUS_TARGET_TICKS = 0
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    locked = None
    if VIRUS_TARGET_KEY is not None and VIRUS_TARGET_TICKS < VIRUS_LOCK_TICKS:
        for item in candidates:
            if item[2] == VIRUS_TARGET_KEY:
                locked = item
                break
    if locked is not None:
        candidates = [locked] + [item for item in candidates if item[2] != VIRUS_TARGET_KEY]

    for _, _, key, virus_pos, _ in candidates[:VIRUS_MAX_CANDIDATES]:
        report = report_towards(
            cache,
            virus_pos - hunter_pos,
            reports=cache["safe_reports"],
            danger_scale=0.001,
        )
        if report is None:
            continue

        if VIRUS_TARGET_KEY == key:
            VIRUS_TARGET_TICKS += 1
        else:
            VIRUS_TARGET_KEY = key
            VIRUS_TARGET_TICKS = 0
        return report_to_result(report, split=False)

    return None


def override_virus(cache, step_distance):
    """Select virus farming, travel, or intentional virus-split behaviour when appropriate."""
    global VIRUS_TARGET_KEY, VIRUS_TARGET_TICKS

    if cache["own_blob_count"] >= MAX_BLOB_COUNT:
        return override_capped_virus_farm(cache)

    if cache["own_blob_count"] != 1:
        VIRUS_TARGET_KEY = None
        VIRUS_TARGET_TICKS = 0
        return None

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    # Protect a winning late-game blob: once this large, do not seek, approach,
    # or intentionally collide with viruses at all.
    if player_radius ** 2 >= VIRUS_FARM_MAX_RADIUS:
        VIRUS_TARGET_KEY = None
        VIRUS_TARGET_TICKS = 0
        return None

    candidates = []
    for key, info in VIRUS_MEMORY.items():
        virus_pos = info["pos"]
        virus_rad = info["radius"]

        if player_radius <= virus_rad * VIRUS_FARM_EAT_RATIO:
            continue

        dist = np.linalg.norm(virus_pos - player_pos)
        candidates.append((dist, key, virus_pos, virus_rad))

    if not candidates:
        VIRUS_TARGET_KEY = None
        VIRUS_TARGET_TICKS = 0
        return None

    candidates.sort(key=lambda item: item[0])
    if len(candidates) > VIRUS_MAX_CANDIDATES:
        candidates = candidates[:VIRUS_MAX_CANDIDATES]

    visible_keys = {virus_key(pos) for pos in cache["virus_locs"]}
    reached_dist = max(1.0, player_radius * VIRUS_MEMORY_REACHED_DIST_MULT)

    # Lock the same virus briefly. This prevents left/right target swapping and
    # the visible-virus wobble that was happening near virus clusters.
    locked = None
    if VIRUS_TARGET_KEY is not None and VIRUS_TARGET_TICKS < VIRUS_LOCK_TICKS:
        for item in candidates:
            if item[1] == VIRUS_TARGET_KEY:
                locked = item
                break

    if locked is not None:
        best_dist = candidates[0][0]
        # Only switch away if another virus is dramatically closer.
        if best_dist < locked[0] * VIRUS_SWITCH_DIST_MULT:
            ordered = candidates
        else:
            ordered = [locked] + [item for item in candidates if item[1] != VIRUS_TARGET_KEY]
    else:
        ordered = candidates

    for dist, key, virus_pos, virus_rad in ordered:
        is_visible = key in visible_keys

        if dist < reached_dist and not is_visible:
            if key in VIRUS_MEMORY:
                del VIRUS_MEMORY[key]
            if VIRUS_TARGET_KEY == key:
                VIRUS_TARGET_KEY = None
                VIRUS_TARGET_TICKS = 0
            continue

        if is_visible and not virus_enemy_safe(cache, virus_pos, virus_rad):
            continue

        direction = vec_norm(virus_pos - player_pos)
        if direction is None:
            continue

        if is_visible:
            # Drive the whole blob directly into the virus. The engine itself
            # performs useful fragmentation up to the 16-blob cap; manually
            # splitting first only reduces the mass of the contacting piece.
            # Ignore only the chosen target virus. Every other visible virus,
            # enemy danger, and wall-clamped displacement remains validated.
            if not exact_move_safe_to_virus(
                cache,
                direction,
                step_distance,
                key,
            ):
                continue

            if VIRUS_TARGET_KEY == key:
                VIRUS_TARGET_TICKS += 1
            else:
                VIRUS_TARGET_KEY = key
                VIRUS_TARGET_TICKS = 0

            return float(direction[0]), float(direction[1]), False

        # Travelling toward remembered unseen virus: stay conservative and keep
        # using cached safe movement, because we are not intentionally popping yet.
        report = report_towards(
            cache,
            virus_pos - player_pos,
            reports=cache["safe_reports"],
            danger_scale=0.001,
        )

        if report is None:
            continue

        if VIRUS_TARGET_KEY == key:
            VIRUS_TARGET_TICKS += 1
        else:
            VIRUS_TARGET_KEY = key
            VIRUS_TARGET_TICKS = 0

        return report_to_result(report, split=False)

    return None


def override_split_chase(cache, step_distance):
    """Keep hunting with the largest own piece while the player is fragmented."""
    global CHASE_TARGET_KEY, CHASE_TARGET_TICKS, CHASE_LOST_TICKS

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    blob_player_ids = cache["blob_player_ids"]
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    if len(blob_locs) == 0 or len(own_locs) == 0 or not cache["safe_reports"]:
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    hunter_idx = int(np.argmax(own_rads))
    hunter_pos = own_locs[hunter_idx]
    hunter_radius = float(own_rads[hunter_idx])

    vectors = blob_locs - hunter_pos
    dists = np.linalg.norm(vectors, axis=1)
    edge_dists = dists - hunter_radius - blob_rads
    edible = can_eat_blob_by_radius(hunter_radius, blob_rads, EAT_RATIO)

    locked_owner = (
        blob_player_ids == CHASE_TARGET_KEY
        if CHASE_TARGET_KEY is not None
        else np.zeros(len(blob_locs), dtype=bool)
    )
    acquisition = edge_dists < hunter_radius * SPLIT_CHASE_RANGE_MULT
    retention = edge_dists < hunter_radius * SPLIT_CHASE_RETENTION_RANGE_MULT
    in_range = acquisition | (locked_owner & retention)

    remerge_not_imminent = np.array([
        not enemy_owner_remerge_imminent(cache, idx, hunter_radius)
        for idx in range(len(blob_locs))
    ], dtype=bool)
    candidates = edible & in_range & remerge_not_imminent

    if not np.any(candidates):
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    candidate_indices = np.where(candidates)[0]
    if CHASE_TARGET_KEY is not None and CHASE_TARGET_TICKS < CHASE_LOCK_TICKS:
        locked = candidate_indices[blob_player_ids[candidate_indices] == CHASE_TARGET_KEY]
        if len(locked) > 0:
            candidate_indices = locked

    best_target_idx = None
    best_target_score = -float("inf")
    for idx in candidate_indices:
        target_score = 4.0 * blob_rads[idx] ** 2 / max(edge_dists[idx], 1.0)
        target_score -= 0.12 * max(edge_dists[idx], 0.0)
        if int(blob_player_ids[idx]) == CHASE_TARGET_KEY:
            target_score += 18.0
        if enemy_near_wall(blob_locs[idx]):
            target_score += 8.0

        if target_score > best_target_score:
            best_target_score = target_score
            best_target_idx = int(idx)

    if best_target_idx is None:
        return None

    report = report_towards(
        cache,
        blob_locs[best_target_idx] - hunter_pos,
        reports=cache["safe_reports"],
        danger_scale=0.001,
    )
    if report is None:
        return None

    chosen_key = int(blob_player_ids[best_target_idx])
    if CHASE_TARGET_KEY == chosen_key:
        CHASE_TARGET_TICKS += 1
    else:
        CHASE_TARGET_KEY = chosen_key
        CHASE_TARGET_TICKS = 0
    CHASE_LOST_TICKS = 0

    return report_to_result(report, split=False)


def override_chase(cache, step_distance):
    """Select and pursue an edible enemy using prediction, interception, and target locking."""
    global CHASE_TARGET_KEY, CHASE_TARGET_TICKS, CHASE_LOST_TICKS

    if cache["own_blob_count"] != 1:
        return override_split_chase(cache, step_distance)

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    blob_player_ids = cache["blob_player_ids"]

    if len(blob_locs) == 0:
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    edge_dists = cache["enemy_edge_dists"]
    edible = cache["enemy_edible"]
    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    blob_vels = cache.get("blob_vels")

    locked_owner = (
        blob_player_ids == CHASE_TARGET_KEY
        if CHASE_TARGET_KEY is not None
        else np.zeros(len(blob_locs), dtype=bool)
    )
    acquisition_range = edge_dists < player_radius * CHASE_RANGE_MULT
    retention_range = edge_dists < player_radius * CHASE_RETENTION_RANGE_MULT
    in_range = acquisition_range | (locked_owner & retention_range)

    # Chase separated fragments aggressively. Back off only when siblings are
    # physically close enough that a dangerous remerge is becoming imminent.
    remerge_not_imminent = np.array([
        not enemy_owner_remerge_imminent(cache, idx, player_radius)
        for idx in range(len(blob_locs))
    ], dtype=bool)
    candidates = edible & in_range & remerge_not_imminent

    if not np.any(candidates):
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    candidate_indices = np.where(candidates)[0]

    if CHASE_TARGET_KEY is not None and CHASE_TARGET_TICKS < CHASE_LOCK_TICKS:
        locked = [
            idx for idx in candidate_indices
            if int(blob_player_ids[idx]) == CHASE_TARGET_KEY
        ]
        if locked:
            candidate_indices = np.array(locked, dtype=int)

    best_score = -float("inf")
    best_report = None
    best_target_idx = None
    best_closing = 0.0
    best_wall = False
    best_aim_point = None
    best_exact_direct = False

    for idx in candidate_indices:
        enemy_vel = blob_vels[idx] if blob_vels is not None and len(blob_vels) > idx else np.array([0.0, 0.0], dtype=float)
        enemy_speed = np.linalg.norm(enemy_vel)

        # Dynamic prediction: lead more when far away, less when already close.
        lead_ticks = np.clip(
            edge_dists[idx] / max(step_distance + enemy_speed, 1.0),
            1.0,
            CHASE_LEAD_TICKS,
        )
        predicted_pos = blob_locs[idx] + enemy_vel * lead_ticks
        predicted_pos[0] = np.clip(predicted_pos[0], 0.0, ARENA_SIZE)
        predicted_pos[1] = np.clip(predicted_pos[1], 0.0, ARENA_SIZE)

        current_edge = edge_dists[idx]
        near_wall = enemy_near_wall(predicted_pos)
        catchable_now = current_edge < max(player_radius * CHASE_DIRECT_EDGE_MULT, step_distance * 1.25)

        block_point, centre_dir = chase_block_point(
            predicted_pos,
            blob_rads[idx],
            player_radius,
            enemy_vel,
        )

        direct_finish = near_wall or catchable_now
        aim_point = predicted_pos if direct_finish else block_point
        direct_dir = vec_norm(predicted_pos - player_pos)

        for report in cache["safe_reports"]:
            future_pos = np.array([report["future_x"], report["future_y"]], dtype=float)
            future_edge = np.linalg.norm(predicted_pos - future_pos) - player_radius - blob_rads[idx]
            closing = current_edge - future_edge

            score = 0.0
            score += CHASE_SIZE_WEIGHT * blob_rads[idx]
            score -= max(current_edge, 0.0) * 0.20
            score += 0.0002 * report["score"]
            score += CHASE_STICKINESS_WEIGHT * np.dot(report["dir"], LAST_DIRECTION)

            if direct_finish:
                # If we probably can catch them, be much more direct/aggressive.
                score += CHASE_CLOSE_WEIGHT * 1.35 * closing
                if direct_dir is not None:
                    score += 120.0 * np.dot(report["dir"], direct_dir)
                if near_wall:
                    score += 18.0
            else:
                # Open space: get to the centre-side blocking point so they are
                # forced outward instead of circling around us forever.
                block_dist = np.linalg.norm(block_point - future_pos)
                score += 1.10 * CHASE_CLOSE_WEIGHT * closing
                score -= 0.70 * CHASE_BLOCK_DIST_WEIGHT * block_dist

                if centre_dir is not None:
                    future_from_enemy = vec_norm(future_pos - predicted_pos)
                    if future_from_enemy is not None:
                        centre_side = np.dot(future_from_enemy, centre_dir)
                        score += 0.80 * CHASE_CENTER_SIDE_WEIGHT * centre_side

            if CHASE_TARGET_KEY == int(blob_player_ids[idx]):
                score += 12.0

            if score > best_score:
                best_score = score
                best_report = report
                best_target_idx = idx
                best_closing = closing
                best_wall = near_wall
                best_aim_point = aim_point
                best_exact_direct = direct_finish

    if best_report is None or best_target_idx is None:
        return None

    # Prefer exact aiming toward the chosen intercept/block point, but only if
    # the exact vector is safe. The cached report remains a safe fallback.
    exact_dir = vec_norm(best_aim_point - player_pos) if best_aim_point is not None else None
    if exact_dir is not None:
        if exact_move_safe(cache, exact_dir, step_distance, check_virus=True, enemy_only=False):
            chosen_result = (float(exact_dir[0]), float(exact_dir[1]), False)
        else:
            chosen_result = report_to_result(best_report, split=False)
    else:
        chosen_result = report_to_result(best_report, split=False)

    # Do not give up too quickly while attempting a trap. Trapping may briefly
    # reduce direct closing distance but still improves position.
    if best_closing < CHASE_MIN_CLOSING_RATE and not best_wall:
        CHASE_LOST_TICKS += 1
    else:
        CHASE_LOST_TICKS = max(0, CHASE_LOST_TICKS - 2)

    if CHASE_LOST_TICKS >= CHASE_LOST_LIMIT:
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    chosen_key = int(blob_player_ids[best_target_idx])
    if CHASE_TARGET_KEY == chosen_key:
        CHASE_TARGET_TICKS += 1
    else:
        CHASE_TARGET_KEY = chosen_key
        CHASE_TARGET_TICKS = 0

    return chosen_result

def override_food_roam(cache, step_distance):
    """Pursue ranked food targets, then fall back to stable centre-biased roaming."""
    global ROAM_DIRECTION, ROAM_TICKS

    food_candidates = food_ranked_targets(cache)

    for _, dx, dy, key in food_candidates:
        target_vec = np.array([dx, dy], dtype=float)
        report = report_towards(cache, target_vec, reports=cache["safe_reports"])

        if report is None:
            food_blacklist(key)
            continue

        return report_to_result(report, split=False)

    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    center_dir = vec_norm(center - cache["player_pos"])

    if ROAM_DIRECTION is not None and ROAM_TICKS < ROAM_LOCK_TICKS:
        report = report_for_direction(cache, ROAM_DIRECTION)
        if report is not None and report["safe"] and report["actual_move"] >= step_distance * 0.20:
            ROAM_TICKS += 1
            return report_to_result(report, split=False)

    best_score = -float("inf")
    best_report = None

    for report in cache["safe_reports"]:
        score = 0.0
        score += report["score"]
        score += ROAM_MOVE_WEIGHT * report["actual_move"]
        score += ROAM_STICKINESS_WEIGHT * np.dot(report["dir"], LAST_DIRECTION)

        if center_dir is not None:
            score += ROAM_CENTER_WEIGHT * np.dot(report["dir"], center_dir)

        if score > best_score:
            best_score = score
            best_report = report

    if best_report is None:
        ROAM_DIRECTION = None
        ROAM_TICKS = 0
        return None

    ROAM_DIRECTION = np.array(best_report["dir"], dtype=float)
    ROAM_TICKS = 0

    return report_to_result(best_report, split=False)


def override_fallback(cache, step_distance):
    """Choose the best remaining safe direction when no higher-priority behaviour acts."""
    reports = cache["safe_reports"]

    if not reports:
        fallback = vec_norm(LAST_DIRECTION)
        if fallback is None:
            return 1.0, 0.0, False
        return float(fallback[0]), float(fallback[1]), False

    best_score = -float("inf")
    best_report = None

    for report in reports:
        score = 0.0
        score += report["score"]
        score += 0.30 * score_food_position(cache, report["future_x"], report["future_y"])
        # Enemy danger is already represented inside report["score"]. Avoid
        # another per-direction enemy scan in the final fallback.
        score += TURN_WEIGHT * np.dot(report["dir"], LAST_DIRECTION)

        if score > best_score:
            best_score = score
            best_report = report

    return report_to_result(best_report, split=False)


# =========================
# Direction choice
# =========================

def choose_direction(game: Game):
    """Build the frame cache and return the first valid action from the priority override chain."""
    global LAST_DIRECTION

    cache = cache_frame(game)
    step_distance = cache["step_distance"]

    overrides = [
        override_escape,
        override_unstuck,
        override_split,
        override_chase,
        override_virus,
        override_food_roam,
        override_fallback,
    ]

    for override in overrides:
        result = override(cache, step_distance)
        if result is None:
            continue

        dx, dy, should_split = result
        final_direction = vec_norm(np.array([dx, dy], dtype=float))

        if final_direction is None:
            final_direction = LAST_DIRECTION

        LAST_DIRECTION = final_direction

        return (
            float(final_direction[0]),
            float(final_direction[1]),
            bool(should_split),
            cache,
        )

    return 1.0, 0.0, False, cache


# =========================
# Main loop
# =========================

def main() -> None:
    """Run the game query loop and send one movement response for every move request."""
    load_tuning_config()
    game = Game()

    while True:
        query = game.get_next_query()

        match query:
            case QueryMovePlayer():
                try:
                    dx, dy, should_split, cache = choose_direction(game)

                    game.send_move(
                        MovePlayer(
                            player_id=cache["player_id"],
                            direction=DirectionModel(
                                x=float(dx),
                                y=float(dy),
                            ),
                            split=bool(should_split),
                        )
                    )

                except Exception:
                    game.send_move(
                        MovePlayer(
                            player_id=game.state.me.player_id,
                            direction=DirectionModel(
                                x=1.0,
                                y=0.0,
                            ),
                            split=False,
                        )
                    )

            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()