from __future__ import annotations

import math
from collections import deque

import numpy as np

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel


TAU = 2.0 * math.pi

# Engine-grounded constants.
BASE_PLAYER_SPEED = 1.1
PLAYER_SPEED_RADIUS_FACTOR = 0.08
MIN_PLAYER_SPEED = 0.25
EAT_SIZE_RATIO = 1.2
MAX_BLOB_COUNT = 16
SPLIT_MIN_MASS = 2.0
SPLIT_COOLDOWN_FRAMES = 18
SPLIT_EJECT_SPEED = 1.6
FOOD_RADIUS = 0.15

# Farming / prey scoring.
FOOD_CLUSTER_RADIUS_MULT = 2.35
FOOD_CLUSTER_MIN_RADIUS = 1.15
FOOD_CLUSTER_MAX_RADIUS = 5.5
FOOD_VALUE_WEIGHT = 7.0
FOOD_COUNT_WEIGHT = 8.0
FOOD_DENSITY_WEIGHT = 13.0
FOOD_DISTANCE_WEIGHT = 0.45
FOOD_LOCAL_FLOW_WEIGHT = 0.18
FOOD_LOCAL_DISTANCE_POWER = 0.85

PREY_VALUE_WEIGHT = 3.0
PREY_RADIUS_RATIO = 0.82

# Phase behaviour.
EARLY_GAME_THRESHOLD = 0.25
EARLY_RADIUS_THRESHOLD = 2.45
EARLY_FARM_STICKINESS = 8.0
EARLY_FARM_SMOOTHING = 0.42
EARLY_PREY_MULT = 0.05
MID_PREY_MULT = 1.0
LATE_PREY_MULT = 1.65

# Threat / survival scoring.
THREAT_RADIUS_RATIO = 1.06
EARLY_THREAT_RATIO = 1.15
SAFETY_MARGIN = 2.2

# Counter-split prediction.
COUNTER_SPLIT_REACTION_MARGIN = 1.45
COUNTER_SPLIT_DETECTION_MULT = 1.38
COUNTER_SPLIT_ESCAPE_WEIGHT = 2.5
COUNTER_SPLIT_MIN_RADIUS_RATIO = 1.52
PREDICTIVE_THREAT_FRAMES = 3.0
ENEMY_APPROACH_WEIGHT = 1.35

SURVIVAL_HEADING_STICKINESS = 1.15
FARM_HEADING_STICKINESS = 18.0
VIRUS_ROUTE_TARGET_WEIGHT = 4.0
VIRUS_ROUTE_HEADING_STICKINESS = 1.8
SURVIVAL_SMOOTHING = 0.12
FARMING_SMOOTHING = 0.52

# Split-chase tuning. The engine splits every eligible blob, so split only while stable.
SPLIT_TARGET_VALUE = 230.0
SPLIT_DANGER_LOOKAHEAD = 3.0
SPLIT_COOLDOWN_ROUNDS = SPLIT_COOLDOWN_FRAMES + 5
SPLIT_ONLY_WHEN_MERGED = True
SPLIT_MIN_TARGET_RADIUS = 0.35
SPLIT_MAX_DISTANCE_PENALTY = 38.0
SPLIT_BAIT_THREAT_MARGIN = 2.2

# Virus handling.
VIRUS_BLOB_LOOKAHEAD_MULT = 3.25
VIRUS_SPLIT_EXTRA_MARGIN = 1.15
VIRUS_BASE_MARGIN = 0.38
VIRUS_PATH_MARGIN = 0.22

# Arena/wall handling. If map size is unavailable these do nothing.
EDGE_AVOID_MARGIN = 4.5
EDGE_BLOCK_RADIUS = 2.2
EDGE_ESCAPE_WEIGHT = 1.4

# Merge/stuck behaviour.
MERGE_READY_COOLDOWN = 3
MERGE_PULL_WEIGHT = 46.0
STUCK_HISTORY_LEN = 16
STUCK_MODE_TICKS = 10
STUCK_MIN_DISPLACEMENT_FACTOR = 0.32
STUCK_TURN_THRESHOLD = 0.55
STUCK_SMOOTHING = 0.20

# Fallback if game.state.round / max_rounds are not provided.
FALLBACK_MAX_ROUNDS = 1000

_LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
_LAST_SPLIT_ROUND = -9999
_TURN_INDEX = 0
_RECENT_POSITIONS: deque[np.ndarray] = deque(maxlen=STUCK_HISTORY_LEN)
_RECENT_ANGLES: deque[float] = deque(maxlen=STUCK_HISTORY_LEN)
_STUCK_MODE_UNTIL = -9999
_ENEMY_MEMORY: dict[object, tuple[np.ndarray, int, float]] = {}
_ENEMY_VELOCITY: dict[object, np.ndarray] = {}


class _Ctx:
    """Per-turn precomputed state shared across helper functions."""

    __slots__ = (
        "player_id",
        "player_pos",
        "player_radius",
        "largest_blob_radius",
        "me",
        "my_blobs",
        "visible_blobs",
        "visible_food",
        "visible_viruses",
        "food_clusters",
        "round_index",
        "round_frac",
        "arena_size",
    )

    def __init__(
        self,
        player_id: int,
        player_pos: np.ndarray,
        player_radius: float,
        largest_blob_radius: float,
        me,
        my_blobs: list,
        visible_blobs: list,
        visible_food: list,
        visible_viruses: list,
        food_clusters: list,
        round_index: int,
        round_frac: float,
        arena_size: float,
    ) -> None:
        self.player_id = player_id
        self.player_pos = player_pos
        self.player_radius = player_radius
        self.largest_blob_radius = largest_blob_radius
        self.me = me
        self.my_blobs = my_blobs
        self.visible_blobs = visible_blobs
        self.visible_food = visible_food
        self.visible_viruses = visible_viruses
        self.food_clusters = food_clusters
        self.round_index = round_index
        self.round_frac = round_frac
        self.arena_size = arena_size


def _pos(obj) -> np.ndarray:
    if hasattr(obj, "pos"):
        return np.array(obj.pos, dtype=float)
    return np.array([float(getattr(obj, "x", 0.0)), float(getattr(obj, "y", 0.0))], dtype=float)


def _radius(obj, default: float = 1.0) -> float:
    return float(getattr(obj, "radius", getattr(obj, "size", default)))


def _owner_id(obj):
    return getattr(obj, "player_id", getattr(obj, "owner_id", getattr(obj, "id", None)))


def _is_ours(ctx: _Ctx, obj) -> bool:
    return obj is ctx.me or _owner_id(obj) == ctx.player_id


def _mass(obj) -> float:
    radius = _radius(obj)
    return radius * radius


def _norm(v: np.ndarray) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _unit(v: np.ndarray) -> np.ndarray:
    length = _norm(v)
    if length <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / length


def _my_blobs(player) -> list:
    blobs = getattr(player, "blobs", {})
    if isinstance(blobs, dict):
        return list(blobs.values())
    return list(blobs or [])


def _active_blobs(player) -> list:
    blobs = _my_blobs(player)
    return blobs if blobs else [player]


def _blob_pos(blob) -> np.ndarray:
    return _pos(blob)


def _largest_owned_radius(blobs: list, fallback_radius: float) -> float:
    if not blobs:
        return fallback_radius
    return max(_radius(blob) for blob in blobs)


def _eligible_split_blobs(player) -> list:
    # Use active blobs, not raw _my_blobs, so this still works if the engine exposes
    # an unsplit player without a .blobs dictionary.
    return [blob for blob in _active_blobs(player) if _mass(blob) >= SPLIT_MIN_MASS]


def _largest_split_blob(player):
    eligible = _eligible_split_blobs(player)
    if not eligible:
        return None
    return max(eligible, key=_radius)


def _is_multi_or_unmerged(ctx: _Ctx) -> bool:
    if len(ctx.my_blobs) > 1:
        return True
    return any(int(getattr(blob, "merge_cooldown", 0)) > 0 for blob in ctx.my_blobs)


def _movement_speed(radius: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR))


def _angle_of(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0])) % TAU


def _angle_to_vec(angle: float) -> tuple[float, float]:
    return (math.cos(angle), math.sin(angle))


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % TAU - math.pi)


def _smoothed_direction(raw_direction: np.ndarray, smoothing: float) -> tuple[float, float]:
    global _LAST_DIRECTION

    raw_direction = _unit(raw_direction)
    previous = _unit(_LAST_DIRECTION)
    direction = previous * smoothing + raw_direction * (1.0 - smoothing)

    if _norm(direction) < 1e-6:
        direction = raw_direction

    _LAST_DIRECTION = _unit(direction)
    return (float(_LAST_DIRECTION[0]), float(_LAST_DIRECTION[1]))


def _edge_distance(
    a_pos: np.ndarray,
    a_radius: float,
    b_pos: np.ndarray,
    b_radius: float,
) -> float:
    return max(0.0, _norm(b_pos - a_pos) - a_radius - b_radius)


def _can_eat(eater_radius: float, target_radius: float) -> bool:
    return eater_radius >= target_radius * EAT_SIZE_RATIO


def _can_consume_virus(blob_radius: float, virus_radius: float) -> bool:
    # Public engine uses blob.mass > virus.radius * EAT_SIZE_RATIO.
    return blob_radius * blob_radius > virus_radius * EAT_SIZE_RATIO


def _blocked_interval(center_angle: float, half_width: float) -> tuple[float, float] | None:
    if half_width >= math.pi:
        return (0.0, TAU)
    return ((center_angle - half_width) % TAU, (center_angle + half_width) % TAU)


def _danger_cone(
    origin_pos: np.ndarray,
    target_pos: np.ndarray,
    inflated_radius: float,
) -> tuple[float, float] | None:
    to_target = target_pos - origin_pos
    distance = _norm(to_target)

    if distance <= 1e-12:
        return (0.0, TAU)

    half_width = math.asin(min(1.0, inflated_radius / distance))
    return _blocked_interval(_angle_of(to_target), half_width)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    expanded: list[tuple[float, float]] = []
    for start, end in intervals:
        start %= TAU
        end %= TAU
        if start == 0.0 and end == TAU:
            return [(0.0, TAU)]
        if start <= end:
            expanded.append((start, end))
        else:
            expanded.append((start, TAU))
            expanded.append((0.0, end))

    expanded.sort()
    merged = [expanded[0]]
    for start, end in expanded[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    if len(merged) > 1 and merged[0][0] == 0.0 and merged[-1][1] == TAU:
        first_start, first_end = merged.pop(0)
        last_start, _ = merged.pop()
        merged.insert(0, (last_start, first_end))

    return merged


def _safe_gaps(blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    blocked = _merge_intervals(blocked)
    if not blocked:
        return [(0.0, TAU)]
    if len(blocked) == 1 and blocked[0] == (0.0, TAU):
        return []

    gaps: list[tuple[float, float]] = []
    for i, (_, end) in enumerate(blocked):
        next_start = blocked[(i + 1) % len(blocked)][0]
        width = (next_start - end) % TAU
        if width > 1e-6:
            gaps.append((end % TAU, width))
    return gaps


def _angle_in_gap(angle: float, gap: tuple[float, float]) -> bool:
    start, width = gap
    return ((angle - start) % TAU) <= width


def _shift_into_gap(
    angle: float,
    gap: tuple[float, float],
    preferred_angle: float | None = None,
) -> float:
    if _angle_in_gap(angle, gap):
        return angle

    start, width = gap
    end = (start + width) % TAU
    dist_to_start = (angle - start) % TAU
    dist_to_end = (end - angle) % TAU

    if preferred_angle is not None:
        target_is_ambiguous = abs(dist_to_start - dist_to_end) < 0.75
        preference_is_safe = _angle_in_gap(preferred_angle, gap)
        if target_is_ambiguous or preference_is_safe:
            return start if _angle_diff(start, preferred_angle) < _angle_diff(end, preferred_angle) else end

    return start if dist_to_start < dist_to_end else end


def _build_food_clusters(visible_food: list, player_radius: float) -> list[dict]:
    """Cluster nearby food pellets into weighted aggregate targets."""
    clusters: list[dict] = []
    merge_distance = min(
        FOOD_CLUSTER_MAX_RADIUS,
        max(FOOD_CLUSTER_MIN_RADIUS, player_radius * FOOD_CLUSTER_RADIUS_MULT),
    )

    for food in visible_food:
        food_pos = _pos(food)
        food_value = max(FOOD_RADIUS, _radius(food, FOOD_RADIUS))

        closest_cluster = None
        closest_distance = float("inf")
        for cluster in clusters:
            distance = _norm(food_pos - cluster["pos"])
            if distance < merge_distance and distance < closest_distance:
                closest_cluster = cluster
                closest_distance = distance

        if closest_cluster is None:
            clusters.append({"pos": food_pos, "value": food_value, "count": 1})
            continue

        old_value = float(closest_cluster["value"])
        new_value = old_value + food_value
        closest_cluster["pos"] = (closest_cluster["pos"] * old_value + food_pos * food_value) / max(new_value, 1e-9)
        closest_cluster["value"] = new_value
        closest_cluster["count"] = int(closest_cluster.get("count", 1)) + 1

    return clusters


def _food_cluster_score(
    cluster: dict,
    player_pos: np.ndarray,
    angle: float | None = None,
) -> float:
    to_food = cluster["pos"] - player_pos
    distance = max(0.1, _norm(to_food))
    value = float(cluster["value"])
    count = int(cluster.get("count", 1))

    base_score = (
        value * FOOD_VALUE_WEIGHT
        + count * FOOD_COUNT_WEIGHT
        + (value / distance) * FOOD_DENSITY_WEIGHT
        - distance * FOOD_DISTANCE_WEIGHT
    )

    if angle is None:
        return base_score

    alignment = math.cos(_angle_diff(angle, _angle_of(to_food)))
    return base_score * alignment


def _food_local_flow_angle(ctx: _Ctx, previous_angle: float) -> tuple[float, bool]:
    """Small continuous food vector used only for early/stuck mode.

    This is deliberately not the main movement logic because a full vector bot gets
    jittery. It just nudges the stable cluster target toward nearby pellets.
    """
    if not ctx.visible_food and not ctx.food_clusters:
        return previous_angle, False

    previous_vec = np.array(_angle_to_vec(previous_angle), dtype=float)
    flow = previous_vec * 0.25

    for food in ctx.visible_food:
        to_food = _pos(food) - ctx.player_pos
        distance = max(0.1, _norm(to_food))
        unit = to_food / distance
        value = max(FOOD_RADIUS, _radius(food, FOOD_RADIUS))
        flow += unit * value / (distance ** FOOD_LOCAL_DISTANCE_POWER + 0.35)

    for cluster in ctx.food_clusters:
        to_cluster = cluster["pos"] - ctx.player_pos
        distance = max(0.1, _norm(to_cluster))
        unit = to_cluster / distance
        value = float(cluster["value"])
        count = int(cluster.get("count", 1))
        flow += unit * FOOD_LOCAL_FLOW_WEIGHT * (value + 0.35 * count) / (distance ** 0.55 + 0.4)

    if _norm(flow) < 1e-7:
        return previous_angle, False
    return _angle_of(flow), True


def _is_early_farm_mode(ctx: _Ctx) -> bool:
    return ctx.round_frac < EARLY_GAME_THRESHOLD or ctx.largest_blob_radius < EARLY_RADIUS_THRESHOLD


def _prey_aggression(ctx: _Ctx) -> float:
    if _is_early_farm_mode(ctx):
        return EARLY_PREY_MULT
    if ctx.largest_blob_radius < 4.0:
        return MID_PREY_MULT
    return min(LATE_PREY_MULT, MID_PREY_MULT + (ctx.largest_blob_radius - 4.0) * 0.20)


def _enemy_key(blob, fallback_index: int):
    for attr in ("blob_id", "cell_id", "id"):
        if hasattr(blob, attr):
            return (attr, getattr(blob, attr))
    # player_id alone is imperfect with split enemies, but using index keeps this conservative.
    return ("player_index", _owner_id(blob), fallback_index)


def _refresh_enemy_memory(ctx: _Ctx) -> None:
    global _ENEMY_MEMORY, _ENEMY_VELOCITY

    new_memory: dict[object, tuple[np.ndarray, int, float]] = {}
    new_velocity: dict[object, np.ndarray] = {}

    for i, blob in enumerate(ctx.visible_blobs):
        if _is_ours(ctx, blob):
            continue

        key = _enemy_key(blob, i)
        pos = _pos(blob)
        radius = _radius(blob)
        velocity = np.array([0.0, 0.0], dtype=float)

        old = _ENEMY_MEMORY.get(key)
        if old is not None:
            old_pos, old_round, old_radius = old
            dt = max(1, ctx.round_index - old_round)

            # Ignore likely identity mismatches after split/merge.
            radius_close = abs(radius - old_radius) <= max(0.35, old_radius * 0.35)
            if radius_close:
                raw_velocity = (pos - old_pos) / dt
                # Clamp impossible/noisy estimates.
                max_reasonable = _movement_speed(radius) + SPLIT_EJECT_SPEED + 0.8
                if _norm(raw_velocity) <= max_reasonable:
                    velocity = raw_velocity

        new_memory[key] = (pos, ctx.round_index, radius)
        new_velocity[key] = velocity

    _ENEMY_MEMORY = new_memory
    _ENEMY_VELOCITY = new_velocity


def _enemy_velocity(blob, fallback_index: int) -> np.ndarray:
    return _ENEMY_VELOCITY.get(_enemy_key(blob, fallback_index), np.array([0.0, 0.0], dtype=float))


def _target_score(angle: float, ctx: _Ctx) -> float:
    score = 0.0

    for cluster in ctx.food_clusters:
        score += _food_cluster_score(cluster, ctx.player_pos, angle)

    prey_mult = _prey_aggression(ctx)
    hunter_radius = ctx.largest_blob_radius

    for blob in ctx.visible_blobs:
        if _is_ours(ctx, blob):
            continue

        blob_pos = _pos(blob)
        blob_radius = _radius(blob)
        if blob_radius < hunter_radius * PREY_RADIUS_RATIO:
            distance = max(0.1, _edge_distance(ctx.player_pos, hunter_radius, blob_pos, blob_radius))
            alignment = math.cos(_angle_diff(angle, _angle_of(blob_pos - ctx.player_pos)))
            size_value = hunter_radius / max(0.1, blob_radius)
            score += alignment * (PREY_VALUE_WEIGHT * prey_mult * size_value / distance)

    return score


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    line = end - start
    length_sq = float(np.dot(line, line))
    if length_sq == 0.0:
        return _norm(point - start)
    t = max(0.0, min(1.0, float(np.dot(point - start, line) / length_sq)))
    projection = start + t * line
    return _norm(point - projection)


def _angle_is_safe(angle: float, gaps: list[tuple[float, float]], blocked: list[tuple[float, float]]) -> bool:
    if not blocked:
        return True
    return any(_angle_in_gap(angle, gap) for gap in gaps)


def _split_center_travel(child_radius: float) -> float:
    # Split child starts about 2r ahead, then immediately gets normal movement + eject velocity.
    return 2.0 * child_radius + SPLIT_EJECT_SPEED + _movement_speed(child_radius)


def _split_capture_reach(child_radius: float) -> float:
    # Target centre only needs to end up inside the child radius. Keep this conservative.
    return _split_center_travel(child_radius) + child_radius * 0.85


def _split_would_hit_bad_virus(
    ctx: _Ctx,
    split_blob,
    child_radius: float,
    direction: np.ndarray,
) -> bool:
    start = _blob_pos(split_blob)
    end = start + direction * _split_center_travel(child_radius)

    for virus in ctx.visible_viruses:
        virus_pos = _pos(virus)
        virus_radius = _radius(virus)
        if _segment_distance(virus_pos, start, end) > child_radius + virus_radius + VIRUS_PATH_MARGIN:
            continue
        if _can_consume_virus(child_radius, virus_radius):
            return True

    return False


def _enemy_split_capture_reach(enemy_child_radius: float) -> float:
    return _split_center_travel(enemy_child_radius) + enemy_child_radius * 0.85


def _counter_split_reach(enemy_radius: float, my_radius: float) -> float:
    """Centre-distance radius where an enemy split child could land on this blob."""
    if enemy_radius * enemy_radius < SPLIT_MIN_MASS:
        return 0.0

    enemy_child_radius = math.sqrt((enemy_radius * enemy_radius) / 2.0)
    if not _can_eat(enemy_child_radius, my_radius):
        return 0.0

    return _enemy_split_capture_reach(enemy_child_radius) + my_radius + COUNTER_SPLIT_REACTION_MARGIN


def _is_counter_split_threat(enemy_radius: float, my_radius: float, center_distance: float) -> tuple[bool, float]:
    split_reach = _counter_split_reach(enemy_radius, my_radius)
    if split_reach <= 0.0:
        return False, 0.0

    ratio_threat = enemy_radius >= my_radius * COUNTER_SPLIT_MIN_RADIUS_RATIO
    distance_threat = center_distance <= split_reach * COUNTER_SPLIT_DETECTION_MULT
    return ratio_threat or distance_threat, split_reach


def _target_is_baited(ctx: _Ctx, target, target_pos: np.ndarray, target_radius: float) -> bool:
    """Avoid chasing/splitting prey that is sitting inside a larger enemy's kill zone."""
    hunter_radius = ctx.largest_blob_radius

    for i, enemy in enumerate(ctx.visible_blobs):
        if _is_ours(ctx, enemy) or enemy is target:
            continue

        enemy_pos = _pos(enemy)
        enemy_radius = _radius(enemy)

        if enemy_radius <= hunter_radius * THREAT_RADIUS_RATIO:
            continue

        enemy_vel = _enemy_velocity(enemy, i)
        predicted_enemy_pos = enemy_pos + enemy_vel * PREDICTIVE_THREAT_FRAMES
        distance_to_target = _norm(predicted_enemy_pos - target_pos)

        direct_reach = enemy_radius + hunter_radius + SPLIT_BAIT_THREAT_MARGIN
        split_reach = _counter_split_reach(enemy_radius, hunter_radius)

        if distance_to_target <= max(direct_reach, split_reach) + target_radius:
            return True

    for virus in ctx.visible_viruses:
        virus_pos = _pos(virus)
        virus_radius = _radius(virus)
        if _can_consume_virus(hunter_radius, virus_radius):
            if _edge_distance(target_pos, hunter_radius, virus_pos, virus_radius) <= hunter_radius * 0.65:
                return True

    return False


def _split_would_be_punished(
    ctx: _Ctx,
    split_blob,
    child_radius: float,
    direction: np.ndarray,
    target,
) -> bool:
    split_pos = _blob_pos(split_blob)
    child_end = split_pos + direction * _split_center_travel(child_radius)

    for i, blob in enumerate(ctx.visible_blobs):
        if _is_ours(ctx, blob):
            continue
        if blob is target:
            continue

        enemy_pos = _pos(blob)
        enemy_radius = _radius(blob)
        enemy_vel = _enemy_velocity(blob, i)
        predicted_enemy_pos = enemy_pos + enemy_vel * PREDICTIVE_THREAT_FRAMES
        distance_to_child = _norm(predicted_enemy_pos - child_end)

        # Enemy can just eat the post-split child by moving normally.
        if _can_eat(enemy_radius, child_radius):
            if distance_to_child <= enemy_radius + _movement_speed(enemy_radius) + SPLIT_DANGER_LOOKAHEAD:
                return True

        # Enemy can counter-split onto the child.
        if enemy_radius * enemy_radius >= SPLIT_MIN_MASS:
            enemy_child_radius = math.sqrt((enemy_radius * enemy_radius) / 2.0)
            if _can_eat(enemy_child_radius, child_radius):
                if distance_to_child <= _enemy_split_capture_reach(enemy_child_radius) + SPLIT_DANGER_LOOKAHEAD:
                    return True

    return False


def _best_split_target(
    ctx: _Ctx,
    blocked: list[tuple[float, float]],
    gaps: list[tuple[float, float]],
) -> tuple[object | None, float | None]:
    global _LAST_SPLIT_ROUND

    if _is_early_farm_mode(ctx):
        return None, None
    if ctx.round_index - _LAST_SPLIT_ROUND < SPLIT_COOLDOWN_ROUNDS:
        return None, None
    if len(ctx.my_blobs) >= MAX_BLOB_COUNT:
        return None, None
    if SPLIT_ONLY_WHEN_MERGED and _is_multi_or_unmerged(ctx):
        return None, None

    # Engine splits every eligible blob in the same direction. If several blobs
    # are eligible, the risk explodes, so do not split unless we are basically one piece.
    eligible = _eligible_split_blobs(ctx.me)
    if len(eligible) != 1:
        return None, None

    split_blob = _largest_split_blob(ctx.me)
    if split_blob is None:
        return None, None

    split_pos = _blob_pos(split_blob)
    split_radius = _radius(split_blob)
    child_radius = math.sqrt((split_radius * split_radius) / 2.0)
    split_reach = _split_capture_reach(child_radius)

    best_target = None
    best_score = -float("inf")
    best_angle = None

    for blob in ctx.visible_blobs:
        if _is_ours(ctx, blob):
            continue

        target_pos = _pos(blob)
        target_radius = _radius(blob)
        center_distance = _norm(target_pos - split_pos)
        angle = _angle_of(target_pos - split_pos)

        if target_radius < SPLIT_MIN_TARGET_RADIUS:
            continue
        if not _can_eat(child_radius, target_radius):
            continue
        if center_distance > split_reach:
            continue
        if not _angle_is_safe(angle, gaps, blocked):
            continue
        if _target_is_baited(ctx, blob, target_pos, target_radius):
            continue

        direction = _unit(target_pos - split_pos)
        if _split_would_hit_bad_virus(ctx, split_blob, child_radius, direction):
            continue
        if _split_would_be_punished(ctx, split_blob, child_radius, direction, blob):
            continue

        score = (
            target_radius * target_radius * SPLIT_TARGET_VALUE
            - center_distance * SPLIT_MAX_DISTANCE_PENALTY
            + max(0.0, split_reach - center_distance) * 18.0
        )
        if score > best_score:
            best_score = score
            best_target = blob
            best_angle = angle

    return best_target, best_angle


def _add_wall_blocks(
    ctx: _Ctx,
    blocked: list[tuple[float, float]],
    emergency_away: list[np.ndarray],
) -> None:
    if ctx.arena_size <= 0:
        return

    for my_blob in ctx.my_blobs:
        pos = _blob_pos(my_blob)
        radius = _radius(my_blob)
        x, y = float(pos[0]), float(pos[1])

        walls = (
            (x - radius, np.array([0.0, y]), np.array([1.0, 0.0])),
            (ctx.arena_size - x - radius, np.array([ctx.arena_size, y]), np.array([-1.0, 0.0])),
            (y - radius, np.array([x, 0.0]), np.array([0.0, 1.0])),
            (ctx.arena_size - y - radius, np.array([x, ctx.arena_size]), np.array([0.0, -1.0])),
        )

        for edge_dist, wall_point, away in walls:
            if edge_dist < EDGE_AVOID_MARGIN:
                inflated = EDGE_BLOCK_RADIUS + radius * 0.25
                interval = _danger_cone(pos, wall_point, inflated)
                if interval is not None:
                    blocked.append(interval)

                strength = EDGE_ESCAPE_WEIGHT * (EDGE_AVOID_MARGIN / max(0.1, edge_dist + 0.1))
                emergency_away.append(away * strength)


def _blocked_angles(ctx: _Ctx) -> tuple[list[tuple[float, float]], list[np.ndarray], bool]:
    blocked: list[tuple[float, float]] = []
    emergency_away: list[np.ndarray] = []
    has_threat = False

    threat_ratio = THREAT_RADIUS_RATIO if ctx.round_frac > EARLY_GAME_THRESHOLD else EARLY_THREAT_RATIO

    # Enemy danger is per-owned-blob. After splitting, one child can be in danger
    # even when the player's mass-weighted centre looks safe.
    for i, enemy in enumerate(ctx.visible_blobs):
        if _is_ours(ctx, enemy):
            continue

        enemy_pos = _pos(enemy)
        enemy_radius = _radius(enemy)
        enemy_velocity = _enemy_velocity(enemy, i)

        for my_blob in ctx.my_blobs:
            my_pos = _blob_pos(my_blob)
            my_radius = _radius(my_blob)

            predicted_enemy_pos = enemy_pos + enemy_velocity * PREDICTIVE_THREAT_FRAMES
            to_enemy = predicted_enemy_pos - my_pos
            center_distance = max(0.1, _norm(to_enemy))
            edge_distance = _edge_distance(my_pos, my_radius, predicted_enemy_pos, enemy_radius)

            direct_threat = enemy_radius > my_radius * threat_ratio
            counter_split_threat, counter_split_reach = _is_counter_split_threat(
                enemy_radius,
                my_radius,
                center_distance,
            )

            if not direct_threat and not counter_split_threat:
                continue

            direct_reach = enemy_radius + my_radius + SAFETY_MARGIN
            reach = direct_reach

            if counter_split_threat:
                reach = max(reach, counter_split_reach)

            detection_distance = reach + my_radius * 3.6
            if counter_split_threat:
                detection_distance = max(
                    detection_distance,
                    counter_split_reach * COUNTER_SPLIT_DETECTION_MULT,
                )

            # If enemy is moving toward this blob, react earlier.
            approach = -float(np.dot(enemy_velocity, _unit(enemy_pos - my_pos)))
            if approach > 0:
                detection_distance += approach * PREDICTIVE_THREAT_FRAMES * ENEMY_APPROACH_WEIGHT

            if center_distance <= detection_distance or edge_distance <= detection_distance:
                interval = _danger_cone(my_pos, predicted_enemy_pos, reach)
                if interval is not None:
                    blocked.append(interval)
                    has_threat = True

                escape_weight = reach / max(0.1, edge_distance)
                if counter_split_threat:
                    escape_weight *= COUNTER_SPLIT_ESCAPE_WEIGHT
                if approach > 0:
                    escape_weight *= 1.15

                emergency_away.append(-_unit(to_enemy) * escape_weight)

    split_or_unmerged = _is_multi_or_unmerged(ctx)

    # Virus danger is also per-owned-blob and uses the engine's mass check.
    for virus in ctx.visible_viruses:
        virus_pos = _pos(virus)
        virus_radius = _radius(virus)

        for my_blob in ctx.my_blobs:
            blob_pos = _blob_pos(my_blob)
            blob_radius = _radius(my_blob)
            if not _can_consume_virus(blob_radius, virus_radius):
                continue

            edge_distance = _edge_distance(blob_pos, blob_radius, virus_pos, virus_radius)
            extra_margin = VIRUS_BASE_MARGIN
            if split_or_unmerged:
                extra_margin += blob_radius * VIRUS_SPLIT_EXTRA_MARGIN

            close_enough = edge_distance <= blob_radius * VIRUS_BLOB_LOOKAHEAD_MULT + extra_margin
            if not close_enough:
                continue

            reach = virus_radius + blob_radius + extra_margin
            interval = _danger_cone(blob_pos, virus_pos, reach)
            if interval is not None:
                blocked.append(interval)
            emergency_away.append(-_unit(virus_pos - blob_pos) * (reach / max(0.1, edge_distance + 0.1)))

    _add_wall_blocks(ctx, blocked, emergency_away)

    return blocked, emergency_away, has_threat


def _best_safe_angle(
    ctx: _Ctx,
    gaps: list[tuple[float, float]],
    previous_angle: float,
) -> float:
    best_angle = gaps[0][0] + gaps[0][1] / 2.0
    best_score = -float("inf")

    candidate_angles: list[float] = []
    for gap in gaps:
        center = (gap[0] + gap[1] / 2.0) % TAU
        candidate_angles.append(center)
        candidate_angles.append(_shift_into_gap(previous_angle, gap, previous_angle))

        food_flow, found_food_flow = _food_local_flow_angle(ctx, previous_angle)
        if found_food_flow:
            candidate_angles.append(_shift_into_gap(food_flow, gap, previous_angle))

        for cluster in ctx.food_clusters:
            candidate_angles.append(
                _shift_into_gap(_angle_of(cluster["pos"] - ctx.player_pos), gap, previous_angle)
            )

        hunter_radius = ctx.largest_blob_radius
        for blob in ctx.visible_blobs:
            if _is_ours(ctx, blob):
                continue

            if _radius(blob) < hunter_radius * PREY_RADIUS_RATIO:
                candidate_angles.append(
                    _shift_into_gap(_angle_of(_pos(blob) - ctx.player_pos), gap, previous_angle)
                )

    for angle in candidate_angles:
        containing_gaps = [gap for gap in gaps if _angle_in_gap(angle, gap)]
        if not containing_gaps:
            continue

        gap = max(containing_gaps, key=lambda item: item[1])
        center = (gap[0] + gap[1] / 2.0) % TAU
        centeredness = math.cos(_angle_diff(angle, center))
        heading_stickiness = SURVIVAL_HEADING_STICKINESS * math.cos(_angle_diff(angle, previous_angle))
        score = gap[1] * 2.0 + centeredness + heading_stickiness + _target_score(angle, ctx)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle % TAU


def _best_food_angle(ctx: _Ctx, previous_angle: float) -> tuple[float, bool]:
    """Food-only early-game goal, built on the stable cluster scoring."""
    best_angle = previous_angle
    best_score = -float("inf")

    for cluster in ctx.food_clusters:
        to_cluster = cluster["pos"] - ctx.player_pos
        angle = _angle_of(to_cluster)
        heading_stickiness = EARLY_FARM_STICKINESS * math.cos(_angle_diff(angle, previous_angle))
        score = _food_cluster_score(cluster, ctx.player_pos) + heading_stickiness
        if score > best_score:
            best_score = score
            best_angle = angle

    # Light local nudge so it sweeps nearby pellets instead of tunnel-visioning one centroid.
    flow_angle, found_flow = _food_local_flow_angle(ctx, previous_angle)
    if found_flow:
        flow_score = _target_score(flow_angle, ctx) + EARLY_FARM_STICKINESS * math.cos(_angle_diff(flow_angle, previous_angle))
        if flow_score > best_score * 0.92:
            best_angle = flow_angle
            best_score = max(best_score, flow_score)

    return best_angle, best_score != -float("inf")


def _merge_drive_angle(ctx: _Ctx, previous_angle: float) -> tuple[float, float | None]:
    if len(ctx.my_blobs) < 2:
        return previous_angle, None

    ready = [blob for blob in ctx.my_blobs if int(getattr(blob, "merge_cooldown", 9999)) <= MERGE_READY_COOLDOWN]
    if len(ready) < 2:
        return previous_angle, None

    total_mass = sum(_mass(blob) for blob in ctx.my_blobs)
    if total_mass <= 0:
        return previous_angle, None

    centroid = sum((_blob_pos(blob) * _mass(blob) for blob in ctx.my_blobs), start=np.array([0.0, 0.0])) / total_mass
    to_centroid = centroid - ctx.player_pos
    distance = _norm(to_centroid)

    if distance <= ctx.player_radius * 0.35:
        return previous_angle, None

    angle = _angle_of(to_centroid)
    stickiness = math.cos(_angle_diff(angle, previous_angle))
    score = MERGE_PULL_WEIGHT + 4.0 * stickiness + min(10.0, distance)
    return angle, score


def _best_goal_angle(ctx: _Ctx, previous_angle: float) -> tuple[float, bool]:
    best_angle = previous_angle
    best_score = -float("inf")
    farm_stickiness = FARM_HEADING_STICKINESS if ctx.round_frac > EARLY_GAME_THRESHOLD else EARLY_FARM_STICKINESS
    hunter_radius = ctx.largest_blob_radius
    prey_mult = _prey_aggression(ctx)

    for cluster in ctx.food_clusters:
        to_cluster = cluster["pos"] - ctx.player_pos
        angle = _angle_of(to_cluster)
        heading_stickiness = farm_stickiness * math.cos(_angle_diff(angle, previous_angle))
        score = _food_cluster_score(cluster, ctx.player_pos) + heading_stickiness
        if score > best_score:
            best_score = score
            best_angle = angle

    for blob in ctx.visible_blobs:
        if _is_ours(ctx, blob):
            continue

        blob_radius = _radius(blob)
        if blob_radius < hunter_radius * PREY_RADIUS_RATIO:
            blob_pos = _pos(blob)
            if _target_is_baited(ctx, blob, blob_pos, blob_radius):
                continue

            to_blob = blob_pos - ctx.player_pos
            distance = max(0.1, _edge_distance(ctx.player_pos, hunter_radius, blob_pos, blob_radius))
            angle = _angle_of(to_blob)
            heading_stickiness = farm_stickiness * math.cos(_angle_diff(angle, previous_angle))
            score = (
                (hunter_radius / max(0.1, blob_radius)) * PREY_VALUE_WEIGHT * prey_mult * 80.0
                - distance
                + heading_stickiness
            )
            if score > best_score:
                best_score = score
                best_angle = angle

    merge_angle, merge_score = _merge_drive_angle(ctx, previous_angle)
    if merge_score is not None and merge_score > best_score:
        best_score = merge_score
        best_angle = merge_angle

    return best_angle, best_score != -float("inf")


def _best_routed_goal_angle(
    ctx: _Ctx,
    gaps: list[tuple[float, float]],
    goal_angle: float,
    previous_angle: float,
) -> float:
    best_angle = previous_angle
    best_score = -float("inf")

    for gap in gaps:
        center = (gap[0] + gap[1] / 2.0) % TAU
        candidates = [
            _shift_into_gap(goal_angle, gap, previous_angle),
            _shift_into_gap(previous_angle, gap, previous_angle),
            center,
        ]

        food_flow, found_food_flow = _food_local_flow_angle(ctx, previous_angle)
        if found_food_flow:
            candidates.append(_shift_into_gap(food_flow, gap, previous_angle))

        for angle in candidates:
            target_alignment = math.cos(_angle_diff(angle, goal_angle))
            heading_stickiness = math.cos(_angle_diff(angle, previous_angle))
            centeredness = math.cos(_angle_diff(angle, center))
            score = (
                VIRUS_ROUTE_TARGET_WEIGHT * target_alignment
                + VIRUS_ROUTE_HEADING_STICKINESS * heading_stickiness
                + 0.25 * centeredness
                + _target_score(angle, ctx)
            )

            if score > best_score:
                best_score = score
                best_angle = angle

    return best_angle % TAU


def _update_stuck_state(ctx: _Ctx, previous_angle: float) -> bool:
    global _STUCK_MODE_UNTIL

    _RECENT_POSITIONS.append(ctx.player_pos.copy())
    _RECENT_ANGLES.append(previous_angle)

    if ctx.round_index <= _STUCK_MODE_UNTIL:
        return True

    if len(_RECENT_POSITIONS) < STUCK_HISTORY_LEN or not ctx.visible_food:
        return False

    displacement = _norm(_RECENT_POSITIONS[-1] - _RECENT_POSITIONS[0])
    min_displacement = max(0.25, ctx.player_radius * STUCK_MIN_DISPLACEMENT_FACTOR)

    total_turn = 0.0
    previous = None
    for angle in _RECENT_ANGLES:
        if previous is not None:
            total_turn += _angle_diff(previous, angle)
        previous = angle

    avg_turn = total_turn / max(1, len(_RECENT_ANGLES) - 1)
    if displacement < min_displacement and avg_turn > STUCK_TURN_THRESHOLD:
        _STUCK_MODE_UNTIL = ctx.round_index + STUCK_MODE_TICKS
        return True

    return False


def _extract_arena_size(game: Game) -> float:
    state_map = getattr(game.state, "map", None)
    if state_map is None:
        return 0.0

    for attr in ("size", "width", "arena_size"):
        value = getattr(state_map, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return 0.0


def choose_direction(game: Game) -> tuple[float, float, bool]:
    global _LAST_SPLIT_ROUND, _TURN_INDEX

    _TURN_INDEX += 1

    me = game.state.me
    previous_angle = _angle_of(_LAST_DIRECTION)

    state_round = getattr(game.state, "round", None)
    state_round_int = int(state_round) if state_round is not None else 0
    cur_round = max(state_round_int, _TURN_INDEX)

    max_rounds = int(getattr(game.state, "max_rounds", FALLBACK_MAX_ROUNDS) or FALLBACK_MAX_ROUNDS)
    round_frac = min(1.0, cur_round / max(1, max_rounds))

    visible_blobs = list(game.state.visible_blobs or [])
    visible_food = list(game.state.visible_food or [])
    visible_viruses = list(game.state.visible_viruses or [])

    player_pos = np.array([me.x, me.y], dtype=float)
    player_radius = _radius(me)
    my_blobs = _active_blobs(me)
    largest_blob_radius = _largest_owned_radius(my_blobs, player_radius)

    ctx = _Ctx(
        player_id=me.player_id,
        player_pos=player_pos,
        player_radius=player_radius,
        largest_blob_radius=largest_blob_radius,
        me=me,
        my_blobs=my_blobs,
        visible_blobs=visible_blobs,
        visible_food=visible_food,
        visible_viruses=visible_viruses,
        food_clusters=_build_food_clusters(visible_food, largest_blob_radius),
        round_index=cur_round,
        round_frac=round_frac,
        arena_size=_extract_arena_size(game),
    )

    _refresh_enemy_memory(ctx)

    blocked, emergency_away, has_threat = _blocked_angles(ctx)
    gaps = _safe_gaps(blocked)

    # Enemy survival always overrides farming and hunting.
    if has_threat:
        if gaps:
            angle = _best_safe_angle(ctx, gaps, previous_angle)
            dx, dy = _smoothed_direction(np.array(_angle_to_vec(angle)), SURVIVAL_SMOOTHING)
            return dx, dy, False

        escape = np.sum(emergency_away, axis=0) if emergency_away else np.array([1.0, 0.0])
        dx, dy = _smoothed_direction(escape, SURVIVAL_SMOOTHING)
        return dx, dy, False

    # If viruses/walls fully close the angular space, do not ignore them just because
    # no enemy threat is active.
    if blocked and not gaps:
        escape = np.sum(emergency_away, axis=0) if emergency_away else -_LAST_DIRECTION
        dx, dy = _smoothed_direction(escape, SURVIVAL_SMOOTHING)
        return dx, dy, False

    stuck_mode = _update_stuck_state(ctx, previous_angle)
    if stuck_mode:
        angle, found_food = _best_food_angle(ctx, previous_angle)
        if found_food:
            if blocked and gaps:
                angle = _best_routed_goal_angle(ctx, gaps, angle, previous_angle)
            dx, dy = _smoothed_direction(np.array(_angle_to_vec(angle)), STUCK_SMOOTHING)
            return dx, dy, False

    # Early/small mode: greedily farm dots. No prey chasing and no splitting unless survival overrides.
    if _is_early_farm_mode(ctx):
        angle, found_food = _best_food_angle(ctx, previous_angle)
        if found_food:
            if blocked and gaps:
                angle = _best_routed_goal_angle(ctx, gaps, angle, previous_angle)
            dx, dy = _smoothed_direction(np.array(_angle_to_vec(angle)), EARLY_FARM_SMOOTHING)
            return dx, dy, False

    split_target, split_angle = _best_split_target(ctx, blocked, gaps)
    if split_target is not None and split_angle is not None:
        _LAST_SPLIT_ROUND = cur_round
        dx, dy = _smoothed_direction(np.array(_angle_to_vec(split_angle)), 0.0)
        return dx, dy, True

    best_angle, found_goal = _best_goal_angle(ctx, previous_angle)
    if not found_goal:
        dx, dy = _smoothed_direction(np.array([1.0, 0.0]), FARMING_SMOOTHING)
        return dx, dy, False

    if blocked and gaps:
        best_angle = _best_routed_goal_angle(ctx, gaps, best_angle, previous_angle)

    dx, dy = _smoothed_direction(np.array(_angle_to_vec(best_angle)), FARMING_SMOOTHING)
    return dx, dy, False


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                dx, dy, split = choose_direction(game)
                game.send_move(
                    MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                        split=split,
                    )
                )
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()