from __future__ import annotations

import math

import numpy as np

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel


TAU = 2.0 * math.pi

# Engine-grounded constants from agario-public-main.
BASE_PLAYER_SPEED = 1.1
PLAYER_SPEED_RADIUS_FACTOR = 0.08
MIN_PLAYER_SPEED = 0.25
EAT_SIZE_RATIO = 1.2
MAX_BLOB_COUNT = 16
SPLIT_MIN_MASS = 2.0
SPLIT_COOLDOWN_FRAMES = 18
SPLIT_EJECT_SPEED = 1.6
SPLIT_EJECT_DRAG = 0.82
FOOD_RADIUS = 0.15

# Farming / prey scoring.
FOOD_CLUSTER_RADIUS_MULT = 2.35
FOOD_CLUSTER_MIN_RADIUS = 1.15
FOOD_CLUSTER_MAX_RADIUS = 5.5
FOOD_VALUE_WEIGHT = 7.0
FOOD_COUNT_WEIGHT = 8.0
FOOD_DENSITY_WEIGHT = 13.0
FOOD_DISTANCE_WEIGHT = 0.45
PREY_VALUE_WEIGHT = 3.0
PREY_RADIUS_RATIO = 0.82

# Threat / survival scoring.
THREAT_RADIUS_RATIO = 1.06
SAFETY_MARGIN = 2.2
SPLIT_THREAT_RATIO = 1.65
SPLIT_REACH_MULT = 3.2

# Counter-split prediction. This is the main anti-"big blob suddenly splits and eats us" mechanism.
COUNTER_SPLIT_REACTION_MARGIN = 1.4
COUNTER_SPLIT_DETECTION_MULT = 1.35
COUNTER_SPLIT_ESCAPE_WEIGHT = 2.4
COUNTER_SPLIT_MIN_RADIUS_RATIO = 1.55

SURVIVAL_HEADING_STICKINESS = 1.2
FARM_HEADING_STICKINESS = 18.0
VIRUS_ROUTE_TARGET_WEIGHT = 4.0
VIRUS_ROUTE_HEADING_STICKINESS = 1.8
SURVIVAL_SMOOTHING = 0.12
FARMING_SMOOTHING = 0.52

# Split-chase tuning. The engine splits every eligible blob, so we only split
# while merged / stable, and we use a cooldown longer than the engine cooldown.
SPLIT_TARGET_VALUE = 230.0
SPLIT_REACH_RADIUS_MULT = 2.85
SPLIT_DANGER_LOOKAHEAD = 2.8
SPLIT_COOLDOWN_ROUNDS = SPLIT_COOLDOWN_FRAMES + 4
SPLIT_ONLY_WHEN_MERGED = True
SPLIT_MIN_TARGET_RADIUS = 0.35
SPLIT_MAX_DISTANCE_PENALTY = 35.0

# Virus handling. Virus consumption is based on blob mass, not player radius.
VIRUS_BLOB_LOOKAHEAD_MULT = 3.2
VIRUS_SPLIT_EXTRA_MARGIN = 1.15
VIRUS_BASE_MARGIN = 0.35
VIRUS_PATH_MARGIN = 0.2

# Early-game aggressiveness. Higher threat ratio = fewer enemies blocked.
EARLY_GAME_THRESHOLD = 0.25
EARLY_THREAT_RATIO = 1.15
EARLY_FARM_STICKINESS = 7.0

_LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
_LAST_SPLIT_ROUND = -9999


class _Ctx:
    """Per-turn precomputed state shared across helper functions."""

    __slots__ = (
        "player_id",
        "player_pos",
        "player_radius",
        "me",
        "my_blobs",
        "visible_blobs",
        "visible_food",
        "visible_viruses",
        "food_clusters",
        "round_index",
        "round_frac",
    )

    def __init__(
        self,
        player_id: int,
        player_pos: np.ndarray,
        player_radius: float,
        me,
        my_blobs: list,
        visible_blobs: list,
        visible_food: list,
        visible_viruses: list,
        food_clusters: list,
        round_index: int,
        round_frac: float,
    ) -> None:
        self.player_id = player_id
        self.player_pos = player_pos
        self.player_radius = player_radius
        self.me = me
        self.my_blobs = my_blobs
        self.visible_blobs = visible_blobs
        self.visible_food = visible_food
        self.visible_viruses = visible_viruses
        self.food_clusters = food_clusters
        self.round_index = round_index
        self.round_frac = round_frac


def _pos(obj) -> np.ndarray:
    return np.array(obj.pos, dtype=float)


def _radius(obj, default: float = 1.0) -> float:
    return float(getattr(obj, "radius", getattr(obj, "size", default)))


def _mass(obj) -> float:
    radius = _radius(obj)
    return radius * radius


def _norm(v: np.ndarray) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _unit(v: np.ndarray) -> np.ndarray:
    length = _norm(v)
    if length == 0:
        return np.array([1.0, 0.0])
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
    if hasattr(blob, "pos"):
        return _pos(blob)
    return np.array([blob.x, blob.y], dtype=float)


def _eligible_split_blobs(player) -> list:
    return [blob for blob in _my_blobs(player) if _mass(blob) >= SPLIT_MIN_MASS]


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
    # Engine uses: blob.mass > virus.radius * EAT_SIZE_RATIO.
    return blob_radius * blob_radius > virus_radius * EAT_SIZE_RATIO


def _blocked_interval(center_angle: float, half_width: float) -> tuple[float, float] | None:
    if half_width >= math.pi:
        return (0.0, TAU)
    return ((center_angle - half_width) % TAU, (center_angle + half_width) % TAU)


def _danger_cone(
    player_pos: np.ndarray,
    target_pos: np.ndarray,
    inflated_radius: float,
) -> tuple[float, float] | None:
    to_target = target_pos - player_pos
    distance = _norm(to_target)

    if distance == 0:
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
        food_value = _radius(food)

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
        if new_value > 0:
            closest_cluster["pos"] = (closest_cluster["pos"] * old_value + food_pos * food_value) / new_value
        else:
            closest_cluster["pos"] = (closest_cluster["pos"] + food_pos) / 2.0
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

    # Value/count/density makes a real cloud beat a single slightly-closer pellet.
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


def _target_score(angle: float, ctx: _Ctx) -> float:
    score = 0.0

    for cluster in ctx.food_clusters:
        score += _food_cluster_score(cluster, ctx.player_pos, angle)

    for blob in ctx.visible_blobs:
        if blob.player_id == ctx.player_id:
            continue
        blob_pos = _pos(blob)
        blob_radius = _radius(blob)
        if blob_radius < ctx.player_radius * PREY_RADIUS_RATIO:
            distance = max(0.1, _edge_distance(ctx.player_pos, ctx.player_radius, blob_pos, blob_radius))
            alignment = math.cos(_angle_diff(angle, _angle_of(blob_pos - ctx.player_pos)))
            size_value = ctx.player_radius / max(0.1, blob_radius)
            score += alignment * (PREY_VALUE_WEIGHT * size_value / distance)

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
    """Return whether enemy can split-kill us soon and the predicted danger reach."""
    split_reach = _counter_split_reach(enemy_radius, my_radius)
    if split_reach <= 0.0:
        return False, 0.0

    # Ratio check catches obvious split threats early.
    # Distance check catches close enemies even when ratios are barely over threshold.
    ratio_threat = enemy_radius >= my_radius * COUNTER_SPLIT_MIN_RADIUS_RATIO
    distance_threat = center_distance <= split_reach * COUNTER_SPLIT_DETECTION_MULT
    return ratio_threat or distance_threat, split_reach


def _split_would_be_punished(
    ctx: _Ctx,
    split_blob,
    child_radius: float,
    direction: np.ndarray,
    target,
) -> bool:
    split_pos = _blob_pos(split_blob)
    child_end = split_pos + direction * _split_center_travel(child_radius)

    for blob in ctx.visible_blobs:
        if blob.player_id == ctx.player_id:
            continue
        if blob is target:
            continue

        enemy_pos = _pos(blob)
        enemy_radius = _radius(blob)
        distance_to_child = _norm(enemy_pos - child_end)

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
        if blob.player_id == ctx.player_id:
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


def _blocked_angles(ctx: _Ctx) -> tuple[list[tuple[float, float]], list[np.ndarray], bool]:
    blocked: list[tuple[float, float]] = []
    emergency_away: list[np.ndarray] = []
    has_threat = False

    threat_ratio = THREAT_RADIUS_RATIO if ctx.round_frac > EARLY_GAME_THRESHOLD else EARLY_THREAT_RATIO

    # Enemy danger is per-owned-blob. After splitting, one child can be in danger
    # even when the player's mass-weighted centre looks safe.
    for enemy in ctx.visible_blobs:
        if enemy.player_id == ctx.player_id:
            continue

        enemy_pos = _pos(enemy)
        enemy_radius = _radius(enemy)

        for my_blob in ctx.my_blobs:
            my_pos = _blob_pos(my_blob)
            my_radius = _radius(my_blob)
            to_enemy = enemy_pos - my_pos
            center_distance = max(0.1, _norm(to_enemy))
            edge_distance = _edge_distance(my_pos, my_radius, enemy_pos, enemy_radius)

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

            # The old code only widened the enemy cone once the enemy was already
            # obviously huge. This predicts the split child's landing zone instead,
            # so we start dodging before the split actually happens.
            if counter_split_threat:
                reach = max(reach, counter_split_reach)

            detection_distance = reach + my_radius * 3.5
            if counter_split_threat:
                detection_distance = max(
                    detection_distance,
                    counter_split_reach * COUNTER_SPLIT_DETECTION_MULT,
                )

            if center_distance <= detection_distance or edge_distance <= detection_distance:
                interval = _danger_cone(my_pos, enemy_pos, reach)
                if interval is not None:
                    blocked.append(interval)
                    has_threat = True

                escape_weight = reach / max(0.1, edge_distance)
                if counter_split_threat:
                    escape_weight *= COUNTER_SPLIT_ESCAPE_WEIGHT

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
        candidate_angles.append((gap[0] + gap[1] / 2.0) % TAU)
        candidate_angles.append(_shift_into_gap(previous_angle, gap, previous_angle))

        for cluster in ctx.food_clusters:
            candidate_angles.append(
                _shift_into_gap(_angle_of(cluster["pos"] - ctx.player_pos), gap, previous_angle)
            )

        for blob in ctx.visible_blobs:
            if blob.player_id == ctx.player_id:
                continue
            if _radius(blob) < ctx.player_radius * PREY_RADIUS_RATIO:
                candidate_angles.append(
                    _shift_into_gap(_angle_of(_pos(blob) - ctx.player_pos), gap, previous_angle)
                )

    for angle in candidate_angles:
        containing_gaps = [gap for gap in gaps if _angle_in_gap(angle, gap)]
        if not containing_gaps:
            continue

        gap = max(containing_gaps, key=lambda item: item[1])
        centeredness = math.cos(_angle_diff(angle, (gap[0] + gap[1] / 2.0) % TAU))
        heading_stickiness = SURVIVAL_HEADING_STICKINESS * math.cos(_angle_diff(angle, previous_angle))
        score = gap[1] * 2.0 + centeredness + heading_stickiness + _target_score(angle, ctx)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle % TAU


def _best_goal_angle(ctx: _Ctx, previous_angle: float) -> tuple[float, bool]:
    best_angle = previous_angle
    best_score = -float("inf")
    farm_stickiness = FARM_HEADING_STICKINESS if ctx.round_frac > EARLY_GAME_THRESHOLD else EARLY_FARM_STICKINESS

    for cluster in ctx.food_clusters:
        to_cluster = cluster["pos"] - ctx.player_pos
        angle = _angle_of(to_cluster)
        heading_stickiness = farm_stickiness * math.cos(_angle_diff(angle, previous_angle))
        score = _food_cluster_score(cluster, ctx.player_pos) + heading_stickiness
        if score > best_score:
            best_score = score
            best_angle = angle

    for blob in ctx.visible_blobs:
        if blob.player_id == ctx.player_id:
            continue

        blob_radius = _radius(blob)
        if blob_radius < ctx.player_radius * PREY_RADIUS_RATIO:
            blob_pos = _pos(blob)
            to_blob = blob_pos - ctx.player_pos
            distance = max(0.1, _edge_distance(ctx.player_pos, ctx.player_radius, blob_pos, blob_radius))
            angle = _angle_of(to_blob)
            heading_stickiness = farm_stickiness * math.cos(_angle_diff(angle, previous_angle))
            score = (
                (ctx.player_radius / max(0.1, blob_radius)) * PREY_VALUE_WEIGHT * 80.0
                - distance
                + heading_stickiness
            )
            if score > best_score:
                best_score = score
                best_angle = angle

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


def choose_direction(game: Game) -> tuple[float, float, bool]:
    global _LAST_SPLIT_ROUND

    me = game.state.me
    previous_angle = _angle_of(_LAST_DIRECTION)

    cur_round = int(getattr(game.state, "round", 0) or 0)
    max_rounds = int(getattr(game.state, "max_rounds", 1) or 1)
    round_frac = cur_round / max(1, max_rounds)

    visible_blobs = list(game.state.visible_blobs or [])
    visible_food = list(game.state.visible_food or [])
    visible_viruses = list(game.state.visible_viruses or [])
    player_pos = np.array([me.x, me.y], dtype=float)
    player_radius = _radius(me)
    my_blobs = _active_blobs(me)

    ctx = _Ctx(
        player_id=me.player_id,
        player_pos=player_pos,
        player_radius=player_radius,
        me=me,
        my_blobs=my_blobs,
        visible_blobs=visible_blobs,
        visible_food=visible_food,
        visible_viruses=visible_viruses,
        food_clusters=_build_food_clusters(visible_food, player_radius),
        round_index=cur_round,
        round_frac=round_frac,
    )

    blocked, emergency_away, has_threat = _blocked_angles(ctx)
    gaps = _safe_gaps(blocked)

    if has_threat:
        if gaps:
            angle = _best_safe_angle(ctx, gaps, previous_angle)
            dx, dy = _smoothed_direction(np.array(_angle_to_vec(angle)), SURVIVAL_SMOOTHING)
            return dx, dy, False

        escape = np.sum(emergency_away, axis=0) if emergency_away else np.array([1.0, 0.0])
        dx, dy = _smoothed_direction(escape, SURVIVAL_SMOOTHING)
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