from __future__ import annotations

import math

import numpy as np

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel


TAU = 2.0 * math.pi

# Hard per-turn budgets. The original implementation clustered every visible
# food item (quadratic work) and compared every enemy with every owned blob.
# Nearest objects are the only ones that can affect the next decision.
MAX_FOOD_CONSIDERED = 28
MAX_BLOBS_CONSIDERED = 20
MAX_VIRUSES_CONSIDERED = 10

FOOD_CLUSTER_RADIUS_MULT = 2.0
FOOD_VALUE_WEIGHT = 6.0
PREY_VALUE_WEIGHT = 3.0
THREAT_RADIUS_RATIO = 1.08
PREY_RADIUS_RATIO = 0.83       # ≈ 1/EAT_SIZE_RATIO = 1/1.2 = 0.833; hunt any blob we can eat
EAT_SIZE_RATIO = 1.2
MAX_BLOB_COUNT = 16
SPLIT_MIN_MASS = 2.0
SPLIT_EJECT_SPEED = 1.6
SPLIT_TARGET_VALUE = 240.0
SPLIT_REACH_EXTRA_TICKS = 3.0
SPLIT_DANGER_LOOKAHEAD = 2.5
SPLIT_COOLDOWN_ROUNDS = 24    # minimum rounds between splits to prevent chain-splitting
SAFETY_MARGIN = 2.0            # buffer above eater.radius – was 18.0 which is 30% of the 60×60 arena
SPLIT_THREAT_RATIO = 1.75      # child after split needs to eat us: 1.2*√2 ≈ 1.697; use 1.75 for margin
SPLIT_EXTRA_REACH = 7.0        # extra cone reach for split threats (≈4 units eject + 3 units directional in 3 ticks)
VIRUS_DANGER_RATIO = 0.75
SURVIVAL_HEADING_STICKINESS = 1.5
FARM_HEADING_STICKINESS = 12.0  # was 25.0 – more responsive in tight 60×60 arena
VIRUS_ROUTE_TARGET_WEIGHT = 4.0
VIRUS_ROUTE_HEADING_STICKINESS = 2.5
SURVIVAL_SMOOTHING = 0.35
FARMING_SMOOTHING = 0.65

# Early-game aggressiveness: looser threat detection and faster direction changes
EARLY_GAME_THRESHOLD = 0.25   # first 25% of rounds counts as early game
EARLY_THREAT_RATIO = 1.3      # higher = fewer angles blocked → more aggressive movement
EARLY_FARM_STICKINESS = 8.0   # lower = faster pivots to chase food/prey

# Arena edge avoidance (arena is 60×60; coordinates go from 0 to ARENA_SIZE)
EDGE_AVOID_MARGIN = 6.0       # wall danger activates within this distance of player center
EDGE_BLOCK_RADIUS = 3.5       # virtual radius for wall danger-cone calculation

# Merge drive: converge split blobs once merge cooldown expires
MERGE_PULL_WEIGHT = 85.0      # goal score weight for blob convergence direction
MERGE_READY_COOLDOWN = 3      # merge_cooldown threshold to count a blob as merge-ready

# Prey hunting urgency scales with size (mass decay = 0.002/tick; large blobs bleed mass)
LARGE_RADIUS_THRESHOLD = 3.0  # radius above which prey-score bonus activates
LARGE_PREY_BONUS = 2.0        # max prey-score multiplier when at or above threshold

_LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
_LAST_SPLIT_ROUND: int = -9999


class _Ctx:
    """Per-turn precomputed state shared across all helper functions."""

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
        "round_frac",
        "round_index",
        "arena_size",
    )

    def __init__(
        self,
        player_id,
        player_pos: np.ndarray,
        player_radius: float,
        me,
        my_blobs: list,
        visible_blobs: list,
        visible_food: list,
        visible_viruses: list,
        food_clusters: list,
        round_frac: float,
        round_index: int,
        arena_size: float,
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
        self.round_frac = round_frac
        self.round_index = round_index
        self.arena_size = arena_size


def _pos(obj) -> np.ndarray:
    return np.array(obj.pos, dtype=float)


def _nearest(objects: list, origin: np.ndarray, limit: int) -> list:
    if len(objects) <= limit:
        return objects
    ox, oy = float(origin[0]), float(origin[1])
    return sorted(
        objects,
        key=lambda obj: (float(obj.pos[0]) - ox) ** 2 + (float(obj.pos[1]) - oy) ** 2,
    )[:limit]


def _radius(obj, default: float = 1.0) -> float:
    return float(getattr(obj, "radius", getattr(obj, "size", default)))


def _mass(obj) -> float:
    r = _radius(obj)
    return r * r


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


def _largest_split_blob(player):
    eligible = [blob for blob in _my_blobs(player) if _mass(blob) >= SPLIT_MIN_MASS]
    if not eligible:
        return None
    return max(eligible, key=_radius)


def _eligible_split_blobs(player) -> list:
    """All owned blobs with enough mass to split."""
    return [blob for blob in _my_blobs(player) if _mass(blob) >= SPLIT_MIN_MASS]


_BASE_PLAYER_SPEED = 1.1
_PLAYER_SPEED_RADIUS_FACTOR = 0.08
_MIN_PLAYER_SPEED = 0.25


def _movement_speed(radius: float) -> float:
    """Engine speed formula: max(MIN, BASE / (1 + radius * FACTOR))."""
    return max(_MIN_PLAYER_SPEED, _BASE_PLAYER_SPEED / (1.0 + radius * _PLAYER_SPEED_RADIUS_FACTOR))


def _split_capture_reach(child_radius: float) -> float:
    """Engine-grounded split capture radius: eject offset + eject speed + child movement speed."""
    return 3.0 * child_radius + SPLIT_EJECT_SPEED + _movement_speed(child_radius)


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
    """Engine condition: blob.mass > virus.radius * EAT_SIZE_RATIO (mass = radius²)."""
    return blob_radius * blob_radius > virus_radius * EAT_SIZE_RATIO


def _blob_pos(blob) -> np.ndarray:
    if hasattr(blob, "pos"):
        return _pos(blob)
    return np.array([blob.x, blob.y], dtype=float)


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
    """Cluster nearby food pellets into aggregate targets. Called once per turn."""
    clusters: list[dict] = []
    merge_distance = max(1.0, player_radius * FOOD_CLUSTER_RADIUS_MULT)

    for food in visible_food:
        food_pos = _pos(food)
        food_value = _radius(food)

        for cluster in clusters:
            if _norm(food_pos - cluster["pos"]) < merge_distance:
                old_value = cluster["value"]
                new_value = old_value + food_value
                cluster["pos"] = (cluster["pos"] * old_value + food_pos * food_value) / new_value
                cluster["value"] = new_value
                cluster["count"] += 1
                break
        else:
            clusters.append({"pos": food_pos, "value": food_value, "count": 1})

    return clusters


def _target_score(angle: float, ctx: _Ctx) -> float:
    score = 0.0

    for cluster in ctx.food_clusters:
        to_food = cluster["pos"] - ctx.player_pos
        distance = max(1.0, _norm(to_food))
        alignment = math.cos(_angle_diff(angle, _angle_of(to_food)))
        score += alignment * (cluster["value"] * FOOD_VALUE_WEIGHT / distance)

    for blob in ctx.visible_blobs:
        if blob.player_id == ctx.player_id:
            continue
        blob_pos = _pos(blob)
        blob_radius = _radius(blob)
        if blob_radius < ctx.player_radius * PREY_RADIUS_RATIO:
            distance = max(1.0, _edge_distance(ctx.player_pos, ctx.player_radius, blob_pos, blob_radius))
            alignment = math.cos(_angle_diff(angle, _angle_of(blob_pos - ctx.player_pos)))
            size_value = ctx.player_radius / max(1.0, blob_radius)
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


def _split_would_hit_bad_virus(
    ctx: _Ctx,
    split_blob,
    child_radius: float,
    direction: np.ndarray,
) -> bool:
    start = _blob_pos(split_blob)
    end = start + direction * (3.0 * child_radius + SPLIT_EJECT_SPEED * SPLIT_REACH_EXTRA_TICKS)

    for virus in ctx.visible_viruses:
        virus_pos = _pos(virus)
        virus_radius = _radius(virus)
        if _segment_distance(virus_pos, start, end) > child_radius + virus_radius:
            continue
        if _can_consume_virus(child_radius, virus_radius):
            return True

    return False


def _split_would_be_punished(
    ctx: _Ctx,
    split_blob,
    child_radius: float,
    target,
) -> bool:
    split_pos = _blob_pos(split_blob)

    for blob in ctx.visible_blobs:
        if blob.player_id == ctx.player_id:
            continue
        if blob is target:
            continue

        enemy_radius = _radius(blob)
        if not _can_eat(enemy_radius, child_radius):
            continue

        enemy_pos = _pos(blob)
        edge_distance = _edge_distance(split_pos, child_radius, enemy_pos, enemy_radius)
        if edge_distance <= enemy_radius + ctx.player_radius * SPLIT_DANGER_LOOKAHEAD:
            return True

    return False


def _best_split_target(
    ctx: _Ctx,
    blocked: list[tuple[float, float]],
    gaps: list[tuple[float, float]],
) -> tuple[object | None, float | None]:
    if len(ctx.my_blobs) >= MAX_BLOB_COUNT:
        return None, None

    # Only split when exactly one blob is eligible: prevents unsafe chain-splits where a
    # second blob also fires and flies into a virus/enemy/wall uncontrolled.
    eligible = _eligible_split_blobs(ctx.me)
    if len(eligible) != 1:
        return None, None

    # Cooldown prevents rapid-fire chain-splitting into chaos.
    if ctx.round_index - _LAST_SPLIT_ROUND < SPLIT_COOLDOWN_ROUNDS:
        return None, None

    split_blob = eligible[0]
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

        if not _can_eat(child_radius, target_radius):
            continue
        if center_distance > split_reach:
            continue
        if not _angle_is_safe(angle, gaps, blocked):
            continue

        direction = _unit(target_pos - split_pos)
        if _split_would_hit_bad_virus(ctx, split_blob, child_radius, direction):
            continue
        if _split_would_be_punished(ctx, split_blob, child_radius, blob):
            continue

        score = (
            target_radius * target_radius * SPLIT_TARGET_VALUE
            - center_distance * 35.0
            + max(0.0, split_reach - center_distance) * 20.0
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

    # Higher threat_ratio → fewer enemies trigger avoidance → more aggressive movement.
    threat_ratio = THREAT_RADIUS_RATIO if ctx.round_frac > EARLY_GAME_THRESHOLD else EARLY_THREAT_RATIO

    # Per-blob enemy danger: check each owned blob against each enemy blob.
    # After splitting, a child blob may be threatened even if the merged centroid looks safe.
    for enemy in ctx.visible_blobs:
        if enemy.player_id == ctx.player_id:
            continue

        enemy_pos = _pos(enemy)
        enemy_radius = _radius(enemy)

        for my_blob in ctx.my_blobs:
            my_blob_pos = _blob_pos(my_blob)
            my_blob_radius = _radius(my_blob)

            if enemy_radius <= my_blob_radius * threat_ratio:
                continue

            can_split_threat = enemy_radius >= my_blob_radius * SPLIT_THREAT_RATIO
            reach = enemy_radius + SAFETY_MARGIN
            if can_split_threat:
                reach += SPLIT_EXTRA_REACH

            edge_distance = _edge_distance(my_blob_pos, my_blob_radius, enemy_pos, enemy_radius)
            detection_distance = reach + my_blob_radius * 4.0
            if edge_distance <= detection_distance:
                interval = _danger_cone(my_blob_pos, enemy_pos, reach)
                if interval is not None:
                    blocked.append(interval)
                    has_threat = True
                emergency_away.append(-_unit(enemy_pos - ctx.player_pos) * (reach / max(1.0, edge_distance)))

    # Per-blob virus danger: each blob may independently pop a virus.
    # Engine condition: blob.mass > virus.radius * EAT_SIZE_RATIO (mass = radius²).
    for virus in ctx.visible_viruses:
        virus_pos = _pos(virus)
        virus_radius = _radius(virus)

        for my_blob in ctx.my_blobs:
            my_blob_pos = _blob_pos(my_blob)
            my_blob_radius = _radius(my_blob)

            if not _can_consume_virus(my_blob_radius, virus_radius):
                continue

            edge_distance = _edge_distance(my_blob_pos, my_blob_radius, virus_pos, virus_radius)
            close_enough = edge_distance <= virus_radius + my_blob_radius * 3.0 + SAFETY_MARGIN
            if close_enough:
                reach = virus_radius + my_blob_radius + SAFETY_MARGIN
                interval = _danger_cone(my_blob_pos, virus_pos, reach)
                if interval is not None:
                    blocked.append(interval)

    # Wall avoidance: treat arena boundaries as virtual obstacles.
    # Engine clamps blobs to [radius, arena_size - radius], so walls are at 0 and arena_size.
    if ctx.arena_size > 0:
        px, py = float(ctx.player_pos[0]), float(ctx.player_pos[1])
        for wall_pt in (
            np.array([0.0, py]),
            np.array([ctx.arena_size, py]),
            np.array([px, 0.0]),
            np.array([px, ctx.arena_size]),
        ):
            dist = _norm(wall_pt - ctx.player_pos)
            if 0 < dist < EDGE_AVOID_MARGIN:
                interval = _danger_cone(ctx.player_pos, wall_pt, EDGE_BLOCK_RADIUS)
                if interval is not None:
                    blocked.append(interval)
                # Push emergency escape vectors away from wall
                emergency_away.append(
                    _unit(ctx.player_pos - wall_pt) * (EDGE_AVOID_MARGIN / max(1.0, dist))
                )

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
    # Early game: lower stickiness → faster pivots to chase food/prey
    farm_stickiness = FARM_HEADING_STICKINESS if ctx.round_frac > EARLY_GAME_THRESHOLD else EARLY_FARM_STICKINESS
    # Large blobs decay at 0.2%/tick: proportionally boost prey attraction to compensate
    prey_mult = (
        min(LARGE_PREY_BONUS, LARGE_PREY_BONUS * ctx.player_radius / LARGE_RADIUS_THRESHOLD)
        if ctx.player_radius > LARGE_RADIUS_THRESHOLD else 1.0
    )

    for cluster in ctx.food_clusters:
        to_cluster = cluster["pos"] - ctx.player_pos
        distance = max(1.0, _norm(to_cluster))
        angle = _angle_of(to_cluster)
        heading_stickiness = farm_stickiness * math.cos(_angle_diff(angle, previous_angle))
        score = (
            cluster["value"] * FOOD_VALUE_WEIGHT
            + cluster["count"] * 8.0
            + (cluster["value"] / distance) * 12.0
            - distance * 0.45
            + heading_stickiness
        )
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
            distance = max(1.0, _edge_distance(ctx.player_pos, ctx.player_radius, blob_pos, blob_radius))
            angle = _angle_of(to_blob)
            heading_stickiness = farm_stickiness * math.cos(_angle_diff(angle, previous_angle))
            score = (ctx.player_radius / max(1.0, blob_radius)) * PREY_VALUE_WEIGHT * prey_mult * 100.0 - distance + heading_stickiness
            if score > best_score:
                best_score = score
                best_angle = angle

    # Merge drive: when ≥2 blobs have expired cooldown, aim toward their mass-weighted centroid
    # to collapse the vulnerable split window as quickly as possible.
    if len(ctx.my_blobs) > 1:
        ready = [b for b in ctx.my_blobs if int(getattr(b, "merge_cooldown", 999)) <= MERGE_READY_COOLDOWN]
        if len(ready) >= 2:
            total_m = sum(_radius(b) ** 2 for b in ctx.my_blobs)
            if total_m > 0:
                cx = sum(_blob_pos(b)[0] * _radius(b) ** 2 for b in ctx.my_blobs) / total_m
                cy = sum(_blob_pos(b)[1] * _radius(b) ** 2 for b in ctx.my_blobs) / total_m
                centroid = np.array([cx, cy])
                to_centroid = centroid - ctx.player_pos
                dist_centroid = _norm(to_centroid)
                if dist_centroid > ctx.player_radius * 0.5:
                    angle = _angle_of(to_centroid)
                    heading_stickiness = farm_stickiness * math.cos(_angle_diff(angle, previous_angle))
                    score = MERGE_PULL_WEIGHT + heading_stickiness
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
    me = game.state.me
    previous_angle = _angle_of(_LAST_DIRECTION)

    cur_round = int(getattr(game.state, "round", 0) or 0)
    max_rounds = int(getattr(game.state, "max_rounds", 1) or 1)
    round_frac = cur_round / max(1, max_rounds)
    arena_size = float(getattr(game.state.map, "size", 0) or 0)

    player_pos = np.array([me.x, me.y], dtype=float)
    player_radius = _radius(me)
    visible_blobs = _nearest(
        list(game.state.visible_blobs or []), player_pos, MAX_BLOBS_CONSIDERED
    )
    visible_food = _nearest(
        list(game.state.visible_food or []), player_pos, MAX_FOOD_CONSIDERED
    )
    visible_viruses = _nearest(
        list(game.state.visible_viruses or []), player_pos, MAX_VIRUSES_CONSIDERED
    )

    ctx = _Ctx(
        player_id=me.player_id,
        player_pos=player_pos,
        player_radius=player_radius,
        me=me,
        my_blobs=_active_blobs(me),
        visible_blobs=visible_blobs,
        visible_food=visible_food,
        visible_viruses=visible_viruses,
        food_clusters=_build_food_clusters(visible_food, player_radius),
        round_frac=round_frac,
        round_index=cur_round,
        arena_size=arena_size,
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
        global _LAST_SPLIT_ROUND
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