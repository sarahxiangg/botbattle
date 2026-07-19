from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np


# Baseline-preserving rank-aware last-stand candidate (submission #123).
# Identical to the successful #114 policy before round 900, apart from bounded
# chase evaluation and correctness fixes. Late leaders remain exactly on the
# baseline policy; only late trailers gain additional mass-positive options.


# =============================================================================
# Tuning surface
# =============================================================================
#
# Every strategically meaningful number remains centralized here so a later
# optimiser can expose selected values.  This competition build is deliberately
# standalone: it performs no filesystem/config reads during startup.

PARAMS = {
    # Engine / bounded work
    "ARENA_SIZE": 60.0,
    "MAX_BLOBS": 16,
    "NUM_DIRECTIONS": 12,
    "MAX_ENEMIES": 24,
    "MAX_FOOD": 40,
    "MAX_VIRUSES": 14,
    "SAFETY_OWN_PIECES": 2,
    "RANK_PRESSURE_START_ROUND": 900,
    "RANK_PRESSURE_MIN_RANK": 7,
    "RANK_DESPERATE_START_ROUND": 1200,
    "RANK_DESPERATE_MIN_RANK": 6,
    "RANK_LAST_STAND_START_ROUND": 1250,
    "RANK_LAST_STAND_MIN_RANK": 7,
    "RANK_PRESSURE_MIN_UTILITY_MULT": 0.80,
    "RANK_DESPERATE_MIN_UTILITY_MULT": 0.55,
    "RANK_LAST_STAND_MIN_UTILITY": 0.0,
    "RANK_TARGET_MASS_BONUS": 0.50,
    "RANK_CHASE_RANGE_BONUS": 0.12,
    "RANK_CHASE_COST_DISCOUNT": 0.10,
    "RANK_DEPTH_COST_DISCOUNT": 0.10,
    "RANK_FRAGMENT_COST_DISCOUNT": 0.12,
    "RANK_EXTERNAL_RISK_DISCOUNT": 0.16,
    "STEP_RADIUS_MULT": 1.2709918139713585,
    "MOVE_LOOKAHEAD_TICKS": 3.0,
    "EAT_MASS_RATIO": 1.20,
    "MASS_DECAY_RATE": 0.002,

    # Shared split geometry
    # Engine SPLIT_MIN_MASS is 2.0, hence radius sqrt(2).
    "SPLIT_MIN_RADIUS": 1.4142135623730951,
    "SPLIT_EJECT_SPEED": 1.6,
    "SPLIT_EJECT_DRAG": 0.82,
    "MERGE_ATTRACTION_SPEED": 0.08,
    "SPLIT_TARGET_EAT_RATIO": 1.20,
    "MAX_SPLIT_DEPTH": 4,

    # Escape: direct and generalized enemy split threats
    "ESCAPE_DIRECT_RANGE_MULT": 5.438491509148152,
    "ESCAPE_SPLIT_REACH_BUFFER": 1.0498944587482988,
    "ESCAPE_DISTANCE_WEIGHT": 900.0,
    "ESCAPE_MARGIN_WEIGHT": 2400.0,
    "ESCAPE_MOVE_WEIGHT": 149.11937669812596,
    "ESCAPE_WALL_WEIGHT": 117.92810824423375,
    "ESCAPE_VIRUS_SHIELD_WEIGHT": 682.4299124894012,
    "ESCAPE_HARD_MARGIN": 0.1404504918882442,
    # Exact virus headings are useful while escaping, but considering every
    # remembered virus makes the emergency path explode in dense late games.
    "ESCAPE_MAX_VIRUS_DIRECTIONS": 2,
    "ESCAPE_MAX_SHIELD_VIRUSES": 4,
    "ESCAPE_COUNTER_MAX_TARGETS": 2,
    "ESCAPE_COUNTER_STEP_BUDGET": 4,
    "ENEMY_MEMORY_TICKS": 18,
    "ENEMY_MEMORY_FULL_TICKS": 7,
    "ENEMY_MEMORY_MIN_CONFIDENCE": 0.40,

    # A tight fragment pack can become one dangerous blob before our pieces
    # do. Model only credible near-merges, and ignore sacrificial fragment risk
    # when our total mass overwhelmingly dominates the owner.
    "MERGE_COMPACT_FULL_RATIO": 1.15,
    "MERGE_COMPACT_ZERO_RATIO": 1.85,
    "MERGE_MIN_CONFIDENCE": 0.15,
    "MERGE_OWN_DOMINANCE_RATIO": 1.45,
    "MERGE_CHASE_REACTION_TICKS": 2.0,
    "MERGE_THREAT_HORIZON_TICKS": 5,

    # Unstuck
    "STUCK_TICKS": 4,
    "STUCK_MOVE_RADIUS_MULT": 0.04,
    "UNSTUCK_MOVE_WEIGHT": 1200.0,
    "UNSTUCK_CENTER_WEIGHT": 300.0,

    # Offensive splitting / bounded rollout
    "SPLIT_MAX_TARGETS": 6,
    "SPLIT_MAX_LAUNCHERS": 2,
    # Sum of simulated split depths per decision. A depth-3 rollout costs three
    # units, keeping exact stabilisation comfortably below the turn timeout.
    "SPLIT_ROLLOUT_STEP_BUDGET": 16,
    "SPLIT_CAPTURE_WEIGHT": 4.799170739486253,
    "SPLIT_TARGET_WEIGHT": 3.0,
    "SPLIT_DEPTH_COST": 4.382917879561087,
    "SPLIT_FRAGMENT_COST": 0.22909141833208407,
    "SPLIT_EXTERNAL_RISK_WEIGHT": 0.9004198132569647,
    "SPLIT_MIN_UTILITY": 4.020980901689232,
    "SPLIT_LARGER_OWNER_RATIO": 1.013412343039414,
    "SPLIT_PLAN_WAIT_TICKS": 3,
    "COUNTER_SPLIT_MIN_GAIN_RATIO": 0.10,

    # Chase / predictive interception
    "CHASE_ACQUIRE_RANGE_MULT": 6.937850048841669,
    "CHASE_DIRECT_RANGE_MULT": 4.767548974591575,
    "CHASE_MAX_INTERCEPT_TICKS": 17.807525662843084,
    "CHASE_LOCK_TICKS": 88,
    "CHASE_WALL_HORIZON": 24.203673370555656,
    "CHASE_BLOCK_OFFSET_MULT": 1.1697134053132832,
    "CHASE_TARGET_MASS_WEIGHT": 5.0,
    "CHASE_DISTANCE_WEIGHT": 0.3655968105851199,
    "CHASE_ETA_WEIGHT": 1.0523219513138513,
    "CHASE_LOCK_BONUS": 19.69155617163071,
    "CHASE_VELOCITY_NEW_WEIGHT": 0.27979115446569386,
    "CHASE_PIECE_MATCH_DISTANCE": 4.0,
    "CHASE_PIECE_MEMORY_TICKS": 3,
    "CHASE_MAX_PAIR_EVALUATIONS": 32,
    "BASE_PLAYER_SPEED": 1.1,
    "PLAYER_SPEED_RADIUS_FACTOR": 0.08,
    "MIN_PLAYER_SPEED": 0.25,

    # Virus farming
    # Engine virus test is blob.mass > virus.radius^2 * this ratio.
    "VIRUS_EAT_RATIO": 1.20,
    "VIRUS_DANGER_BUFFER_MULT": 1.10,
    "VIRUS_ENEMY_RANGE_MULT": 5.324337587731354,
    "VIRUS_MANUAL_SPLIT_TIME_BONUS": 0.7800153459936319,
    "VIRUS_MANUAL_SPLIT_RISK_COST": 3.290572756798085,
    "VIRUS_MEMORY_TICKS": 35,
    "VIRUS_MAX_FARM_MASS": 79.67688162750461,
    "VIRUS_DOMINANCE_RATIO": 2.0,
    "VIRUS_CHAIN_RADIUS": 10.0,
    "VIRUS_CHAIN_WEIGHT": 0.85,
    "VIRUS_TARGET_LOCK_BONUS": 1.75,

    # Food / default roaming
    "FOOD_CLUSTER_RADIUS": 6.026448514674607,
    "FOOD_CLUSTER_WEIGHT": 0.7227495410377542,
    "FOOD_DISTANCE_POWER": 1.973767217454007,
    "FOOD_LOCK_TICKS": 6,
    "FOOD_REACHED_DISTANCE": 0.55,
    "FOOD_CORNER_PENALTY": 0.35,
    "ROAM_LOCK_TICKS": 8,
    "ROAM_CENTER_WEIGHT": 1.152743368148301,
    "ROAM_MOMENTUM_WEIGHT": 0.5075238366840418,
}

OVERRIDE_ORDER = ["escape", "split", "virus", "chase", "unstuck", "food"]


P = PARAMS
SQRT2 = np.sqrt(2.0)
DIRECTIONS = np.empty((0, 2), dtype=float)
PAIR_INDICES = {
    count: np.triu_indices(count, k=1)
    for count in range(2, PARAMS["MAX_BLOBS"] + 1)
}


def refresh_derived():
    global DIRECTIONS
    DIRECTIONS = np.array([
        [
            np.cos(2.0 * np.pi * i / P["NUM_DIRECTIONS"]),
            np.sin(2.0 * np.pi * i / P["NUM_DIRECTIONS"]),
        ]
        for i in range(P["NUM_DIRECTIONS"])
    ], dtype=float)


refresh_derived()


# =============================================================================
# Minimal persistent state
# =============================================================================

LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
LAST_SPLIT_REQUESTED = False
LAST_POSITION = None
STUCK_COUNT = 0

OWN_EJECT_ESTIMATES = {}

ENEMY_TRACKS = {}
ENEMY_PIECE_TRACKS = {}
NEXT_ENEMY_PIECE_TRACK_ID = 1
FRAME_TICK = 0
LAST_ENGINE_ROUND = None

CHASE_TARGET_TRACK_ID = None
CHASE_TICKS = 0

SPLIT_PLAN_PID = None
SPLIT_PLAN_POSITION = None
SPLIT_PLAN_STEPS = 0
SPLIT_PLAN_WAIT = 0

VIRUS_MEMORY = {}
VIRUS_TARGET_KEY = None
VIRUS_CHAIN_SIGNATURE = None
VIRUS_CHAIN_VALUES = {}
FOOD_TARGET = None
FOOD_TARGET_TICKS = 0
ROAM_DIRECTION = None
ROAM_TICKS = 0


# =============================================================================
# Geometry and cache
# =============================================================================

def unit(vector):
    vector = np.asarray(vector, dtype=float)
    x = float(vector[0])
    y = float(vector[1])
    length = (x * x + y * y) ** 0.5
    return None if length <= 1e-9 else vector / length


def distance_2d(point_a, point_b):
    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    return (dx * dx + dy * dy) ** 0.5


def stable_position_key(position):
    """One-decimal identity without allocating a NumPy rounding array."""
    return (round(float(position[0]), 1), round(float(position[1]), 1))


def can_eat_mass(eater_mass, target_mass, ratio=None):
    if ratio is None:
        ratio = P["EAT_MASS_RATIO"]
    return eater_mass >= target_mass * ratio


def movement_speed(radius):
    """Actual engine centre speed for a blob of this radius."""
    return max(
        P["MIN_PLAYER_SPEED"],
        P["BASE_PLAYER_SPEED"]
        / (1.0 + float(radius) * P["PLAYER_SPEED_RADIUS_FACTOR"]),
    )


def can_pop_virus(blob_radius, virus_radius):
    """Apply the engine's intentionally asymmetric virus collision test."""
    return blob_radius * blob_radius > (
        virus_radius * virus_radius * P["VIRUS_EAT_RATIO"]
    )


def clamp_points(points, radii):
    points = np.asarray(points, dtype=float).copy()
    radii = np.asarray(radii, dtype=float)
    np.maximum(points[:, 0], radii, out=points[:, 0])
    np.minimum(points[:, 0], P["ARENA_SIZE"] - radii, out=points[:, 0])
    np.maximum(points[:, 1], radii, out=points[:, 1])
    np.minimum(points[:, 1], P["ARENA_SIZE"] - radii, out=points[:, 1])
    return points


def clamp_point(point, radius):
    return clamp_points(np.asarray(point, dtype=float)[None, :], np.array([radius]))[0]


def cap_nearest(locs, limit, origin, *arrays):
    if len(locs) <= limit:
        return (locs, *arrays)
    keep = np.argpartition(np.sum((locs - origin) ** 2, axis=1), limit - 1)[:limit]
    return (locs[keep], *(array[keep] for array in arrays))


def segment_distances(start, end, points):
    """Distance and progress of points along one finite segment."""
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-9:
        deltas = points - start
        return np.sqrt(np.sum(deltas * deltas, axis=1)), np.zeros(len(points))
    progress = np.sum((points - start) * segment, axis=1) / length_sq
    progress = np.clip(progress, 0.0, 1.0)
    closest = start + progress[:, None] * segment
    deltas = points - closest
    return np.sqrt(np.sum(deltas * deltas, axis=1)), progress


def max_split_depth(piece_count):
    if piece_count <= 0 or piece_count >= P["MAX_BLOBS"]:
        return 0
    # A global split can partially fill the 16-piece cap.  floor(log2(cap/n))
    # incorrectly says that 3->6->12 may split only twice, even though a third
    # command validly produces 16.  Simulate the bounded count exactly.
    depth = 0
    count = int(piece_count)
    while count < P["MAX_BLOBS"] and depth < P["MAX_SPLIT_DEPTH"]:
        count = min(P["MAX_BLOBS"], count * 2)
        depth += 1
    return depth


def split_profile(radius, depth):
    """Exact engine radii and forward centre advance for a launched lineage."""
    before = radius / (SQRT2 ** np.arange(depth, dtype=float))
    after = before / SQRT2
    # Child spawns two child-radii forward, then receives normal movement and
    # SPLIT_EJECT_SPEED in the same tick. Attraction/decay can only reduce the
    # offensive reach slightly, so this is also the correct safe threat bound.
    travels = np.array([
        2.0 * child_radius
        + movement_speed(float(child_radius))
        + P["SPLIT_EJECT_SPEED"]
        for child_radius in after
    ], dtype=float)
    return before, after, travels


def player_summaries(blobs):
    summaries = {}
    for blob in blobs:
        pid = int(blob.player_id)
        summaries.setdefault(pid, []).append(blob)
    result = {}
    for pid, pieces in summaries.items():
        locs = np.array([piece.pos for piece in pieces], dtype=float)
        rads = np.array([piece.radius for piece in pieces], dtype=float)
        cooldowns = np.array([
            getattr(piece, "merge_cooldown", 0) for piece in pieces
        ], dtype=int)
        masses = rads ** 2
        total = float(np.sum(masses))
        centroid = np.sum(locs * masses[:, None], axis=0) / max(total, 1e-9)
        cover_deltas = locs - centroid
        cover_radius = float(np.max(
            np.sqrt(np.sum(cover_deltas * cover_deltas, axis=1)) + rads
        ))
        result[pid] = {
            "mass": total,
            "count": len(pieces),
            "centroid": centroid,
            "max_radius": float(np.max(rads)),
            "cover_radius": cover_radius,
            "locs": locs,
            "rads": rads,
            "cooldowns": cooldowns,
        }
    return result


def circle_is_visible(position, radius, view_center, vision_size):
    """Mirror GameState.player_can_see_circle's rounded-square test."""
    outside = np.maximum(
        np.abs(np.asarray(position, dtype=float) - view_center)
        - float(vision_size) * 0.5,
        0.0,
    )
    return float(np.dot(outside, outside)) <= float(radius) ** 2


def force_outside_vision(position, radius, view_center, vision_size, velocity):
    """Move a remembered-but-unseen piece to the nearest credible blind side."""
    position = np.asarray(position, dtype=float).copy()
    if not circle_is_visible(position, radius, view_center, vision_size):
        return np.clip(position, radius, P["ARENA_SIZE"] - radius)

    radial = position - view_center
    velocity = np.asarray(velocity, dtype=float)
    if np.dot(velocity, radial) > 0.0:
        direction = unit(velocity)
    else:
        direction = unit(radial)
    if direction is None:
        direction = LAST_DIRECTION
    # Crossing one expanded square side guarantees the whole circle is no
    # longer visible. The arena clamp handles wall-pinned view centres.
    dominant = max(float(np.max(np.abs(direction))), 1e-6)
    blind_distance = (
        float(vision_size) * 0.5 + float(radius) + 0.05
    ) / dominant
    projected = view_center + direction * blind_distance
    return np.clip(projected, radius, P["ARENA_SIZE"] - radius)


def update_enemy_tracks(summaries, view_center, vision_size):
    global ENEMY_TRACKS
    updated = {}
    alpha = P["CHASE_VELOCITY_NEW_WEIGHT"]
    for pid, summary in summaries.items():
        prior = ENEMY_TRACKS.get(pid)
        if prior is None:
            velocity = np.zeros(2, dtype=float)
            hidden_mass = 0.0
            hidden_radius = 0.0
            hidden_position = summary["centroid"].copy()
            hidden_age = 0
            hidden_merge_virtual = False
            estimated_mass = float(summary["mass"])
        else:
            elapsed = max(1, FRAME_TICK - prior["tick"])
            measured = (summary["centroid"] - prior["position"]) / elapsed
            velocity = alpha * measured + (1.0 - alpha) * prior["velocity"]
            decay = (1.0 - P["MASS_DECAY_RATE"]) ** elapsed
            prior_mass = float(prior["mass"]) * decay
            observed_mass = float(summary["mass"])
            hidden_mass = max(0.0, prior_mass - observed_mass)
            # The largest previously seen fragment cannot contribute more mass
            # than the entire hidden remainder. Radius decays at sqrt(mass).
            prior_max_radius = float(prior["max_radius"]) * np.sqrt(decay)
            hidden_cooldown = max(
                0, int(prior.get("max_cooldown", 0)) - elapsed
            )
            hidden_merge_virtual = (
                hidden_cooldown <= P["MERGE_THREAT_HORIZON_TICKS"]
                and float(prior.get("cover_ratio", float("inf")))
                <= P["MERGE_COMPACT_ZERO_RATIO"]
            )
            hidden_radius = (
                np.sqrt(hidden_mass)
                if hidden_merge_virtual
                else min(prior_max_radius, np.sqrt(hidden_mass))
            )
            if hidden_mass > 0.05 and hidden_radius > 0.0:
                raw_hidden_position = (
                    prior.get("hidden_position", prior["position"])
                    + prior["velocity"] * elapsed
                )
                hidden_position = force_outside_vision(
                    raw_hidden_position,
                    hidden_radius,
                    view_center,
                    vision_size,
                    prior["velocity"],
                )
                hidden_age = int(prior.get("hidden_age", 0)) + elapsed
                if hidden_age > P["ENEMY_MEMORY_TICKS"]:
                    hidden_mass = 0.0
                    hidden_radius = 0.0
                    hidden_position = summary["centroid"].copy()
                    hidden_age = 0
                    hidden_merge_virtual = False
            else:
                hidden_mass = 0.0
                hidden_radius = 0.0
                hidden_position = summary["centroid"].copy()
                hidden_age = 0
                hidden_merge_virtual = False
            estimated_mass = observed_mass + hidden_mass
        updated[pid] = {
            "tick": FRAME_TICK,
            "position": summary["centroid"].copy(),
            "velocity": velocity,
            "mass": estimated_mass,
            "observed_mass": float(summary["mass"]),
            "max_radius": max(float(summary["max_radius"]), hidden_radius),
            "count": max(
                int(summary["count"]),
                int(prior["count"]) if prior is not None and hidden_mass > 0.0
                else int(summary["count"]),
            ),
            "hidden_mass": float(hidden_mass),
            "hidden_radius": float(hidden_radius),
            "hidden_position": hidden_position,
            "hidden_age": int(hidden_age),
            "hidden_merge_virtual": bool(hidden_merge_virtual),
            "max_cooldown": int(np.max(summary["cooldowns"])),
            "cover_ratio": float(
                summary["cover_radius"] / max(np.sqrt(summary["mass"]), 1e-6)
            ),
        }
    # Short continuity helps chase through a brief visibility gap without
    # turning stale players into strategic hard vetoes.
    for pid, prior in ENEMY_TRACKS.items():
        if (
            pid not in updated
            and FRAME_TICK - prior["tick"] <= P["ENEMY_MEMORY_TICKS"]
        ):
            updated[pid] = prior
    ENEMY_TRACKS = updated


def update_enemy_piece_tracks(enemy_locs, enemy_rads, enemy_pids):
    """Match individual enemy fragments and return per-piece velocity.

    Owner-centroid velocity is wrong when a player has several fragments moving
    in different directions. A small greedy nearest-prediction match is bounded
    by MAX_ENEMIES and gives chase a stable identity for each visible piece.
    """
    global ENEMY_PIECE_TRACKS, NEXT_ENEMY_PIECE_TRACK_ID

    previous = {
        key: value for key, value in ENEMY_PIECE_TRACKS.items()
        if FRAME_TICK - value["tick"] <= P["CHASE_PIECE_MEMORY_TICKS"]
    }
    previous_by_pid = {}
    for key, prior in previous.items():
        previous_by_pid.setdefault(int(prior["pid"]), []).append((key, prior))
    assigned_current = set()
    assigned_previous = set()
    matches = {}

    pairs = []
    for pid in np.unique(enemy_pids):
        current_indices = np.where(enemy_pids == pid)[0]
        prior_items = previous_by_pid.get(int(pid), ())
        if len(current_indices) == 0 or not prior_items:
            continue
        prior_keys = np.array([item[0] for item in prior_items], dtype=int)
        prior_positions = np.array([
            item[1]["position"] for item in prior_items
        ], dtype=float)
        prior_velocities = np.array([
            item[1]["velocity"] for item in prior_items
        ], dtype=float)
        prior_radii = np.array([
            item[1]["radius"] for item in prior_items
        ], dtype=float)
        ages = np.array([
            max(1, FRAME_TICK - item[1]["tick"])
            for item in prior_items
        ], dtype=float)
        predicted = prior_positions + prior_velocities * ages[:, None]
        deltas = (
            enemy_locs[current_indices, None, :]
            - predicted[None, :, :]
        )
        distance_sq = np.sum(deltas * deltas, axis=2)
        gates = np.maximum(
            P["CHASE_PIECE_MATCH_DISTANCE"],
            1.5 * (
                enemy_rads[current_indices, None]
                + prior_radii[None, :]
            ),
        )
        current_local, prior_local = np.where(distance_sq <= gates * gates)
        pairs.extend(
            (
                float(distance_sq[current_i, prior_i]),
                int(current_indices[current_i]),
                int(prior_keys[prior_i]),
            )
            for current_i, prior_i in zip(current_local, prior_local)
        )

    for _, current_i, key in sorted(pairs, key=lambda item: item[0]):
        if current_i in assigned_current or key in assigned_previous:
            continue
        matches[current_i] = key
        assigned_current.add(current_i)
        assigned_previous.add(key)

    track_ids = np.empty(len(enemy_locs), dtype=int)
    velocities = np.zeros((len(enemy_locs), 2), dtype=float)
    updated = {}
    alpha = P["CHASE_VELOCITY_NEW_WEIGHT"]
    for current_i, (pos, radius, pid) in enumerate(zip(
        enemy_locs, enemy_rads, enemy_pids
    )):
        key = matches.get(current_i)
        if key is None:
            key = NEXT_ENEMY_PIECE_TRACK_ID
            NEXT_ENEMY_PIECE_TRACK_ID += 1
            owner = ENEMY_TRACKS.get(int(pid))
            velocity = (
                np.zeros(2, dtype=float)
                if owner is None else owner["velocity"].copy()
            )
        else:
            prior = previous[key]
            age = max(1, FRAME_TICK - prior["tick"])
            measured = (pos - prior["position"]) / age
            velocity = alpha * measured + (1.0 - alpha) * prior["velocity"]
        track_ids[current_i] = int(key)
        velocities[current_i] = velocity
        updated[int(key)] = {
            "pid": int(pid),
            "position": pos.copy(),
            "radius": float(radius),
            "velocity": velocity.copy(),
            "tick": FRAME_TICK,
        }

    # Briefly retain unmatched tracks only to reconnect temporary occlusion.
    for key, prior in previous.items():
        if key not in updated:
            updated[key] = prior
    ENEMY_PIECE_TRACKS = updated
    return track_ids, velocities


def update_virus_memory(virus_locs, virus_rads):
    global VIRUS_MEMORY, VIRUS_CHAIN_SIGNATURE, VIRUS_CHAIN_VALUES
    for pos, radius in zip(virus_locs, virus_rads):
        key = stable_position_key(pos)
        VIRUS_MEMORY[key] = {
            "position": pos.copy(),
            "radius": float(radius),
            "tick": FRAME_TICK,
        }
    VIRUS_MEMORY = {
        key: value for key, value in VIRUS_MEMORY.items()
        if FRAME_TICK - value["tick"] <= P["VIRUS_MEMORY_TICKS"]
    }

    # Virus coordinates never move.  Chain payoff therefore changes only
    # when the remembered set changes, not on every decision tick.
    signature = tuple(sorted(
        (key, float(value["radius"]))
        for key, value in VIRUS_MEMORY.items()
    ))
    if signature != VIRUS_CHAIN_SIGNATURE:
        chain_values = {}
        for key, info in VIRUS_MEMORY.items():
            value = 0.0
            for other_key, other in VIRUS_MEMORY.items():
                if other_key == key:
                    continue
                separation = distance_2d(
                    other["position"], info["position"]
                )
                if separation <= P["VIRUS_CHAIN_RADIUS"]:
                    value += (
                        other["radius"] ** 2
                        * (1.0 - separation / P["VIRUS_CHAIN_RADIUS"])
                    )
            chain_values[key] = value
        VIRUS_CHAIN_VALUES = chain_values
        VIRUS_CHAIN_SIGNATURE = signature


def update_own_eject_estimates(blob_ids, cooldowns):
    """Reconstruct hidden own eject momentum from stable IDs and cooldowns."""
    global OWN_EJECT_ESTIMATES
    updated = {}
    for blob_id, cooldown in zip(blob_ids, cooldowns):
        blob_id = int(blob_id)
        cooldown = int(cooldown)
        if cooldown == 18:
            # Virus replacement happens after movement and explicitly zeros
            # eject velocity while resetting cooldown to 18.
            estimate = np.zeros(2, dtype=float)
        elif blob_id in OWN_EJECT_ESTIMATES:
            estimate = (
                OWN_EJECT_ESTIMATES[blob_id] * P["SPLIT_EJECT_DRAG"]
            )
        elif LAST_SPLIT_REQUESTED and cooldown == 17:
            # A normal split child moved once already; its visible next-tick
            # eject is 1.6 * 0.82 in the submitted split direction.
            estimate = (
                LAST_DIRECTION
                * P["SPLIT_EJECT_SPEED"]
                * P["SPLIT_EJECT_DRAG"]
            )
        else:
            estimate = np.zeros(2, dtype=float)
        updated[blob_id] = estimate
    OWN_EJECT_ESTIMATES = updated
    return np.array([updated[int(blob_id)] for blob_id in blob_ids], dtype=float)


def reset_transient_state_after_round_gap():
    """Clear observations made stale by the engine's 30-round death gap."""
    global LAST_DIRECTION, LAST_SPLIT_REQUESTED, LAST_POSITION, STUCK_COUNT
    global OWN_EJECT_ESTIMATES, ENEMY_TRACKS, ENEMY_PIECE_TRACKS
    global NEXT_ENEMY_PIECE_TRACK_ID, CHASE_TARGET_TRACK_ID, CHASE_TICKS
    global SPLIT_PLAN_PID, SPLIT_PLAN_POSITION, SPLIT_PLAN_STEPS, SPLIT_PLAN_WAIT
    global VIRUS_MEMORY, VIRUS_TARGET_KEY, VIRUS_CHAIN_SIGNATURE
    global VIRUS_CHAIN_VALUES, FOOD_TARGET, FOOD_TARGET_TICKS
    global ROAM_DIRECTION, ROAM_TICKS

    LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
    LAST_SPLIT_REQUESTED = False
    LAST_POSITION = None
    STUCK_COUNT = 0
    OWN_EJECT_ESTIMATES = {}
    ENEMY_TRACKS = {}
    ENEMY_PIECE_TRACKS = {}
    NEXT_ENEMY_PIECE_TRACK_ID = 1
    CHASE_TARGET_TRACK_ID = None
    CHASE_TICKS = 0
    SPLIT_PLAN_PID = None
    SPLIT_PLAN_POSITION = None
    SPLIT_PLAN_STEPS = 0
    SPLIT_PLAN_WAIT = 0
    VIRUS_MEMORY = {}
    VIRUS_TARGET_KEY = None
    VIRUS_CHAIN_SIGNATURE = None
    VIRUS_CHAIN_VALUES = {}
    FOOD_TARGET = None
    FOOD_TARGET_TICKS = 0
    ROAM_DIRECTION = None
    ROAM_TICKS = 0


def build_cache(game):
    global FRAME_TICK, LAST_ENGINE_ROUND
    engine_round = int(game.state.round)
    if (
        LAST_ENGINE_ROUND is not None
        and engine_round > LAST_ENGINE_ROUND + 1
    ):
        reset_transient_state_after_round_gap()
    LAST_ENGINE_ROUND = engine_round
    FRAME_TICK += 1
    me = game.state.me
    rankings = [int(player_id) for player_id in game.state.rankings]
    rank_by_pid = {
        player_id: index + 1 for index, player_id in enumerate(rankings)
    }
    live_rank = rank_by_pid.get(int(me.player_id), len(rankings))
    rank_mode = 0
    if (
        engine_round >= P["RANK_LAST_STAND_START_ROUND"]
        and live_rank >= P["RANK_LAST_STAND_MIN_RANK"]
    ):
        rank_mode = 3
    elif (
        engine_round >= P["RANK_DESPERATE_START_ROUND"]
        and live_rank >= P["RANK_DESPERATE_MIN_RANK"]
    ):
        rank_mode = 2
    elif (
        engine_round >= P["RANK_PRESSURE_START_ROUND"]
        and live_rank >= P["RANK_PRESSURE_MIN_RANK"]
    ):
        rank_mode = 1
    origin = np.array([me.x, me.y], dtype=float)

    own = list(me.blobs.values())
    if own:
        own_ids_all = np.array([blob.blob_id for blob in own], dtype=int)
        own_locs_all = np.array([blob.pos for blob in own], dtype=float)
        own_rads_all = np.array([blob.radius for blob in own], dtype=float)
        own_cooldowns_all = np.array([
            getattr(blob, "merge_cooldown", 0) for blob in own
        ], dtype=int)
    else:
        own_ids_all = np.array([0], dtype=int)
        own_locs_all = np.array([[me.x, me.y]], dtype=float)
        own_rads_all = np.array([me.radius], dtype=float)
        own_cooldowns_all = np.zeros(1, dtype=int)
    own_eject_all = update_own_eject_estimates(
        own_ids_all, own_cooldowns_all
    )

    own_order = np.argsort(own_rads_all)[::-1]

    enemies_all = [
        blob for blob in game.state.visible_blobs
        if getattr(blob, "player_id", None) != me.player_id
    ]
    summaries = player_summaries(enemies_all)
    view_center = np.asarray(game.state.view_center, dtype=float)
    vision_size = float(game.state.vision_size)
    update_enemy_tracks(summaries, view_center, vision_size)

    if enemies_all:
        enemy_locs = np.array([blob.pos for blob in enemies_all], dtype=float)
        enemy_rads = np.array([blob.radius for blob in enemies_all], dtype=float)
        enemy_pids = np.array([blob.player_id for blob in enemies_all], dtype=int)
        enemy_cooldowns = np.array([
            getattr(blob, "merge_cooldown", 0) for blob in enemies_all
        ], dtype=int)
        enemy_locs, enemy_rads, enemy_pids, enemy_cooldowns = cap_nearest(
            enemy_locs,
            P["MAX_ENEMIES"],
            origin,
            enemy_rads,
            enemy_pids,
            enemy_cooldowns,
        )
    else:
        enemy_locs = np.empty((0, 2), dtype=float)
        enemy_rads = np.empty(0, dtype=float)
        enemy_pids = np.empty(0, dtype=int)
        enemy_cooldowns = np.empty(0, dtype=int)

    # Keep safety work bounded, but make the selection meaningful: protect the
    # largest strategic piece and the fragment physically closest to enemy
    # contact. With equal-sized swarms, selecting only by radius was arbitrary
    # and frequently ignored the exact front-line blob being lured forward.
    safety_limit = min(P["SAFETY_OWN_PIECES"], len(own_locs_all))
    keep = [int(own_order[0])]
    if safety_limit > 1 and len(enemy_locs) > 0:
        edge_gaps = (
            np.sqrt(np.sum(
                (
                    own_locs_all[:, None, :] - enemy_locs[None, :, :]
                ) ** 2,
                axis=2,
            ))
            - own_rads_all[:, None]
            - enemy_rads[None, :]
        )
        exposed_order = np.argsort(np.min(edge_gaps, axis=1))
        for index in exposed_order:
            index = int(index)
            if index not in keep:
                keep.append(index)
            if len(keep) >= safety_limit:
                break
    for index in own_order:
        index = int(index)
        if index not in keep:
            keep.append(index)
        if len(keep) >= safety_limit:
            break
    keep = np.array(keep, dtype=int)
    own_locs = own_locs_all[keep]
    own_rads = own_rads_all[keep]
    own_eject = own_eject_all[keep]

    enemy_track_ids, enemy_velocities = update_enemy_piece_tracks(
        enemy_locs, enemy_rads, enemy_pids
    )

    # Players that just left vision remain escape/split-risk threats, but are
    # deliberately not inserted into visible targets for chase or capture.
    memory_threats = []
    for pid, track in ENEMY_TRACKS.items():
        if pid in summaries:
            # A stable owner ID may be represented by only one visible piece.
            # Preserve a recently observed large sibling just outside the exact
            # current view instead of pretending the rest of the owner vanished.
            hidden_age = int(track.get("hidden_age", 0))
            hidden_radius = float(track.get("hidden_radius", 0.0))
            if hidden_radius > 0.0 and hidden_age <= P["ENEMY_MEMORY_TICKS"]:
                if hidden_age <= P["ENEMY_MEMORY_FULL_TICKS"]:
                    confidence = 1.0
                else:
                    decay_span = max(
                        1.0,
                        P["ENEMY_MEMORY_TICKS"]
                        - P["ENEMY_MEMORY_FULL_TICKS"],
                    )
                    confidence = max(
                        P["ENEMY_MEMORY_MIN_CONFIDENCE"],
                        1.0
                        - (
                            hidden_age - P["ENEMY_MEMORY_FULL_TICKS"]
                        ) / decay_span,
                    )
                memory_threats.append({
                    "pid": int(pid),
                    "position": track["hidden_position"].copy(),
                    "radius": hidden_radius,
                    "count": (
                        P["MAX_BLOBS"]
                        if track.get("hidden_merge_virtual", False)
                        else max(1, int(track["count"]))
                    ),
                    "confidence": float(confidence),
                    "age": hidden_age,
                    "merge_virtual": bool(
                        track.get("hidden_merge_virtual", False)
                    ),
                    "hidden_remainder": True,
                })
            continue
        age = FRAME_TICK - track["tick"]
        if age <= 0 or age > P["ENEMY_MEMORY_TICKS"]:
            continue
        predicted = track["position"] + track["velocity"] * age
        predicted = np.clip(predicted, 0.0, P["ARENA_SIZE"])
        if age <= P["ENEMY_MEMORY_FULL_TICKS"]:
            confidence = 1.0
        else:
            decay_span = max(
                1.0,
                P["ENEMY_MEMORY_TICKS"] - P["ENEMY_MEMORY_FULL_TICKS"],
            )
            confidence = max(
                P["ENEMY_MEMORY_MIN_CONFIDENCE"],
                1.0 - (age - P["ENEMY_MEMORY_FULL_TICKS"]) / decay_span,
            )
        cooldown_remaining = max(
            0, int(track.get("max_cooldown", 0)) - int(age)
        )
        merge_credible = (
            cooldown_remaining <= P["MERGE_THREAT_HORIZON_TICKS"]
            and track.get("cover_ratio", float("inf"))
            <= P["MERGE_COMPACT_ZERO_RATIO"]
        )
        memory_threats.append({
            "pid": int(pid),
            "position": predicted,
            "radius": float(
                np.sqrt(track["mass"])
                if merge_credible else track["max_radius"]
            ),
            # A credible virtual merge is direct-only; otherwise retain the
            # last real fragment count for exact remaining split capacity.
            "count": (
                P["MAX_BLOBS"] if merge_credible else int(track["count"])
            ),
            "confidence": float(confidence),
            "age": int(age),
            "merge_virtual": bool(merge_credible),
        })

    # Food is the final override and is unused on most offensive/virus ticks.
    # Keep the current query objects and materialise the bounded NumPy array
    # only if food logic is actually reached.
    foods = game.state.visible_food

    viruses = game.state.visible_viruses
    if viruses:
        virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
        virus_rads = np.array([virus.radius for virus in viruses], dtype=float)
        virus_locs, virus_rads = cap_nearest(
            virus_locs,
            P["MAX_VIRUSES"],
            origin,
            virus_rads,
        )
    else:
        virus_locs = np.empty((0, 2), dtype=float)
        virus_rads = np.empty(0, dtype=float)
    update_virus_memory(virus_locs, virus_rads)

    total_mass = float(np.sum(own_rads_all ** 2))
    position = np.sum(
        own_locs_all * (own_rads_all ** 2)[:, None], axis=0
    ) / max(total_mass, 1e-9)

    return {
        "round": engine_round,
        "live_rank": live_rank,
        "rank_by_pid": rank_by_pid,
        "rank_mode": rank_mode,
        "player_id": int(me.player_id),
        "position": position,
        "radius": float(np.sqrt(total_mass)),
        "mass": total_mass,
        "own_count": len(own_rads_all),
        "own_locs": own_locs,
        "own_rads": own_rads,
        "own_eject": own_eject,
        "own_locs_all": own_locs_all,
        "own_rads_all": own_rads_all,
        "own_ids_all": own_ids_all,
        "own_cooldowns_all": own_cooldowns_all,
        "own_eject_all": own_eject_all,
        "enemy_locs": enemy_locs,
        "enemy_rads": enemy_rads,
        "enemy_pids": enemy_pids,
        "enemy_cooldowns": enemy_cooldowns,
        "enemy_track_ids": enemy_track_ids,
        "enemy_velocities": enemy_velocities,
        "memory_threats": memory_threats,
        "summaries": summaries,
        "food_objects": foods,
        "virus_locs": virus_locs,
        "virus_rads": virus_rads,
        "visible_virus_keys": {
            stable_position_key(pos) for pos in virus_locs
        },
        # Kept for backwards-compatible tuning files; movement projection is
        # now per-piece and engine-speed based in future_own_positions().
        "step": P["STEP_RADIUS_MULT"] * float(np.sqrt(total_mass)),
    }


# =============================================================================
# Shared threat model and movement safety
# =============================================================================

def owner_visible_count(cache, player_id):
    summary = cache["summaries"].get(int(player_id))
    if summary is not None:
        return max(1, int(summary["count"]))
    track = ENEMY_TRACKS.get(int(player_id))
    return 1 if track is None else max(1, int(track["count"]))


def compact_effective_mass(locs, rads):
    """Mass that a geometrically compact set can credibly consolidate."""
    masses = np.asarray(rads, dtype=float) ** 2
    if len(masses) == 0:
        return 0.0, 0.0, float("inf"), np.zeros(2, dtype=float)
    total = float(np.sum(masses))
    centroid = np.sum(
        np.asarray(locs, dtype=float) * masses[:, None], axis=0
    ) / max(total, 1e-9)
    merged_radius = float(np.sqrt(total))
    cover_deltas = np.asarray(locs, dtype=float) - centroid
    cover_radius = float(np.max(
        np.sqrt(np.sum(cover_deltas * cover_deltas, axis=1))
        + np.asarray(rads, dtype=float)
    ))
    cover_ratio = cover_radius / max(merged_radius, 1e-6)
    span = max(
        P["MERGE_COMPACT_ZERO_RATIO"] - P["MERGE_COMPACT_FULL_RATIO"],
        1e-6,
    )
    geometry = float(np.clip(
        (P["MERGE_COMPACT_ZERO_RATIO"] - cover_ratio) / span,
        0.0,
        1.0,
    ))
    largest_mass = float(np.max(masses))
    effective_mass = largest_mass + geometry * (total - largest_mass)
    return effective_mass, geometry, cover_ratio, centroid


def owner_merge_profile(cache, player_id):
    """Estimate how much of a visible owner could credibly merge very soon.

    The engine exposes each visible blob's exact merge_cooldown. Combine that
    with geometric compactness: newly split pieces with cooldown 18 remain
    attackable, while a tight pack at cooldown 0-2 is an imminent merged blob.
    """
    player_id = int(player_id)
    profile_cache = cache.setdefault("_owner_merge_profiles", {})
    if player_id in profile_cache:
        return profile_cache[player_id]

    summary = cache["summaries"].get(player_id)
    if summary is None or int(summary["count"]) <= 1:
        profile_cache[player_id] = None
        return None
    if cache["mass"] >= (
        float(summary["mass"]) * P["MERGE_OWN_DOMINANCE_RATIO"]
    ):
        profile_cache[player_id] = None
        return None

    best = None
    horizon = int(P["MERGE_THREAT_HORIZON_TICKS"])
    cooldowns = summary["cooldowns"]
    for ready_tick in range(horizon + 1):
        ready = cooldowns <= ready_tick
        if np.count_nonzero(ready) < 2:
            continue
        locs = summary["locs"][ready]
        rads = summary["rads"][ready]
        masses = rads ** 2
        total = float(np.sum(masses))
        _, geometry, cover_ratio, centroid = compact_effective_mass(locs, rads)
        time_readiness = 1.0 - ready_tick / max(horizon + 1.0, 1.0)
        confidence = geometry * time_readiness
        if confidence < P["MERGE_MIN_CONFIDENCE"]:
            continue
        largest_mass = float(np.max(masses))
        effective_mass = largest_mass + confidence * (total - largest_mass)

        # If our own compact pieces become merge-ready no later and their
        # consolidated mass cannot be eaten, both owners merge before the
        # engine resolves player eating; this is not a punishment scenario.
        own_ready_profiles = cache.setdefault("_own_ready_profiles", {})
        own_profile = own_ready_profiles.get(ready_tick)
        if own_profile is None:
            own_ready = cache["own_cooldowns_all"] <= ready_tick
            if np.count_nonzero(own_ready) >= 2:
                own_mass, own_geometry, _, _ = compact_effective_mass(
                    cache["own_locs_all"][own_ready],
                    cache["own_rads_all"][own_ready],
                )
                own_profile = (own_mass, own_geometry)
            else:
                own_profile = (0.0, 0.0)
            own_ready_profiles[ready_tick] = own_profile
        own_mass, own_geometry = own_profile
        if own_mass > 0.0:
            if (
                own_geometry >= P["MERGE_MIN_CONFIDENCE"]
                and not can_eat_mass(
                    effective_mass, own_mass, P["EAT_MASS_RATIO"]
                )
            ):
                continue
        candidate = {
            "position": centroid,
            "radius": float(np.sqrt(max(effective_mass, largest_mass))),
            "mass": float(effective_mass),
            "confidence": confidence,
            "cover_ratio": cover_ratio,
            "ready_tick": ready_tick,
        }
        if best is None or candidate["mass"] > best["mass"]:
            best = candidate
    profile_cache[player_id] = best
    return best


def enemy_reach_options(enemy_radius, owner_count, target_radius):
    """Yield (depth, resulting radius, centre reach) able to eat a target."""
    enemy_radius = float(enemy_radius)
    enemy_mass = enemy_radius ** 2
    target_mass = float(target_radius ** 2)

    # Splitting only makes an individual attacker smaller. If the current blob
    # cannot eat the target, no deeper split from that blob can either.
    if enemy_mass < target_mass * P["EAT_MASS_RATIO"]:
        return

    # ESCAPE_DIRECT_RANGE_MULT is a reaction horizon in ticks.  The previous
    # target_radius * multiplier inflated a direct attack by 20-30 world units
    # and made escape dominate almost every encounter.  Direct capture is the
    # enemy radius plus the distance that enemy can actually move soon.
    direct_motion = (
        movement_speed(enemy_radius) * P["ESCAPE_DIRECT_RANGE_MULT"]
    )
    yield 0, enemy_radius, enemy_radius + direct_motion

    for depth in range(1, max_split_depth(owner_count) + 1):
        before, after, travels = split_profile(enemy_radius, depth)
        if before[-1] < P["SPLIT_MIN_RADIUS"]:
            break
        resulting_radius = float(after[-1])
        if resulting_radius ** 2 < target_mass * P["EAT_MASS_RATIO"]:
            continue
        centre_reach = (
            float(np.sum(travels)) + resulting_radius
        ) * P["ESCAPE_SPLIT_REACH_BUFFER"]
        yield depth, resulting_radius, centre_reach


def enemy_threat_sources(cache):
    """Yield pieces, credible near-merges, and decaying last-seen owners."""
    cached = cache.get("_threat_sources")
    if cached is not None:
        return cached
    sources = []
    for index, (position, radius, pid) in enumerate(zip(
        cache["enemy_locs"], cache["enemy_rads"], cache["enemy_pids"]
    )):
        sources.append({
            "position": position,
            "radius": float(radius),
            "pid": int(pid),
            "count": owner_visible_count(cache, pid),
            "confidence": 1.0,
            "visible_index": int(index),
            "merge_virtual": False,
        })
    # Add one direct-only virtual blob for a compact owner. Setting count to
    # MAX_BLOBS prevents the model from imagining an immediate merge *and*
    # another multi-split, which would be needlessly timid.
    for pid in cache["summaries"]:
        profile = owner_merge_profile(cache, pid)
        if profile is None:
            continue
        member_indices = np.where(cache["enemy_pids"] == int(pid))[0]
        sources.append({
            "position": profile["position"],
            "radius": profile["radius"],
            "pid": int(pid),
            "count": P["MAX_BLOBS"],
            "confidence": 1.0,
            "visible_index": None,
            "merge_virtual": True,
            "member_indices": member_indices,
        })
    for threat in cache["memory_threats"]:
        sources.append({
            **threat,
            "visible_index": None,
            "merge_virtual": bool(threat.get("merge_virtual", False)),
        })
    cache["_threat_sources"] = sources
    cache["_threat_positions"] = np.array(
        [source["position"] for source in sources], dtype=float
    )
    return sources


def threat_profiles(cache, own_rads):
    """Cache attack reaches; only positions change across direction scoring."""
    cached = cache.get("_threat_profiles")
    if cached is not None:
        return cached
    sources = enemy_threat_sources(cache)
    profiles = []
    for own_radius in own_rads:
        source_indices = []
        depths = []
        attack_radii = []
        reaches = []
        for source_index, enemy in enumerate(sources):
            for depth, attack_radius, reach in enemy_reach_options(
                enemy["radius"], enemy["count"], float(own_radius)
            ):
                source_indices.append(source_index)
                depths.append(depth)
                attack_radii.append(attack_radius)
                reaches.append(reach * enemy["confidence"])
        profiles.append({
            "source_indices": np.asarray(source_indices, dtype=int),
            "depths": np.asarray(depths, dtype=int),
            "attack_radii": np.asarray(attack_radii, dtype=float),
            "reaches": np.asarray(reaches, dtype=float),
        })
    cache["_threat_profiles"] = profiles
    return profiles


def threat_state(cache, own_locs=None, own_rads=None):
    """Return the closest generalized enemy attack margin and active attacks."""
    if own_locs is None:
        own_locs = cache["own_locs"]
        own_rads = cache["own_rads"]
    sources = enemy_threat_sources(cache)
    if not sources or len(own_locs) == 0:
        return float("inf"), []

    best_margin = float("inf")
    active = []
    profiles = threat_profiles(cache, own_rads)
    source_positions = cache["_threat_positions"]
    for own_index, (own_pos, own_radius) in enumerate(zip(own_locs, own_rads)):
        profile = profiles[own_index]
        if len(profile["reaches"]) == 0:
            continue
        deltas = source_positions - own_pos
        source_distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        source_indices = profile["source_indices"]
        margins = source_distances[source_indices] - profile["reaches"]
        best_margin = min(best_margin, float(np.min(margins)))
        active_profile_indices = np.where(
            margins <= own_radius * P["ESCAPE_HARD_MARGIN"]
        )[0]
        for profile_index in active_profile_indices:
            source_index = int(source_indices[profile_index])
            enemy = sources[source_index]
            active.append({
                "own_index": own_index,
                "enemy_position": enemy["position"],
                "enemy_radius": enemy["radius"],
                "enemy_pid": enemy["pid"],
                "enemy_visible_index": enemy["visible_index"],
                "depth": int(profile["depths"][profile_index]),
                "attack_radius": float(
                    profile["attack_radii"][profile_index]
                ),
                "reach": float(profile["reaches"][profile_index]),
                "margin": float(margins[profile_index]),
            })
    return best_margin, active


def future_own_positions(cache, direction):
    starts = cache["own_locs"]
    normal_steps = cache.get("_future_normal_steps")
    eject_offset = cache.get("_future_eject_offset")
    if normal_steps is None:
        normal_steps = np.array([
            movement_speed(radius) * P["MOVE_LOOKAHEAD_TICKS"]
            for radius in cache["own_rads"]
        ], dtype=float)
        horizon = int(np.ceil(P["MOVE_LOOKAHEAD_TICKS"]))
        eject_factor = (
            (1.0 - P["SPLIT_EJECT_DRAG"] ** horizon)
            / max(1.0 - P["SPLIT_EJECT_DRAG"], 1e-9)
        )
        eject_offset = cache["own_eject"] * eject_factor
        cache["_future_normal_steps"] = normal_steps
        cache["_future_eject_offset"] = eject_offset
    ends = (
        starts
        + direction[None, :] * normal_steps[:, None]
        + eject_offset
    )
    return clamp_points(ends, cache["own_rads"])


def virus_paths_safe(cache, direction, ends=None):
    if cache["own_count"] >= P["MAX_BLOBS"] or len(cache["virus_locs"]) == 0:
        return True
    starts = cache["own_locs"]
    if ends is None:
        ends = future_own_positions(cache, direction)
    dangerous_masks = cache.get("_virus_dangerous_masks")
    if dangerous_masks is None:
        dangerous_masks = [
            can_pop_virus(float(own_radius), cache["virus_rads"])
            for own_radius in cache["own_rads"]
        ]
        cache["_virus_dangerous_masks"] = dangerous_masks
    for own_index, (start, end, own_radius) in enumerate(zip(
        starts, ends, cache["own_rads"]
    )):
        dangerous = dangerous_masks[own_index]
        if not np.any(dangerous):
            continue
        distances, _ = segment_distances(start, end, cache["virus_locs"])
        required = (
            own_radius * P["VIRUS_DANGER_BUFFER_MULT"]
        )
        if np.any(dangerous & (distances <= required)):
            return False
    return True


def movement_safe(cache, direction):
    if direction is None:
        return False
    futures = future_own_positions(cache, direction)
    if not virus_paths_safe(cache, direction, futures):
        return False
    _, active = threat_state(cache, futures, cache["own_rads"])
    return not active


def wall_clearance(positions, radii):
    left_bottom = positions - radii[:, None]
    right_top = P["ARENA_SIZE"] - positions - radii[:, None]
    return float(np.min(np.concatenate((left_bottom, right_top), axis=1)))


def virus_shield_score(cache, futures, active_threats):
    """Reward safe small blobs for putting a virus on the pursuer's path."""
    if not active_threats or len(cache["virus_locs"]) == 0:
        return 0.0
    score = 0.0
    shield_indices = cache.get("_escape_shield_indices")
    if shield_indices is None:
        virus_count = len(cache["virus_locs"])
        limit = min(P["ESCAPE_MAX_SHIELD_VIRUSES"], virus_count)
        if limit < virus_count:
            distance_sq = np.min(np.sum(
                (
                    cache["own_locs"][:, None, :]
                    - cache["virus_locs"][None, :, :]
                ) ** 2,
                axis=2,
            ), axis=0)
            shield_indices = np.argpartition(
                distance_sq, limit - 1
            )[:limit]
        else:
            shield_indices = np.arange(virus_count, dtype=int)
        cache["_escape_shield_indices"] = shield_indices
    virus_locs = cache["virus_locs"][shield_indices]
    virus_rads = cache["virus_rads"][shield_indices]
    contribution_cache = {}
    dangerous_masks = cache.get("_escape_virus_dangerous_masks")
    if dangerous_masks is None:
        dangerous_masks = [
            can_pop_virus(float(own_radius), virus_rads)
            for own_radius in cache["own_rads"]
        ]
        cache["_escape_virus_dangerous_masks"] = dangerous_masks
    enemy_pop_masks = cache.setdefault("_escape_virus_enemy_pop_masks", {})
    for threat in active_threats:
        own_i = threat["own_index"]
        own_radius = float(cache["own_rads"][own_i])
        enemy_radius = float(threat["enemy_radius"])
        enemy_pos = threat["enemy_position"]
        geometry_key = (
            int(own_i),
            float(enemy_pos[0]),
            float(enemy_pos[1]),
            enemy_radius,
        )
        cached_contribution = contribution_cache.get(geometry_key)
        if cached_contribution is not None:
            score += cached_contribution
            continue
        own_safe = ~dangerous_masks[own_i]
        enemy_pops = enemy_pop_masks.get(enemy_radius)
        if enemy_pops is None:
            enemy_pops = can_pop_virus(enemy_radius, virus_rads)
            enemy_pop_masks[enemy_radius] = enemy_pops
        useful = own_safe & enemy_pops
        if not np.any(useful):
            contribution_cache[geometry_key] = 0.0
            continue
        future = futures[own_i]
        distances, progress = segment_distances(
            enemy_pos,
            future,
            virus_locs,
        )
        shield = useful & (progress > 0.15) & (progress < 0.95)
        contribution = 0.0
        if np.any(shield):
            contribution = float(np.max(
                np.maximum(0.0, enemy_radius - distances[shield])
            ))
        contribution_cache[geometry_key] = contribution
        score += contribution
    return score


def escape_candidate_futures(cache, directions):
    """Project every escape heading together instead of one NumPy call each."""
    if cache.get("_future_normal_steps") is None:
        future_own_positions(cache, directions[0])
    normal_steps = cache["_future_normal_steps"]
    eject_offset = cache["_future_eject_offset"]
    futures = (
        cache["own_locs"][None, :, :]
        + directions[:, None, :] * normal_steps[None, :, None]
        + eject_offset[None, :, :]
    )
    radii = cache["own_rads"][None, :]
    futures[:, :, 0] = np.minimum(
        np.maximum(futures[:, :, 0], radii),
        P["ARENA_SIZE"] - radii,
    )
    futures[:, :, 1] = np.minimum(
        np.maximum(futures[:, :, 1], radii),
        P["ARENA_SIZE"] - radii,
    )
    return futures


def escape_virus_paths_safe(cache, futures):
    """Vectorized equivalent of virus_paths_safe for escape headings."""
    candidate_count = len(futures)
    if cache["own_count"] >= P["MAX_BLOBS"] or len(cache["virus_locs"]) == 0:
        return np.ones(candidate_count, dtype=bool)

    safe = np.ones(candidate_count, dtype=bool)
    starts = cache["own_locs"]
    points = cache["virus_locs"]
    for own_index, (start, own_radius) in enumerate(zip(
        starts, cache["own_rads"]
    )):
        dangerous = can_pop_virus(float(own_radius), cache["virus_rads"])
        if not np.any(dangerous):
            continue
        segments = futures[:, own_index, :] - start
        length_sq = np.sum(segments * segments, axis=1)
        offsets = points[None, :, :] - start[None, None, :]
        progress = np.sum(
            offsets * segments[:, None, :], axis=2
        ) / np.maximum(length_sq[:, None], 1e-9)
        progress = np.clip(progress, 0.0, 1.0)
        closest = start[None, None, :] + progress[:, :, None] * segments[:, None, :]
        deltas = points[None, :, :] - closest
        distance_sq = np.sum(deltas * deltas, axis=2)
        blocked = np.any(
            dangerous[None, :]
            & (distance_sq <= (own_radius * P["VIRUS_DANGER_BUFFER_MULT"]) ** 2),
            axis=1,
        )
        safe &= ~blocked
    return safe


def escape_threat_metrics(cache, futures):
    """Return best margin and active-envelope count for every heading."""
    candidate_count = len(futures)
    margins_out = np.full(candidate_count, float("inf"), dtype=float)
    active_counts = np.zeros(candidate_count, dtype=int)
    sources = enemy_threat_sources(cache)
    if not sources:
        return margins_out, active_counts

    profiles = threat_profiles(cache, cache["own_rads"])
    source_positions = cache["_threat_positions"]
    for own_index, own_radius in enumerate(cache["own_rads"]):
        profile = profiles[own_index]
        if len(profile["reaches"]) == 0:
            continue
        deltas = (
            futures[:, own_index, None, :]
            - source_positions[None, :, :]
        )
        source_distance = np.sqrt(np.sum(deltas * deltas, axis=2))
        margins = (
            source_distance[:, profile["source_indices"]]
            - profile["reaches"][None, :]
        )
        margins_out = np.minimum(margins_out, np.min(margins, axis=1))
        active_counts += np.sum(
            margins <= float(own_radius) * P["ESCAPE_HARD_MARGIN"],
            axis=1,
        )
    return margins_out, active_counts


def escape_virus_shield_scores(cache, futures, active_threats):
    """Score all headings at once while preserving threat-depth weighting."""
    scores = np.zeros(len(futures), dtype=float)
    if not active_threats or len(cache["virus_locs"]) == 0:
        return scores

    shield_indices = cache.get("_escape_shield_indices")
    if shield_indices is None:
        virus_count = len(cache["virus_locs"])
        limit = min(P["ESCAPE_MAX_SHIELD_VIRUSES"], virus_count)
        if limit < virus_count:
            distance_sq = np.min(np.sum(
                (
                    cache["own_locs"][:, None, :]
                    - cache["virus_locs"][None, :, :]
                ) ** 2,
                axis=2,
            ), axis=0)
            shield_indices = np.argpartition(distance_sq, limit - 1)[:limit]
        else:
            shield_indices = np.arange(virus_count, dtype=int)
        cache["_escape_shield_indices"] = shield_indices
    virus_locs = cache["virus_locs"][shield_indices]
    virus_rads = cache["virus_rads"][shield_indices]

    own_safe_masks = [
        ~can_pop_virus(float(radius), virus_rads)
        for radius in cache["own_rads"]
    ]
    enemy_pop_masks = {}
    for threat in active_threats:
        own_index = int(threat["own_index"])
        enemy_radius = float(threat["enemy_radius"])
        enemy_pops = enemy_pop_masks.get(enemy_radius)
        if enemy_pops is None:
            enemy_pops = can_pop_virus(enemy_radius, virus_rads)
            enemy_pop_masks[enemy_radius] = enemy_pops
        useful = own_safe_masks[own_index] & enemy_pops
        if not np.any(useful):
            continue

        start = np.asarray(threat["enemy_position"], dtype=float)
        segments = futures[:, own_index, :] - start
        length_sq = np.sum(segments * segments, axis=1)
        offsets = virus_locs[None, :, :] - start[None, None, :]
        progress = np.sum(
            offsets * segments[:, None, :], axis=2
        ) / np.maximum(length_sq[:, None], 1e-9)
        clipped = np.clip(progress, 0.0, 1.0)
        closest = start[None, None, :] + clipped[:, :, None] * segments[:, None, :]
        deltas = virus_locs[None, :, :] - closest
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        shield = (
            useful[None, :]
            & (clipped > 0.15)
            & (clipped < 0.95)
        )
        contributions = np.where(
            shield, np.maximum(0.0, enemy_radius - distances), 0.0
        )
        scores += np.max(contributions, axis=1)
    return scores


def override_escape(cache):
    """Escape direct and generalized multi-split kill envelopes."""
    global SPLIT_PLAN_PID, SPLIT_PLAN_STEPS
    _, active = threat_state(cache)
    if not active:
        return None

    # Escape is not an unconditional veto on offense.  If one immediate split
    # eats enough nearby mass that the fed piece survives every remaining
    # attack envelope, taking the food is the stronger escape.
    counter = best_escape_counter_split(cache)
    if counter is not None:
        SPLIT_PLAN_PID = None
        SPLIT_PLAN_STEPS = 0
        direction = counter["direction"]
        return float(direction[0]), float(direction[1]), True

    SPLIT_PLAN_PID = None
    SPLIT_PLAN_STEPS = 0

    weighted_away = np.zeros(2, dtype=float)
    for threat in active:
        own_pos = cache["own_locs"][threat["own_index"]]
        enemy_pos = threat["enemy_position"]
        away = unit(own_pos - enemy_pos)
        if away is not None:
            weighted_away += away / max(threat["reach"] + threat["margin"], 1.0)

    candidates = [direction for direction in DIRECTIONS]
    exact_away = unit(weighted_away)
    if exact_away is not None:
        candidates.append(exact_away)
    # Small blobs may deliberately run through a virus that would pop the
    # pursuer. These candidates are scored, never blindly selected.
    if len(cache["virus_locs"]) > P["ESCAPE_MAX_VIRUS_DIRECTIONS"]:
        virus_order = np.argpartition(
            np.sum(
                (cache["virus_locs"] - cache["position"]) ** 2,
                axis=1,
            ),
            P["ESCAPE_MAX_VIRUS_DIRECTIONS"] - 1,
        )[:P["ESCAPE_MAX_VIRUS_DIRECTIONS"]]
    else:
        virus_order = range(len(cache["virus_locs"]))
    for virus_index in virus_order:
        virus_pos = cache["virus_locs"][virus_index]
        toward = unit(virus_pos - cache["position"])
        if toward is not None:
            candidates.append(toward)

    directions = np.asarray(candidates, dtype=float)
    futures = escape_candidate_futures(cache, directions)
    safe = escape_virus_paths_safe(cache, futures)
    margins, active_counts = escape_threat_metrics(cache, futures)
    move_deltas = futures - cache["own_locs"][None, :, :]
    actual_move = np.mean(np.sqrt(np.sum(
        move_deltas * move_deltas, axis=2
    )), axis=1)
    radii = cache["own_rads"][None, :]
    clearances = np.min(np.concatenate((
        futures - radii[:, :, None],
        P["ARENA_SIZE"] - futures - radii[:, :, None],
    ), axis=2), axis=(1, 2))
    shield_scores = escape_virus_shield_scores(cache, futures, active)
    scores = P["ESCAPE_MARGIN_WEIGHT"] * margins
    scores += P["ESCAPE_MOVE_WEIGHT"] * actual_move
    scores += P["ESCAPE_WALL_WEIGHT"] * clearances
    scores += P["ESCAPE_VIRUS_SHIELD_WEIGHT"] * shield_scores
    scores -= P["ESCAPE_DISTANCE_WEIGHT"] * active_counts
    scores[~safe] = -float("inf")
    best_index = int(np.argmax(scores))
    if not np.isfinite(scores[best_index]):
        return None
    best = directions[best_index]
    return float(best[0]), float(best[1]), False


def override_unstuck(cache):
    """Rare recovery for genuine low-displacement states."""
    global LAST_POSITION, STUCK_COUNT
    if LAST_POSITION is None:
        LAST_POSITION = cache["position"].copy()
        return None

    moved = distance_2d(cache["position"], LAST_POSITION)
    LAST_POSITION = cache["position"].copy()
    if moved < cache["radius"] * P["STUCK_MOVE_RADIUS_MULT"]:
        STUCK_COUNT += 1
    else:
        STUCK_COUNT = 0
    if STUCK_COUNT < P["STUCK_TICKS"]:
        return None

    center = np.array([P["ARENA_SIZE"] / 2.0] * 2, dtype=float)
    center_direction = unit(center - cache["position"])
    best = None
    best_score = -float("inf")
    for direction in DIRECTIONS:
        if not movement_safe(cache, direction):
            continue
        futures = future_own_positions(cache, direction)
        move_deltas = futures - cache["own_locs"]
        displacement = float(np.mean(np.sqrt(np.sum(
            move_deltas * move_deltas, axis=1
        ))))
        score = P["UNSTUCK_MOVE_WEIGHT"] * displacement
        if center_direction is not None:
            score += P["UNSTUCK_CENTER_WEIGHT"] * float(np.dot(
                direction, center_direction
            ))
        if score > best_score:
            best_score = score
            best = direction
    if best is None:
        return None
    STUCK_COUNT = 0
    return float(best[0]), float(best[1]), False


# =============================================================================
# Offensive split rollout
# =============================================================================

def stabilise_projected_owner(positions, masses, blob_ids, eject, cooldowns):
    """Mirror StateMutator._stabilise_same_player_blobs for <=16 pieces."""
    if len(masses) <= 1:
        return positions, masses, blob_ids, eject, cooldowns

    total_mass = float(np.sum(masses))
    centroid = np.sum(positions * masses[:, None], axis=0) / max(total_mass, 1e-9)
    deltas = centroid[None, :] - positions
    distances = np.sqrt(np.sum(deltas * deltas, axis=1))
    moving = distances > 1e-12
    steps = np.minimum(P["MERGE_ATTRACTION_SPEED"], distances[moving])
    positions[moving] += (
        deltas[moving] / distances[moving, None]
    ) * steps[:, None]
    positions = clamp_points(positions, np.sqrt(masses))

    def clamp_index(index):
        radius = float(masses[index] ** 0.5)
        positions[index, 0] = min(
            max(radius, float(positions[index, 0])),
            P["ARENA_SIZE"] - radius,
        )
        positions[index, 1] = min(
            max(radius, float(positions[index, 1])),
            P["ARENA_SIZE"] - radius,
        )

    def merge_touching_once():
        nonlocal positions, masses, blob_ids, eject, cooldowns
        if np.count_nonzero(cooldowns == 0) < 2:
            return False
        radii = np.sqrt(masses)
        # Query snapshots and every appended child are already in increasing
        # engine blob-ID order; filtering a consumed index preserves it.
        for index_a in range(len(masses)):
            for index_b in range(index_a + 1, len(masses)):
                if cooldowns[index_a] > 0 or cooldowns[index_b] > 0:
                    continue
                dx = float(positions[index_b, 0] - positions[index_a, 0])
                dy = float(positions[index_b, 1] - positions[index_a, 1])
                minimum = float(radii[index_a] + radii[index_b] + 1e-4)
                if dx * dx + dy * dy > minimum * minimum:
                    continue
                if (
                    masses[index_a] > masses[index_b]
                    or (
                        masses[index_a] == masses[index_b]
                        and blob_ids[index_a] < blob_ids[index_b]
                    )
                ):
                    survivor, consumed = int(index_a), int(index_b)
                else:
                    survivor, consumed = int(index_b), int(index_a)
                combined = float(masses[survivor] + masses[consumed])
                positions[survivor] = (
                    positions[survivor] * masses[survivor]
                    + positions[consumed] * masses[consumed]
                ) / combined
                eject[survivor] = (
                    eject[survivor] * masses[survivor]
                    + eject[consumed] * masses[consumed]
                ) / combined
                masses[survivor] = combined
                cooldowns[survivor] = 0
                clamp_index(survivor)
                keep = np.arange(len(masses)) != consumed
                positions = positions[keep]
                masses = masses[keep]
                blob_ids = blob_ids[keep]
                eject = eject[keep]
                cooldowns = cooldowns[keep]
                return True
        return False

    def separate(iterations=4):
        nonlocal positions
        pair_a, pair_b = PAIR_INDICES[len(masses)]
        for _ in range(iterations):
            radii = np.sqrt(masses)
            pair_deltas = positions[pair_b] - positions[pair_a]
            pairwise_sq = np.sum(pair_deltas * pair_deltas, axis=1)
            pair_minimum = radii[pair_a] + radii[pair_b] + 1e-4
            initial_overlaps = pairwise_sq < pair_minimum * pair_minimum
            if not np.any(initial_overlaps):
                break
            changed = False
            # A pair that was initially clear only needs a new scalar distance
            # test if an earlier overlap moved one of its members. This keeps
            # the engine's exact sequential order while skipping most of the
            # 120 pair calculations in a 16-piece swarm.
            moved = np.zeros(len(masses), dtype=bool)
            pair_index = 0
            for index_a in range(len(masses)):
                for index_b in range(index_a + 1, len(masses)):
                    initially_overlapping = bool(initial_overlaps[pair_index])
                    pair_index += 1
                    if (
                        not initially_overlapping
                        and not moved[index_a]
                        and not moved[index_b]
                    ):
                        continue
                    dx = float(positions[index_b, 0] - positions[index_a, 0])
                    dy = float(positions[index_b, 1] - positions[index_a, 1])
                    distance_sq = dx * dx + dy * dy
                    minimum = float(radii[index_a] + radii[index_b] + 1e-4)
                    if distance_sq >= minimum * minimum:
                        continue
                    distance = distance_sq ** 0.5
                    if distance == 0.0:
                        direction_x, direction_y = (1.0, 0.0)
                    else:
                        direction_x = dx / distance
                        direction_y = dy / distance
                    overlap = minimum - distance
                    combined = float(masses[index_a] + masses[index_b])
                    move_a = overlap * float(masses[index_b]) / combined
                    move_b = overlap * float(masses[index_a]) / combined
                    positions[index_a, 0] -= direction_x * move_a
                    positions[index_a, 1] -= direction_y * move_a
                    positions[index_b, 0] += direction_x * move_b
                    positions[index_b, 1] += direction_y * move_b
                    clamp_index(index_a)
                    clamp_index(index_b)
                    moved[index_a] = True
                    moved[index_b] = True
                    changed = True
            if not changed:
                break

    while merge_touching_once():
        pass
    separate()
    while merge_touching_once():
        pass
    separate()
    return positions, masses, blob_ids, eject, cooldowns


def rollout_split(cache, direction, depth):
    """Simulate repeated global splits using the engine's tick equations."""
    positions = cache["own_locs_all"].copy()
    masses = cache["own_rads_all"] ** 2
    blob_ids = cache.get(
        "own_ids_all", np.arange(len(positions), dtype=int)
    ).copy()
    next_blob_id = int(np.max(blob_ids, initial=-1)) + 1
    eject = cache.get("own_eject_all", np.zeros_like(positions)).copy()
    cooldowns = cache.get(
        "own_cooldowns_all", np.zeros(len(positions), dtype=int)
    ).copy()
    captured = np.zeros(len(cache["enemy_locs"]), dtype=bool)
    capture_steps = []

    for step in range(depth):
        radii = np.sqrt(masses)
        capacity = P["MAX_BLOBS"] - len(masses)
        # StateMutator snapshots the starting IDs, walks them in sorted order,
        # and appends each child with a fresh increasing ID. This matters for
        # partial 12->16 splits: children created earlier in this command are
        # never eligible until a later command.
        eligible = [
            index for index in range(len(blob_ids))
            if radii[index] >= P["SPLIT_MIN_RADIUS"]
        ][:capacity]
        if len(eligible) == 0 or capacity <= 0:
            return None

        child_positions = []
        child_masses = []
        child_eject = []
        child_ids = []
        child_cooldowns = []
        for index in eligible:
            half_mass = float(masses[index] * 0.5)
            half_radius = np.sqrt(half_mass)

            # The existing blob stays in place with its old ID/eject velocity;
            # the child is appended after every pre-existing blob.
            masses[index] = half_mass
            cooldowns[index] = 18
            child_position = clamp_point(
                positions[index] + direction * (2.0 * half_radius + 1e-4),
                half_radius,
            )
            child_positions.append(child_position)
            child_masses.append(half_mass)
            child_eject.append(direction * P["SPLIT_EJECT_SPEED"])
            child_ids.append(next_blob_id)
            child_cooldowns.append(18)
            next_blob_id += 1

        positions = np.concatenate(
            (positions, np.asarray(child_positions, dtype=float)), axis=0
        )
        masses = np.concatenate(
            (masses, np.asarray(child_masses, dtype=float)), axis=0
        )
        eject = np.concatenate(
            (eject, np.asarray(child_eject, dtype=float)), axis=0
        )
        blob_ids = np.concatenate(
            (blob_ids, np.asarray(child_ids, dtype=int)), axis=0
        )
        cooldowns = np.concatenate(
            (cooldowns, np.asarray(child_cooldowns, dtype=int)), axis=0
        )

        # Every piece receives normal movement, then its current eject impulse.
        radii = np.sqrt(masses)
        normal_steps = np.array([
            movement_speed(radius) for radius in radii
        ], dtype=float)
        positions += direction[None, :] * normal_steps[:, None] + eject
        eject *= P["SPLIT_EJECT_DRAG"]
        cooldowns = np.maximum(cooldowns - 1, 0)
        positions = clamp_points(positions, radii)

        # Engine decays mass, then attracts all sibling pieces by 0.08 toward
        # their mass centroid before resolving viruses and player eating.
        minimum_mass = 0.9 ** 2
        decayed = masses * (1.0 - P["MASS_DECAY_RATE"])
        masses = np.where(
            masses > minimum_mass,
            np.maximum(decayed, minimum_mass),
            masses,
        )
        positions, masses, blob_ids, eject, cooldowns = stabilise_projected_owner(
            positions, masses, blob_ids, eject, cooldowns
        )

        projected_count = len(masses)
        if len(cache["virus_locs"]) > 0:
            for virus_pos, virus_radius in zip(
                cache["virus_locs"], cache["virus_rads"]
            ):
                radii = np.sqrt(masses)
                virus_deltas = positions - virus_pos
                distance_sq = np.sum(virus_deltas * virus_deltas, axis=1)
                candidates = np.where(
                    (distance_sq <= masses)
                    & can_pop_virus(radii, virus_radius)
                )[0]
                if len(candidates) == 0:
                    continue
                # Below the cap the virus replaces the hitting blob with enough
                # pieces to fill all 16 slots, invalidating an enemy split plan.
                if projected_count < P["MAX_BLOBS"]:
                    return None
                eater = int(candidates[np.argmax(radii[candidates])])
                masses[eater] += float(virus_radius ** 2)

        positions, masses, blob_ids, eject, cooldowns = stabilise_projected_owner(
            positions, masses, blob_ids, eject, cooldowns
        )

        eaten_this_step = []
        if len(cache["enemy_locs"]) > 0:
            # The split resolves against where prey is moving, not a frozen
            # screenshot.  One prediction tick per split command is enough to
            # catch lateral runners without expensive branching.
            enemy_positions = clamp_points(
                cache["enemy_locs"]
                + cache["enemy_velocities"] * float(step + 1),
                cache["enemy_rads"],
            )
            # Eating is endpoint-only; crossing over a target during eject does
            # not count. Engine considers larger target blobs first.
            enemy_deltas = (
                positions[:, None, :] - enemy_positions[None, :, :]
            )
            distance_sq = np.sum(enemy_deltas * enemy_deltas, axis=2)
            target_order = np.argsort(cache["enemy_rads"])[::-1]
            for enemy_index in target_order:
                if captured[enemy_index]:
                    continue
                initial_enemy_mass = float(
                    cache["enemy_rads"][enemy_index] ** 2
                )
                if initial_enemy_mass > minimum_mass:
                    enemy_mass = max(
                        minimum_mass,
                        initial_enemy_mass
                        * (1.0 - P["MASS_DECAY_RATE"]) ** (step + 1),
                    )
                else:
                    enemy_mass = initial_enemy_mass
                possible = (
                    (distance_sq[:, enemy_index] <= masses)
                    & can_eat_mass(
                        masses,
                        enemy_mass,
                        P["SPLIT_TARGET_EAT_RATIO"],
                    )
                )
                if not np.any(possible):
                    continue
                possible_indices = np.where(possible)[0]
                piece_index = int(possible_indices[
                    np.argmax(masses[possible_indices])
                ])
                masses[piece_index] += enemy_mass
                captured[enemy_index] = True
                eaten_this_step.append(int(enemy_index))
        positions, masses, blob_ids, eject, cooldowns = stabilise_projected_owner(
            positions, masses, blob_ids, eject, cooldowns
        )
        capture_steps.append(eaten_this_step)

    captured_mass = float(np.sum(cache["enemy_rads"][captured] ** 2))
    largest_index = int(np.argmax(masses))
    return {
        "position": positions[largest_index].copy(),
        "mass": float(masses[largest_index]),
        "radius": float(np.sqrt(masses[largest_index])),
        "positions": positions,
        "masses": masses,
        "captured": captured,
        "captured_mass": captured_mass,
        "capture_steps": capture_steps,
    }


def split_external_risk(cache, rollout, target_pid):
    """Estimate—not veto—the attacker mass exposed to other owners."""
    risk = 0.0
    final_pos = rollout["position"]
    final_radius = rollout["radius"]
    for enemy in enemy_threat_sources(cache):
        enemy_index = enemy["visible_index"]
        if int(enemy["pid"]) == int(target_pid):
            continue
        if enemy_index is not None and rollout["captured"][enemy_index]:
            continue
        enemy_pos = enemy["position"]
        distance = distance_2d(enemy_pos, final_pos)
        for _, _, reach in enemy_reach_options(
            enemy["radius"], enemy["count"], final_radius
        ):
            reach *= enemy["confidence"]
            if distance <= reach:
                risk = max(risk, min(
                    rollout["mass"],
                    float(enemy["radius"] ** 2),
                ))
                break
    return risk


def analytical_split_capture_possible(cache, target_index, depth):
    """Cheap conservative gate before an exact multi-piece rollout.

    The circular reach bound deliberately overestimates real directional
    reach, so a rejected depth cannot capture the target. This avoids paying
    for depth 1, 2 and 3 simulations when only depth 4 is even geometrically
    possible.
    """
    target_radius = float(cache["enemy_rads"][target_index])
    target_mass = target_radius ** 2
    predicted = clamp_point(
        cache["enemy_locs"][target_index]
        + cache["enemy_velocities"][target_index] * float(depth),
        target_radius,
    )
    profile_cache = cache.setdefault("_split_priority_profiles", {})
    for own_pos, own_radius in zip(
        cache["own_locs_all"], cache["own_rads_all"]
    ):
        radius_key = float(own_radius)
        radius_profiles = profile_cache.get(radius_key)
        if radius_profiles is None:
            radius_profiles = []
            for profile_depth in range(1, max_split_depth(cache["own_count"]) + 1):
                radius_profiles.append(split_profile(radius_key, profile_depth))
            profile_cache[radius_key] = radius_profiles
        if depth > len(radius_profiles):
            continue
        before, after, travels = radius_profiles[depth - 1]
        if np.any(before < P["SPLIT_MIN_RADIUS"]):
            continue
        final_radius = float(after[-1])
        if not can_eat_mass(
            final_radius ** 2,
            target_mass,
            P["SPLIT_TARGET_EAT_RATIO"],
        ):
            continue
        # Circular reach is a superset of the actual directed endpoint.
        if distance_2d(predicted, own_pos) <= float(np.sum(travels) + final_radius):
            return True
    return False


def split_target_priority(cache, target_index, maximum_depth):
    """Cheap reach/payoff ranking before running the bounded split rollout."""
    target_pos = cache["enemy_locs"][target_index]
    target_radius = float(cache["enemy_rads"][target_index])
    target_mass = target_radius ** 2
    velocity = cache["enemy_velocities"][target_index]
    best_gap = float("inf")
    feasible = False

    profile_cache = cache.setdefault("_split_priority_profiles", {})
    for own_pos, own_radius in zip(
        cache["own_locs_all"], cache["own_rads_all"]
    ):
        radius_key = float(own_radius)
        radius_profiles = profile_cache.get(radius_key)
        if radius_profiles is None:
            radius_profiles = []
            profile_cache[radius_key] = radius_profiles
        if len(radius_profiles) < maximum_depth:
            for depth in range(len(radius_profiles) + 1, maximum_depth + 1):
                before, after, travels = split_profile(radius_key, depth)
                radius_profiles.append((before, after, travels))
        for depth in range(1, maximum_depth + 1):
            before, after, travels = radius_profiles[depth - 1]
            if np.any(before < P["SPLIT_MIN_RADIUS"]):
                break
            final_radius = float(after[-1])
            if not can_eat_mass(
                final_radius ** 2,
                target_mass,
                P["SPLIT_TARGET_EAT_RATIO"],
            ):
                continue
            predicted = clamp_point(
                target_pos + velocity * float(depth), target_radius
            )
            distance = distance_2d(predicted, own_pos)
            reach = float(np.sum(travels) + final_radius)
            gap = max(0.0, distance - reach)
            best_gap = min(best_gap, gap)
            feasible = feasible or gap <= movement_speed(final_radius) * 1.5

    # Nearby edible crumbs must outrank distant large targets, while a target
    # already inside a credible split envelope receives a decisive boost.
    value = target_mass / (1.0 + best_gap)
    return (1 if feasible else 0, value)


def split_opportunities(
    cache,
    required_pid=None,
    max_depth_override=None,
    target_limit=None,
    rollout_step_budget=None,
):
    if len(cache["enemy_locs"]) == 0:
        return []
    maximum_depth = max_split_depth(cache["own_count"])
    if max_depth_override is not None:
        maximum_depth = min(maximum_depth, int(max_depth_override))
    if maximum_depth <= 0:
        return []

    target_indices = np.arange(len(cache["enemy_locs"]), dtype=int)
    if required_pid is not None:
        target_indices = target_indices[
            cache["enemy_pids"][target_indices] == int(required_pid)
        ]
    prioritised = sorted(
        (
            (split_target_priority(cache, int(index), maximum_depth), int(index))
            for index in target_indices
        ),
        reverse=True,
    )
    # The generous 1.5-tick tolerance in split_target_priority makes this a
    # safe compute filter: an endpoint rollout cannot capture targets outside
    # every analytical split envelope.
    if target_limit is None:
        target_limit = P["SPLIT_MAX_TARGETS"]
    if rollout_step_budget is None:
        rollout_step_budget = P["SPLIT_ROLLOUT_STEP_BUDGET"]
    target_indices = [
        index for priority, index in prioritised if priority[0] > 0
    ][:target_limit]

    opportunities = []
    rollout_step_cost = 0
    budget_exhausted = False
    for target_index in target_indices:
        target_pos = cache["enemy_locs"][target_index]
        target_mass = float(cache["enemy_rads"][target_index] ** 2)
        target_pid = int(cache["enemy_pids"][target_index])
        owner_mass = cache["summaries"].get(
            target_pid, {"mass": target_mass}
        )["mass"]

        launcher_order = np.argsort(np.sum(
            (cache["own_locs_all"] - target_pos) ** 2, axis=1
        ))[:P["SPLIT_MAX_LAUNCHERS"]]

        for launcher_index in launcher_order:
            for depth in range(1, maximum_depth + 1):
                predicted_target = clamp_point(
                    target_pos
                    + cache["enemy_velocities"][target_index] * float(depth),
                    float(cache["enemy_rads"][target_index]),
                )
                direction = unit(
                    predicted_target - cache["own_locs_all"][launcher_index]
                )
                if direction is None:
                    continue
                if not analytical_split_capture_possible(
                    cache, target_index, depth
                ):
                    continue
                if (
                    rollout_step_cost + depth
                    > rollout_step_budget
                ):
                    budget_exhausted = True
                    break
                rollout_step_cost += depth
                rollout = rollout_split(cache, direction, depth)
                if rollout is None or not rollout["captured"][target_index]:
                    continue

                owner_mask = cache["enemy_pids"] == target_pid
                captured_owner_mass = float(np.sum(
                    cache["enemy_rads"][owner_mask & rollout["captured"]] ** 2
                ))
                owner_remaining = max(0.0, owner_mass - captured_owner_mass)

                # When attacking upward, the complete rollout must eat enough
                # that the fed leading piece survives the owner's remainder.
                if (
                    cache["rank_mode"] < 3
                    and owner_mass
                    > cache["mass"] * P["SPLIT_LARGER_OWNER_RATIO"]
                    and can_eat_mass(
                        owner_remaining,
                        rollout["mass"],
                        P["EAT_MASS_RATIO"],
                    )
                ):
                    continue

                external_risk = split_external_risk(cache, rollout, target_pid)
                projected_count = min(
                    P["MAX_BLOBS"], cache["own_count"] * (2 ** depth)
                )
                fragmentation = max(0, projected_count - cache["own_count"])
                risk_level = max(0, cache["rank_mode"])
                if risk_level >= 3:
                    depth_cost = 0.0
                    fragment_cost = 0.0
                    external_risk_weight = 0.0
                else:
                    depth_cost = P["SPLIT_DEPTH_COST"] * max(
                        0.5,
                        1.0 - P["RANK_DEPTH_COST_DISCOUNT"] * risk_level,
                    )
                    fragment_cost = P["SPLIT_FRAGMENT_COST"] * max(
                        0.45,
                        1.0 - P["RANK_FRAGMENT_COST_DISCOUNT"] * risk_level,
                    )
                    external_risk_weight = (
                        P["SPLIT_EXTERNAL_RISK_WEIGHT"] * max(
                            0.4,
                            1.0 - (
                                P["RANK_EXTERNAL_RISK_DISCOUNT"] * risk_level
                            ),
                        )
                    )
                utility = P["SPLIT_CAPTURE_WEIGHT"] * rollout["captured_mass"]
                utility += P["SPLIT_TARGET_WEIGHT"] * target_mass
                target_rank = cache["rank_by_pid"].get(
                    target_pid, cache["live_rank"]
                )
                ranks_above = max(0, cache["live_rank"] - target_rank)
                utility += (
                    P["RANK_TARGET_MASS_BONUS"]
                    * risk_level
                    * min(ranks_above, 3)
                    * target_mass
                )
                utility -= depth_cost * depth
                utility -= (
                    fragment_cost
                    * fragmentation
                    * cache["mass"]
                    / max(P["MAX_BLOBS"], 1)
                )
                utility -= external_risk_weight * external_risk

                opportunities.append({
                    "utility": float(utility),
                    "direction": direction,
                    "depth": depth,
                    "target_index": int(target_index),
                    "target_pid": target_pid,
                    "target_position": target_pos.copy(),
                    "launcher_index": int(launcher_index),
                    "rollout": rollout,
                })
                # Continue deeper only on a later tick if it remains useful;
                # this bounds work and avoids pre-committing fragmentation.
                break
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    opportunities.sort(key=lambda item: item["utility"], reverse=True)
    return opportunities


def counter_split_survives(cache, opportunity):
    """Whether an immediate mass-gaining split escapes all known punishers."""
    rollout = opportunity["rollout"]
    if (
        rollout["captured_mass"]
        < cache["mass"] * P["COUNTER_SPLIT_MIN_GAIN_RATIO"]
    ):
        return False

    fed_pos = rollout["position"]
    fed_radius = rollout["radius"]
    for enemy in enemy_threat_sources(cache):
        visible_index = enemy["visible_index"]
        if (
            enemy.get("merge_virtual", False)
            and int(enemy["pid"]) in cache["summaries"]
        ):
            member_mask = cache["enemy_pids"] == int(enemy["pid"])
            captured_mass = float(np.sum(
                cache["enemy_rads"][member_mask & rollout["captured"]] ** 2
            ))
            owner_mass = float(cache["summaries"][int(enemy["pid"])]["mass"])
            if captured_mass >= owner_mass - 1e-6:
                continue
        if (
            visible_index is not None
            and rollout["captured"][visible_index]
        ):
            continue
        distance = distance_2d(enemy["position"], fed_pos)
        for _, _, reach in enemy_reach_options(
            enemy["radius"], enemy["count"], fed_radius
        ):
            if distance <= reach * enemy["confidence"]:
                return False
    return True


def split_min_utility(cache):
    if cache["rank_mode"] >= 3:
        return P["RANK_LAST_STAND_MIN_UTILITY"]
    if cache["rank_mode"] >= 2:
        multiplier = P["RANK_DESPERATE_MIN_UTILITY_MULT"]
    elif cache["rank_mode"] == 1:
        multiplier = P["RANK_PRESSURE_MIN_UTILITY_MULT"]
    else:
        multiplier = 1.0
    return P["SPLIT_MIN_UTILITY"] * multiplier


def best_escape_counter_split(cache):
    """Return the best one-tick counter-capture, never a speculative combo."""
    for option in split_opportunities(
        cache,
        max_depth_override=1,
        target_limit=P["ESCAPE_COUNTER_MAX_TARGETS"],
        rollout_step_budget=P["ESCAPE_COUNTER_STEP_BUDGET"],
    ):
        if option["utility"] < split_min_utility(cache):
            break
        if counter_split_survives(cache, option):
            return option
    return None


def clear_split_plan():
    global SPLIT_PLAN_PID, SPLIT_PLAN_POSITION
    global SPLIT_PLAN_STEPS, SPLIT_PLAN_WAIT
    SPLIT_PLAN_PID = None
    SPLIT_PLAN_POSITION = None
    SPLIT_PLAN_STEPS = 0
    SPLIT_PLAN_WAIT = 0


def override_split(cache):
    """Execute or continue the best bounded mass-positive split rollout."""
    global SPLIT_PLAN_PID, SPLIT_PLAN_POSITION
    global SPLIT_PLAN_STEPS, SPLIT_PLAN_WAIT

    if SPLIT_PLAN_PID is not None and SPLIT_PLAN_STEPS > 0:
        options = split_opportunities(
            cache,
            required_pid=SPLIT_PLAN_PID,
            max_depth_override=SPLIT_PLAN_STEPS,
        )
        if options and options[0]["utility"] >= split_min_utility(cache):
            best = options[0]
            SPLIT_PLAN_STEPS = max(0, best["depth"] - 1)
            SPLIT_PLAN_POSITION = best["target_position"].copy()
            SPLIT_PLAN_WAIT = 0
            if SPLIT_PLAN_STEPS == 0:
                clear_split_plan()
            direction = best["direction"]
            return float(direction[0]), float(direction[1]), True

        SPLIT_PLAN_WAIT += 1
        visible = np.where(cache["enemy_pids"] == int(SPLIT_PLAN_PID))[0]
        if SPLIT_PLAN_WAIT <= P["SPLIT_PLAN_WAIT_TICKS"] and len(visible) > 0:
            target = cache["enemy_locs"][visible[np.argmin(np.sum(
                (cache["enemy_locs"][visible] - SPLIT_PLAN_POSITION) ** 2,
                axis=1,
            ))]]
            direction = unit(target - cache["position"])
            if direction is not None and movement_safe(cache, direction):
                return float(direction[0]), float(direction[1]), False
        clear_split_plan()

    options = split_opportunities(cache)
    if not options or options[0]["utility"] < split_min_utility(cache):
        return None

    best = options[0]
    if best["depth"] > 1:
        SPLIT_PLAN_PID = best["target_pid"]
        SPLIT_PLAN_POSITION = best["target_position"].copy()
        SPLIT_PLAN_STEPS = best["depth"] - 1
        SPLIT_PLAN_WAIT = 0
    direction = best["direction"]
    return float(direction[0]), float(direction[1]), True


# =============================================================================
# Predictive chase
# =============================================================================

def intercept_point(hunter_pos, hunter_radius, hunter_step, prey_pos, prey_velocity):
    """Solve a constant-velocity engine-radius intercept.

    Return whether a real non-negative intercept exists; blindly substituting
    the maximum horizon when it does not is what made the bot pursue faster
    prey forever in open space.
    """
    relative = prey_pos - hunter_pos
    speed = max(float(hunter_step), 1e-6)
    capture_radius = float(hunter_radius)
    a = float(np.dot(prey_velocity, prey_velocity) - speed ** 2)
    b = float(2.0 * (
        np.dot(relative, prey_velocity) - capture_radius * speed
    ))
    c = float(np.dot(relative, relative) - capture_radius ** 2)

    if c <= 0.0:
        eta = 0.0
    else:
        roots = []
        if abs(a) <= 1e-9:
            if abs(b) > 1e-9:
                roots.append(-c / b)
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = np.sqrt(discriminant)
                roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        positive = [root for root in roots if root >= 0.0 and np.isfinite(root)]
        has_intercept = bool(positive)
        eta = min(positive) if positive else P["CHASE_MAX_INTERCEPT_TICKS"]

    if c <= 0.0:
        has_intercept = True

    eta = float(np.clip(eta, 0.0, P["CHASE_MAX_INTERCEPT_TICKS"]))
    point = prey_pos + prey_velocity * eta
    point = np.clip(point, 0.0, P["ARENA_SIZE"])
    return point, eta, has_intercept


def capture_position_feasible(hunter_radius, prey_position):
    """Whether any legal hunter centre can eat a target at this position.

    The engine eats when centre distance is at most the eater radius. Near a
    corner, a large hunter's centre cannot approach the walls. Consequently a
    tiny target can be permanently unreachable even while the circles appear
    close. This projection computes the exact minimum legal centre distance.
    """
    nearest_legal_center = clamp_point(prey_position, float(hunter_radius))
    minimum_distance = distance_2d(nearest_legal_center, prey_position)
    return minimum_distance <= float(hunter_radius) + 1e-6


def ray_wall_distance(position, direction):
    """Distance from a prey position to the wall it is being driven toward."""
    direction = unit(direction)
    if direction is None:
        return float("inf")
    distances = []
    for axis in range(2):
        component = direction[axis]
        if component > 1e-9:
            distances.append((P["ARENA_SIZE"] - position[axis]) / component)
        elif component < -1e-9:
            distances.append(-position[axis] / component)
    positive = [distance for distance in distances if distance >= 0.0]
    return min(positive) if positive else float("inf")


def chase_direction_safe(cache, desired):
    if desired is not None and movement_safe(cache, desired):
        return desired

    # When a virus blocks the direct chase line, keep taking the same safe side
    # around it. Re-selecting from symmetric headings every tick caused the
    # observed left-right oscillation. This fast path also avoids the 12-way
    # safety scan while the existing detour remains useful.
    held = unit(LAST_DIRECTION)
    if (
        desired is not None
        and held is not None
        and float(np.dot(held, desired)) >= 0.50
        and movement_safe(cache, held)
    ):
        return held

    best = None
    best_dot = -float("inf")
    for direction in DIRECTIONS:
        if not movement_safe(cache, direction):
            continue
        alignment = -1.0 if desired is None else float(np.dot(direction, desired))
        if alignment > best_dot:
            best_dot = alignment
            best = direction
    return best


def chase_enters_merge_trap(
    cache,
    hunter_pos,
    hunter_radius,
    prey_index,
    capture_point,
    eta,
):
    """Reject a fragment chase that feeds its hunter into a merging owner."""
    pid = int(cache["enemy_pids"][prey_index])
    profile = owner_merge_profile(cache, pid)
    if profile is None:
        return False

    prey_mass = float(cache["enemy_rads"][prey_index] ** 2)
    post_capture_mass = float(hunter_radius ** 2 + prey_mass)
    # Remove the prey's estimated contribution from the virtual merge. If the
    # hunter becomes too large for the remaining pack, the chase is self-safe.
    remaining_mass = max(
        0.0,
        profile["mass"] - prey_mass * profile["confidence"],
    )
    if not can_eat_mass(
        remaining_mass, post_capture_mass, P["EAT_MASS_RATIO"]
    ):
        return False

    owner_velocity = ENEMY_TRACKS.get(pid, {}).get(
        "velocity", np.zeros(2, dtype=float)
    )
    pack_position = profile["position"] + owner_velocity * min(float(eta), 3.0)
    pack_position = np.clip(pack_position, 0.0, P["ARENA_SIZE"])
    remaining_radius = float(np.sqrt(remaining_mass))
    reaction_ticks = min(
        float(eta), P["CHASE_MAX_INTERCEPT_TICKS"]
    ) + P["MERGE_CHASE_REACTION_TICKS"]
    punish_reach = (
        remaining_radius
        + movement_speed(remaining_radius) * reaction_ticks
    )
    return distance_2d(capture_point, pack_position) <= punish_reach


def override_chase(cache):
    """Predict prey motion and retain only chases with a credible wall finish."""
    global CHASE_TARGET_TRACK_ID, CHASE_TICKS
    if len(cache["enemy_locs"]) == 0:
        CHASE_TARGET_TRACK_ID = None
        CHASE_TICKS = 0
        return None

    risk_level = max(0, cache["rank_mode"])
    chase_range_scale = 1.0 + P["RANK_CHASE_RANGE_BONUS"] * risk_level
    chase_cost_scale = max(
        0.55, 1.0 - P["RANK_CHASE_COST_DISCOUNT"] * risk_level
    )
    acquire_range = P["CHASE_ACQUIRE_RANGE_MULT"] * chase_range_scale
    direct_range = P["CHASE_DIRECT_RANGE_MULT"] * chase_range_scale
    wall_horizon = min(
        P["ARENA_SIZE"], P["CHASE_WALL_HORIZON"] * chase_range_scale
    )

    own_locs = cache["own_locs_all"]
    own_rads = cache["own_rads_all"]
    enemy_locs = cache["enemy_locs"]
    enemy_rads = cache["enemy_rads"]
    pair_deltas = enemy_locs[None, :, :] - own_locs[:, None, :]
    pair_distances = np.sqrt(np.sum(pair_deltas * pair_deltas, axis=2))
    pair_gaps = pair_distances - own_rads[:, None]
    edible = (
        own_rads[:, None] ** 2
        >= enemy_rads[None, :] ** 2 * P["EAT_MASS_RATIO"]
    )
    locked_enemies = cache["enemy_track_ids"] == CHASE_TARGET_TRACK_ID
    valid = edible & (
        (pair_gaps <= own_rads[:, None] * acquire_range)
        | locked_enemies[None, :]
    )
    if not np.any(valid):
        CHASE_TARGET_TRACK_ID = None
        CHASE_TICKS = 0
        return None
    cheap_scores = (
        P["CHASE_TARGET_MASS_WEIGHT"] * enemy_rads[None, :] ** 2
        - P["CHASE_DISTANCE_WEIGHT"] * chase_cost_scale
        * np.maximum(pair_gaps, 0.0)
        + P["CHASE_LOCK_BONUS"] * locked_enemies[None, :]
    )
    cheap_scores[~valid] = -float("inf")
    selected_pairs = set()
    for enemy_index in range(len(enemy_locs)):
        if np.any(valid[:, enemy_index]):
            selected_pairs.add((
                int(np.argmax(cheap_scores[:, enemy_index])), enemy_index
            ))
    remaining = max(
        0, P["CHASE_MAX_PAIR_EVALUATIONS"] - len(selected_pairs)
    )
    if remaining > 0:
        valid_flat = np.flatnonzero(valid)
        if len(valid_flat) > remaining:
            flat_scores = cheap_scores.ravel()[valid_flat]
            top = np.argpartition(
                flat_scores, len(flat_scores) - remaining
            )[-remaining:]
            valid_flat = valid_flat[top]
        for flat_index in valid_flat:
            selected_pairs.add(tuple(int(value) for value in np.unravel_index(
                int(flat_index), valid.shape
            )))
            if len(selected_pairs) >= P["CHASE_MAX_PAIR_EVALUATIONS"]:
                break

    candidates = []
    # Every fragment is a potential hunter.  Restricting this to the two
    # largest blobs wastes most of a 16-piece swarm even when a local fragment
    # can finish prey immediately.
    for own_index, (hunter_pos, hunter_radius) in enumerate(zip(
        cache["own_locs_all"], cache["own_rads_all"]
    )):
        hunter_mass = float(hunter_radius ** 2)
        hunter_step = movement_speed(hunter_radius)
        for enemy_index, (prey_pos, prey_radius, pid, track_id) in enumerate(zip(
            cache["enemy_locs"], cache["enemy_rads"], cache["enemy_pids"],
            cache["enemy_track_ids"],
        )):
            if (own_index, enemy_index) not in selected_pairs:
                continue
            prey_mass = float(prey_radius ** 2)
            if not can_eat_mass(hunter_mass, prey_mass):
                continue

            vector = prey_pos - hunter_pos
            centre_distance = distance_2d(prey_pos, hunter_pos)
            # Engine capture ignores target radius: target centre must enter
            # the hunter radius. This is the true remaining capture gap.
            edge_distance = centre_distance - hunter_radius
            locked = int(track_id) == CHASE_TARGET_TRACK_ID
            close = edge_distance <= hunter_radius * acquire_range
            if not close and not locked:
                continue

            velocity = cache["enemy_velocities"][enemy_index]
            escape_direction = unit(velocity)
            if escape_direction is None:
                escape_direction = unit(vector)
            wall_distance = ray_wall_distance(prey_pos, escape_direction)

            predicted, eta, has_intercept = intercept_point(
                hunter_pos,
                float(hunter_radius),
                hunter_step,
                prey_pos,
                velocity,
            )
            predicted = np.clip(
                predicted,
                float(prey_radius),
                P["ARENA_SIZE"] - float(prey_radius),
            )

            # A target in a corner may be geometrically impossible to overlap
            # because our centre is radius-constrained away from both walls.
            # This is target-specific and clears immediately if that piece
            # moves back into a capturable position.
            if not capture_position_feasible(hunter_radius, predicted):
                continue

            # Piece-level edibility is insufficient when that piece belongs to
            # a much larger compact owner. Do not walk a small hunter into its
            # siblings unless eating the target makes the hunter unpunishable.
            if chase_enters_merge_trap(
                cache,
                hunter_pos,
                float(hunter_radius),
                enemy_index,
                predicted,
                eta,
            ):
                continue

            # If constant-velocity interception is impossible, only a nearby
            # wall can eventually change the geometry in our favour.
            if not has_intercept and wall_distance > wall_horizon:
                continue
            direct = (
                edge_distance <= hunter_radius * direct_range
                or wall_distance <= wall_horizon * 0.35
            )
            if direct:
                aim = predicted
            else:
                center = np.array([P["ARENA_SIZE"] / 2.0] * 2, dtype=float)
                center_side = unit(center - predicted)
                if center_side is None:
                    center_side = unit(predicted - hunter_pos)
                if center_side is None:
                    center_side = np.zeros(2, dtype=float)
                offset = min(
                    hunter_radius * P["CHASE_BLOCK_OFFSET_MULT"],
                    prey_radius * 2.0,
                )
                aim = predicted + center_side * offset

            score = P["CHASE_TARGET_MASS_WEIGHT"] * prey_mass
            target_rank = cache["rank_by_pid"].get(
                int(pid), cache["live_rank"]
            )
            ranks_above = max(0, cache["live_rank"] - target_rank)
            score += (
                P["RANK_TARGET_MASS_BONUS"]
                * risk_level
                * min(ranks_above, 3)
                * prey_mass
            )
            score -= (
                P["CHASE_DISTANCE_WEIGHT"]
                * chase_cost_scale
                * max(edge_distance, 0.0)
            )
            score -= P["CHASE_ETA_WEIGHT"] * chase_cost_scale * eta
            if locked:
                score += P["CHASE_LOCK_BONUS"]
            if wall_distance <= wall_horizon:
                score += P["CHASE_LOCK_BONUS"] * (
                    1.0 - wall_distance / max(wall_horizon, 1e-6)
                )

            candidates.append({
                "score": float(score),
                "pid": int(pid),
                "track_id": int(track_id),
                "aim": aim,
                "hunter_pos": hunter_pos,
                "wall_distance": wall_distance,
            })

    if not candidates:
        CHASE_TARGET_TRACK_ID = None
        CHASE_TICKS = 0
        return None

    locked_candidates = [
        item for item in candidates
        if (
            item["track_id"] == CHASE_TARGET_TRACK_ID
            and CHASE_TICKS < P["CHASE_LOCK_TICKS"]
        )
    ]
    pool = locked_candidates or candidates
    best = max(pool, key=lambda item: item["score"])
    desired = unit(best["aim"] - best["hunter_pos"])
    direction = chase_direction_safe(cache, desired)
    if direction is None:
        return None

    if best["track_id"] == CHASE_TARGET_TRACK_ID:
        CHASE_TICKS += 1
    else:
        CHASE_TARGET_TRACK_ID = best["track_id"]
        CHASE_TICKS = 0
    return float(direction[0]), float(direction[1]), False


# =============================================================================
# Virus farming
# =============================================================================

def virus_action_safe(cache, virus_pos, piece_radius):
    """Check generalized enemy reach to the expected post-virus pieces."""
    for enemy in enemy_threat_sources(cache):
        enemy_pos = enemy["position"]
        distance = min(
            distance_2d(enemy_pos, cache["position"]),
            distance_2d(enemy_pos, virus_pos),
        )
        for _, _, reach in enemy_reach_options(
            enemy["radius"], enemy["count"], piece_radius
        ):
            reach *= enemy["confidence"]
            reaction = (
                movement_speed(enemy["radius"])
                * P["VIRUS_ENEMY_RANGE_MULT"]
            )
            if distance <= reach + reaction:
                return False
    return True


def movement_safe_to_virus(cache, direction, target_pos, expected_piece_radius):
    """Allow safe virus chains instead of treating neighbouring farms as walls."""
    futures = future_own_positions(cache, direction)
    _, threats = threat_state(cache, futures, cache["own_rads"])
    if threats:
        return False
    if cache["own_count"] >= P["MAX_BLOBS"]:
        return True
    for start, end, own_radius in zip(
        cache["own_locs"], futures, cache["own_rads"]
    ):
        if len(cache["virus_locs"]) == 0:
            continue
        distances, _ = segment_distances(start, end, cache["virus_locs"])
        dangerous = can_pop_virus(own_radius, cache["virus_rads"])
        for index, virus_pos in enumerate(cache["virus_locs"]):
            if distance_2d(virus_pos, target_pos) <= 0.2:
                dangerous[index] = False
        required = np.full(
            len(cache["virus_rads"]), own_radius, dtype=float
        )
        encountered = np.where(dangerous & (distances <= required))[0]
        # Hitting another safe virus first is progress, not a blocked route.
        for index in encountered:
            if not virus_action_safe(
                cache, cache["virus_locs"][index], expected_piece_radius
            ):
                return False
    return True


def virus_farming_allowed(cache):
    """Farm while small, capped at 16, or safely dominant over all known owners."""
    if cache["own_count"] >= P["MAX_BLOBS"]:
        return True
    if cache["rank_mode"] > 0:
        return True
    if cache["mass"] < P["VIRUS_MAX_FARM_MASS"]:
        return True
    largest_enemy_mass = max(
        (float(track["mass"]) for track in ENEMY_TRACKS.values()),
        default=0.0,
    )
    return cache["mass"] >= largest_enemy_mass * P["VIRUS_DOMINANCE_RATIO"]


def override_virus(cache):
    """Choose direct collision or an intentional split into a useful virus."""
    global VIRUS_TARGET_KEY
    if not virus_farming_allowed(cache):
        VIRUS_TARGET_KEY = None
        return None
    if not VIRUS_MEMORY:
        VIRUS_TARGET_KEY = None
        return None

    opportunities = []
    memory_items = VIRUS_MEMORY.items()
    if VIRUS_TARGET_KEY in VIRUS_MEMORY:
        # Target continuity is already a core part of the policy.  Evaluating
        # every other remembered virus on every travel tick added substantial
        # cumulative work and was the source of occasional target oscillation.
        memory_items = ((VIRUS_TARGET_KEY, VIRUS_MEMORY[VIRUS_TARGET_KEY]),)
    for key, info in memory_items:
        virus_pos = info["position"]
        virus_radius = info["radius"]
        visible = key in cache["visible_virus_keys"]
        # At the cap use every fragment as a possible collector. Eligibility
        # still applies: the engine does not waive the 1.2 mass ratio at 16.
        if cache["own_count"] >= P["MAX_BLOBS"]:
            virus_own_locs = cache["own_locs_all"]
            virus_own_rads = cache["own_rads_all"]
        else:
            virus_own_locs = cache["own_locs"]
            virus_own_rads = cache["own_rads"]
        for own_index, (own_pos, own_radius) in enumerate(zip(
            virus_own_locs, virus_own_rads
        )):
            distance = distance_2d(virus_pos, own_pos)
            # Engine virus contact is centre distance <= blob radius; virus
            # radius is deliberately absent from this collision test.
            edge_distance = max(distance - own_radius, 0.0)
            travel_ticks = edge_distance / max(movement_speed(own_radius), 1e-6)
            arrival_mass = max(
                0.9 ** 2,
                float(own_radius ** 2)
                * (1.0 - P["MASS_DECAY_RATE"]) ** max(1, int(np.ceil(travel_ticks))),
            )
            arrival_radius = float(np.sqrt(arrival_mass))
            if not can_pop_virus(arrival_radius, virus_radius):
                continue
            chain_value = VIRUS_CHAIN_VALUES.get(key, 0.0)
            value = (
                virus_radius ** 2
                + P["VIRUS_CHAIN_WEIGHT"] * chain_value
            ) / max(travel_ticks + 1.0, 1.0)
            if not visible:
                value *= 0.65 if key == VIRUS_TARGET_KEY else 0.50
            if key == VIRUS_TARGET_KEY:
                value *= P["VIRUS_TARGET_LOCK_BONUS"]
            opportunities.append({
                "key": key,
                "position": virus_pos,
                "radius": virus_radius,
                "own_index": own_index,
                "distance": distance,
                "edge_distance": edge_distance,
                "travel_ticks": travel_ticks,
                "arrival_mass": arrival_mass,
                "value": value,
                "visible": visible,
            })
    if not opportunities:
        VIRUS_TARGET_KEY = None
        return None
    # Rank by payoff with a continuity bonus instead of making visibility an
    # absolute priority.  The latter caused target flips every time a virus
    # crossed the vision boundary—the observed back-and-forth behaviour.
    opportunities.sort(key=lambda item: item["value"], reverse=True)

    for item in opportunities:
        if cache["own_count"] >= P["MAX_BLOBS"]:
            own_pos = cache["own_locs_all"][item["own_index"]]
            own_radius = float(cache["own_rads_all"][item["own_index"]])
        else:
            own_pos = cache["own_locs"][item["own_index"]]
            own_radius = float(cache["own_rads"][item["own_index"]])
        direction = unit(item["position"] - own_pos)
        if direction is None:
            continue

        if (
            not item["visible"]
            and item["distance"] <= own_radius + 0.5
        ):
            VIRUS_MEMORY.pop(item["key"], None)
            if item["key"] == VIRUS_TARGET_KEY:
                VIRUS_TARGET_KEY = None
            continue

        virus_piece_count = max(1, P["MAX_BLOBS"] - cache["own_count"] + 1)
        expected_piece_radius = float(np.sqrt(
            (item["arrival_mass"] + item["radius"] ** 2)
            / virus_piece_count
        ))
        if not virus_action_safe(cache, item["position"], expected_piece_radius):
            continue

        # Manual splitting is worthwhile only when it reaches a visible virus
        # materially sooner and the launched half remains large enough to pop.
        can_split = (
            item["visible"]
            and cache["own_count"] < P["MAX_BLOBS"]
            and own_radius >= P["SPLIT_MIN_RADIUS"]
        )
        split_spawn_radius = own_radius / SQRT2
        split_radius = float(np.sqrt(
            own_radius ** 2 * 0.5 * (1.0 - P["MASS_DECAY_RATE"])
        ))
        split_reach = (
            2.0 * split_spawn_radius
            + movement_speed(split_spawn_radius)
            + P["SPLIT_EJECT_SPEED"]
            + split_radius
        )
        direct_ticks = item["travel_ticks"]
        manual_utility = (
            P["VIRUS_MANUAL_SPLIT_TIME_BONUS"] * max(0.0, direct_ticks - 1.0)
            - P["VIRUS_MANUAL_SPLIT_RISK_COST"] * cache["own_count"]
        )
        if (
            can_split
            and can_pop_virus(split_radius, item["radius"])
            and item["distance"] <= split_reach
            and manual_utility > 0.0
        ):
            VIRUS_TARGET_KEY = item["key"]
            return float(direction[0]), float(direction[1]), True

        if movement_safe_to_virus(
            cache, direction, item["position"], expected_piece_radius
        ):
            VIRUS_TARGET_KEY = item["key"]
            return float(direction[0]), float(direction[1]), False
    VIRUS_TARGET_KEY = None
    return None


# =============================================================================
# Food farming and final default
# =============================================================================

def food_key(position):
    return stable_position_key(position)


def ranked_food(cache):
    foods = cache.get("food_locs")
    if foods is None:
        food_objects = cache["food_objects"]
        if food_objects:
            foods = np.array([food.pos for food in food_objects], dtype=float)
            foods, = cap_nearest(
                foods, P["MAX_FOOD"], cache["position"]
            )
        else:
            foods = np.empty((0, 2), dtype=float)
        cache["food_locs"] = foods
    if len(foods) == 0:
        return []
    food_deltas = foods - cache["position"]
    distances = np.sqrt(np.sum(food_deltas * food_deltas, axis=1))
    pair_deltas = foods[:, None, :] - foods[None, :, :]
    pairwise_sq = np.sum(pair_deltas * pair_deltas, axis=2)
    density = np.sum(
        pairwise_sq <= P["FOOD_CLUSTER_RADIUS"] ** 2, axis=1
    )

    wall_dist = np.minimum.reduce((
        foods[:, 0],
        P["ARENA_SIZE"] - foods[:, 0],
        foods[:, 1],
        P["ARENA_SIZE"] - foods[:, 1],
    ))
    corner_factor = np.where(
        wall_dist < P["FOOD_CORNER_PENALTY"], 0.45, 1.0
    )
    scores = (
        (1.0 + P["FOOD_CLUSTER_WEIGHT"] * density)
        * corner_factor
        / np.maximum(distances, 0.25) ** P["FOOD_DISTANCE_POWER"]
    )
    order = np.argsort(scores)[::-1]
    return [
        (float(scores[index]), foods[index], food_key(foods[index]))
        for index in order
    ]


def safest_toward(cache, desired):
    desired = unit(desired)
    if desired is not None and movement_safe(cache, desired):
        return desired
    best = None
    best_score = -float("inf")
    center = np.array([P["ARENA_SIZE"] / 2.0] * 2, dtype=float)
    center_dir = unit(center - cache["position"])
    for direction in DIRECTIONS:
        if not movement_safe(cache, direction):
            continue
        score = 0.0
        if desired is not None:
            score += float(np.dot(direction, desired))
        if center_dir is not None:
            score += 0.08 * float(np.dot(direction, center_dir))
        if score > best_score:
            best_score = score
            best = direction
    return best


def override_food(cache):
    """Cluster-aware food collection; stable roaming is the only default."""
    global FOOD_TARGET, FOOD_TARGET_TICKS, ROAM_DIRECTION, ROAM_TICKS
    # Honour the short food lock directly. Rebuilding the 40x40 density matrix
    # while travelling toward the same visible pellet cannot improve the move
    # unless that pellet disappears or the lock expires.
    if FOOD_TARGET is not None and FOOD_TARGET_TICKS < P["FOOD_LOCK_TICKS"]:
        locked_target = next((
            np.asarray(food.pos, dtype=float)
            for food in cache["food_objects"]
            if food_key(food.pos) == FOOD_TARGET
        ), None)
        if locked_target is not None:
            if distance_2d(
                locked_target, cache["position"]
            ) > P["FOOD_REACHED_DISTANCE"]:
                direction = safest_toward(
                    cache, locked_target - cache["position"]
                )
                if direction is not None:
                    FOOD_TARGET_TICKS += 1
                    return float(direction[0]), float(direction[1]), False
            else:
                FOOD_TARGET = None
                FOOD_TARGET_TICKS = 0

    ranked = ranked_food(cache)
    if ranked:
        selected = ranked[0]
        if FOOD_TARGET is not None and FOOD_TARGET_TICKS < P["FOOD_LOCK_TICKS"]:
            locked = next((item for item in ranked if item[2] == FOOD_TARGET), None)
            if locked is not None and locked[0] >= selected[0] * 0.60:
                selected = locked

        _, target, key = selected
        if distance_2d(target, cache["position"]) <= P["FOOD_REACHED_DISTANCE"]:
            FOOD_TARGET = None
            FOOD_TARGET_TICKS = 0
        else:
            if key == FOOD_TARGET:
                FOOD_TARGET_TICKS += 1
            else:
                FOOD_TARGET = key
                FOOD_TARGET_TICKS = 0
            direction = safest_toward(cache, target - cache["position"])
            if direction is not None:
                return float(direction[0]), float(direction[1]), False

    # No visible/reachable food: roam stably toward open central space. This is
    # intentionally not a beam search and has constant bounded cost.
    if ROAM_DIRECTION is not None and ROAM_TICKS < P["ROAM_LOCK_TICKS"]:
        if movement_safe(cache, ROAM_DIRECTION):
            ROAM_TICKS += 1
            return float(ROAM_DIRECTION[0]), float(ROAM_DIRECTION[1]), False

    center = np.array([P["ARENA_SIZE"] / 2.0] * 2, dtype=float)
    center_dir = unit(center - cache["position"])
    best = None
    best_score = -float("inf")
    for direction in DIRECTIONS:
        if not movement_safe(cache, direction):
            continue
        futures = future_own_positions(cache, direction)
        score = wall_clearance(futures, cache["own_rads"])
        if center_dir is not None:
            score += P["ROAM_CENTER_WEIGHT"] * float(np.dot(direction, center_dir))
        score += P["ROAM_MOMENTUM_WEIGHT"] * float(np.dot(
            direction, LAST_DIRECTION
        ))
        if score > best_score:
            best_score = score
            best = direction
    if best is None:
        best = center_dir if center_dir is not None else LAST_DIRECTION
    ROAM_DIRECTION = best.copy()
    ROAM_TICKS = 0
    return float(best[0]), float(best[1]), False


# =============================================================================
# Priority and engine loop
# =============================================================================

OVERRIDES = {
    "escape": override_escape,
    "unstuck": override_unstuck,
    "split": override_split,
    "chase": override_chase,
    "virus": override_virus,
    "food": override_food,
}


def choose_direction(game):
    global LAST_DIRECTION, LAST_SPLIT_REQUESTED
    cache = build_cache(game)
    for name in OVERRIDE_ORDER:
        result = OVERRIDES[name](cache)
        if result is None:
            continue
        dx, dy, should_split = result
        direction = unit(np.array([dx, dy], dtype=float))
        if direction is None:
            continue
        LAST_DIRECTION = direction
        LAST_SPLIT_REQUESTED = bool(should_split)
        return float(direction[0]), float(direction[1]), bool(should_split), cache

    # Food is designed to return a stable roam even without visible food.
    direction = unit(LAST_DIRECTION)
    LAST_SPLIT_REQUESTED = False
    return float(direction[0]), float(direction[1]), False, cache


def main():
    global LAST_SPLIT_REQUESTED
    game = Game()
    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                try:
                    dx, dy, should_split, cache = choose_direction(game)
                    game.send_move(MovePlayer(
                        player_id=cache["player_id"],
                        direction=DirectionModel(x=dx, y=dy),
                        split=should_split,
                    ))
                except Exception:
                    LAST_SPLIT_REQUESTED = False
                    game.send_move(MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=1.0, y=0.0),
                        split=False,
                    ))
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()
