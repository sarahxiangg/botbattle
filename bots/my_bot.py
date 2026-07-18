from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import os
import json

import numpy as np


# =============================================================================
# Tuning surface
# =============================================================================
#
# Every strategically meaningful number lives here. An optimiser can write a
# JSON object to BOT_TUNING_PATH; unknown keys fail loudly so experiments never
# silently tune a misspelled parameter. OVERRIDE_ORDER is tunable as a list.

PARAMS = {
    # Engine / bounded work
    "ARENA_SIZE": 60.0,
    "MAX_BLOBS": 16,
    "NUM_DIRECTIONS": 12,
    "MAX_ENEMIES": 24,
    "MAX_FOOD": 40,
    "MAX_VIRUSES": 14,
    "SAFETY_OWN_PIECES": 2,
    "STEP_RADIUS_MULT": 1.2709918139713585,
    "EAT_MASS_RATIO": 1.12,

    # Shared split geometry
    "SPLIT_MIN_RADIUS": 2.0,
    "SPLIT_TRAVEL_MULT": 2.50,
    "SPLIT_REACH_RELIABILITY": 0.6931558907420518,
    "SPLIT_TARGET_EAT_RATIO": 1.10,
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
    "ENEMY_MEMORY_TICKS": 18,
    "ENEMY_MEMORY_FULL_TICKS": 7,
    "ENEMY_MEMORY_MIN_CONFIDENCE": 0.40,

    # Unstuck
    "STUCK_TICKS": 4,
    "STUCK_MOVE_RADIUS_MULT": 0.04,
    "UNSTUCK_MOVE_WEIGHT": 1200.0,
    "UNSTUCK_CENTER_WEIGHT": 300.0,

    # Offensive splitting / bounded rollout
    "SPLIT_MAX_TARGETS": 6,
    "SPLIT_MAX_LAUNCHERS": 2,
    "SPLIT_CAPTURE_WEIGHT": 4.799170739486253,
    "SPLIT_TARGET_WEIGHT": 3.0,
    "SPLIT_DEPTH_COST": 4.382917879561087,
    "SPLIT_FRAGMENT_COST": 0.22909141833208407,
    "SPLIT_EXTERNAL_RISK_WEIGHT": 0.9004198132569647,
    "SPLIT_MIN_UTILITY": 4.020980901689232,
    "SPLIT_LARGER_OWNER_RATIO": 1.013412343039414,
    "SPLIT_PLAN_WAIT_TICKS": 3,

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
    "BASE_PLAYER_SPEED": 1.1,
    "PLAYER_SPEED_RADIUS_FACTOR": 0.08,
    "MIN_PLAYER_SPEED": 0.25,

    # Virus farming
    "VIRUS_EAT_RATIO": 1.10,
    "VIRUS_DANGER_BUFFER_MULT": 1.10,
    "VIRUS_ENEMY_RANGE_MULT": 5.324337587731354,
    "VIRUS_PIECE_RADIUS_MULT": 0.35,
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

OVERRIDE_ORDER = ["escape", "unstuck", "split", "virus", "chase", "food"]


# def load_tuning_config():
#     """Load flat numeric parameters and optional override order from JSON."""
#     global OVERRIDE_ORDER
#     # BOT_CONFIG is accepted for compatibility with earlier tournament
#     # wrappers; BOT_TUNING_PATH is the canonical name for the clean bot.
#     path = os.getenv("BOT_TUNING_PATH") or os.getenv("BOT_CONFIG")
#     if not path:
#         return

#     with open(path, "r", encoding="utf-8") as handle:
#         values = json.load(handle)
#     if not isinstance(values, dict):
#         raise ValueError("Tuning config must be a JSON object")

#     allowed = set(PARAMS) | {"OVERRIDE_ORDER"}
#     unknown = set(values) - allowed
#     if unknown:
#         raise ValueError(f"Unknown tuning parameters: {sorted(unknown)}")

#     for name, value in values.items():
#         if name == "OVERRIDE_ORDER":
#             if sorted(value) != sorted(OVERRIDE_ORDER):
#                 raise ValueError("OVERRIDE_ORDER must contain each strategy exactly once")
#             OVERRIDE_ORDER = list(value)
#             continue
#         default = PARAMS[name]
#         PARAMS[name] = type(default)(value)
#     refresh_derived()


P = PARAMS
SQRT2 = np.sqrt(2.0)
DIRECTIONS = np.empty((0, 2), dtype=float)


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
LAST_POSITION = None
STUCK_COUNT = 0

ENEMY_TRACKS = {}
ENEMY_PIECE_TRACKS = {}
NEXT_ENEMY_PIECE_TRACK_ID = 1
FRAME_TICK = 0

CHASE_TARGET_TRACK_ID = None
CHASE_TICKS = 0

SPLIT_PLAN_PID = None
SPLIT_PLAN_POSITION = None
SPLIT_PLAN_STEPS = 0
SPLIT_PLAN_WAIT = 0

VIRUS_MEMORY = {}
VIRUS_TARGET_KEY = None
FOOD_TARGET = None
FOOD_TARGET_TICKS = 0
ROAM_DIRECTION = None
ROAM_TICKS = 0


# =============================================================================
# Geometry and cache
# =============================================================================

def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    return None if length <= 1e-9 else vector / length


def can_eat_mass(eater_mass, target_mass, ratio=None):
    if ratio is None:
        ratio = P["EAT_MASS_RATIO"]
    return np.asarray(eater_mass) >= np.asarray(target_mass) * ratio


def movement_speed(radius):
    """Actual engine centre speed for a blob of this radius."""
    return max(
        P["MIN_PLAYER_SPEED"],
        P["BASE_PLAYER_SPEED"]
        / (1.0 + float(radius) * P["PLAYER_SPEED_RADIUS_FACTOR"]),
    )


def clamp_points(points, radii):
    points = np.asarray(points, dtype=float).copy()
    radii = np.asarray(radii, dtype=float)
    points[:, 0] = np.clip(points[:, 0], radii, P["ARENA_SIZE"] - radii)
    points[:, 1] = np.clip(points[:, 1], radii, P["ARENA_SIZE"] - radii)
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
        return np.linalg.norm(points - start, axis=1), np.zeros(len(points))
    progress = np.sum((points - start) * segment, axis=1) / length_sq
    progress = np.clip(progress, 0.0, 1.0)
    closest = start + progress[:, None] * segment
    return np.linalg.norm(points - closest, axis=1), progress


def max_split_depth(piece_count):
    if piece_count <= 0 or piece_count >= P["MAX_BLOBS"]:
        return 0
    cap_depth = int(np.floor(np.log2(P["MAX_BLOBS"] / piece_count)))
    return max(0, min(P["MAX_SPLIT_DEPTH"], cap_depth))


def split_profile(radius, depth):
    """Radii before/after and cumulative reliable reach for repeated splits."""
    before = radius / (SQRT2 ** np.arange(depth, dtype=float))
    after = before / SQRT2
    travels = before * P["SPLIT_TRAVEL_MULT"] * P["SPLIT_REACH_RELIABILITY"]
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
        masses = rads ** 2
        total = float(np.sum(masses))
        centroid = np.sum(locs * masses[:, None], axis=0) / max(total, 1e-9)
        result[pid] = {
            "mass": total,
            "count": len(pieces),
            "centroid": centroid,
            "max_radius": float(np.max(rads)),
        }
    return result


def update_enemy_tracks(summaries):
    global ENEMY_TRACKS
    updated = {}
    alpha = P["CHASE_VELOCITY_NEW_WEIGHT"]
    for pid, summary in summaries.items():
        prior = ENEMY_TRACKS.get(pid)
        if prior is None:
            velocity = np.zeros(2, dtype=float)
        else:
            elapsed = max(1, FRAME_TICK - prior["tick"])
            measured = (summary["centroid"] - prior["position"]) / elapsed
            velocity = alpha * measured + (1.0 - alpha) * prior["velocity"]
        updated[pid] = {
            "tick": FRAME_TICK,
            "position": summary["centroid"].copy(),
            "velocity": velocity,
            "mass": summary["mass"],
            "max_radius": summary["max_radius"],
            "count": summary["count"],
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
    assigned_current = set()
    assigned_previous = set()
    matches = {}

    pairs = []
    for current_i, (pos, radius, pid) in enumerate(zip(
        enemy_locs, enemy_rads, enemy_pids
    )):
        for key, prior in previous.items():
            if int(prior["pid"]) != int(pid):
                continue
            age = max(1, FRAME_TICK - prior["tick"])
            predicted = prior["position"] + prior["velocity"] * age
            distance = float(np.linalg.norm(pos - predicted))
            gate = max(
                P["CHASE_PIECE_MATCH_DISTANCE"],
                1.5 * (float(radius) + float(prior["radius"])),
            )
            if distance <= gate:
                pairs.append((distance, current_i, key))

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
    global VIRUS_MEMORY
    for pos, radius in zip(virus_locs, virus_rads):
        key = tuple(np.round(pos, 1))
        VIRUS_MEMORY[key] = {
            "position": pos.copy(),
            "radius": float(radius),
            "tick": FRAME_TICK,
        }
    VIRUS_MEMORY = {
        key: value for key, value in VIRUS_MEMORY.items()
        if FRAME_TICK - value["tick"] <= P["VIRUS_MEMORY_TICKS"]
    }


def build_cache(game):
    global FRAME_TICK
    FRAME_TICK += 1
    me = game.state.me

    own = list(me.blobs.values())
    if own:
        own_locs_all = np.array([blob.pos for blob in own], dtype=float)
        own_rads_all = np.array([blob.radius for blob in own], dtype=float)
    else:
        own_locs_all = np.array([[me.x, me.y]], dtype=float)
        own_rads_all = np.array([me.radius], dtype=float)

    own_order = np.argsort(own_rads_all)[::-1]
    keep = own_order[:P["SAFETY_OWN_PIECES"]]
    own_locs = own_locs_all[keep]
    own_rads = own_rads_all[keep]

    enemies_all = [
        blob for blob in game.state.visible_blobs
        if getattr(blob, "player_id", None) != me.player_id
    ]
    summaries = player_summaries(enemies_all)
    update_enemy_tracks(summaries)

    if enemies_all:
        enemy_locs = np.array([blob.pos for blob in enemies_all], dtype=float)
        enemy_rads = np.array([blob.radius for blob in enemies_all], dtype=float)
        enemy_pids = np.array([blob.player_id for blob in enemies_all], dtype=int)
        enemy_locs, enemy_rads, enemy_pids = cap_nearest(
            enemy_locs,
            P["MAX_ENEMIES"],
            np.array([me.x, me.y], dtype=float),
            enemy_rads,
            enemy_pids,
        )
    else:
        enemy_locs = np.empty((0, 2), dtype=float)
        enemy_rads = np.empty(0, dtype=float)
        enemy_pids = np.empty(0, dtype=int)

    enemy_track_ids, enemy_velocities = update_enemy_piece_tracks(
        enemy_locs, enemy_rads, enemy_pids
    )

    # Players that just left vision remain escape/split-risk threats, but are
    # deliberately not inserted into visible targets for chase or capture.
    memory_threats = []
    for pid, track in ENEMY_TRACKS.items():
        if pid in summaries:
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
        memory_threats.append({
            "pid": int(pid),
            "position": predicted,
            # Total last-seen mass covers the dangerous merge-back case.
            "radius": float(np.sqrt(track["mass"])),
            "count": int(track["count"]),
            "confidence": float(confidence),
            "age": int(age),
        })

    foods = game.state.visible_food
    if foods:
        food_locs = np.array([food.pos for food in foods], dtype=float)
        food_locs, = cap_nearest(
            food_locs,
            P["MAX_FOOD"],
            np.array([me.x, me.y], dtype=float),
        )
    else:
        food_locs = np.empty((0, 2), dtype=float)

    viruses = game.state.visible_viruses
    if viruses:
        virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
        virus_rads = np.array([virus.radius for virus in viruses], dtype=float)
        virus_locs, virus_rads = cap_nearest(
            virus_locs,
            P["MAX_VIRUSES"],
            np.array([me.x, me.y], dtype=float),
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
        "player_id": int(me.player_id),
        "position": position,
        "radius": float(np.sqrt(total_mass)),
        "mass": total_mass,
        "own_count": len(own_rads_all),
        "own_locs": own_locs,
        "own_rads": own_rads,
        "own_locs_all": own_locs_all,
        "own_rads_all": own_rads_all,
        "enemy_locs": enemy_locs,
        "enemy_rads": enemy_rads,
        "enemy_pids": enemy_pids,
        "enemy_track_ids": enemy_track_ids,
        "enemy_velocities": enemy_velocities,
        "memory_threats": memory_threats,
        "summaries": summaries,
        "food_locs": food_locs,
        "virus_locs": virus_locs,
        "virus_rads": virus_rads,
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


def enemy_reach_options(enemy_radius, owner_count, target_radius):
    """Yield (depth, resulting radius, centre reach) able to eat a target."""
    enemy_radius = float(enemy_radius)
    enemy_mass = enemy_radius ** 2
    target_mass = float(target_radius ** 2)

    # Splitting only makes an individual attacker smaller. If the current blob
    # cannot eat the target, no deeper split from that blob can either.
    if enemy_mass < target_mass * P["EAT_MASS_RATIO"]:
        return

    proximity = target_radius * P["ESCAPE_DIRECT_RANGE_MULT"]
    yield 0, enemy_radius, enemy_radius + proximity

    for depth in range(1, max_split_depth(owner_count) + 1):
        before, after, _ = split_profile(enemy_radius, depth)
        if before[-1] < P["SPLIT_MIN_RADIUS"]:
            break
        resulting_radius = float(after[-1])
        if resulting_radius ** 2 < target_mass * P["EAT_MASS_RATIO"]:
            continue
        theoretical_travel = float(np.sum(
            before * P["SPLIT_TRAVEL_MULT"]
        ))
        centre_reach = (
            theoretical_travel + resulting_radius
        ) * P["ESCAPE_SPLIT_REACH_BUFFER"]
        yield depth, resulting_radius, centre_reach


def enemy_threat_sources(cache):
    """Yield visible enemies plus decaying, escape-only last-seen owners."""
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
        })
    for threat in cache["memory_threats"]:
        sources.append({
            **threat,
            "visible_index": None,
        })
    cache["_threat_sources"] = sources
    return sources


def threat_profiles(cache, own_rads):
    """Cache attack reaches; only positions change across direction scoring."""
    key = tuple(float(radius) for radius in own_rads)
    profiles_by_radius = cache.setdefault("_threat_profiles", {})
    cached = profiles_by_radius.get(key)
    if cached is not None:
        return cached
    sources = enemy_threat_sources(cache)
    profiles = []
    for own_radius in own_rads:
        per_enemy = []
        for enemy in sources:
            per_enemy.append([
                (depth, attack_radius, reach * enemy["confidence"])
                for depth, attack_radius, reach in enemy_reach_options(
                    enemy["radius"], enemy["count"], float(own_radius)
                )
            ])
        profiles.append(per_enemy)
    profiles_by_radius[key] = profiles
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
    for own_index, (own_pos, own_radius) in enumerate(zip(own_locs, own_rads)):
        for enemy_index, enemy in enumerate(sources):
            enemy_pos = enemy["position"]
            delta = enemy_pos - own_pos
            distance = float(np.hypot(delta[0], delta[1]))
            for depth, attack_radius, reach in profiles[own_index][enemy_index]:
                margin = distance - reach
                best_margin = min(best_margin, margin)
                if margin <= own_radius * P["ESCAPE_HARD_MARGIN"]:
                    active.append({
                        "own_index": own_index,
                        "enemy_position": enemy_pos,
                        "enemy_radius": enemy["radius"],
                        "enemy_pid": enemy["pid"],
                        "enemy_visible_index": enemy["visible_index"],
                        "depth": depth,
                        "attack_radius": attack_radius,
                        "reach": reach,
                        "margin": margin,
                    })
    return best_margin, active


def future_own_positions(cache, direction):
    starts = cache["own_locs"]
    ends = starts + direction[None, :] * cache["step"]
    return clamp_points(ends, cache["own_rads"])


def virus_paths_safe(cache, direction):
    if cache["own_count"] >= P["MAX_BLOBS"] or len(cache["virus_locs"]) == 0:
        return True
    starts = cache["own_locs"]
    ends = future_own_positions(cache, direction)
    for start, end, own_radius in zip(starts, ends, cache["own_rads"]):
        dangerous = own_radius > cache["virus_rads"] * P["VIRUS_EAT_RATIO"]
        if not np.any(dangerous):
            continue
        distances, _ = segment_distances(start, end, cache["virus_locs"])
        required = (
            own_radius + cache["virus_rads"]
        ) * P["VIRUS_DANGER_BUFFER_MULT"]
        if np.any(dangerous & (distances <= required)):
            return False
    return True


def movement_safe(cache, direction):
    if direction is None or not virus_paths_safe(cache, direction):
        return False
    futures = future_own_positions(cache, direction)
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
    for threat in active_threats:
        own_i = threat["own_index"]
        own_radius = float(cache["own_rads"][own_i])
        enemy_radius = float(threat["enemy_radius"])
        own_safe = own_radius <= cache["virus_rads"] * P["VIRUS_EAT_RATIO"]
        enemy_pops = enemy_radius > cache["virus_rads"] * P["VIRUS_EAT_RATIO"]
        useful = own_safe & enemy_pops
        if not np.any(useful):
            continue
        enemy_pos = threat["enemy_position"]
        future = futures[own_i]
        distances, progress = segment_distances(
            enemy_pos,
            future,
            cache["virus_locs"],
        )
        shield = useful & (progress > 0.15) & (progress < 0.95)
        if np.any(shield):
            score += float(np.max(
                np.maximum(0.0, enemy_radius + cache["virus_rads"][shield] - distances[shield])
            ))
    return score


def override_escape(cache):
    """Escape direct and generalized multi-split kill envelopes."""
    global SPLIT_PLAN_PID, SPLIT_PLAN_STEPS
    _, active = threat_state(cache)
    if not active:
        return None

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
    for virus_pos in cache["virus_locs"]:
        toward = unit(virus_pos - cache["position"])
        if toward is not None:
            candidates.append(toward)

    best = None
    best_score = -float("inf")
    for direction in candidates:
        if not virus_paths_safe(cache, direction):
            continue
        futures = future_own_positions(cache, direction)
        margin, remaining = threat_state(cache, futures, cache["own_rads"])
        actual_move = float(np.mean(np.linalg.norm(
            futures - cache["own_locs"], axis=1
        )))
        score = P["ESCAPE_MARGIN_WEIGHT"] * margin
        score += P["ESCAPE_MOVE_WEIGHT"] * actual_move
        score += P["ESCAPE_WALL_WEIGHT"] * wall_clearance(
            futures, cache["own_rads"]
        )
        score += P["ESCAPE_VIRUS_SHIELD_WEIGHT"] * virus_shield_score(
            cache, futures, active
        )
        score -= P["ESCAPE_DISTANCE_WEIGHT"] * len(remaining)
        if score > best_score:
            best_score = score
            best = direction

    return None if best is None else (float(best[0]), float(best[1]), False)


def override_unstuck(cache):
    """Rare recovery for genuine low-displacement states."""
    global LAST_POSITION, STUCK_COUNT
    if LAST_POSITION is None:
        LAST_POSITION = cache["position"].copy()
        return None

    moved = float(np.linalg.norm(cache["position"] - LAST_POSITION))
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
        displacement = float(np.mean(np.linalg.norm(
            futures - cache["own_locs"], axis=1
        )))
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

def rollout_split(cache, direction, depth):
    """Simulate every branch created by repeated global split commands."""
    positions = cache["own_locs_all"].copy()
    masses = cache["own_rads_all"] ** 2
    captured = np.zeros(len(cache["enemy_locs"]), dtype=bool)
    capture_steps = []

    for step in range(depth):
        radii = np.sqrt(masses)
        eligible = np.where(radii >= P["SPLIT_MIN_RADIUS"])[0]
        capacity = P["MAX_BLOBS"] - len(masses)
        if len(eligible) == 0 or capacity <= 0:
            return None

        # If the cap would be exceeded, model the largest eligible pieces as
        # the ones receiving the remaining split slots.
        eligible = eligible[np.argsort(masses[eligible])[::-1]][:capacity]
        split_set = set(int(index) for index in eligible)
        next_positions = []
        next_masses = []
        path_starts = []
        path_ends = []

        for index, (position, mass, radius) in enumerate(zip(
            positions, masses, radii
        )):
            if index not in split_set:
                next_positions.append(position.copy())
                next_masses.append(float(mass))
                path_starts.append(position.copy())
                path_ends.append(position.copy())
                continue

            half_mass = float(mass * 0.5)
            half_radius = np.sqrt(half_mass)
            travel = (
                radius
                * P["SPLIT_TRAVEL_MULT"]
                * P["SPLIT_REACH_RELIABILITY"]
            )
            landing = clamp_point(position + direction * travel, half_radius)

            # Stationary half.
            next_positions.append(position.copy())
            next_masses.append(half_mass)
            path_starts.append(position.copy())
            path_ends.append(position.copy())
            # Launched half.
            next_positions.append(landing)
            next_masses.append(half_mass)
            path_starts.append(position.copy())
            path_ends.append(landing)

        positions = np.array(next_positions, dtype=float)
        masses = np.array(next_masses, dtype=float)
        path_starts = np.array(path_starts, dtype=float)
        path_ends = np.array(path_ends, dtype=float)

        projected_count = len(masses)
        if projected_count < P["MAX_BLOBS"] and len(cache["virus_locs"]) > 0:
            for start, end, mass in zip(path_starts, path_ends, masses):
                radius = np.sqrt(mass)
                distances, _ = segment_distances(
                    start, end, cache["virus_locs"]
                )
                dangerous = radius > cache["virus_rads"] * P["VIRUS_EAT_RATIO"]
                if np.any(dangerous & (
                    distances <= radius + cache["virus_rads"]
                )):
                    return None

        eaten_this_step = []
        if len(cache["enemy_locs"]) > 0:
            # Precompute target distance/progress to every new piece path.
            distance_matrix = np.empty((len(masses), len(cache["enemy_locs"])))
            progress_matrix = np.empty_like(distance_matrix)
            for piece_index, (start, end) in enumerate(zip(path_starts, path_ends)):
                distance_matrix[piece_index], progress_matrix[piece_index] = (
                    segment_distances(start, end, cache["enemy_locs"])
                )

            target_order = np.argsort(np.min(progress_matrix, axis=0))
            for enemy_index in target_order:
                if captured[enemy_index]:
                    continue
                enemy_mass = float(cache["enemy_rads"][enemy_index] ** 2)
                piece_radii = np.sqrt(masses)
                possible = (
                    (distance_matrix[:, enemy_index] <= piece_radii)
                    & can_eat_mass(
                        masses,
                        enemy_mass,
                        P["SPLIT_TARGET_EAT_RATIO"],
                    )
                )
                if not np.any(possible):
                    continue
                possible_indices = np.where(possible)[0]
                margins = (
                    piece_radii[possible_indices]
                    - distance_matrix[possible_indices, enemy_index]
                )
                piece_index = int(possible_indices[np.argmax(margins)])
                masses[piece_index] += enemy_mass
                captured[enemy_index] = True
                eaten_this_step.append(int(enemy_index))
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
        distance = float(np.linalg.norm(enemy_pos - final_pos))
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


def split_opportunities(cache, required_pid=None, max_depth_override=None):
    if len(cache["enemy_locs"]) == 0:
        return []
    maximum_depth = max_split_depth(cache["own_count"])
    if max_depth_override is not None:
        maximum_depth = min(maximum_depth, int(max_depth_override))
    if maximum_depth <= 0:
        return []

    target_indices = np.argsort(cache["enemy_rads"])[::-1]
    if required_pid is not None:
        target_indices = target_indices[
            cache["enemy_pids"][target_indices] == int(required_pid)
        ]
    target_indices = target_indices[:P["SPLIT_MAX_TARGETS"]]

    opportunities = []
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
            launcher_mass = float(cache["own_rads_all"][launcher_index] ** 2)
            direction = unit(target_pos - cache["own_locs_all"][launcher_index])
            if direction is None:
                continue

            for depth in range(1, maximum_depth + 1):
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
                    owner_mass > cache["mass"] * P["SPLIT_LARGER_OWNER_RATIO"]
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
                utility = P["SPLIT_CAPTURE_WEIGHT"] * rollout["captured_mass"]
                utility += P["SPLIT_TARGET_WEIGHT"] * target_mass
                utility -= P["SPLIT_DEPTH_COST"] * depth
                utility -= (
                    P["SPLIT_FRAGMENT_COST"]
                    * fragmentation
                    * cache["mass"]
                    / max(P["MAX_BLOBS"], 1)
                )
                utility -= P["SPLIT_EXTERNAL_RISK_WEIGHT"] * external_risk

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

    opportunities.sort(key=lambda item: item["utility"], reverse=True)
    return opportunities


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
        if options and options[0]["utility"] >= P["SPLIT_MIN_UTILITY"]:
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
    if not options or options[0]["utility"] < P["SPLIT_MIN_UTILITY"]:
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
    minimum_distance = float(np.linalg.norm(
        nearest_legal_center - np.asarray(prey_position, dtype=float)
    ))
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


def override_chase(cache):
    """Predict prey motion and retain only chases with a credible wall finish."""
    global CHASE_TARGET_TRACK_ID, CHASE_TICKS
    if len(cache["enemy_locs"]) == 0:
        CHASE_TARGET_TRACK_ID = None
        CHASE_TICKS = 0
        return None

    candidates = []
    for own_index, (hunter_pos, hunter_radius) in enumerate(zip(
        cache["own_locs"], cache["own_rads"]
    )):
        hunter_mass = float(hunter_radius ** 2)
        hunter_step = movement_speed(hunter_radius)
        for enemy_index, (prey_pos, prey_radius, pid, track_id) in enumerate(zip(
            cache["enemy_locs"], cache["enemy_rads"], cache["enemy_pids"],
            cache["enemy_track_ids"],
        )):
            prey_mass = float(prey_radius ** 2)
            if not can_eat_mass(hunter_mass, prey_mass):
                continue

            vector = prey_pos - hunter_pos
            centre_distance = float(np.linalg.norm(vector))
            edge_distance = centre_distance - hunter_radius - prey_radius
            locked = int(track_id) == CHASE_TARGET_TRACK_ID
            close = edge_distance <= hunter_radius * P["CHASE_ACQUIRE_RANGE_MULT"]
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

            # If constant-velocity interception is impossible, only a nearby
            # wall can eventually change the geometry in our favour.
            if not has_intercept and wall_distance > P["CHASE_WALL_HORIZON"]:
                continue
            direct = (
                edge_distance <= hunter_radius * P["CHASE_DIRECT_RANGE_MULT"]
                or wall_distance <= P["CHASE_WALL_HORIZON"] * 0.35
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
            score -= P["CHASE_DISTANCE_WEIGHT"] * max(edge_distance, 0.0)
            score -= P["CHASE_ETA_WEIGHT"] * eta
            if locked:
                score += P["CHASE_LOCK_BONUS"]
            if wall_distance <= P["CHASE_WALL_HORIZON"]:
                score += P["CHASE_LOCK_BONUS"] * (
                    1.0 - wall_distance / max(P["CHASE_WALL_HORIZON"], 1e-6)
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
            float(np.linalg.norm(enemy_pos - cache["position"])),
            float(np.linalg.norm(enemy_pos - virus_pos)),
        )
        for _, _, reach in enemy_reach_options(
            enemy["radius"], enemy["count"], piece_radius
        ):
            reach *= enemy["confidence"]
            if distance <= reach + cache["radius"] * P["VIRUS_ENEMY_RANGE_MULT"]:
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
        dangerous = own_radius > cache["virus_rads"] * P["VIRUS_EAT_RATIO"]
        for index, virus_pos in enumerate(cache["virus_locs"]):
            if np.linalg.norm(virus_pos - target_pos) <= 0.2:
                dangerous[index] = False
        required = own_radius + cache["virus_rads"]
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
    for key, info in VIRUS_MEMORY.items():
        virus_pos = info["position"]
        virus_radius = info["radius"]
        visible = any(
            np.linalg.norm(pos - virus_pos) <= 0.2
            for pos in cache["virus_locs"]
        )
        for own_index, (own_pos, own_radius) in enumerate(zip(
            cache["own_locs"], cache["own_rads"]
        )):
            if own_radius <= virus_radius * P["VIRUS_EAT_RATIO"]:
                continue
            distance = float(np.linalg.norm(virus_pos - own_pos))
            edge_distance = max(distance - own_radius - virus_radius, 0.0)
            travel_ticks = edge_distance / max(movement_speed(own_radius), 1e-6)
            chain_value = 0.0
            for other_key, other in VIRUS_MEMORY.items():
                if other_key == key:
                    continue
                separation = float(np.linalg.norm(
                    other["position"] - virus_pos
                ))
                if separation <= P["VIRUS_CHAIN_RADIUS"]:
                    chain_value += (
                        other["radius"] ** 2
                        * (1.0 - separation / P["VIRUS_CHAIN_RADIUS"])
                    )
            value = (
                virus_radius ** 2
                + P["VIRUS_CHAIN_WEIGHT"] * chain_value
            ) / max(travel_ticks + 1.0, 1.0)
            if not visible:
                value *= 0.65
            if key == VIRUS_TARGET_KEY and visible:
                value *= P["VIRUS_TARGET_LOCK_BONUS"]
            opportunities.append({
                "key": key,
                "position": virus_pos,
                "radius": virus_radius,
                "own_index": own_index,
                "distance": distance,
                "edge_distance": edge_distance,
                "travel_ticks": travel_ticks,
                "value": value,
                "visible": visible,
            })
    if not opportunities:
        VIRUS_TARGET_KEY = None
        return None
    # A visible locked target wins ties and score fluctuations outright. If it
    # disappeared far away, visible alternatives may supersede stale memory.
    opportunities.sort(
        key=lambda item: (
            item["key"] == VIRUS_TARGET_KEY and item["visible"],
            item["visible"],
            item["value"],
        ),
        reverse=True,
    )

    for item in opportunities:
        own_pos = cache["own_locs"][item["own_index"]]
        own_radius = float(cache["own_rads"][item["own_index"]])
        direction = unit(item["position"] - own_pos)
        if direction is None:
            continue

        if (
            not item["visible"]
            and item["distance"] <= own_radius + item["radius"] + 0.5
        ):
            VIRUS_MEMORY.pop(item["key"], None)
            if item["key"] == VIRUS_TARGET_KEY:
                VIRUS_TARGET_KEY = None
            continue

        if cache["own_count"] >= P["MAX_BLOBS"]:
            if movement_safe_to_virus(
                cache, direction, item["position"], own_radius
            ):
                VIRUS_TARGET_KEY = item["key"]
                return float(direction[0]), float(direction[1]), False
            continue

        expected_piece_radius = cache["radius"] * P["VIRUS_PIECE_RADIUS_MULT"]
        if not virus_action_safe(cache, item["position"], expected_piece_radius):
            continue

        # Manual splitting is worthwhile only when it reaches a visible virus
        # materially sooner and the launched half remains large enough to pop.
        can_split = (
            item["visible"]
            and cache["own_count"] * 2 <= P["MAX_BLOBS"]
            and own_radius >= P["SPLIT_MIN_RADIUS"]
        )
        split_radius = own_radius / SQRT2
        split_reach = (
            own_radius
            * P["SPLIT_TRAVEL_MULT"]
            * P["SPLIT_REACH_RELIABILITY"]
            + split_radius
            + item["radius"]
        )
        direct_ticks = item["travel_ticks"]
        manual_utility = (
            P["VIRUS_MANUAL_SPLIT_TIME_BONUS"] * max(0.0, direct_ticks - 1.0)
            - P["VIRUS_MANUAL_SPLIT_RISK_COST"] * cache["own_count"]
        )
        if (
            can_split
            and split_radius > item["radius"] * P["VIRUS_EAT_RATIO"]
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
    return tuple(np.round(position, 1))


def ranked_food(cache):
    foods = cache["food_locs"]
    if len(foods) == 0:
        return []
    distances = np.linalg.norm(foods - cache["position"], axis=1)
    pairwise = np.linalg.norm(foods[:, None, :] - foods[None, :, :], axis=2)
    density = np.sum(pairwise <= P["FOOD_CLUSTER_RADIUS"], axis=1)

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
    ranked = ranked_food(cache)
    if ranked:
        selected = ranked[0]
        if FOOD_TARGET is not None and FOOD_TARGET_TICKS < P["FOOD_LOCK_TICKS"]:
            locked = next((item for item in ranked if item[2] == FOOD_TARGET), None)
            if locked is not None and locked[0] >= selected[0] * 0.60:
                selected = locked

        _, target, key = selected
        if np.linalg.norm(target - cache["position"]) <= P["FOOD_REACHED_DISTANCE"]:
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
    global LAST_DIRECTION
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
        return float(direction[0]), float(direction[1]), bool(should_split), cache

    # Food is designed to return a stable roam even without visible food.
    direction = unit(LAST_DIRECTION)
    return float(direction[0]), float(direction[1]), False, cache


def main():
    #load_tuning_config()
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
                    game.send_move(MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=1.0, y=0.0),
                        split=False,
                    ))
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()