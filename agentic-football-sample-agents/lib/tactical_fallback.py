"""Phase-aware rule-based fallback for the diamond team.

The sample teams all share one set of position fallbacks, so an aggressive team
whose LLM fails plays balanced football without anything looking wrong. This
module keys off the same PhaseView the prompts are written against, so when the
model drops out the shape and the phase behaviour survive.

Every function returns exactly one command, matching the response contract the
prompts ask the model for.
"""

from __future__ import annotations

from phase import COUNTER, DEFEND, LOOSE, POSSESS, RESTART, PhaseView, attack_dir, side_of
from state import _is_my_team, _player_idx, dist
from tactical_state import dist_to_opp_goal, dist_to_own_goal

HALF_LENGTH = 55.0

# Every tunable number below comes from calibration.py. The shot gates use true
# distance to the goal centre plus a lateral gate, which is what throws out the
# wide-angle shots an x-axis "distance" would have allowed.
from calibration import (  # noqa: E402
    DEF_PRESS_INTENSITY,
    DEF_PRESS_RADIUS,
    FWD_PRESS_MIN_PROGRESS,
    FWD_SHOOT_DIST,
    FWD_SHOOT_LATERAL,
    GK_LOOSE_BALL_RADIUS,
    GK_SWEEP_RADIUS,
    MID_PRESS_INTENSITY,
    MID_SHOOT_DIST,
    MID_SHOOT_LATERAL,
)

GK_ID, DEF_ID, ML_ID, MR_ID, FWD_ID = 0, 1, 2, 3, 4


# ── geometry helpers ─────────────────────────────────────────────────────────

def anchor(team_id: int, prog_frac: float, lateral: float) -> tuple[float, float]:
    """Team-relative position → absolute (x, y).

    prog_frac: -1.0 is our own goal line, +1.0 is the opponent goal line.
    lateral:   negative is our LEFT channel, positive our RIGHT, facing forward.
    """
    d = attack_dir(team_id)
    return d * prog_frac * HALF_LENGTH, d * lateral


def _move(team_id: int, prog_frac: float, lateral: float, sprint: bool = False) -> dict:
    x, y = anchor(team_id, prog_frac, lateral)
    return {
        "commandType": "MOVE_TO",
        "parameters": {"target_x": round(x, 1), "target_y": round(y, 1), "sprint": sprint},
        "duration": 0,
    }


def _cmd(command_type: str, params: dict, duration: int = 0) -> dict:
    return {"commandType": command_type, "parameters": params, "duration": duration}


def _teammates(game_state: dict, team_id: int) -> dict[int, dict]:
    return {
        _player_idx(p): p
        for p in game_state.get("players", [])
        if _is_my_team(p, team_id)
    }


def _opponents(game_state: dict, team_id: int) -> dict[int, dict]:
    return {
        _player_idx(p): p
        for p in game_state.get("players", [])
        if not _is_my_team(p, team_id)
    }


def _ball_lateral(game_state: dict, team_id: int) -> float:
    """Ball's y in team-relative terms — negative is our LEFT."""
    y = game_state.get("ball", {}).get("position", {}).get("y", 0)
    return attack_dir(team_id) * y


def _most_advanced(mates: dict[int, dict], team_id: int, exclude: set[int]) -> int | None:
    candidates = {pid: p for pid, p in mates.items() if pid not in exclude and pid != GK_ID}
    if not candidates:
        return None
    return min(candidates, key=lambda pid: dist_to_opp_goal(candidates[pid].get("position", {}), team_id))


def _most_dangerous_opponent(game_state: dict, team_id: int) -> int:
    opps = _opponents(game_state, team_id)
    if not opps:
        return 0
    return min(opps, key=lambda pid: dist_to_own_goal(opps[pid].get("position", {}), team_id))


def _is_pusher(my_side: int, game_state: dict, team_id: int, my_player_id: int) -> bool:
    """One mid pushes, the other holds as the pivot — never both at once.

    The mid on the ball's side pushes. With the ball central, the mid nearer to
    it pushes. This is the adjustment to "both mids flank": with four outfield
    players, two mids leaving the middle at the same time puts one defender
    alone against the counter.
    """
    lateral = _ball_lateral(game_state, team_id)
    if abs(lateral) > 6:
        return (lateral < 0) == (my_side < 0)
    mates = _teammates(game_state, team_id)
    ball_pos = game_state.get("ball", {}).get("position", {})
    other = MR_ID if my_player_id == ML_ID else ML_ID
    if my_player_id not in mates or other not in mates:
        return True
    return dist(mates[my_player_id].get("position", {}), ball_pos) <= dist(
        mates[other].get("position", {}), ball_pos
    )


def _shoot_or(team_id: int, pos: dict, max_dist: float, max_lateral: float, otherwise: dict) -> dict:
    if dist_to_opp_goal(pos, team_id) <= max_dist and abs(pos.get("y", 0)) <= max_lateral:
        aim = "TR" if attack_dir(team_id) * pos.get("y", 0) < 0 else "BL"
        return _cmd("SHOOT", {"aim_location": aim, "power": 0.85})
    return otherwise


# ── per-role fallbacks ───────────────────────────────────────────────────────

def gk_fallback(game_state: dict, team_id: int, my_player_id: int, view: PhaseView) -> dict:
    mates = _teammates(game_state, team_id)
    ball_pos = game_state.get("ball", {}).get("position", {"x": 0, "y": 0})

    if view.i_have_ball:
        if view.phase == COUNTER:
            target = _most_advanced(mates, team_id, exclude={my_player_id})
            return _cmd("GK_DISTRIBUTE", {"target_player_id": target if target is not None else FWD_ID,
                                          "method": "KICK"})
        return _cmd("GK_DISTRIBUTE", {"target_player_id": DEF_ID, "method": "THROW"})

    # Sweeper duty. With a single defender the space behind the line is the
    # standing risk of this formation, so the keeper covers it rather than
    # holding the goal line unconditionally.
    if view.opp_carrier_id is not None and DEF_ID in mates:
        carrier = _opponents(game_state, team_id).get(view.opp_carrier_id, {})
        carrier_pos = carrier.get("position", {})
        beat_defender = dist_to_own_goal(carrier_pos, team_id) < dist_to_own_goal(
            mates[DEF_ID].get("position", {}), team_id
        )
        if beat_defender and dist_to_own_goal(carrier_pos, team_id) < GK_SWEEP_RADIUS:
            return _cmd("INTERCEPT", {"aggressive": True}, duration=2)

    if view.phase == LOOSE and dist(mates.get(my_player_id, {}).get("position", {}), ball_pos) < GK_LOOSE_BALL_RADIUS:
        return _cmd("INTERCEPT", {"aggressive": False}, duration=2)

    if view.phase in (POSSESS, COUNTER):
        return _move(team_id, -0.75, 0.0)

    track = max(-8.0, min(8.0, _ball_lateral(game_state, team_id) * 0.25))
    return _move(team_id, -0.93, track)


def def_fallback(game_state: dict, team_id: int, my_player_id: int, view: PhaseView) -> dict:
    mates = _teammates(game_state, team_id)
    my_pos = mates.get(my_player_id, {}).get("position", {"x": 0, "y": 0})
    lateral = _ball_lateral(game_state, team_id)

    if view.i_have_ball:
        outlet = ML_ID if lateral < 0 else MR_ID
        if view.phase == COUNTER:
            target = _most_advanced(mates, team_id, exclude={my_player_id})
            return _cmd("PASS", {"target_player_id": target if target is not None else FWD_ID,
                                 "type": "THROUGH"})
        return _cmd("PASS", {"target_player_id": outlet, "type": "GROUND"})

    if view.phase == DEFEND:
        carrier = _opponents(game_state, team_id).get(view.opp_carrier_id, {}) if view.opp_carrier_id is not None else {}
        if carrier and dist(my_pos, carrier.get("position", {})) < DEF_PRESS_RADIUS:
            return _cmd("PRESS_BALL", {"intensity": DEF_PRESS_INTENSITY}, duration=3)
        return _cmd("MARK", {"target_player_id": _most_dangerous_opponent(game_state, team_id),
                             "tightness": "TIGHT"}, duration=4)

    if view.phase == LOOSE:
        if view.i_am_nearest_to_ball:
            return _cmd("INTERCEPT", {"aggressive": True}, duration=2)
        return _move(team_id, -0.50, lateral * 0.3)

    if view.phase == COUNTER:
        return _move(team_id, -0.30, 0.0, sprint=True)

    if view.phase == RESTART:
        return _move(team_id, -0.60, 0.0)

    return _move(team_id, -0.45, lateral * 0.3)


def mid_fallback(game_state: dict, team_id: int, my_player_id: int, view: PhaseView) -> dict:
    my_side = -1 if my_player_id == ML_ID else 1
    pusher = _is_pusher(my_side, game_state, team_id, my_player_id)
    mates = _teammates(game_state, team_id)
    my_pos = mates.get(my_player_id, {}).get("position", {"x": 0, "y": 0})

    if view.i_have_ball:
        forward = _most_advanced(mates, team_id, exclude={my_player_id})
        pass_cmd = _cmd("PASS", {"target_player_id": forward if forward is not None else FWD_ID,
                                 "type": "THROUGH" if view.phase == COUNTER else "GROUND"})
        return _shoot_or(team_id, my_pos, MID_SHOOT_DIST, MID_SHOOT_LATERAL, pass_cmd)

    if view.phase == DEFEND:
        if pusher:
            return _cmd("PRESS_BALL", {"intensity": MID_PRESS_INTENSITY}, duration=3)
        return _move(team_id, -0.28, my_side * 6.0)

    if view.phase == LOOSE:
        if view.i_am_nearest_to_ball:
            return _cmd("INTERCEPT", {"aggressive": True}, duration=2)
        return _move(team_id, -0.15, my_side * 8.0)

    if view.phase == COUNTER:
        if pusher:
            return _move(team_id, 0.55, my_side * 20.0, sprint=True)
        return _move(team_id, 0.15, my_side * 6.0, sprint=True)

    if view.phase == RESTART:
        return _move(team_id, 0.0, my_side * 12.0)

    # POSSESS — hold the shape but keep offering an angle rather than standing
    # still; a static diamond gives the ball carrier nothing to pass to.
    if pusher:
        return _move(team_id, 0.20, my_side * 16.0)
    return _move(team_id, 0.05, my_side * 8.0)


def fwd_fallback(game_state: dict, team_id: int, my_player_id: int, view: PhaseView) -> dict:
    mates = _teammates(game_state, team_id)
    my_pos = mates.get(my_player_id, {}).get("position", {"x": 0, "y": 0})
    lateral = _ball_lateral(game_state, team_id)
    away_from_ball = -12.0 if lateral > 0 else 12.0  # stretch the far channel

    if view.i_have_ball:
        advance = _move(team_id, 0.80, my_pos.get("y", 0) * 0.5 * attack_dir(team_id), sprint=True)
        return _shoot_or(team_id, my_pos, FWD_SHOOT_DIST, FWD_SHOOT_LATERAL, advance)

    if view.phase == DEFEND:
        # Press only where it pays: when they are building from deep. Chasing
        # back is the mids' job in this shape.
        if view.opp_carrier_progress is not None and view.opp_carrier_progress > FWD_PRESS_MIN_PROGRESS:
            return _cmd("PRESS_BALL", {"intensity": 0.7}, duration=3)
        return _move(team_id, 0.15, away_from_ball * 0.5)

    if view.phase == LOOSE:
        if view.i_am_nearest_to_ball:
            return _cmd("INTERCEPT", {"aggressive": True}, duration=2)
        return _move(team_id, 0.35, 0.0)

    if view.phase == COUNTER:
        return _move(team_id, 0.75, away_from_ball, sprint=True)

    if view.phase == RESTART:
        return _move(team_id, 0.25, 0.0)

    return _move(team_id, 0.55, away_from_ball * 0.8)


ROLE_FALLBACKS = {
    "GK": gk_fallback,
    "DEF": def_fallback,
    "ML": mid_fallback,
    "MR": mid_fallback,
    "FWD": fwd_fallback,
}

LAST_RESORT = {
    "GK": _cmd("SET_STANCE", {"stance": 2}),
    "DEF": _cmd("MARK", {"target_player_id": 0, "tightness": "TIGHT"}, duration=4),
    "ML": _cmd("PRESS_BALL", {"intensity": 0.6}, duration=3),
    "MR": _cmd("PRESS_BALL", {"intensity": 0.6}, duration=3),
    "FWD": _cmd("PRESS_BALL", {"intensity": 0.6}, duration=3),
}


def build_tactical_fallback(role: str):
    """Return fallback(game_state, team_id, my_player_id, view) -> list[dict]."""
    fn = ROLE_FALLBACKS[role]

    def fallback(game_state: dict, team_id: int, my_player_id: int, view: PhaseView) -> list[dict]:
        cmd = dict(fn(game_state, team_id, my_player_id, view))
        cmd["playerId"] = my_player_id
        cmd["teamId"] = team_id
        return [cmd]

    return fallback
