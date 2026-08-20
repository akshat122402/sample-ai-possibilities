"""Tactical geometry, computed in the harness instead of called as tools.

The gateway team exposes these four calculations as Lambda tools behind an
AgentCore Gateway. Each use costs two extra model turns plus a Gateway->Lambda
hop, which alone blows a sub-second tick budget. The math is deterministic and
needs nothing the payload doesn't already carry, so the diamond team computes
it here and prints the results into the state summary — the model reads the
answers instead of asking for them.

The formulas match the gateway tools (calculate_pass_options, evaluate_shot,
find_open_space, get_defensive_assignment) so the two teams stay comparable.

tactical_hints() is the entry point: it picks which calculations matter this
tick from the phase and the role, so the summary never grows more than a few
lines.
"""

from __future__ import annotations

import math

from calibration import (
    GK_BEHIND_DEF_MARGIN,
    GK_DEPTH_ATTACK,
    GK_DEPTH_DEFEND,
    GK_MAX_FROM_GOAL,
    PASS_LANE_RADIUS,
)
from phase import COUNTER, DEFEND, POSSESS, PhaseView, attack_dir, progress
from state import _is_my_team, _player_idx, dist, get_goal_positions
from tactical_state import ROLE_NAMES, goal_centre

GOAL_HALF_WIDTH = 5.0


# ── pass options ─────────────────────────────────────────────────────────────

def _interception_risk(passer: dict, receiver: dict, opponents: list) -> float:
    """Probability (0-1) an opponent near the pass lane cuts the ball out."""
    pass_dist = dist(passer, receiver)
    if pass_dist < 1:
        return 0.0
    dx = receiver.get("x", 0) - passer.get("x", 0)
    dy = receiver.get("y", 0) - passer.get("y", 0)
    risk = 0.0
    for opp in opponents:
        ox = opp.get("x", 0) - passer.get("x", 0)
        oy = opp.get("y", 0) - passer.get("y", 0)
        t = max(0.0, min(1.0, (ox * dx + oy * dy) / (pass_dist ** 2)))
        lane = {"x": passer.get("x", 0) + t * dx, "y": passer.get("y", 0) + t * dy}
        d = dist(opp, lane)
        if d < PASS_LANE_RADIUS:
            risk = max(risk, 1.0 - (d / PASS_LANE_RADIUS))
    return min(risk, 0.95)


def pass_options(my_pos: dict, teammates: list, opponents: list) -> list:
    """Success odds for a pass to each teammate, best first.

    teammates/opponents are player dicts; the caller excludes the passer.
    """
    opp_positions = [p.get("position", {}) for p in opponents]
    options = []
    for tm in teammates:
        pos = tm.get("position", {})
        d = dist(my_pos, pos)
        risk = _interception_risk(my_pos, pos, opp_positions)
        options.append({
            "player_id": _player_idx(tm),
            "dist": round(d, 1),
            "risk": round(risk, 2),
            "success": round(max(0.05, 1.0 - risk - (d / 120.0)), 2),
        })
    options.sort(key=lambda o: o["success"], reverse=True)
    return options


# ── shot quality ─────────────────────────────────────────────────────────────

def shot_quality(shooter: dict, gk_pos: dict, blockers: list, team_id: int) -> dict:
    goal = goal_centre(team_id, opponent=True)
    dist_to_goal = dist(shooter, goal)
    dist_gk_to_goal = dist(gk_pos, goal)

    angle_to_goal = math.atan2(GOAL_HALF_WIDTH, max(dist_to_goal, 0.1))
    angle_factor = min(1.0, angle_to_goal / 0.15)
    distance_factor = max(0.0, 1.0 - (dist_to_goal / 55.0))
    gk_factor = min(1.0, abs(gk_pos.get("y", 0)) / 8.0) * 0.3
    gk_dist_factor = min(1.0, dist_gk_to_goal / 15.0) * 0.2

    blocker_penalty = 0.0
    for b in blockers:
        b_dist = dist(shooter, b)
        if b_dist < 10:
            blocker_penalty += (10 - b_dist) / 10.0 * 0.15
    blocker_penalty = min(blocker_penalty, 0.4)

    probability = round(
        max(0.02, min(0.95,
            (distance_factor * 0.45) + (angle_factor * 0.25) + gk_factor + gk_dist_factor - blocker_penalty
        )), 2
    )

    # Aim away from the keeper. T/B/L/R are from the shooter's viewpoint.
    if gk_pos.get("y", 0) > 1:
        aim = "BL" if shooter.get("y", 0) > 0 else "BR"
    elif gk_pos.get("y", 0) < -1:
        aim = "TL" if shooter.get("y", 0) > 0 else "TR"
    else:
        aim = "TR" if shooter.get("y", 0) <= 0 else "TL"

    return {
        "probability": probability,
        "dist": round(dist_to_goal, 1),
        "aim": aim,
        "power": round(min(1.0, 0.6 + (dist_to_goal / 80.0)), 2),
    }


# ── open space ───────────────────────────────────────────────────────────────

def best_open_space(opponents: list, my_pos: dict, zone: str, team_id: int,
                    y_range: tuple = (-30, 30)) -> dict:
    """Grid-search the zone for the point clearest of opponents but reachable.

    y_range confines the search to the caller's lane or pocket, so the hint can
    never contradict a role's positioning rules.
    """
    if team_id == 0:
        zones = {"defense": (-55, -15), "midfield": (-15, 15), "attack": (15, 52)}
    else:
        zones = {"defense": (15, 55), "midfield": (-15, 15), "attack": (-52, -15)}
    x_min, x_max = zones.get(zone, zones["midfield"])
    y_min, y_max = y_range

    opp_positions = [p.get("position", {}) for p in opponents]
    best, best_score = None, -1e9
    x = x_min
    while x <= x_max:
        y = y_min
        while y <= y_max:
            point = {"x": x, "y": y}
            clearance = min((dist(point, o) for o in opp_positions), default=999.0)
            score = clearance - dist(point, my_pos) * 0.15
            if score > best_score:
                best_score = score
                best = point
            y += 5
        x += 5

    clearance = min((dist(best, o) for o in opp_positions), default=999.0)
    return {"x": round(best["x"], 1), "y": round(best["y"], 1), "clearance": round(clearance, 1)}


# ── defensive assignment ─────────────────────────────────────────────────────

def mark_targets(opponents: list, ball_pos: dict, my_pos: dict, team_id: int,
                 carrier_id) -> list:
    """Opponents ranked by threat to our goal, most dangerous first."""
    my_goal = goal_centre(team_id, opponent=False)
    threats = []
    for opp in opponents:
        pos = opp.get("position", {})
        pid = _player_idx(opp)
        d_goal = dist(pos, my_goal)
        d_ball = dist(pos, ball_pos)
        score = max(0.0, 1.0 - (d_goal / 80.0))
        score += 0.3 if d_ball < 10 else (0.15 if d_ball < 20 else 0.0)
        has_ball = pid == carrier_id
        if has_ball:
            score += 0.4
        threats.append({
            "player_id": pid,
            "threat": round(score, 2),
            "dist_to_me": round(dist(pos, my_pos), 1),
            "has_ball": has_ball,
            "tightness": "TIGHT" if score > 0.7 else "LOOSE",
        })
    threats.sort(key=lambda t: t["threat"], reverse=True)
    return threats


# ── goalkeeper line target ───────────────────────────────────────────────────

def gk_line_target(ball_pos: dict, team_id: int, def_pos, we_have_ball: bool) -> dict:
    """Where the GK should stand: on the goal-centre→ball line, at a depth set
    by possession, capped, and always GK_BEHIND_DEF_MARGIN closer to goal than
    DEF. Computed here so the model never does per-tick trigonometry."""
    my_goal = goal_centre(team_id, opponent=False)
    d_gb = max(dist(my_goal, ball_pos), 0.1)
    depth = GK_DEPTH_ATTACK if we_have_ball else GK_DEPTH_DEFEND
    if def_pos is not None:
        depth = min(depth, max(2.0, dist(my_goal, def_pos) - GK_BEHIND_DEF_MARGIN))
    depth = min(depth, GK_MAX_FROM_GOAL, d_gb)
    return {
        "x": round(my_goal["x"] + depth * (ball_pos.get("x", 0) - my_goal["x"]) / d_gb, 1),
        "y": round(depth * ball_pos.get("y", 0) / d_gb, 1),
        "depth": round(depth, 1),
    }


# ── the per-tick hint block ──────────────────────────────────────────────────

def _space_y_range(role: str, my_pos: dict) -> tuple:
    """Where a role may be sent for space, mirroring its prompt's rules:
    mids stay in their lane (their current side of y=0), FWD holds the central
    pocket, DEF stays out of the corners."""
    if role in ("ML", "MR"):
        return (2, 30) if my_pos.get("y", 0) >= 0 else (-30, -2)
    if role == "FWD":
        return (-14, 14)
    return (-18, 18)  # DEF: the box's side walls


# Which zone a role offers into when its team has the ball.
_SPACE_ZONE = {
    "DEF": {POSSESS: "midfield", COUNTER: "midfield"},
    "ML": {POSSESS: "midfield", COUNTER: "attack"},
    "MR": {POSSESS: "midfield", COUNTER: "attack"},
    "FWD": {POSSESS: "attack", COUNTER: "attack"},
}


def tactical_hints(game_state: dict, team_id: int, my_player_id: int,
                   role: str, view: PhaseView) -> list:
    """Lines for the state summary. Empty when nothing computed applies."""
    players = game_state.get("players", [])
    ball_pos = game_state.get("ball", {}).get("position", {"x": 0, "y": 0})
    mine = [p for p in players if _is_my_team(p, team_id)]
    opponents = [p for p in players if not _is_my_team(p, team_id)]
    me = next((p for p in mine if _player_idx(p) == my_player_id), None)
    if me is None:
        return []
    my_pos = me.get("position", {"x": 0, "y": 0})

    lines = []

    if view.i_have_ball:
        teammates = [p for p in mine if _player_idx(p) != my_player_id]
        opts = pass_options(my_pos, teammates, opponents)[:3]
        if opts:
            lines.append("Pass: " + " | ".join(
                f"{ROLE_NAMES.get(o['player_id'], 'P%d' % o['player_id'])}(id{o['player_id']}) "
                f"{o['success']:.0%} dist {o['dist']}" for o in opts
            ))
        if role != "GK":
            opp_gk = next((p for p in opponents if _player_idx(p) == 0), None)
            gk_pos = opp_gk.get("position", goal_centre(team_id)) if opp_gk else goal_centre(team_id)
            blockers = [p.get("position", {}) for p in opponents if _player_idx(p) != 0]
            shot = shot_quality(my_pos, gk_pos, blockers, team_id)
            verdict = "take it" if shot["probability"] > 0.25 else "poor — pass instead"
            lines.append(
                f"Shot: {shot['probability']:.0%} from {shot['dist']} out — {verdict}"
                + (f" (aim {shot['aim']}, power {shot['power']})" if shot["probability"] > 0.25 else "")
            )
    elif view.we_have_ball and role in _SPACE_ZONE:
        zone = _SPACE_ZONE[role].get(view.phase)
        if zone:
            space = best_open_space(opponents, my_pos, zone, team_id,
                                    y_range=_space_y_range(role, my_pos))
            lines.append(
                f"Space: ({space['x']},{space['y']}) is the clearest {zone} point, "
                f"{space['clearance']} from the nearest opponent"
            )

    if view.phase == DEFEND and role in ("DEF", "ML", "MR"):
        threats = mark_targets(opponents, ball_pos, my_pos, team_id, view.opp_carrier_id)[:2]
        lines.append("Mark: " + " | ".join(
            f"P{t['player_id']} threat {t['threat']}{' (has ball)' if t['has_ball'] else ''} "
            f"{t['tightness']}, {t['dist_to_me']} from you" for t in threats
        ))

    if role == "GK" and not view.i_have_ball:
        defender = next((p for p in mine if _player_idx(p) == 1), None)
        spot = gk_line_target(ball_pos, team_id,
                              defender.get("position") if defender else None,
                              view.we_have_ball)
        lines.append(
            f"GK line: ({spot['x']},{spot['y']}) — on the ball-goal line at depth {spot['depth']}"
        )

    return lines
