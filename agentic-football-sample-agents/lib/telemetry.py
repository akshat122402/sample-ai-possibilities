"""Structured decision/outcome logging for offline calibration.

The constants in calibration.py are guesses. This module produces the data that
replaces them with measurements, and it does so without adding any runtime
dependency: it writes JSON lines through the agent's existing logger, which the
AgentCore runtime already ships to CloudWatch (`observability: enabled: true` in
every runtime template). There is no store to read from during a match — the
learning happens between matches, offline.

Two record kinds:

  kinematics — one per tick per agent. Player speed, sprint flag, stamina and
               its delta, ball speed and its deceleration. This is the raw
               physics: how fast a player moves, what sprinting costs, how
               quickly the ball slows down.

  decision   — what the agent chose, with the features that should predict
               whether it works (distance to goal, lateral offset, nearest
               opponent, stamina, phase).

  outcome    — how a tracked decision resolved: a shot scored or was lost, a
               pass was completed or intercepted, a press won the ball or did
               not. Paired to its decision by `id`.

Aggregate with CloudWatch Logs Insights:

    fields @message
    | filter @message like /TELEMETRY/
    | parse @message 'TELEMETRY *' as body
    | filter body like /"kind":"outcome"/

Everything here is best-effort. Telemetry must never be able to break a match,
so every entry point swallows its own exceptions.
"""

from __future__ import annotations

import json
import os

from calibration import DEFAULT_RESOLVE_WINDOW, RESOLVE_WINDOW
from phase import resolve_holder
from state import _is_my_team, _player_idx, dist
from tactical_state import dist_to_opp_goal

SCHEMA = 1
PREFIX = "TELEMETRY"

# Commands whose success or failure is observable from the tick stream. MOVE_TO
# is deliberately absent: the kinematics stream already measures movement, and
# tracking one per tick would drown the useful records.
TRACKED = {"SHOOT", "PASS", "GK_DISTRIBUTE", "INTERCEPT", "SLIDE_TACKLE", "PRESS_BALL"}

MAX_PENDING = 8

_state: dict[tuple[int, int], dict] = {}
_seq = [0]


def enabled() -> bool:
    return os.environ.get("TELEMETRY_ENABLED", "1") not in ("0", "false", "False")


def reset() -> None:
    """Clear tracker state — used by tests."""
    _state.clear()
    _seq[0] = 0


def _emit(log, kind: str, fields: dict) -> None:
    if not enabled():
        return
    record = {"tel": SCHEMA, "kind": kind}
    record.update(fields)
    log.info(f"{PREFIX} {json.dumps(record, separators=(',', ':'), default=str)}")


def _next_id() -> str:
    _seq[0] += 1
    return f"d{_seq[0]}"


def _snapshot(game_state: dict, team_id: int, player_id: int) -> dict:
    players = game_state.get("players", [])
    ball = game_state.get("ball", {})
    ball_pos = ball.get("position", {"x": 0, "y": 0})
    ball_vel = ball.get("velocity", {"x": 0, "y": 0})
    score = game_state.get("score", {})
    me = next(
        (p for p in players if _is_my_team(p, team_id) and _player_idx(p) == player_id), {}
    )
    _, holder = resolve_holder(ball, players)
    return {
        "t": float(game_state.get("gameTime", 0) or 0),
        "myPos": dict(me.get("position", {"x": 0, "y": 0})),
        "speed": float(me.get("speed", 0) or 0),
        "sprint": bool(me.get("isSprinting", False)),
        "stam": float(me.get("stamina", 1.0) or 0),
        "ballPos": dict(ball_pos),
        "ballSpeed": (ball_vel.get("x", 0) ** 2 + ball_vel.get("y", 0) ** 2) ** 0.5,
        "holder": _player_idx(holder) if holder else None,
        "holderMine": bool(holder is not None and _is_my_team(holder, team_id)),
        "us": score.get("home", 0) if team_id == 0 else score.get("away", 0),
        "them": score.get("away", 0) if team_id == 0 else score.get("home", 0),
    }


def _nearest_opponent(game_state: dict, team_id: int, my_pos: dict) -> float:
    opps = [p for p in game_state.get("players", []) if not _is_my_team(p, team_id)]
    if not opps:
        return -1.0
    return min(dist(my_pos, p.get("position", {})) for p in opps)


# ── resolution ──────────────────────────────────────────────────────────────

def _resolve(pending: dict, snap: dict, player_id: int) -> str | None:
    """Return an outcome label, or None if the decision is still open."""
    cmd = pending["cmd"]
    at = pending["snap"]
    expired = snap["t"] - pending["t"] >= pending["window"]

    if cmd == "SHOOT":
        if snap["us"] > at["us"]:
            return "goal"
        if snap["holder"] is not None and not snap["holderMine"]:
            return "lost"
        if snap["holderMine"] and snap["holder"] != player_id:
            return "rebound_retained"
        return "no_goal" if expired else None

    if cmd in ("PASS", "GK_DISTRIBUTE"):
        if snap["holderMine"] and snap["holder"] != player_id:
            return "completed"
        if snap["holder"] is not None and not snap["holderMine"]:
            return "intercepted"
        return "unresolved" if expired else None

    if cmd in ("INTERCEPT", "PRESS_BALL", "SLIDE_TACKLE"):
        if snap["holderMine"]:
            return "won_ball_self" if snap["holder"] == player_id else "won_ball_team"
        return "not_won" if expired else None

    return "unresolved" if expired else None


# ── public API ──────────────────────────────────────────────────────────────

def observe(log, game_state: dict, team_id: int, player_id: int, role: str, view) -> None:
    """Called once per tick, before the decision. Emits kinematics and resolves
    anything still pending from earlier ticks."""
    if not enabled():
        return
    try:
        key = (team_id, player_id)
        entry = _state.setdefault(key, {"prev": None, "pending": []})
        snap = _snapshot(game_state, team_id, player_id)
        prev = entry["prev"]

        if prev is not None:
            dt = snap["t"] - prev["t"]
            if dt > 0:
                _emit(log, "kinematics", {
                    "role": role, "player": player_id, "team": team_id,
                    "t": snap["t"], "dt": round(dt, 3), "phase": view.phase,
                    "speed": round(snap["speed"], 3),
                    "obsSpeed": round(dist(snap["myPos"], prev["myPos"]) / dt, 3),
                    "sprint": snap["sprint"],
                    "stam": round(snap["stam"], 4),
                    "stamRate": round((snap["stam"] - prev["stam"]) / dt, 5),
                    "ballSpeed": round(snap["ballSpeed"], 3),
                    "ballAccel": round((snap["ballSpeed"] - prev["ballSpeed"]) / dt, 3),
                    "oppBeaten": view.opponents_beaten,
                })

        still_open = []
        for pending in entry["pending"]:
            label = _resolve(pending, snap, player_id)
            if label is None:
                still_open.append(pending)
                continue
            _emit(log, "outcome", {
                "id": pending["id"], "role": role, "player": player_id, "team": team_id,
                "cmd": pending["cmd"], "phase": pending["phase"],
                "result": label,
                "elapsed": round(snap["t"] - pending["t"], 3),
                **pending["features"],
            })
        entry["pending"] = still_open
        entry["prev"] = snap
    except Exception as e:  # never let telemetry break a tick
        log.warn(f"telemetry observe failed: {e}")


def record_decision(log, game_state, team_id, player_id, role, view, command, source) -> None:
    """Called once per tick, after the command is chosen."""
    if not enabled():
        return
    try:
        key = (team_id, player_id)
        entry = _state.setdefault(key, {"prev": None, "pending": []})
        snap = entry["prev"] or _snapshot(game_state, team_id, player_id)
        cmd = command.get("commandType", "?")

        features = {
            "distGoal": round(dist_to_opp_goal(snap["myPos"], team_id), 2),
            "lateral": round(abs(snap["myPos"].get("y", 0)), 2),
            "nearestOpp": round(_nearest_opponent(game_state, team_id, snap["myPos"]), 2),
            "stam": round(snap["stam"], 4),
            "distBall": round(dist(snap["myPos"], snap["ballPos"]), 2),
            "hasBall": view.i_have_ball,
        }
        decision_id = _next_id()
        _emit(log, "decision", {
            "id": decision_id, "role": role, "player": player_id, "team": team_id,
            "t": snap["t"], "phase": view.phase, "source": source, "cmd": cmd,
            "params": command.get("parameters", {}),
            **features,
        })

        # Only open a pending outcome for commands whose result is observable,
        # and never a second one for a command type already in flight — a
        # maintained PRESS_BALL reissued every tick should be one record.
        if cmd in TRACKED and len(entry["pending"]) < MAX_PENDING:
            if not any(p["cmd"] == cmd for p in entry["pending"]):
                entry["pending"].append({
                    "id": decision_id, "cmd": cmd, "t": snap["t"], "phase": view.phase,
                    "window": RESOLVE_WINDOW.get(cmd, DEFAULT_RESOLVE_WINDOW),
                    "snap": snap, "features": features,
                })
    except Exception as e:
        log.warn(f"telemetry record_decision failed: {e}")
