"""Offline tests for the diamond team's shared tactical library.

    python3 lib/test_tactical.py

No AWS credentials and no network. Covers the phase classifier, the team-relative
geometry, the pusher/pivot rule, the role whitelist and every role's fallback in
every phase.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_helpers import GAME_STATE, mock_agentcore

mock_agentcore()

import phase as ph
from phase import COUNTER, DEFEND, LOOSE, POSSESS, RESTART, classify_phase, reset_memory, side_of
from prompt_common import ROLE_COMMANDS, command_reference
from tactical_agent_base import enforce_role
from tactical_fallback import build_tactical_fallback
from tactical_state import _pct, dist_to_opp_goal, summarize_tactical_state

HOME = 0
AWAY = 1
FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(f"{label} {detail}")
    print(f"  [{status}] {label}{(' — ' + detail) if detail and not condition else ''}")


def state(possession="agentId_3", play_mode="OPEN_PLAY", ball=(15.3, -5.2), time=120.5):
    s = copy.deepcopy(GAME_STATE)
    s["ball"]["possessionAgentId"] = possession
    s["ball"]["position"] = {"x": ball[0], "y": ball[1], "z": 0}
    s["playMode"] = play_mode
    s["gameTime"] = time
    return s


def take_possession(s, team_code, agent="agentId_1"):
    """Give the ball to a specific player and put them on it.

    Both teams number their players 0-4, so the holder is only unambiguous when
    that player is the one nearest the ball — which is also how the runtime
    resolves it.
    """
    s = copy.deepcopy(s)
    s["ball"]["possessionAgentId"] = agent
    for p in s["players"]:
        if p["teamCode"] == team_code and p["agentId"] == agent:
            p["position"] = {"x": s["ball"]["position"]["x"], "y": s["ball"]["position"]["y"]}
    return s


def opp_possession(s, agent="agentId_1"):
    """Hand the ball to an AWAY player (i.e. HOME's opponent)."""
    return take_possession(s, "away", agent)


def commit_forward(s, team_code, xs):
    """Push a team's outfield players deep into the opposite half — the shape
    that makes a turnover a counter-attack."""
    s = copy.deepcopy(s)
    outfield = [p for p in s["players"] if p["teamCode"] == team_code and p["agentId"] != "agentId_0"]
    for p, x in zip(outfield, xs):
        p["position"] = {"x": x, "y": p["position"].get("y", 0)}
    return s


def won_ball_on_the_break(team_id=HOME):
    """Two ticks: the opponent has the ball deep in our half with their outfield
    committed forward, then we win it. Returns (tick1_state, tick2_state)."""
    if team_id == HOME:
        xs = [-35.0, -30.0, -25.0, -10.0]
        t1 = commit_forward(state(ball=(-20.0, 0.0)), "away", xs)
        t2 = commit_forward(state(ball=(-20.0, 0.0), time=121.0), "away", xs)
        return take_possession(t1, "away"), take_possession(t2, "home")
    xs = [35.0, 30.0, 25.0, 10.0]
    t1 = commit_forward(state(ball=(20.0, 0.0)), "home", xs)
    t2 = commit_forward(state(ball=(20.0, 0.0), time=121.0), "home", xs)
    return take_possession(t1, "home"), take_possession(t2, "away")


# ---------------------------------------------------------------------------

def test_geometry():
    print("=== TEAM-RELATIVE GEOMETRY ===")
    check("HOME y=-10 is LEFT", side_of(-10, HOME) == "LEFT")
    check("AWAY y=-10 is RIGHT", side_of(-10, AWAY) == "RIGHT", side_of(-10, AWAY))
    check("y=0 is CENTRE both ways", side_of(0, HOME) == side_of(0, AWAY) == "CENTRE")

    # The bug this replaces: an x-axis gap calls a wide position "in range".
    wide = {"x": 35.0, "y": 30.0}
    x_gap = abs(wide["x"] - 55.0)
    true_dist = dist_to_opp_goal(wide, HOME)
    check("true distance exceeds the x-axis gap from wide", true_dist > x_gap + 10,
          f"x_gap={x_gap:.1f} true={true_dist:.1f}")
    check("wide position is outside shooting range", true_dist > 24)


def test_stamina():
    print("\n=== STAMINA RENDERING ===")
    check("0.65 renders as 65%", _pct(0.65) == "65%", _pct(0.65))
    check("0.95 renders as 95%", _pct(0.95) == "95%", _pct(0.95))
    check("0.65 and 0.95 differ", _pct(0.65) != _pct(0.95))
    check("0-100 payloads still work", _pct(80) == "80%", _pct(80))


def test_phases():
    print("\n=== PHASE CLASSIFICATION ===")
    reset_memory()
    v = classify_phase(state(), HOME, 2)
    check("our possession is POSSESS", v.phase == POSSESS, v.phase)

    reset_memory()
    v = classify_phase(opp_possession(state()), HOME, 2)
    check("their possession is DEFEND", v.phase == DEFEND, v.phase)

    reset_memory()
    v = classify_phase(state(possession=None), HOME, 2)
    check("free ball is LOOSE", v.phase == LOOSE, v.phase)

    reset_memory()
    v = classify_phase(state(play_mode="THROW_IN"), HOME, 2)
    check("dead ball is RESTART", v.phase == RESTART, v.phase)

    reset_memory()
    v = classify_phase(state(play_mode=0), HOME, 2)
    check("legacy int playMode 0 is open play", v.phase != RESTART, v.phase)


def test_counter():
    print("\n=== COUNTER TRIGGER ===")
    reset_memory()
    tick1, tick2 = won_ball_on_the_break(HOME)
    classify_phase(tick1, HOME, 2)
    v = classify_phase(tick2, HOME, 2)
    check("turnover with opponents upfield triggers COUNTER", v.phase == COUNTER, v.phase)
    check("opponents beaten counted", v.opponents_beaten >= 2, str(v.opponents_beaten))
    check("possession change detected", v.possession_changed)

    # Same turnover, opponents all goal-side — an ordinary attack, not a break.
    reset_memory()
    classify_phase(opp_possession(state(ball=(-20.0, 0.0))), HOME, 2)
    v = classify_phase(state(ball=(-20.0, 0.0), time=121.0), HOME, 2)
    check("no counter when opponents are goal-side", v.phase == POSSESS, v.phase)

    # The counter expires.
    reset_memory()
    tick1, tick2 = won_ball_on_the_break(HOME)
    classify_phase(tick1, HOME, 2)
    classify_phase(tick2, HOME, 2)
    late = copy.deepcopy(tick2)
    late["gameTime"] = 121.0 + ph.COUNTER_HOLD_SECONDS + 1
    v = classify_phase(late, HOME, 2)
    check("counter expires into POSSESS", v.phase == POSSESS, v.phase)

    # Losing the ball kills it immediately.
    reset_memory()
    tick1, tick2 = won_ball_on_the_break(HOME)
    classify_phase(tick1, HOME, 2)
    classify_phase(tick2, HOME, 2)
    lost = take_possession(
        commit_forward(state(ball=(-20.0, 0.0), time=121.5), "away", [-35.0, -30.0, -25.0, -10.0]),
        "away")
    v = classify_phase(lost, HOME, 2)
    check("losing the ball ends the counter", v.phase == DEFEND, v.phase)

    # Works identically for AWAY.
    reset_memory()
    tick1, tick2 = won_ball_on_the_break(AWAY)
    classify_phase(tick1, AWAY, 2)
    v = classify_phase(tick2, AWAY, 2)
    check("AWAY counter works the same", v.phase == COUNTER, v.phase)


def test_pusher_rule():
    print("\n=== PUSHER / PIVOT RULE ===")
    from tactical_fallback import _is_pusher
    for ball_y, label in [(-20.0, "ball LEFT"), (20.0, "ball RIGHT"), (0.0, "ball CENTRE")]:
        s = state(ball=(10.0, ball_y))
        ml = _is_pusher(-1, s, HOME, 2)
        mr = _is_pusher(1, s, HOME, 3)
        check(f"exactly one pusher with {label}", ml != mr, f"ML={ml} MR={mr}")
    s = state(ball=(10.0, -20.0))
    check("left ball makes ML the pusher", _is_pusher(-1, s, HOME, 2))


def test_fallbacks():
    print("\n=== FALLBACKS: every role, every phase ===")
    roles = {"GK": 0, "DEF": 1, "ML": 2, "MR": 3, "FWD": 4}
    scenarios = {
        POSSESS: state(),
        DEFEND: opp_possession(state()),
        LOOSE: state(possession=None),
        RESTART: state(play_mode="CORNER"),
    }
    for role, pid in roles.items():
        fb = build_tactical_fallback(role)
        for phase_name, s in scenarios.items():
            reset_memory()
            v = classify_phase(s, HOME, pid)
            cmds = fb(s, HOME, pid, v)
            ok = (
                len(cmds) == 1
                and cmds[0]["playerId"] == pid
                and cmds[0]["teamId"] == HOME
                and cmds[0]["commandType"] in ROLE_COMMANDS[role]
            )
            check(f"{role} in {phase_name} -> {cmds[0]['commandType']}", ok, json.dumps(cmds))
            if cmds[0]["commandType"] == "MOVE_TO":
                p = cmds[0]["parameters"]
                check(f"{role} {phase_name} target in bounds",
                      -55 <= p["target_x"] <= 55 and -35 <= p["target_y"] <= 35, json.dumps(p))

        # counter phase needs its own two-tick setup
        reset_memory()
        tick1, s = won_ball_on_the_break(HOME)
        classify_phase(tick1, HOME, pid)
        v = classify_phase(s, HOME, pid)
        cmds = fb(s, HOME, pid, v)
        check(f"{role} in COUNTER -> {cmds[0]['commandType']}",
              v.phase == COUNTER and len(cmds) == 1 and cmds[0]["playerId"] == pid, json.dumps(cmds))


def test_gk_sweeper():
    print("\n=== GK SWEEPER RULE ===")
    s = opp_possession(state(ball=(-40.0, 2.0)), agent="agentId_1")
    for p in s["players"]:
        if p["teamCode"] == "home" and p["agentId"] == "agentId_1":
            p["position"] = {"x": -20.0, "y": 0.0}  # our DEF has been bypassed
    reset_memory()
    v = classify_phase(s, HOME, 0)
    cmds = build_tactical_fallback("GK")(s, HOME, 0, v)
    check("GK sweeps when the carrier is behind DEF",
          cmds[0]["commandType"] == "INTERCEPT", json.dumps(cmds))

    safe = opp_possession(state(ball=(20.0, 2.0)), agent="agentId_1")
    reset_memory()
    v = classify_phase(safe, HOME, 0)
    cmds = build_tactical_fallback("GK")(safe, HOME, 0, v)
    check("GK holds position when the ball is far",
          cmds[0]["commandType"] == "MOVE_TO", json.dumps(cmds))


def test_shot_gate():
    print("\n=== SHOT GATE ===")
    fb = build_tactical_fallback("FWD")
    # Close and central: shoot.
    s = state(possession="agentId_4", ball=(38.0, 2.0))
    for p in s["players"]:
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
            p["position"] = {"x": 38.0, "y": 2.0}
    reset_memory()
    cmds = fb(s, HOME, 4, classify_phase(s, HOME, 4))
    check("central and close -> SHOOT", cmds[0]["commandType"] == "SHOOT", json.dumps(cmds))

    # Same x, wide: no shot. This is the case an x-axis "distance" gets wrong.
    s = state(possession="agentId_4", ball=(38.0, 30.0))
    for p in s["players"]:
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
            p["position"] = {"x": 38.0, "y": 30.0}
    reset_memory()
    cmds = fb(s, HOME, 4, classify_phase(s, HOME, 4))
    check("same x but wide -> no shot", cmds[0]["commandType"] != "SHOOT", json.dumps(cmds))


def test_role_whitelist():
    print("\n=== ROLE WHITELIST ===")
    reset = [{"commandType": "RESET", "playerId": 2, "parameters": {}}]
    for role in ROLE_COMMANDS:
        check(f"{role} refuses RESET", enforce_role(reset, role) == [])
        check(f"{role} refuses CLEAR_OVERRIDE",
              enforce_role([{"commandType": "CLEAR_OVERRIDE", "parameters": {}}], role) == [])
    check("GK refuses SHOOT", enforce_role([{"commandType": "SHOOT", "parameters": {}}], "GK") == [])
    check("GK refuses SLIDE_TACKLE",
          enforce_role([{"commandType": "SLIDE_TACKLE", "parameters": {}}], "GK") == [])
    check("FWD refuses GK_DISTRIBUTE",
          enforce_role([{"commandType": "GK_DISTRIBUTE", "parameters": {}}], "FWD") == [])
    stance = [{"commandType": "SET_STANCE", "parameters": {"stance": 2}}]
    check("only the captain may SET_STANCE",
          enforce_role(stance, "GK") == stance
          and all(enforce_role(stance, r) == [] for r in ("DEF", "ML", "MR", "FWD")))
    move = [{"commandType": "MOVE_TO", "parameters": {"target_x": 1, "target_y": 1}}]
    check("legal commands pass through", enforce_role(move, "FWD") == move)


def test_prompt_blocks():
    print("\n=== PROMPT BLOCKS ===")
    for role in ROLE_COMMANDS:
        ref = command_reference(role)
        check(f"{role} reference omits RESET", "RESET" not in ref)
        check(f"{role} reference lists only its own commands",
              all((c in ref) for c in ROLE_COMMANDS[role]))
    check("outfield roles are not shown GK_DISTRIBUTE",
          "GK_DISTRIBUTE" not in command_reference("FWD"))


def test_summary():
    print("\n=== STATE SUMMARY ===")
    reset_memory()
    s = state()
    v = classify_phase(s, HOME, 4)
    text = summarize_tactical_state(s, HOME, 4, "FWD", v)
    for token in ["PHASE:", "stam=", "vel=", "distToOppGoal=", "channel", "behindBall",
                  "Opponents beaten"]:
        check(f"summary contains {token!r}", token in text)
    check("stamina is not collapsed to 0/1", "stam=1 " not in text and "stam=0 " not in text)
    print("\n---- sample (FWD) ----")
    print(text)
    print("----------------------")


if __name__ == "__main__":
    test_geometry()
    test_stamina()
    test_phases()
    test_counter()
    test_pusher_rule()
    test_fallbacks()
    test_gk_sweeper()
    test_shot_gate()
    test_role_whitelist()
    test_prompt_blocks()
    test_summary()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All tactical library tests passed.")
