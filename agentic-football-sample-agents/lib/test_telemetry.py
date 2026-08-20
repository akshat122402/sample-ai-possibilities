"""Offline tests for the calibration telemetry.

    python3 lib/test_telemetry.py

Checks that the records needed to fit calibration.py actually come out, that
outcomes are paired to their decisions, and that telemetry can never take a
match down with it.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_helpers import GAME_STATE, mock_agentcore

mock_agentcore()

import telemetry
from calibration import PROVENANCE, unmeasured
from phase import classify_phase, reset_memory

HOME = 0
FAILURES = []


class Log:
    """Captures what telemetry writes."""

    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    def warn(self, msg):
        self.lines.append(msg)

    def error(self, msg):
        self.lines.append(msg)

    def records(self, kind=None):
        out = []
        for line in self.lines:
            if not line.startswith(telemetry.PREFIX):
                continue
            rec = json.loads(line[len(telemetry.PREFIX) + 1:])
            if kind is None or rec["kind"] == kind:
                out.append(rec)
        return out


def check(label, condition, detail=""):
    if not condition:
        FAILURES.append(f"{label} {detail}")
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          f"{(' — ' + str(detail)) if detail and not condition else ''}")


def state(t=120.5, possession="agentId_3", score=(1, 0), ball=(15.3, -5.2)):
    s = copy.deepcopy(GAME_STATE)
    s["gameTime"] = t
    s["ball"]["possessionAgentId"] = possession
    s["ball"]["position"] = {"x": ball[0], "y": ball[1], "z": 0}
    s["score"] = {"home": score[0], "away": score[1]}
    return s


def hand_to(s, team_code, agent):
    s = copy.deepcopy(s)
    s["ball"]["possessionAgentId"] = agent
    for p in s["players"]:
        if p["teamCode"] == team_code and p["agentId"] == agent:
            p["position"] = {"x": s["ball"]["position"]["x"], "y": s["ball"]["position"]["y"]}
    return s


def tick(log, s, pid=4, role="FWD", command=None, source="llm"):
    view = classify_phase(s, HOME, pid)
    telemetry.observe(log, s, HOME, pid, role, view)
    if command:
        telemetry.record_decision(log, s, HOME, pid, role, view, command, source)
    return view


def fresh():
    telemetry.reset()
    reset_memory()
    return Log()


SHOOT = {"commandType": "SHOOT", "parameters": {"aim_location": "TR", "power": 0.9}}
PASS = {"commandType": "PASS", "parameters": {"target_player_id": 2, "type": "GROUND"}}
PRESS = {"commandType": "PRESS_BALL", "parameters": {"intensity": 0.8}}
MOVE = {"commandType": "MOVE_TO", "parameters": {"target_x": 10, "target_y": 0, "sprint": True}}


def test_kinematics():
    print("=== KINEMATICS ===")
    log = fresh()
    tick(log, state(t=100.0))
    s2 = state(t=100.5)
    for p in s2["players"]:
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
            p["position"] = {"x": 25.0, "y": 15.0}
            p["stamina"] = 0.80
    s2["ball"]["velocity"] = {"x": 1.0, "y": 0.0, "z": 0}
    tick(log, s2)

    recs = log.records("kinematics")
    check("a kinematics record is emitted", len(recs) == 1, len(recs))
    if recs:
        r = recs[0]
        check("dt is measured", r["dt"] == 0.5, r["dt"])
        check("observed speed is derived from displacement", r["obsSpeed"] > 0, r["obsSpeed"])
        check("stamina rate is signed per second", r["stamRate"] < 0, r["stamRate"])
        check("ball acceleration is recorded", "ballAccel" in r)
        check("sprint flag carried", "sprint" in r)
    check("no kinematics on the first tick of a session",
          len(log.records("kinematics")) == 1)


def test_decision_record():
    print("\n=== DECISION RECORD ===")
    log = fresh()
    tick(log, state(), command=SHOOT)
    recs = log.records("decision")
    check("a decision record is emitted", len(recs) == 1, len(recs))
    if recs:
        r = recs[0]
        for field in ("id", "role", "phase", "source", "cmd", "distGoal", "lateral",
                      "nearestOpp", "stam", "hasBall"):
            check(f"decision carries {field}", field in r)
        check("command type recorded", r["cmd"] == "SHOOT", r["cmd"])
        check("source recorded", r["source"] == "llm", r["source"])


def test_shot_outcomes():
    print("\n=== SHOT OUTCOMES ===")
    log = fresh()
    tick(log, state(t=100.0, score=(1, 0)), command=SHOOT)
    tick(log, state(t=100.4, score=(2, 0)))          # we scored
    outs = log.records("outcome")
    check("a goal resolves as goal", outs and outs[0]["result"] == "goal",
          outs[0]["result"] if outs else "none")
    check("outcome is paired to its decision",
          outs and outs[0]["id"] == log.records("decision")[0]["id"])
    check("outcome carries the decision's features", outs and "distGoal" in outs[0])

    log = fresh()
    tick(log, state(t=100.0), command=SHOOT)
    tick(log, hand_to(state(t=100.4), "away", "agentId_0"))   # their keeper has it
    outs = log.records("outcome")
    check("a save resolves as lost", outs and outs[0]["result"] == "lost",
          outs[0]["result"] if outs else "none")

    log = fresh()
    tick(log, state(t=100.0, possession=None), command=SHOOT)
    tick(log, state(t=100.4, possession=None))
    check("a shot still in flight stays open", not log.records("outcome"))
    tick(log, state(t=110.0, possession=None))                # past the window
    outs = log.records("outcome")
    check("an unresolved shot closes as no_goal after the window",
          outs and outs[0]["result"] == "no_goal", outs[0]["result"] if outs else "none")


def test_pass_outcomes():
    print("\n=== PASS OUTCOMES ===")
    log = fresh()
    tick(log, state(t=100.0), command=PASS)
    tick(log, hand_to(state(t=100.5), "home", "agentId_2"))
    outs = log.records("outcome")
    check("reaching a team-mate is completed", outs and outs[0]["result"] == "completed",
          outs[0]["result"] if outs else "none")

    log = fresh()
    tick(log, state(t=100.0), command=PASS)
    tick(log, hand_to(state(t=100.5), "away", "agentId_2"))
    outs = log.records("outcome")
    check("reaching an opponent is intercepted",
          outs and outs[0]["result"] == "intercepted", outs[0]["result"] if outs else "none")


def test_press_outcomes():
    print("\n=== PRESS OUTCOMES ===")
    log = fresh()
    opp_ball = hand_to(state(t=100.0), "away", "agentId_1")
    tick(log, opp_ball, pid=2, role="ML", command=PRESS)
    tick(log, hand_to(state(t=100.6), "home", "agentId_2"), pid=2, role="ML")
    outs = log.records("outcome")
    check("winning the ball yourself is won_ball_self",
          outs and outs[0]["result"] == "won_ball_self", outs[0]["result"] if outs else "none")


def test_pending_discipline():
    print("\n=== PENDING DISCIPLINE ===")
    log = fresh()
    opp_ball = hand_to(state(t=100.0), "away", "agentId_1")
    for i in range(5):                       # a maintained press reissued every tick
        s = hand_to(state(t=100.0 + i * 0.1), "away", "agentId_1")
        tick(log, s, pid=2, role="ML", command=PRESS)
    check("a reissued maintained command opens one pending, not five",
          len(log.records("outcome")) == 0 and len(log.records("decision")) == 5)
    tick(log, hand_to(state(t=100.6), "home", "agentId_2"), pid=2, role="ML")
    check("and resolves exactly once", len(log.records("outcome")) == 1,
          len(log.records("outcome")))

    log = fresh()
    for i in range(4):
        tick(log, state(t=100.0 + i * 0.1), command=MOVE)
    check("MOVE_TO opens no pending outcome", len(log.records("outcome")) == 0)
    check("MOVE_TO is still recorded as a decision", len(log.records("decision")) == 4)


def test_disable_and_safety():
    print("\n=== DISABLE + SAFETY ===")
    os.environ["TELEMETRY_ENABLED"] = "0"
    log = fresh()
    tick(log, state(), command=SHOOT)
    check("TELEMETRY_ENABLED=0 emits nothing", log.records() == [])
    os.environ.pop("TELEMETRY_ENABLED")

    log = fresh()

    class Broken:
        phase = "POSSESS"
        opponents_beaten = 0
        i_have_ball = False

    try:
        telemetry.observe(log, {"players": "not a list"}, HOME, 4, "FWD", Broken())
        telemetry.record_decision(log, {}, HOME, 4, "FWD", Broken(), {"commandType": "SHOOT"}, "llm")
        survived = True
    except Exception as e:
        survived = False
        print(f"    raised: {e}")
    check("malformed state never raises", survived)


def test_calibration_registry():
    print("\n=== CALIBRATION REGISTRY ===")
    import tactical_fallback as tf
    check("fallback reads its gates from calibration",
          tf.FWD_SHOOT_DIST == __import__("calibration").FWD_SHOOT_DIST)
    check("every tunable has a provenance entry", len(PROVENANCE) >= 12, len(PROVENANCE))
    check("all constants are still marked unmeasured", len(unmeasured()) == len(PROVENANCE),
          f"{len(unmeasured())} of {len(PROVENANCE)}")


if __name__ == "__main__":
    test_kinematics()
    test_decision_record()
    test_shot_outcomes()
    test_pass_outcomes()
    test_press_outcomes()
    test_pending_discipline()
    test_disable_and_safety()
    test_calibration_registry()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All telemetry tests passed.")
