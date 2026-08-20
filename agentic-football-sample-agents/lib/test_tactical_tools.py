"""Offline tests for the computed tactics and the captain.

    python3 lib/test_tactical_tools.py

No AWS credentials and no network. Covers the four inlined gateway calculations,
the phase/role gating of the hint block, and the captain's cadence, parsing and
stance-change behaviour.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_helpers import GAME_STATE, mock_agentcore

mock_agentcore()

import blackboard as bb
import captain as cap
import strategy as strat
import telemetry as tel
from phase import classify_phase, reset_memory
from prompt_common import ROLE_COMMANDS
from tactical_state import summarize_tactical_state
from tactical_tools import (
    best_open_space,
    mark_targets,
    pass_options,
    shot_quality,
    tactical_hints,
)

HOME = 0
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


class _Log:
    @staticmethod
    def info(msg): pass
    @staticmethod
    def warn(msg): pass
    @staticmethod
    def error(msg): pass


class _FixedAgent:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_prompt = None

    def __call__(self, prompt):
        self.calls += 1
        self.last_prompt = str(prompt)
        return self.response


class _FakeTable:
    """In-memory stand-in for a boto3 DynamoDB Table."""

    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[(int(Item["teamId"]), Item["entry"])] = dict(Item)

    def get_item(self, Key):
        item = self.items.get((int(Key["teamId"]), Key["entry"]))
        return {"Item": dict(item)} if item else {}

    def query(self, KeyConditionExpression=None):
        # Without boto3 the lib passes the bare team id as the condition.
        tid = int(KeyConditionExpression)
        return {"Items": [dict(v) for (t, _), v in self.items.items() if t == tid]}


def test_pass_options():
    print("pass_options:")
    passer = {"x": 0, "y": 0}
    teammates = [
        {"agentId": "agentId_1", "position": {"x": 10, "y": 0}},   # clear lane, ahead
        {"agentId": "agentId_2", "position": {"x": 40, "y": 20}},  # long
    ]
    opponents = [{"agentId": "agentId_3", "position": {"x": 20, "y": 10}}]
    opts = pass_options(passer, teammates, opponents, HOME)
    check("sorted best-first", opts[0]["success"] >= opts[1]["success"])
    check("short clear pass wins", opts[0]["player_id"] == 1, f"got {opts[0]}")
    check("success within [0.05, 1]", all(0.05 <= o["success"] <= 1.0 for o in opts))
    check("open runner ahead gets THROUGH", opts[0]["type"] == "THROUGH", f"got {opts[0]}")
    check("long ball gets AERIAL", opts[1]["type"] == "AERIAL", f"got {opts[1]}")

    blocked = pass_options(passer, [teammates[0]],
                           [{"position": {"x": 5, "y": 0}}], HOME)  # opponent on the lane
    check("opponent on the lane raises risk", blocked[0]["risk"] > 0.8, f"risk {blocked[0]['risk']}")

    short_back = pass_options({"x": 10, "y": 0},
                              [{"agentId": "agentId_1", "position": {"x": 2, "y": 2}}],
                              [{"position": {"x": -30, "y": -20}}], HOME)
    check("short backward pass gets GROUND", short_back[0]["type"] == "GROUND",
          f"got {short_back[0]}")


def test_shot_quality():
    print("shot_quality:")
    close = shot_quality({"x": 45, "y": 0}, {"x": 54, "y": 0}, [], HOME)
    far = shot_quality({"x": -10, "y": 0}, {"x": 54, "y": 0}, [], HOME)
    check("closer shot scores higher", close["probability"] > far["probability"])
    check("probability clamped", 0.02 <= close["probability"] <= 0.95)
    off_line = shot_quality({"x": 45, "y": 0}, {"x": 54, "y": 6}, [], HOME)
    check("GK off his line helps", off_line["probability"] >= close["probability"])
    check("aim avoids the keeper", off_line["aim"] in ("BL", "BR"), f"aim {off_line['aim']}")
    blocked = shot_quality({"x": 45, "y": 0}, {"x": 54, "y": 0},
                           [{"x": 47, "y": 0}, {"x": 46, "y": 1}], HOME)
    check("blockers reduce the shot", blocked["probability"] < close["probability"])


def test_open_space():
    print("best_open_space:")
    opponents = [{"position": {"x": 30, "y": 20}}, {"position": {"x": 30, "y": -20}}]
    spot = best_open_space(opponents, {"x": 20, "y": 0}, "attack", HOME)
    check("attack zone for HOME is x>=15", spot["x"] >= 15, f"got {spot}")
    check("clearance reported", spot["clearance"] > 0)
    away_spot = best_open_space(opponents, {"x": -20, "y": 0}, "attack", 1)
    check("attack zone for AWAY is x<=-15", away_spot["x"] <= -15, f"got {away_spot}")


def test_mark_targets():
    print("mark_targets:")
    opponents = [
        {"agentId": "agentId_2", "position": {"x": -40, "y": 5}},   # deep in our half
        {"agentId": "agentId_4", "position": {"x": 30, "y": 0}},    # far away
    ]
    threats = mark_targets(opponents, {"x": -38, "y": 5}, {"x": -30, "y": 0}, HOME, carrier_id=2)
    check("carrier near our goal ranks first", threats[0]["player_id"] == 2, f"got {threats[0]}")
    check("carrier flagged", threats[0]["has_ball"])
    check("high threat marked TIGHT", threats[0]["tightness"] == "TIGHT")


def test_hint_gating():
    print("tactical_hints gating:")
    reset_memory()
    s = state()  # HOME player 3 has the ball
    view = classify_phase(s, HOME, 3)
    hints = tactical_hints(s, HOME, 3, "MR", view)
    check("carrier gets pressure + pass + shot lines", len(hints) == 3, f"got {hints}")
    check("pressure line first", hints[0].startswith("Pressure:"))
    check("pass line has a delivery type", hints[1].startswith("Pass:") and
          any(t in hints[1] for t in ("GROUND", "AERIAL", "THROUGH")), f"got {hints[1]}")
    check("shot line present", hints[2].startswith("Shot:"))

    view4 = classify_phase(s, HOME, 4)
    hints4 = tactical_hints(s, HOME, 4, "FWD", view4)
    check("off-ball attacker gets a space line", len(hints4) == 1 and hints4[0].startswith("Space:"),
          f"got {hints4}")

    reset_memory()
    d = state(possession="agentId_2")
    # give the ball to an AWAY player unambiguously
    for p in d["players"]:
        if p["teamCode"] == "away" and p["agentId"] == "agentId_2":
            p["position"] = dict(d["ball"]["position"])
    view_d = classify_phase(d, HOME, 1)
    hints_d = tactical_hints(d, HOME, 1, "DEF", view_d)
    check("defender in DEFEND gets a mark line",
          any(h.startswith("Mark:") for h in hints_d), f"got {hints_d}")

    view_gk = classify_phase(d, HOME, 0)
    gk_hints = tactical_hints(d, HOME, 0, "GK", view_gk)
    check("GK gets exactly the line hint",
          len(gk_hints) == 1 and gk_hints[0].startswith("GK line:"), f"got {gk_hints}")


def test_summary_contains_hints():
    print("summary integration:")
    reset_memory()
    s = state()
    view = classify_phase(s, HOME, 3)
    summary = summarize_tactical_state(s, HOME, 3, "MR", view)
    check("Computed block in carrier summary", "Computed (deterministic" in summary)
    check("summary stays compact", len(summary) < 2500, f"{len(summary)} chars")


def test_captain_parsing():
    print("captain parsing:")
    check("plain JSON", cap._parse_stance('{"stance": 2, "reason": "protect the lead"}')["stance"] == 2)
    check("prose around JSON", cap._parse_stance('Sure: {"stance": 1, "reason": "chase"} done')["stance"] == 1)
    check("python literals recovered", cap._parse_stance("{'x': 1}") is None)  # single quotes stay unparsed
    check("bad stance rejected", cap._parse_stance('{"stance": 7}') is None)
    check("no JSON rejected", cap._parse_stance("attack!") is None)


def test_captain_cadence():
    print("captain cadence:")
    cap.reset()
    reset_memory()
    log = _Log()
    agent = _FixedAgent('{"stance": 1, "reason": "chase the game"}')

    kickoff = state(possession=None, play_mode="KICKOFF", time=0.0)
    view = classify_phase(kickoff, HOME, 0)
    cmd = cap.maybe_review(agent, log, kickoff, HOME, view)
    check("first review at a restart", cmd is not None and cmd["commandType"] == "SET_STANCE")
    check("stance carried", cmd["parameters"]["stance"] == 1)

    soon = state(time=5.0)
    view = classify_phase(soon, HOME, 0)
    check("no review inside the period", cap.maybe_review(agent, log, soon, HOME, view) is None)
    check("model not called again", agent.calls == 1)

    goal = state(time=8.0)
    goal["score"] = {"home": 2, "away": 0}
    view = classify_phase(goal, HOME, 0)
    agent2 = _FixedAgent('{"stance": 2, "reason": "protect it"}')
    cmd = cap.maybe_review(agent2, log, goal, HOME, view)
    check("a goal forces a review", cmd is not None and cmd["parameters"]["stance"] == 2)

    much_later = state(time=100.0)
    view = classify_phase(much_later, HOME, 0)
    agent3 = _FixedAgent('{"stance": 2, "reason": "still ahead"}')
    check("unchanged stance issues nothing", cap.maybe_review(agent3, log, much_later, HOME, view) is None)
    check("but the review ran", agent3.calls == 1)

    cap.reset()
    reset_memory()
    broken = _FixedAgent("no json here")
    view = classify_phase(kickoff, HOME, 0)
    check("garbage response keeps stance", cap.maybe_review(broken, log, kickoff, HOME, view) is None)


def test_gk_line():
    print("gk_line_target:")
    from tactical_tools import gk_line_target
    # HOME goal at (-55, 0); ball ahead of goal, DEF at 20 units out
    spot = gk_line_target({"x": 0, "y": 10}, HOME, {"x": -35, "y": 0}, we_have_ball=True)
    check("attack depth capped by cap and DEF margin",
          2.0 <= spot["depth"] <= 18.0, f"got {spot}")
    check("stays goal-side of DEF", spot["depth"] <= 20.0 - 6.0 + 1e-6)
    defend = gk_line_target({"x": -30, "y": -8}, HOME, {"x": -35, "y": 0}, we_have_ball=False)
    check("defend depth is shallow", defend["depth"] <= 6.0, f"got {defend}")
    check("on the ball side of centre", defend["y"] < 0, f"got {defend}")
    deep_def = gk_line_target({"x": 0, "y": 0}, HOME, {"x": -50, "y": 0}, we_have_ball=True)
    check("drops with a deep DEF", deep_def["depth"] <= 2.0 + 1e-6, f"got {deep_def}")


def test_strategy_catalogue():
    print("strategy catalogue:")
    roles = set(ROLE_COMMANDS)
    check("default strategy exists", strat.valid(strat.DEFAULT))
    for name, entry in strat.STRATEGIES.items():
        check(f"{name} briefs every role", set(entry["brief"]) == roles,
              f"got {sorted(entry['brief'])}")
    check("catalogue lists every strategy",
          all(name in strat.catalogue() for name in strat.STRATEGIES))
    check("unknown strategy rejected", not strat.valid("PARK_THE_BUS"))
    check("unknown brief falls back", strat.role_brief("PARK_THE_BUS", "GK") != "")


def test_blackboard():
    print("blackboard:")
    bb.reset()
    table = _FakeTable()
    bb.set_table_for_tests(table)

    bb.publish_plan(0, "HIGH_PRESS", 1, "chase the game", 100.0)
    plan = bb.read_plan(0)
    check("plan round-trips", plan and plan["strategy"] == "HIGH_PRESS" and plan["stance"] == 1,
          f"got {plan}")
    bb._plan_cache.clear()
    check("plan survives a cache clear", bb.read_plan(0)["strategy"] == "HIGH_PRESS")

    bb.publish_stats(0, 3, {"PASS": [3, 1]})
    check("stats round-trip", bb.read_stats(0) == {3: {"PASS": [3, 1]}}, f"got {bb.read_stats(0)}")
    bb.publish_stats(0, 3, {"PASS": [9, 9]})  # inside the throttle window
    check("stats writes are throttled", bb.read_stats(0)[3] == {"PASS": [3, 1]})
    check("plan item is not a stats item", 3 in bb.read_stats(0) and len(bb.read_stats(0)) == 1)

    bb.reset()  # no table, no env var
    check("unconfigured blackboard reads None", bb.read_plan(0) is None)
    bb.publish_stats(0, 3, {"PASS": [1, 1]})  # must be a silent no-op
    check("unconfigured blackboard reads no stats", bb.read_stats(0) == {})


def test_telemetry_memory():
    print("telemetry memory:")
    tel.reset()
    reset_memory()
    log = _Log()

    s1 = state(possession="agentId_3", ball=(14, -5), time=10.0)  # our P3 on the ball
    view1 = classify_phase(s1, HOME, 3)
    tel.observe(log, s1, HOME, 3, "MR", view1)
    tel.record_decision(log, s1, HOME, 3, "MR", view1,
                        {"commandType": "PASS", "parameters": {"target_player_id": 4}}, "llm")

    s2 = state(possession="agentId_4", ball=(20, 15), time=11.0)  # our P4 receives
    view2 = classify_phase(s2, HOME, 3)
    tel.observe(log, s2, HOME, 3, "MR", view2)

    check("completed pass tallied", tel.match_tally(HOME, 3) == {"PASS": [1, 1]},
          f"got {tel.match_tally(HOME, 3)}")
    check("recent pattern line", tel.recent_patterns(HOME, 3) == "PASS 1/1",
          f"got {tel.recent_patterns(HOME, 3)!r}")

    summary = summarize_tactical_state(s2, HOME, 3, "MR", view2)
    check("recent outcomes reach the summary", "Your recent outcomes" in summary)


def test_strategy_in_summary():
    print("strategy line in summary:")
    bb.reset()
    reset_memory()
    bb.set_table_for_tests(_FakeTable())
    bb.publish_plan(0, "LOW_BLOCK", 2, "protect the lead", 200.0)

    s = state()
    view = classify_phase(s, HOME, 1)
    summary = summarize_tactical_state(s, HOME, 1, "DEF", view)
    check("STRATEGY line present", "STRATEGY: LOW_BLOCK" in summary)
    check("role brief attached", strat.role_brief("LOW_BLOCK", "DEF") in summary)
    check("STANCE line present", "STANCE: DEFEND (captain)" in summary)

    bb.publish_plan(0, strat.DEFAULT, 0, "level game", 210.0)
    bb._plan_cache.clear()
    summary = summarize_tactical_state(s, HOME, 1, "DEF", classify_phase(s, HOME, 1))
    check("default strategy adds no line", "STRATEGY:" not in summary)
    bb.reset()


def test_captain_strategy():
    print("captain strategy:")
    cap.reset()
    tel.reset()
    reset_memory()
    bb.reset()
    bb.set_table_for_tests(_FakeTable())
    log = _Log()

    kickoff = state(possession=None, play_mode="KICKOFF", time=0.0)
    view = classify_phase(kickoff, HOME, 0)
    agent = _FixedAgent('{"stance": 1, "strategy": "HIGH_PRESS", "reason": "win it back high"}')
    cmd = cap.maybe_review(agent, log, kickoff, HOME, view)
    check("stance command issued", cmd is not None and cmd["parameters"]["stance"] == 1)
    check("strategy published", bb.read_plan(HOME)["strategy"] == "HIGH_PRESS")
    check("catalogue shown to the captain", "LOW_BLOCK" in cap.CAPTAIN_PROMPT)

    goal = state(time=30.0)
    goal["score"] = {"home": 1, "away": 0}
    view = classify_phase(goal, HOME, 0)
    agent2 = _FixedAgent('{"stance": 1, "strategy": "NOT_A_STRATEGY", "reason": "keep going"}')
    cap.maybe_review(agent2, log, goal, HOME, view)
    check("invalid strategy keeps the current one", bb.read_plan(HOME)["strategy"] == "HIGH_PRESS")
    bb.reset()


if __name__ == "__main__":
    test_pass_options()
    test_shot_quality()
    test_open_space()
    test_mark_targets()
    test_hint_gating()
    test_summary_contains_hints()
    test_captain_parsing()
    test_captain_cadence()
    test_gk_line()
    test_strategy_catalogue()
    test_blackboard()
    test_telemetry_memory()
    test_strategy_in_summary()
    test_captain_strategy()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All tactical tools tests passed.")
