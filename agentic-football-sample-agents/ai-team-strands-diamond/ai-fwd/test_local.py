"""Local test for the FWD agent — phases, fallback, role whitelist, parsing.

    python3 ai-fwd/test_local.py           # offline
    python3 ai-fwd/test_local.py --llm     # adds one real Bedrock call
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from test_helpers import GAME_STATE, TEAM_ID, mock_agentcore

mock_agentcore()

from phase import classify_phase, reset_memory
from parsing import parse_commands
from prompt_common import ROLE_COMMANDS
from tactical_agent_base import enforce_role
from tactical_fallback import build_tactical_fallback
from tactical_state import summarize_tactical_state

from main import MY_PLAYER_ID, ROLE, SYSTEM_PROMPT

fallback = build_tactical_fallback(ROLE)
failures = []


def scenario(possession="agentId_3", play_mode="OPEN_PLAY", ball=(15.3, -5.2)):
    s = json.loads(json.dumps(GAME_STATE))
    s["ball"]["possessionAgentId"] = possession
    s["ball"]["position"] = {"x": ball[0], "y": ball[1], "z": 0}
    s["playMode"] = play_mode
    return s


def opponent_has_ball(s, agent="agentId_1"):
    """Give the ball to an AWAY player and move them onto it. Both teams number
    their players 0-4, so the holder is only unambiguous when they are nearest
    the ball — which is how the runtime resolves it too."""
    s["ball"]["possessionAgentId"] = agent
    for p in s["players"]:
        if p["teamCode"] == "away" and p["agentId"] == agent:
            p["position"] = {"x": s["ball"]["position"]["x"], "y": s["ball"]["position"]["y"]}
    return s


def test_summary():
    print(f"=== STATE SUMMARY ({ROLE}, player {MY_PLAYER_ID}) ===")
    reset_memory()
    s = scenario()
    view = classify_phase(s, TEAM_ID, MY_PLAYER_ID)
    print(summarize_tactical_state(s, TEAM_ID, MY_PLAYER_ID, ROLE, view))
    print()


def test_fallback_every_phase():
    print(f"=== FALLBACK IN EVERY PHASE ({ROLE}) ===")
    cases = {
        "possession": scenario(),
        "defending": opponent_has_ball(scenario(ball=(-25.0, 3.0))),
        "loose ball": scenario(possession=None),
        "restart": scenario(play_mode="GOAL_KICK"),
    }
    for label, s in cases.items():
        reset_memory()
        view = classify_phase(s, TEAM_ID, MY_PLAYER_ID)
        cmds = fallback(s, TEAM_ID, MY_PLAYER_ID, view)
        c = cmds[0]
        ok = (
            len(cmds) == 1
            and c["playerId"] == MY_PLAYER_ID
            and c["teamId"] == TEAM_ID
            and c["commandType"] in ROLE_COMMANDS[ROLE]
        )
        if not ok:
            failures.append(f"fallback {label}: {json.dumps(cmds)}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:11s} phase={view.phase:8s} -> "
              f"{c['commandType']} {c.get('parameters', {})}")
    print()


def test_role_whitelist():
    print(f"=== ROLE WHITELIST ({ROLE}) ===")
    banned = ["RESET", "CLEAR_OVERRIDE"] + [
        c for c in ("SHOOT", "GK_DISTRIBUTE", "SET_STANCE", "SLIDE_TACKLE")
        if c not in ROLE_COMMANDS[ROLE]
    ]
    for cmd in banned:
        kept = enforce_role([{"commandType": cmd, "parameters": {}}], ROLE)
        ok = kept == []
        if not ok:
            failures.append(f"whitelist let {cmd} through")
        print(f"  [{'PASS' if ok else 'FAIL'}] refuses {cmd}")
    legal = ROLE_COMMANDS[ROLE][0]
    kept = enforce_role([{"commandType": legal, "parameters": {}}], ROLE)
    ok = len(kept) == 1
    if not ok:
        failures.append(f"whitelist dropped legal command {legal}")
    print(f"  [{'PASS' if ok else 'FAIL'}] allows {legal}")
    print()


def test_parse():
    print("=== PARSING ===")
    legal = ROLE_COMMANDS[ROLE][0]
    tests = [
        (f'[{{"commandType":"{legal}","playerId":9,"parameters":{{}},"duration":0}}]', 1),
        (f'Sure!\n[{{"commandType":"{legal}","playerId":9,"parameters":{{}},"duration":0}}]\nDone', 1),
        ('[{"commandType":"RESET","parameters":{}}]', 0),
        ("not json at all", 0),
    ]
    for text, expected in tests:
        cmds = enforce_role(parse_commands(text, TEAM_ID, MY_PLAYER_ID), ROLE)
        ok = len(cmds) == expected and all(c["playerId"] == MY_PLAYER_ID for c in cmds)
        if not ok:
            failures.append(f"parse {text[:40]} -> {len(cmds)}, expected {expected}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {text[:46]!r} -> {len(cmds)} (expected {expected})")
    print()


def test_llm():
    print(f"=== LLM ({ROLE}) ===")
    try:
        from strands import Agent
        from strands.models import BedrockModel

        reset_memory()
        s = scenario()
        view = classify_phase(s, TEAM_ID, MY_PLAYER_ID)
        summary = summarize_tactical_state(s, TEAM_ID, MY_PLAYER_ID, ROLE, view)
        agent = Agent(model=BedrockModel(model_id="us.amazon.nova-pro-v1:0"), system_prompt=SYSTEM_PROMPT)
        text = str(agent(summary))
        print(text[:400])
        cmds = enforce_role(parse_commands(text, TEAM_ID, MY_PLAYER_ID), ROLE)
        print(f"\nParsed {len(cmds)} allowed command(s): "
              f"{[c.get('commandType') for c in cmds]}")
        if not cmds:
            print("LLM produced nothing usable — the fallback would have covered this tick.")
    except Exception as e:
        print(f"LLM test error: {e}")


if __name__ == "__main__":
    test_summary()
    test_fallback_every_phase()
    test_role_whitelist()
    test_parse()

    if "--llm" in sys.argv:
        test_llm()
    else:
        print("Skipping LLM test. Run with --llm to test.")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll offline tests passed.")
