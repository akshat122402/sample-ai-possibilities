"""Shared prompt blocks for the diamond team.

In the sample teams roughly half of every system prompt is byte-identical
boilerplate, copy-pasted across five files. It lives here once instead, which
also makes the per-role command list a function of the role: an agent is only
shown the commands it is allowed to issue, so there is nothing to hallucinate
from.
"""

from __future__ import annotations

# ── command catalogue ────────────────────────────────────────────────────────
# duration is documented here because the sample prompts show it only inside a
# single example, leaving the model to guess what it means.

_COMMANDS = {
    "MOVE_TO": 'MOVE_TO — target_x (float), target_y (float), sprint (bool). One-shot: runs until reached or replaced.',
    "PASS": 'PASS — target_player_id (int), type ("GROUND"|"AERIAL"|"THROUGH"). Only when you have the ball.',
    "SHOOT": 'SHOOT — aim_location ("TL"|"TR"|"BL"|"BR"|"CENTER"), power (0.0-1.0). Only when you have the ball.',
    "SLIDE_TACKLE": 'SLIDE_TACKLE — target_player_id (int), sprint (bool), distance (float). High risk: if you miss, you are out of the play.',
    "GK_DISTRIBUTE": 'GK_DISTRIBUTE — target_player_id (int), method ("THROW"|"KICK"). THROW is short and safe, KICK is long.',
    "PRESS_BALL": 'PRESS_BALL — intensity (0.0-1.0). Maintained: set duration to the seconds it should persist.',
    "MARK": 'MARK — target_player_id (int), tightness ("LOOSE"|"TIGHT"). Maintained: set duration.',
    "INTERCEPT": 'INTERCEPT — aggressive (bool). Maintained: set duration. Uses ball velocity to cut the ball off.',
    "FOLLOW_PLAYER": 'FOLLOW_PLAYER — target_player_id (int), target_team ("HOME"|"AWAY"), distance (float). Maintained: set duration.',
    "SET_STANCE": 'SET_STANCE — stance (0=Balanced, 1=Attack, 2=Defend).',
}

# RESET and CLEAR_OVERRIDE are deliberately absent from every role. RESET is
# team-scoped, so one bad command from any of the five would wipe the whole
# team's overrides; nothing in this design needs either of them.
ROLE_COMMANDS = {
    "GK":  ["MOVE_TO", "GK_DISTRIBUTE", "PASS", "INTERCEPT", "PRESS_BALL", "MARK", "SET_STANCE"],
    "DEF": ["MOVE_TO", "PASS", "SHOOT", "SLIDE_TACKLE", "PRESS_BALL", "MARK", "INTERCEPT", "FOLLOW_PLAYER"],
    "ML":  ["MOVE_TO", "PASS", "SHOOT", "SLIDE_TACKLE", "PRESS_BALL", "MARK", "INTERCEPT", "FOLLOW_PLAYER"],
    "MR":  ["MOVE_TO", "PASS", "SHOOT", "SLIDE_TACKLE", "PRESS_BALL", "MARK", "INTERCEPT", "FOLLOW_PLAYER"],
    "FWD": ["MOVE_TO", "PASS", "SHOOT", "PRESS_BALL", "MARK", "INTERCEPT", "FOLLOW_PLAYER"],
}


def command_reference(role: str) -> str:
    lines = [f"- {_COMMANDS[c]}" for c in ROLE_COMMANDS[role]]
    return "## Commands you may issue\n" + "\n".join(lines)


FIELD = """## The pitch
- x runs -55 to +55, y runs -35 to +35. The goal centres are at (-55,0) and (+55,0).
- Your attacking direction is stated on the first lines of every game state. Never infer it.
- LEFT and RIGHT always mean your left and right while facing the opponent's goal. The
  state gives you a `channel` (LEFT/CENTRE/RIGHT) for the ball and for every player, so
  you never have to work out a side from a y value.
- aim_location is from your point of view facing the opponent's goal: T/B are the top and
  bottom corners, L/R your left and right.
- `distToOppGoal` is the true distance to the goal centre, so ~24 really is shooting range."""


HINTS = """## Computed tactics, strategy and memory
Some ticks include extra pre-computed lines. When present, they are more reliable than
your own estimates — use them:
- A "Computed" block: pass success odds, shot quality, the clearest open point, or
  threat-ranked marking targets, worked out deterministically from the same state you see.
  Advice, not orders: the phase and your role still decide what to do with them.
- A STRATEGY line: the captain's current plan for the whole team, with your part in it.
  Follow its emphasis within your role — it outranks your default phase behaviour where
  the two disagree, but never the response format or your command whitelist.
- A "Your recent outcomes" line: how your own recent commands actually went
  (e.g. "PASS 1/3" = one of your last three passes completed). If something keeps
  failing, change it — a different target, a safer type, a different position."""


PHASES = """## Phases
The PHASE on the first line is computed from the game state by the same function every
one of your team-mates runs, so all five of you agree on it without talking. Play the
phase you are given; do not second-guess it.
- DEFEND  — the opponent has the ball
- LOOSE   — nobody has the ball
- COUNTER — we have just won the ball with opponents caught upfield; this lasts a few
            seconds and is the most valuable phase on the pitch
- POSSESS — we have the ball in a settled attack
- RESTART — dead ball (kickoff, throw-in, corner, goal kick)"""


def output_contract(player_id: int, examples: list[str]) -> str:
    joined = "\n".join(examples)
    return f"""## Response format
Return ONE JSON array containing exactly ONE command for player {player_id}. No prose, no
markdown fences, nothing before or after the array.
- JSON booleans are lowercase `true` / `false` — never Python's `True` / `False`.
- `duration` is seconds. Use 0 for one-shot commands (MOVE_TO, PASS, SHOOT) and 2-5 for
  maintained ones (PRESS_BALL, MARK, INTERCEPT, FOLLOW_PLAYER); a maintained command with
  duration 0 expires immediately and wastes the tick.
- Most ticks you will not have the ball. Off-ball positioning is the normal answer.

Examples:
{joined}"""
