"""Builds harness system prompts from the runtime ones.

A harness is a managed agent loop: AWS owns the loop, and the only thing that
runs is a system prompt plus whatever tools are declared. None of this team's
Python executes there — no classify_phase, no summarize_tactical_state, no
enforce_role, no fallback. The model receives the caller's payload as-is.

So a harness prompt has to do in prose what tactical_state.py does in code:
explain the raw payload and ask the model to derive the same quantities. That
is strictly weaker, and the differences are listed in HARNESS_CAVEATS below.

The runtime prompt stays the single source of truth for tactics — this module
only prepends the input contract, so a change to a role's play is picked up by
both without editing anything twice.
"""

from __future__ import annotations

# Things the runtime does in code that a harness cannot do at all. Kept here so
# the gap is documented next to the thing that creates it.
HARNESS_CAVEATS = [
    "no turnover detection — COUNTER is approximated from a single frame, so a "
    "settled attack against a stretched defence looks the same as a break",
    "no role whitelist — nothing refuses RESET or an out-of-role command",
    "no rule-based fallback — a failed or malformed model response is simply lost",
    "no telemetry — the calibration records are produced by runtime code",
    "no clamping of MOVE_TO targets to the pitch",
]


INPUT_CONTRACT = """## Your input

You receive the raw match state as JSON. Nothing pre-processes it for you, so
derive what you need from these fields:

- `gameTime` (seconds), `score` {{home, away}}, `playMode` (e.g. "OPEN_PLAY")
- `ball.position` {{x, y}}, `ball.velocity` {{x, y}}, `ball.possessionAgentId`
- `players[]`, each with `agentId` ("agentId_0" … "agentId_4"), `teamCode`
  ("home"/"away"), `position`, `velocity`, `stamina` (0.0-1.0), `isSprinting`

You are **player {player_id}** on team **{team_word}** (`teamCode` = "{team_code}").

### Deriving the things this prompt refers to

- **Who has the ball.** Both teams number their players 0-4, so
  `possessionAgentId` of "agentId_3" matches one player per team. The holder is
  whichever of those two is closest to `ball.position`. If `possessionAgentId`
  is null, the ball is free.
- **Attacking direction.** home attacks +x (opponent goal at (55, 0)); away
  attacks -x (opponent goal at (-55, 0)).
- **distToOppGoal.** True distance from a position to the opponent goal centre:
  sqrt((x - goalX)^2 + y^2). Not the difference in x — a wide position is much
  further from goal than its x suggests, and the shooting ranges in this prompt
  assume the true distance.
- **Channel (LEFT / CENTRE / RIGHT).** Facing the opponent goal. For home:
  y < -6 is LEFT, y > 6 is RIGHT. For away the signs are reversed. Never work a
  side out from a raw y without applying this.
- **Opponents beaten.** Opponents who are further from their own goal than the
  ball is — i.e. caught behind the play.
- **Stamina** is 0.0-1.0. 0.65 means 65%, not "1".

### Deriving the PHASE

The phase is not given to you here. Work it out, in this order:

1. `playMode` is not "OPEN_PLAY" -> **RESTART**
2. an opponent holds the ball -> **DEFEND**
3. nobody holds the ball -> **LOOSE**
4. you hold it or a team-mate does, and at least 2 opponents are behind the
   ball -> **COUNTER**
5. otherwise -> **POSSESS**

Rule 4 is an approximation. It cannot tell a break from a settled attack against
a stretched defence, because that needs the previous tick and you only get this
one. When in doubt between COUNTER and POSSESS, prefer POSSESS — over-committing
on a counter that is not on leaves your defender isolated.

"""


def build(role: str, player_id: int, runtime_prompt: str, team_id: int = 0) -> str:
    """Compose a harness system prompt: input contract + the runtime's tactics."""
    contract = INPUT_CONTRACT.format(
        player_id=player_id,
        team_word="HOME" if team_id == 0 else "AWAY",
        team_code="home" if team_id == 0 else "away",
    )
    return (
        f"{contract}"
        "---\n\n"
        "Everything below is your role. It is the same brief the runtime version of\n"
        "this agent uses; where it refers to a value like `distToOppGoal`, `channel`\n"
        "or `PHASE`, derive it as described above.\n\n"
        f"{runtime_prompt}"
    )
