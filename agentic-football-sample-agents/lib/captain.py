"""LLM captain for the diamond team.

classify_phase() coordinates the tick; this is the slow loop above it. A second,
stronger model rides in the GK runtime (the one role allowed SET_STANCE), reviews
the match every CAPTAIN_PERIOD_SECONDS, and sets two things: the stance (via a
SET_STANCE command in the GK's own array) and a named strategy (published to the
blackboard, where every player's next state summary reads it). Reviews see the
score, the phase mix, and the team's measured outcome tallies, so the plan
adapts to what is actually working, not just the scoreline.

Cost: the GK tick that carries a review waits on a Sonnet call (~1-3s). Reviews
therefore prefer RESTART ticks, where nobody is moving; an overdue or urgent
review (a goal) fires in open play and accepts one slow tick every ~20s.

Everything here is best-effort: a captain failure costs a review, never a tick.
"""

from __future__ import annotations

import blackboard
import strategy as strategy_mod
import telemetry
from agent_base import create_agent
from calibration import CAPTAIN_PERIOD_SECONDS
from json_tolerant import parse_json_tolerant
from models import CAPTAIN_MODEL_ID
from phase import RESTART

STANCE_NAMES = {0: "BALANCED", 1: "ATTACK", 2: "DEFEND"}

CAPTAIN_PROMPT = f"""You are the captain of a 5-a-side football team playing a 1-2-1 diamond.
Every ~20 seconds you review the match and set the team's plan. Your five players already
coordinate tick-to-tick through a shared phase system; your job is the slower question
their reflexes cannot answer: given the score, the clock, and what has actually been
working, how should the team play the next stretch?

You set two things.

Stance (a game-engine lever):
- 0 BALANCED — the default; keep it unless the review gives a reason not to
- 1 ATTACK   — chase the game: behind late, or dominating and a second goal kills the match
- 2 DEFEND   — protect the lead: ahead late, or being overrun and the priority is survival

Strategy (a plan every player reads in their next state summary):
{strategy_mod.catalogue()}

Use the outcome numbers when you have them — they say what is WORKING, which the score
alone does not. Presses being lost and passes completing points to POSSESS_WIDE, not
HIGH_PRESS; passes being intercepted in build-up points to DIRECT_COUNTER. Change strategy
when the evidence says the current one is failing, not on every review — a plan the team
holds beats a better plan they never settle into.

Respond with ONLY a JSON object, no prose before or after:
{{"stance": 0, "strategy": "BALANCED_DEFAULT", "reason": "one short sentence"}}"""


def create_captain_agent():
    return create_agent(CAPTAIN_PROMPT, model_id=CAPTAIN_MODEL_ID)


# Review state per team. A module global survives between invocations in the
# same container (see phase.py); losing it on a recycle just means one early
# review.
_state: dict = {}


def reset() -> None:
    _state.clear()


def _entry(team_id: int) -> dict:
    return _state.setdefault(team_id, {
        "last_review": None, "last_score": None, "last_stance": None,
        "last_strategy": strategy_mod.DEFAULT, "phase_ticks": {},
    })


def observe(team_id: int, view) -> None:
    """Accumulate phase counts between reviews — the captain's match memory."""
    entry = _entry(team_id)
    entry["phase_ticks"][view.phase] = entry["phase_ticks"].get(view.phase, 0) + 1


def _team_outcomes(team_id: int) -> str:
    """Cumulative outcome tallies across the team, or '' when none exist yet.

    The other four players' tallies arrive over the blackboard; the GK's own
    are local. Without a blackboard this degrades to GK-only numbers.
    """
    combined: dict = {}
    per_player = blackboard.read_stats(team_id)
    per_player[0] = telemetry.match_tally(team_id, 0)  # local GK, always fresh
    for tally in per_player.values():
        for cmd, (n, ok) in tally.items():
            counts = combined.setdefault(cmd, [0, 0])
            counts[0] += n
            counts[1] += ok
    return ", ".join(f"{cmd} {ok}/{n} worked" for cmd, (n, ok) in sorted(combined.items()))


def _build_review(game_state: dict, team_id: int, view, entry: dict) -> str:
    score = game_state.get("score", {})
    home, away = score.get("home", 0), score.get("away", 0)
    ours, theirs = (home, away) if team_id == 0 else (away, home)
    ticks = entry["phase_ticks"]
    total = sum(ticks.values()) or 1
    shares = ", ".join(
        f"{p} {ticks.get(p, 0) * 100 // total}%" for p in ("POSSESS", "DEFEND", "COUNTER", "LOOSE")
    )
    stance = entry["last_stance"]
    lines = [
        f"Time: {float(game_state.get('gameTime', 0) or 0):.0f}s",
        f"Score: us {ours} - {theirs} them",
        f"Play since your last review: {shares}",
        f"Current stance: {STANCE_NAMES.get(stance, 'BALANCED (never set)')}",
        f"Current strategy: {entry['last_strategy']}",
        f"Right now: {view.phase} — {view.reason}",
    ]
    outcomes = _team_outcomes(team_id)
    if outcomes:
        lines.append(f"Team outcomes this match: {outcomes}")
    return "\n".join(lines)


def _parse_stance(text: str):
    """Pull {"stance": n} out of the response; None if it isn't there."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    parsed = parse_json_tolerant(text[start:end + 1])
    if parsed is None:
        return None
    data, _recovered = parsed
    if not isinstance(data, dict):
        return None
    stance = data.get("stance")
    if stance not in (0, 1, 2):
        return None
    chosen = data.get("strategy")
    return {
        "stance": stance,
        # An unknown or missing strategy keeps the current one, not the default.
        "strategy": chosen if strategy_mod.valid(chosen) else None,
        "reason": str(data.get("reason", ""))[:120],
    }


def maybe_review(agent, log, game_state: dict, team_id: int, view):
    """Run a review if one is due. Returns a SET_STANCE command dict or None."""
    entry = _entry(team_id)
    now = float(game_state.get("gameTime", 0) or 0)
    score = game_state.get("score", {})
    score_key = (score.get("home", 0), score.get("away", 0))

    last = entry["last_review"]
    elapsed = None if last is None else now - last
    scored = entry["last_score"] is not None and score_key != entry["last_score"]
    due = elapsed is None or elapsed >= CAPTAIN_PERIOD_SECONDS

    # Prefer a dead ball; go anyway when a goal changed the match or the review
    # is a full period overdue.
    if not (scored or (due and view.phase == RESTART) or (elapsed is not None and elapsed >= 2 * CAPTAIN_PERIOD_SECONDS) or (last is None and view.phase == RESTART)):
        entry["last_score"] = entry["last_score"] or score_key
        return None

    entry["last_review"] = now
    entry["last_score"] = score_key
    entry["phase_ticks"] = {}

    try:
        response = str(agent(_build_review(game_state, team_id, view, entry)))
        decision = _parse_stance(response)
    except Exception as e:
        log.warn(f"captain review failed, keeping stance: {e}")
        return None
    if decision is None:
        log.warn("captain returned no usable decision, keeping stance and strategy")
        return None

    new_strategy = decision["strategy"] or entry["last_strategy"]
    if new_strategy != entry["last_strategy"]:
        log.info(f"captain switches strategy to {new_strategy}: {decision['reason']}")
    entry["last_strategy"] = new_strategy
    blackboard.publish_plan(team_id, new_strategy, decision["stance"],
                            decision["reason"], now)

    if decision["stance"] == entry["last_stance"]:
        log.info(f"captain holds {STANCE_NAMES[decision['stance']]}: {decision['reason']}")
        return None

    entry["last_stance"] = decision["stance"]
    log.info(f"captain sets {STANCE_NAMES[decision['stance']]}: {decision['reason']}")
    return {
        "commandType": "SET_STANCE",
        "parameters": {"stance": decision["stance"]},
        "duration": 0,
    }
