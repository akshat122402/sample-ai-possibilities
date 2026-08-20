"""Named strategies for the diamond team.

A strategy is data, not prose: a catalogue entry the captain chooses from, and a
one-line brief per role that lands in that player's state summary. The captain
picks WHICH plan; the phase layer and each role's prompt still decide HOW it is
executed tick to tick. Strategies deliberately do not touch the rule-based
fallbacks — those stay the phase-pure safe default, so a model dropout degrades
to conservative play, never to a stale plan.

Distribution is the blackboard (lib/blackboard.py): the captain publishes the
chosen strategy, every player's summary reads it back. Without a blackboard the
team simply plays BALANCED_DEFAULT — the same football it played before this
module existed.
"""

from __future__ import annotations

DEFAULT = "BALANCED_DEFAULT"

# name -> {"summary": for the captain's catalogue, "brief": {role: summary line}}
STRATEGIES = {
    "BALANCED_DEFAULT": {
        "summary": "the formation's normal game — phase behaviour as briefed, no extra emphasis",
        "brief": {
            "GK": "play your normal game",
            "DEF": "play your normal game",
            "ML": "play your normal game",
            "MR": "play your normal game",
            "FWD": "play your normal game",
        },
    },
    "HIGH_PRESS": {
        "summary": "win the ball in their half; accept space behind — use when chasing or when they crumble under pressure",
        "brief": {
            "GK": "hold a higher line (~25 out) to sweep the space behind the press",
            "DEF": "step up to the halfway line; press the carrier 5 units earlier than usual",
            "ML": "press their build-up in THEIR half — in your lane press the carrier at 0.9",
            "MR": "press their build-up in THEIR half — in your lane press the carrier at 0.9",
            "FWD": "press their GK and DEF on every distribution; force the long ball",
        },
    },
    "LOW_BLOCK": {
        "summary": "concede territory, protect the box; use when protecting a lead or being overrun",
        "brief": {
            "GK": "stay on your line; sweep only inside ~20 units",
            "DEF": "drop 10 units deeper than usual; never leave the central lane",
            "ML": "both mids sit within 20 units of DEF; screen passes, do not chase wide",
            "MR": "both mids sit within 20 units of DEF; screen passes, do not chase wide",
            "FWD": "stay central near halfway as the outlet; do not press past their DEF",
        },
    },
    "DIRECT_COUNTER": {
        "summary": "skip the build-up: first pass forward, FWD runs early — use against a team that overcommits",
        "brief": {
            "GK": "distribute long (KICK) to the FWD or pushing mid, never short",
            "DEF": "first pass forward on winning the ball — a THROUGH to FWD beats a safe square ball",
            "ML": "on any turnover sprint the wide channel immediately, don't wait for COUNTER phase",
            "MR": "on any turnover sprint the wide channel immediately, don't wait for COUNTER phase",
            "FWD": "stay on the last defender's shoulder; every won ball is coming to you fast",
        },
    },
    "POSSESS_WIDE": {
        "summary": "keep the ball, stretch them side to side — use when ahead on play but not on goals",
        "brief": {
            "GK": "always offer the safe pass back; distribute short (THROW) to restart the pattern",
            "DEF": "hold the ball an extra beat; switch the play to the far channel when it comes back",
            "ML": "hold maximum width (~20 off centre) in POSSESS; pull their block apart",
            "MR": "hold maximum width (~20 off centre) in POSSESS; pull their block apart",
            "FWD": "keep moving across the last line; the chance comes from the switch, not the dribble",
        },
    },
}


def valid(name) -> bool:
    return name in STRATEGIES


def role_brief(name: str, role: str) -> str:
    entry = STRATEGIES.get(name) or STRATEGIES[DEFAULT]
    return entry["brief"].get(role, "play your normal game")


def catalogue() -> str:
    """The strategy menu, formatted for the captain's system prompt."""
    return "\n".join(f"- {name}: {s['summary']}" for name, s in STRATEGIES.items())
