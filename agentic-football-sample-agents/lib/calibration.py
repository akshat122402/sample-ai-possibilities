"""Tunable constants for the diamond team, in one place so they can be measured.

Every number here started as a guess. None of them have been fitted against
match data yet — PROVENANCE records that, and it is the point of this file:
when telemetry.py's logs have been aggregated, a value moves from "unmeasured"
to a measurement with a date and a sample size, and nothing else in the codebase
has to change.

Physics in this game is a property of the engine, so these are constants, not
something to look up at runtime. They are learned between matches and compiled
in — see the README section on calibration for the loop.

To update one:
  1. run matches with TELEMETRY_ENABLED=1 (the default)
  2. aggregate the TELEMETRY lines out of CloudWatch (see lib/telemetry.py)
  3. change the value here and add its PROVENANCE entry
"""

from __future__ import annotations

# ── shooting ────────────────────────────────────────────────────────────────
# True distance to the goal centre, and how far off centre is still worth a
# shot. Fit these from kind="outcome" records where cmd="SHOOT": bucket by
# distGoal and |lateral|, and take the distance at which conversion stops
# beating the expected value of passing.
FWD_SHOOT_DIST = 24.0
FWD_SHOOT_LATERAL = 20.0
MID_SHOOT_DIST = 26.0
MID_SHOOT_LATERAL = 18.0

# ── counter-attack ──────────────────────────────────────────────────────────
# How long a break stays live, and how many opponents must be caught behind the
# ball to call it one. Fit COUNTER_HOLD_SECONDS from how long it actually takes
# a beaten defence to get goal-side again: track opponentsBeaten in the
# kinematics stream after a turnover and find where it decays.
COUNTER_HOLD_SECONDS = 4.0
COUNTER_MIN_OPPONENTS_BEATEN = 2

# ── pressing and defending ──────────────────────────────────────────────────
# DEF_PRESS_RADIUS: how close the carrier must be before the defender leaves
# shape. Fit from outcome records where cmd="PRESS_BALL", bucketed by the
# distance at which it was issued — the radius where winning the ball stops
# being more likely than being beaten.
DEF_PRESS_RADIUS = 18.0
MID_PRESS_INTENSITY = 0.75
DEF_PRESS_INTENSITY = 0.8

# FWD presses only when the opponent is still building from deep; this is the
# carrier's progress toward our goal, so a larger number means further from us.
FWD_PRESS_MIN_PROGRESS = 18.0

# ── computed tactics (tactical_tools.py) ────────────────────────────────────
# PASS_LANE_RADIUS: how close to the passing lane an opponent must be to count
# as interception risk. Fit from outcome records where cmd="PASS": bucket by
# the lane distance of the nearest opponent and find where completion drops.
PASS_LANE_RADIUS = 8.0

# ── spacing ─────────────────────────────────────────────────────────────────
# Minimum separation between team-mates (prompt-enforced). Fit from kinematics:
# the pairwise distance below which two team-mates' pass options collapse.
TEAMMATE_SPACING = 10.0

# ── captain ─────────────────────────────────────────────────────────────────
# How often the captain reviews the match. Each review costs the GK one slow
# tick, so shorter periods trade goalkeeper responsiveness for adaptability.
CAPTAIN_PERIOD_SECONDS = 20.0

# ── goalkeeper line positioning (tactical_tools.gk_line_target) ─────────────
# Depth = how far off the goal centre the GK stands along the ball-goal line.
# Fit from outcome records where role="GK": shots conceded vs depth at the time.
GK_DEPTH_ATTACK = 10.0    # we have the ball — step out to sweep behind DEF
GK_DEPTH_DEFEND = 5.0     # they have the ball in our half
GK_MAX_FROM_GOAL = 18.0   # hard cap, both styles
GK_BEHIND_DEF_MARGIN = 6.0  # always this much closer to goal than DEF

# ── goalkeeper ──────────────────────────────────────────────────────────────
# GK_SWEEP_RADIUS: how far out the keeper will come once the carrier is past
# the defender. This is the formation's biggest single risk, so it is the value
# most worth measuring — too small concedes through balls, too large concedes
# lobs. Fit from outcome records where cmd="INTERCEPT" and role="GK".
GK_SWEEP_RADIUS = 30.0
GK_LOOSE_BALL_RADIUS = 16.0

# ── outcome resolution windows (seconds) ────────────────────────────────────
# How long telemetry waits for a decision to resolve before recording it as
# unresolved. These are bookkeeping, not tactics, but they bound the data.
RESOLVE_WINDOW = {
    "SHOOT": 3.0,
    "PASS": 3.0,
    "GK_DISTRIBUTE": 3.0,
    "MOVE_TO": 1.5,
    "PRESS_BALL": 4.0,
    "INTERCEPT": 4.0,
    "MARK": 4.0,
    "SLIDE_TACKLE": 2.0,
    "FOLLOW_PLAYER": 4.0,
    "SET_STANCE": 0.5,
}
DEFAULT_RESOLVE_WINDOW = 3.0


PROVENANCE = {
    "FWD_SHOOT_DIST": "unmeasured — initial guess",
    "FWD_SHOOT_LATERAL": "unmeasured — initial guess",
    "MID_SHOOT_DIST": "unmeasured — initial guess",
    "MID_SHOOT_LATERAL": "unmeasured — initial guess",
    "COUNTER_HOLD_SECONDS": "unmeasured — initial guess",
    "COUNTER_MIN_OPPONENTS_BEATEN": "unmeasured — initial guess",
    "DEF_PRESS_RADIUS": "unmeasured — initial guess",
    "MID_PRESS_INTENSITY": "unmeasured — initial guess",
    "DEF_PRESS_INTENSITY": "unmeasured — initial guess",
    "FWD_PRESS_MIN_PROGRESS": "unmeasured — initial guess",
    "GK_SWEEP_RADIUS": "unmeasured — initial guess",
    "GK_LOOSE_BALL_RADIUS": "unmeasured — initial guess",
    "PASS_LANE_RADIUS": "unmeasured — inherited from the gateway tool's guess",
    "CAPTAIN_PERIOD_SECONDS": "unmeasured — initial guess",
    "GK_DEPTH_ATTACK": "unmeasured — initial guess",
    "GK_DEPTH_DEFEND": "unmeasured — initial guess",
    "GK_MAX_FROM_GOAL": "unmeasured — initial guess",
    "GK_BEHIND_DEF_MARGIN": "unmeasured — initial guess",
    "TEAMMATE_SPACING": "unmeasured — initial guess",
}


def unmeasured() -> list[str]:
    """Constants still carrying their initial guess."""
    return sorted(k for k, v in PROVENANCE.items() if v.startswith("unmeasured"))
