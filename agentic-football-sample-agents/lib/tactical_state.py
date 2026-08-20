"""State summary for the diamond team.

A phase-aware sibling of state.summarize_state(). Kept separate rather than
changing the shared summariser because four sample teams read that output and
their prompts are written against its exact wording.

What this one adds, and why each matters:
  - stamina as a percentage   — the shared summary formats a 0.0-1.0 float with
                                "%.0f", so 0.95 and 0.65 both render as "1"
  - ball velocity             — present in the payload, dropped by the shared
                                summary, and required by INTERCEPT
  - true distance to goal     — the shared summary reports an x-axis gap, which
                                understates range badly from wide positions
  - opponents beaten          — the measurable half of "counter if it's on"
  - team-relative sides       — LEFT/RIGHT that mean the same thing both ways
  - playMode + set-piece owner— the shared summary prints the mode and explains
                                nothing; dead balls are otherwise played as if
                                the ball were live
"""

from __future__ import annotations

from phase import PhaseView, attack_dir, progress, side_of
from state import _is_my_team, _player_idx, dist, get_goal_positions

ROLE_NAMES = {0: "GK", 1: "DEF", 2: "ML", 3: "MR", 4: "FWD"}


def goal_centre(team_id: int, opponent: bool = True) -> dict:
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    return {"x": opp_goal_x if opponent else my_goal_x, "y": 0.0}


def dist_to_opp_goal(pos: dict, team_id: int) -> float:
    """True distance to the opponent goal centre, not the x-axis gap."""
    return dist(pos, goal_centre(team_id, opponent=True))


def dist_to_own_goal(pos: dict, team_id: int) -> float:
    return dist(pos, goal_centre(team_id, opponent=False))


def _pct(stamina) -> str:
    """Stamina arrives as 0.0-1.0; older payloads used 0-100."""
    try:
        v = float(stamina)
    except (TypeError, ValueError):
        return "?"
    if v <= 1.0:
        v *= 100.0
    return f"{v:.0f}%"


def _speed(vel: dict) -> float:
    return (vel.get("x", 0) ** 2 + vel.get("y", 0) ** 2) ** 0.5


def summarize_tactical_state(
    game_state: dict,
    team_id: int,
    my_player_id: int,
    role: str,
    view: PhaseView,
) -> str:
    """Build the per-tick user message for one diamond agent."""
    ball = game_state.get("ball", {})
    ball_pos = ball.get("position", {"x": 0, "y": 0})
    ball_vel = ball.get("velocity", {"x": 0, "y": 0})
    score = game_state.get("score", {})
    players = game_state.get("players", [])
    my_goal_x, opp_goal_x = get_goal_positions(team_id)

    mine = sorted([p for p in players if _is_my_team(p, team_id)], key=_player_idx)
    opponents = sorted([p for p in players if not _is_my_team(p, team_id)], key=_player_idx)
    me = next((p for p in mine if _player_idx(p) == my_player_id), None)
    my_pos = me.get("position", {"x": 0, "y": 0}) if me else {"x": 0, "y": 0}

    if view.holder_id is None:
        holder = "nobody (ball is free)"
    elif view.holder_is_mine:
        holder = f"MY player {view.holder_id}"
    else:
        holder = f"OPPONENT player {view.holder_id}"

    mode_owner = game_state.get("modeTeamId")
    mode_note = ""
    if mode_owner is not None:
        mode_note = " (ours)" if mode_owner == team_id else " (theirs)"

    # telemetry imports this module's goal helpers, so it can only be imported
    # here; blackboard and strategy ride along to keep the plan lookup together.
    import blackboard
    import strategy as strategy_mod
    import telemetry

    lines = [
        f"PHASE: {view.phase} — {view.reason}",
    ]

    plan = blackboard.read_plan(team_id)
    if plan:
        if strategy_mod.valid(plan.get("strategy")) and plan["strategy"] != strategy_mod.DEFAULT:
            lines.append(
                f"STRATEGY: {plan['strategy']} (captain) — {strategy_mod.role_brief(plan['strategy'], role)}"
            )
        stance = plan.get("stance", 0)
        if stance in (1, 2):
            lines.append(f"STANCE: {strategy_mod.STANCE_NAMES[stance]} (captain)")

    lines += [
        f"Time {float(game_state.get('gameTime', 0)):.0f}s | Score {score.get('home', 0)}-{score.get('away', 0)} "
        f"| You are {'HOME' if team_id == 0 else 'AWAY'} | PlayMode {game_state.get('playMode', 'OPEN_PLAY')}{mode_note}",
        f"You attack toward x={opp_goal_x:+.0f}, you defend x={my_goal_x:+.0f}.",
        f"Ball ({ball_pos.get('x', 0):.1f},{ball_pos.get('y', 0):.1f}) "
        f"vel=({ball_vel.get('x', 0):.1f},{ball_vel.get('y', 0):.1f}) speed={_speed(ball_vel):.1f} "
        f"| channel {view.ball_side} | held by {holder}",
        f"Opponents beaten (behind the ball): {view.opponents_beaten} of {len(opponents)}",
        "",
    ]

    if me:
        lines.append(
            f">>> YOU ({role}, id={my_player_id}) pos=({my_pos.get('x', 0):.1f},{my_pos.get('y', 0):.1f}) "
            f"channel={side_of(my_pos.get('y', 0), team_id)} stam={_pct(me.get('stamina', 1.0))} "
            f"hasBall={'true' if view.i_have_ball else 'false'} "
            f"distToBall={dist(my_pos, ball_pos):.1f} "
            f"distToOppGoal={dist_to_opp_goal(my_pos, team_id):.1f} "
            f"closestToBallOnMyTeam={'true' if view.i_am_nearest_to_ball else 'false'}"
        )
        patterns = telemetry.recent_patterns(team_id, my_player_id)
        if patterns:
            lines.append(f"Your recent outcomes (worked/tried): {patterns}")
        lines.append("")

    lines.append("Teammates:")
    for p in mine:
        pid = _player_idx(p)
        if pid == my_player_id:
            continue
        pos = p.get("position", {})
        lines.append(
            f"  {ROLE_NAMES.get(pid, f'P{pid}'):3s}(id={pid}) ({pos.get('x', 0):.1f},{pos.get('y', 0):.1f}) "
            f"channel={side_of(pos.get('y', 0), team_id)} stam={_pct(p.get('stamina', 1.0))} "
            f"distToOppGoal={dist_to_opp_goal(pos, team_id):.1f} distToBall={dist(pos, ball_pos):.1f}"
        )

    lines.append("")
    lines.append("Opponents:")
    for p in opponents:
        pid = _player_idx(p)
        pos = p.get("position", {})
        beaten = progress(pos.get("x", 0), team_id) < view.ball_progress
        lines.append(
            f"  P{pid} ({pos.get('x', 0):.1f},{pos.get('y', 0):.1f}) "
            f"distToMe={dist(pos, my_pos):.1f} distToOurGoal={dist_to_own_goal(pos, team_id):.1f} "
            f"behindBall={'yes' if beaten else 'no'}"
            + (" <-- has the ball" if pid == view.opp_carrier_id else "")
        )

    # Imported here, not at the top: tactical_tools imports goal_centre and
    # ROLE_NAMES from this module.
    from tactical_tools import tactical_hints

    hints = tactical_hints(game_state, team_id, my_player_id, role, view)
    if hints:
        lines.append("")
        lines.append("Computed (deterministic — trust these numbers):")
        lines.extend(f"  {h}" for h in hints)

    return "\n".join(lines)
