"""Match-phase classification — the shared "captain" for the diamond team.

Every agent calls classify_phase() on the same game state and gets the same
answer, so all five players act on one plan without talking to each other.
There is no inter-agent channel in this architecture (parse_commands stamps
each command with the issuing player's own id, and no command type writes to
teamChat), so a deterministic pure function is the only coordination that is
both instant and guaranteed consistent.

Phases:
    RESTART  — dead ball (kickoff, throw-in, corner, goal kick)
    COUNTER  — we just won the ball with opponents caught upfield
    POSSESS  — we have the ball in a settled attack
    LOOSE    — nobody has the ball
    DEFEND   — the opponent has the ball

Direction handling: every position is normalised to "progress", the distance
travelled toward the opponent goal, so the same rule works for HOME and AWAY.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from state import _is_my_team, _player_idx, _possession_idx, dist, get_goal_positions

RESTART = "RESTART"
COUNTER = "COUNTER"
POSSESS = "POSSESS"
LOOSE = "LOOSE"
DEFEND = "DEFEND"

# How long a counter stays live once triggered. The transition is worth a few
# seconds of committed running; after that the opponent has recovered shape and
# it is an ordinary attack.
COUNTER_HOLD_SECONDS = 4.0

# Opponents caught on the wrong side of the ball before a turnover counts as a
# counter-attack opportunity.
COUNTER_MIN_OPPONENTS_BEATEN = 2


def attack_dir(team_id: int) -> int:
    """+1 if this team attacks toward +x, -1 if toward -x."""
    return 1 if team_id == 0 else -1


def progress(x: float, team_id: int) -> float:
    """Distance travelled toward the opponent goal, sign-normalised per team."""
    return attack_dir(team_id) * x


def side_of(y: float, team_id: int) -> str:
    """Team-relative channel: LEFT / RIGHT / CENTRE, facing the opponent goal.

    Defined explicitly because "left" is otherwise meaningless — it flips with
    the direction of attack, and the balanced team's prompts never define it.
    """
    lateral = attack_dir(team_id) * y
    if lateral < -6:
        return "LEFT"
    if lateral > 6:
        return "RIGHT"
    return "CENTRE"


@dataclass
class PhaseView:
    """Everything derived from the game state that the prompts and the fallback
    both key off. Built once per tick and shared by both, so the LLM and the
    rule-based layer can never disagree about the situation."""

    phase: str = LOOSE
    we_have_ball: bool = False
    holder_id: int | None = None
    holder_is_mine: bool = False
    possession_changed: bool = False
    opponents_beaten: int = 0
    ball_side: str = "CENTRE"
    ball_progress: float = 0.0
    i_have_ball: bool = False
    seconds_since_turnover: float | None = None
    reason: str = ""
    nearest_teammate_to_ball: int | None = None
    i_am_nearest_to_ball: bool = False
    opp_carrier_id: int | None = None
    opp_carrier_progress: float | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tick memory
#
# The AgentCore container is reused between ticks, so a module global survives
# from one invocation to the next and gives us the previous frame — enough to
# detect a turnover without adding an AgentCore Memory resource. If the
# container is recycled we simply see no transition on the first tick after,
# which degrades to POSSESS/DEFEND rather than misfiring. Keyed by team so one
# runtime serving both sides cannot cross-contaminate.
# ---------------------------------------------------------------------------

_prev: dict[int, dict] = {}


def reset_memory() -> None:
    """Clear tick memory — used by tests, and safe to call between matches."""
    _prev.clear()


def _play_mode_is_open(play_mode) -> bool:
    """The game server sends a string ("OPEN_PLAY"); older payloads sent 0."""
    if isinstance(play_mode, str):
        return play_mode.upper() in ("OPEN_PLAY", "OPENPLAY", "PLAY")
    return play_mode in (0, None)


def resolve_holder(ball: dict, players: list) -> tuple[int | None, dict | None]:
    """Find the player actually holding the ball.

    Both teams number their players 0-4 and possessionAgentId carries only that
    index ("agentId_1"), so the id alone matches one player per team. The shared
    helpers take the first match, which is always the HOME player — so an AWAY
    player carrying the ball reads as HOME possession. Where an id is ambiguous,
    the candidate closest to the ball is the holder.
    """
    idx = _possession_idx(ball)
    if idx is None:
        return None, None
    candidates = [p for p in players if _player_idx(p) == idx]
    if not candidates:
        return idx, None
    if len(candidates) == 1:
        return idx, candidates[0]
    ball_pos = ball.get("position", {})
    return idx, min(candidates, key=lambda p: dist(p.get("position", {}), ball_pos))


def classify_phase(game_state: dict, team_id: int, my_player_id: int) -> PhaseView:
    """Classify the current tick. Pure apart from the turnover memory."""
    ball = game_state.get("ball", {})
    ball_pos = ball.get("position", {"x": 0, "y": 0})
    players = game_state.get("players", [])
    game_time = float(game_state.get("gameTime", 0) or 0)

    view = PhaseView()
    view.ball_side = side_of(ball_pos.get("y", 0), team_id)
    view.ball_progress = progress(ball_pos.get("x", 0), team_id)

    holder_id, holder = resolve_holder(ball, players)
    view.holder_id = holder_id
    view.holder_is_mine = bool(holder is not None and _is_my_team(holder, team_id))
    view.we_have_ball = view.holder_is_mine
    view.i_have_ball = view.holder_is_mine and holder_id == my_player_id

    if holder is not None and not view.holder_is_mine:
        view.opp_carrier_id = holder_id
        view.opp_carrier_progress = progress(holder.get("position", {}).get("x", 0), team_id)

    # Opponents caught on the wrong side of the ball.
    view.opponents_beaten = sum(
        1
        for p in players
        if not _is_my_team(p, team_id)
        and progress(p.get("position", {}).get("x", 0), team_id) < view.ball_progress
    )

    # Who on our side is closest to the ball — decides who chases a loose ball.
    mine = [p for p in players if _is_my_team(p, team_id)]
    if mine:
        nearest = min(mine, key=lambda p: dist(p.get("position", {}), ball_pos))
        view.nearest_teammate_to_ball = _player_idx(nearest)
        view.i_am_nearest_to_ball = view.nearest_teammate_to_ball == my_player_id

    # --- turnover detection against the previous tick -----------------------
    owner = "us" if view.holder_is_mine else ("them" if holder is not None else None)
    prev = _prev.get(team_id)
    counter_until = prev.get("counter_until", 0.0) if prev else 0.0

    if prev is not None and prev.get("owner") != owner:
        view.possession_changed = True
        if owner == "us" and prev.get("owner") == "them":
            if view.opponents_beaten >= COUNTER_MIN_OPPONENTS_BEATEN:
                counter_until = game_time + COUNTER_HOLD_SECONDS
    if owner != "us":
        counter_until = 0.0  # a counter dies the moment the ball is lost

    _prev[team_id] = {"owner": owner, "counter_until": counter_until, "time": game_time}

    # --- phase --------------------------------------------------------------
    if not _play_mode_is_open(game_state.get("playMode", 0)):
        view.phase = RESTART
        view.reason = "dead ball"
    elif owner == "us" and game_time < counter_until:
        view.phase = COUNTER
        view.seconds_since_turnover = max(0.0, COUNTER_HOLD_SECONDS - (counter_until - game_time))
        view.reason = f"{view.opponents_beaten} opponents behind the ball"
    elif owner == "us":
        view.phase = POSSESS
        view.reason = "settled possession"
    elif owner is None:
        view.phase = LOOSE
        view.reason = "ball is free"
    else:
        view.phase = DEFEND
        view.reason = "opponent in possession"

    return view
