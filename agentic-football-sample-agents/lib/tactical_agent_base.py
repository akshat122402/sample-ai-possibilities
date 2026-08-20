"""Agent factory and invoke handler for the diamond team.

Same three-layer structure as the shared agent_base (LLM → rule-based → last
resort), with three changes:

  1. The phase is computed before the model is called, and both the prompt and
     the fallback are written against it — so a dropped LLM response keeps the
     shape and the phase behaviour instead of silently reverting to a different
     formation.
  2. Commands are filtered against a per-role whitelist. The model is only shown
     the commands its role may issue, and this enforces it rather than trusting
     the prompt. RESET in particular is refused for everyone: it is team-scoped,
     so one hallucination from any of the five would clear the whole team's
     overrides.
  3. Only the GK may set team stance. That is the whole of the captaincy that
     this architecture can actually support — see phase.py for the rest.
"""

from __future__ import annotations

import json

from strands import Agent

from agent_base import create_agent as create_tactical_agent  # noqa: F401  (re-exported)
from parsing import parse_commands
from phase import classify_phase
from prompt_common import ROLE_COMMANDS
import telemetry
from tactical_fallback import LAST_RESORT, build_tactical_fallback
from tactical_state import summarize_tactical_state


def enforce_role(commands: list[dict], role: str) -> list[dict]:
    """Drop anything this role is not allowed to issue."""
    allowed = set(ROLE_COMMANDS[role])
    return [c for c in commands if c.get("commandType") in allowed]


def create_tactical_invoke_handler(
    app,
    agent: Agent,
    role: str,
    my_player_id: int,
):
    log = app.logger
    fallback = build_tactical_fallback(role)

    def _last_resort(team_id: int, player_id: int) -> list[dict]:
        cmd = dict(LAST_RESORT[role])
        cmd["parameters"] = dict(cmd["parameters"])
        cmd["playerId"] = player_id
        cmd["teamId"] = team_id
        return [cmd]

    @app.entrypoint
    async def invoke(payload, context):
        team_id, effective_pid = 0, my_player_id
        try:
            prompt = payload.get("prompt", "{}")
            prompt_data = json.loads(prompt) if isinstance(prompt, str) else prompt

            game_state = prompt_data.get("gameState", {})
            team_id = prompt_data.get("teamId", 0)

            my_players = prompt_data.get("myPlayers") or [my_player_id]
            effective_pid = my_players[0]
            if effective_pid != my_player_id:
                # The system prompt names this agent's player, so a payload that
                # reassigns it leaves the prompt describing someone else.
                log.warn(
                    f"{role} agent configured for player {my_player_id} but the payload "
                    f"assigned player {effective_pid}; the system prompt still describes "
                    f"player {my_player_id}"
                )

            view = classify_phase(game_state, team_id, effective_pid)
            telemetry.observe(log, game_state, team_id, effective_pid, role, view)
            summary = summarize_tactical_state(game_state, team_id, effective_pid, role, view)
            log.info(f"{role} p{effective_pid} team {team_id} phase={view.phase} ({view.reason})")

            def on_recovered(raw: str) -> None:
                log.warn(f"{role} recovered malformed JSON from the model: {raw[:200]}")

            response_text = str(agent(summary))
            commands = parse_commands(response_text, team_id, effective_pid, on_recovered)
            allowed = enforce_role(commands, role)

            if len(commands) != len(allowed):
                refused = [c.get("commandType") for c in commands if c not in allowed]
                log.warn(f"{role} refused commands outside its role: {refused}")

            if allowed:
                if len(allowed) > 1:
                    log.warn(f"{role} returned {len(allowed)} commands; using the first")
                chosen = allowed[:1]
                log.info(f"LLM: {chosen[0].get('commandType')} in phase {view.phase}")
                telemetry.record_decision(log, game_state, team_id, effective_pid, role,
                                          view, chosen[0], "llm")
                yield json.dumps(chosen)
                return

            log.warn(f"{role} LLM parse failed in phase {view.phase}; using fallback. "
                     f"Response: {response_text[:200]}")
            commands = fallback(game_state, team_id, effective_pid, view)
            telemetry.record_decision(log, game_state, team_id, effective_pid, role,
                                      view, commands[0], "fallback")
            yield json.dumps(commands)

        except Exception as e:
            log.error(f"{role} agent error: {e}")
            try:
                prompt_data = json.loads(payload.get("prompt", "{}"))
                game_state = prompt_data.get("gameState", {})
                team_id = prompt_data.get("teamId", 0)
                my_players = prompt_data.get("myPlayers") or [my_player_id]
                effective_pid = my_players[0]
                view = classify_phase(game_state, team_id, effective_pid)
                commands = fallback(game_state, team_id, effective_pid, view)
                telemetry.record_decision(log, game_state, team_id, effective_pid, role,
                                          view, commands[0], "fallback_after_error")
                yield json.dumps(commands)
            except Exception as inner:
                log.error(f"{role} fallback also failed: {inner}")
                yield json.dumps(_last_resort(team_id, effective_pid))

    return invoke
