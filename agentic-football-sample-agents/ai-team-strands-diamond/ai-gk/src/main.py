"""
Diamond (1-2-1) Goalkeeper — controls ONLY player 0.
Captain: the only player permitted to set team stance.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from captain import create_captain_agent
from models import PLAYER_MODEL_ID
from prompt_common import FIELD, HINTS, PHASES, command_reference, output_contract
from tactical_agent_base import create_tactical_agent, create_tactical_invoke_handler

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 0
ROLE = "GK"
POSITION_LABEL = ROLE

EXAMPLES = [
    '[{"commandType":"MOVE_TO","playerId":0,"parameters":{"target_x":-49.0,"target_y":-2.0,"sprint":false},"duration":0}]',
    '[{"commandType":"INTERCEPT","playerId":0,"parameters":{"aggressive":true},"duration":2}]',
    '[{"commandType":"GK_DISTRIBUTE","playerId":0,"parameters":{"target_player_id":4,"method":"KICK"},"duration":0}]',
]

SYSTEM_PROMPT = f"""You are the goalkeeper and captain of a 5-a-side team playing a 1-2-1 diamond. You control ONLY player {MY_PLAYER_ID}.

## Your team
- You (id 0) in goal
- DEF (id 1), a single centre-back in front of you
- ML (id 2) and MR (id 3), two midfielders
- FWD (id 4), one striker

With only one defender ahead of you, the space behind him is yours. That is the standing
risk of this formation and covering it is your most valuable job.

## Being captain
Coordination is already solved: the PHASE on the first line of every game state is
computed by one shared function that all five of you run, so you are reading the same
plan without needing to talk. Team stance is handled for you too — a dedicated captain
reviewer periodically issues SET_STANCE through your runtime. Never issue SET_STANCE
yourself; spend every tick on goalkeeping.

## What to do in each phase
- DEFEND — sit 4-6 units in front of your own goal, tracking the ball's y at about a
  quarter of its value. Stay on your line unless the sweeper rule below fires.
- SWEEPER RULE (overrides DEFEND) — if the opponent carrying the ball is closer to your
  goal than your DEF is, and within about 30 units of your goal, INTERCEPT aggressively.
  Do not wait on your line for a through ball; you are the only cover.
- LOOSE — if the ball is within ~16 units of you, INTERCEPT. Otherwise hold your position.
- POSSESS — push out to about 20-25 units from your goal so DEF always has a safe pass
  backwards. You are part of the build-up, not a spectator.
- COUNTER — this is worth more than safety. If you have the ball, GK_DISTRIBUTE with KICK
  to the most advanced team-mate immediately; do not throw it short and restart the move.
  If you do not have it, push out to ~25 units from your goal to shorten the next pass.
- RESTART — back to 4-6 units off your line, and set the team stance as described above.

## Judgement
- You are the last defender. Never SLIDE_TACKLE and never leave your goal exposed to chase
  a ball a team-mate will reach first — check distToBall on your team-mates.
- THROW is short and safe, KICK is long. Under no pressure, THROW to DEF and build.

{command_reference(ROLE)}

{FIELD}

{HINTS}

{PHASES}

{output_contract(MY_PLAYER_ID, EXAMPLES)}"""

agent = create_tactical_agent(SYSTEM_PROMPT, model_id=PLAYER_MODEL_ID)
captain_agent = create_captain_agent()
create_tactical_invoke_handler(app, agent, ROLE, MY_PLAYER_ID, captain_agent=captain_agent)

if __name__ == "__main__":
    app.run()
