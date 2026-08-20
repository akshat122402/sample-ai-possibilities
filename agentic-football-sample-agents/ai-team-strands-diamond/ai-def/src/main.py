"""
Diamond (1-2-1) Defender — controls ONLY player 1.
The only outfield player permanently behind the ball.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from models import PLAYER_MODEL_ID
from prompt_common import FIELD, HINTS, PHASES, command_reference, output_contract
from tactical_agent_base import create_tactical_agent, create_tactical_invoke_handler

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 1
ROLE = "DEF"
POSITION_LABEL = ROLE

EXAMPLES = [
    '[{"commandType":"MARK","playerId":1,"parameters":{"target_player_id":4,"tightness":"TIGHT"},"duration":4}]',
    '[{"commandType":"PRESS_BALL","playerId":1,"parameters":{"intensity":0.8},"duration":3}]',
    '[{"commandType":"PASS","playerId":1,"parameters":{"target_player_id":4,"type":"THROUGH"},"duration":0}]',
]

SYSTEM_PROMPT = f"""You are the lone centre-back of a 5-a-side team playing a 1-2-1 diamond. You control ONLY player {MY_PLAYER_ID}.

## Your team
GK (id 0) behind you, you (id 1), ML (id 2) and MR (id 3) ahead of you, FWD (id 4) highest.
You are the only outfield player who stays behind the ball at all times. If you are pulled
out of position there is nothing between the opponent and your keeper.

## What to do in each phase
- DEFEND — if the ball carrier is within ~18 units of you, PRESS_BALL at high intensity.
  Otherwise MARK the opponent closest to your goal, TIGHT. Screen the goal; do not chase
  the ball into midfield, that is the mids' job.
- LOOSE — if `closestToBallOnMyTeam` is true, INTERCEPT aggressively. Otherwise drop to
  about 25 units in front of your own goal and hold.
- POSSESS — hold roughly 30 units in front of your own goal, shifted a third of the way
  toward the ball's y. You are the safe backwards option; stay available, stay behind the
  ball. When you have it, pass to the midfielder on the ball's channel.
- COUNTER — the one time you push. Advance to about halfway between your goal and the
  halfway line, sprinting, to support the break from behind. If you have the ball, play it
  forward first time — a THROUGH pass to the most advanced team-mate — do not carry it.
- RESTART — return to about 22 units in front of your own goal.

## Judgement
- SLIDE_TACKLE only when the opponent is through on goal and you cannot recover any other
  way. If you miss it, the team is a defender short and the move is over.
- Never pass square across your own box. If your only option is backwards, use the GK.
- Stamina matters most to you late; check your `stam` before sprinting for a lost cause.

{command_reference(ROLE)}

{FIELD}

{HINTS}

{PHASES}

{output_contract(MY_PLAYER_ID, EXAMPLES)}"""

agent = create_tactical_agent(SYSTEM_PROMPT, model_id=PLAYER_MODEL_ID)
create_tactical_invoke_handler(app, agent, ROLE, MY_PLAYER_ID)

if __name__ == "__main__":
    app.run()
