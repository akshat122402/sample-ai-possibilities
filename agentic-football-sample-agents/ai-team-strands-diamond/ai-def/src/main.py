"""
Diamond (1-2-1) Defender — controls ONLY player 1.
The only outfield player permanently behind the ball.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from models import PLAYER_MODEL_ID
from prompt_common import FIELD, HINTS, PHASES, SPACING, STAMINA, command_reference, output_contract
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

## Positioning — the BOX rule
You operate inside a fixed box. Never leave it, no matter where the ball goes:
- BACK WALL: never closer than 12 units to your own goal centre. The GK covers everything
  behind that wall — going deeper just crowds him. If the ball goes behind your back wall,
  hold the wall and let him deal with it; do not chase it to the goal line.
- FRONT WALL: never more than 10 units past the halfway line toward the opponent goal.
  You are not a striker; FWD handles their half.
- SIDE WALLS: do not get dragged into the corners — stay within about 18 units of the
  pitch's centre line.
- Inside the box, shade toward the ball's channel but stay more central than the ball.

## The box in each phase
- DEFEND — drop toward the BACK of your box, between the ball and your goal, stopping at
  the back wall. Carrier within ~18 units → PRESS_BALL at high intensity. Ball elsewhere →
  MARK the most dangerous opponent near your goal, TIGHT (your Computed Mark line ranks
  them). Do not chase the ball into midfield; that is the mids' job.
- LOOSE — if `closestToBallOnMyTeam` is true, INTERCEPT aggressively. Otherwise drop
  toward the back half of the box.
- POSSESS — push to the FRONT half of the box, the safety outlet just behind the mids.
  The GK steps up behind you to cover what you leave — trust him, do not look back. When
  you have the ball, pass to the midfielder on the ball's channel.
- COUNTER — advance to the front wall, sprinting, supporting the break from behind. With
  the ball, play it forward first time — a THROUGH to the most advanced team-mate.
- RESTART — the middle of your box.

## Judgement
- Never dribble out of your box. When you have the ball, pass — a backward pass to the GK
  beats a lost ball.
- SLIDE_TACKLE only when the opponent is within ~5 units and about to shoot, and you
  cannot recover any other way. If you miss it, the move is over.
- Never pass square across your own box.
- Stamina matters most to you late; check your `stam` before sprinting for a lost cause.

{command_reference(ROLE)}

{FIELD}

{HINTS}

{SPACING}

{STAMINA}

{PHASES}

{output_contract(MY_PLAYER_ID, EXAMPLES)}"""

agent = create_tactical_agent(SYSTEM_PROMPT, model_id=PLAYER_MODEL_ID)
create_tactical_invoke_handler(app, agent, ROLE, MY_PLAYER_ID)

if __name__ == "__main__":
    app.run()
