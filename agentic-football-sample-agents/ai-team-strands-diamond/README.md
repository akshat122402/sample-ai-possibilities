# AI Team (Strands) — Diamond 1-2-1

Five agents playing a 1-2-1 diamond: balanced by default, with a counter-attack phase that
fires on turnovers. Built on the same Strands + Bedrock AgentCore foundation as the other
sample teams in this folder, with a phase layer on top.

```
GK (0) — DEF (1) — ML (2) / MR (3) — FWD (4)
```

| id | Role | Model | Why |
|----|------|-------|-----|
| 0 | GK, captain | Nova Lite | Sweeper judgement needs more than Micro, but must stay fast |
| 1 | DEF | Nova Lite | Lone centre-back; its mistakes are the expensive ones |
| 2 | ML | Nova Micro | The phase layer removes most of the reasoning; speed wins |
| 3 | MR | Nova Micro | Same |
| 4 | FWD | Nova Pro | Final-third choices (shoot vs pass vs carry) have the widest option space |

## How coordination works

There is no agent-to-agent channel in this architecture. `parse_commands` stamps every
command with the issuing player's own id, no command type writes to `teamChat`, and each
agent is invoked independently per tick. A captain that issues orders is therefore not
possible.

Instead the captain is a pure function. `lib/phase.py` classifies every tick, and all five
agents call it on the same game state, so they reach the same conclusion with no messaging
and no latency:

| Phase | Trigger |
|-------|---------|
| `RESTART` | `playMode` is not open play |
| `COUNTER` | we just won the ball **and** at least 2 opponents are behind it — holds ~4s |
| `POSSESS` | we have the ball, settled |
| `LOOSE` | nobody has the ball |
| `DEFEND` | the opponent has the ball |

The GK's captaincy is the one team-wide lever the platform does support: it is the only
role permitted to issue `SET_STANCE`, which it spends at a restart based on the scoreline.

## The pusher rule

Exactly one midfielder pushes at a time; the other holds the centre as the pivot. The mid
on the ball's channel pushes, and when the ball is central the one nearer to it pushes.

This is the main change from "both mids flank when attacking". With four outfield players,
two midfielders leaving the middle at the same time strands DEF alone and the ball comes
back through the space they both left. The asymmetric version keeps a 1-1 spine at all
times and gives up very little width.

The other tactical adjustment: in `POSSESS` the mids hold *spacing*, not a fixed spot. A
diamond that stands still gives the carrier nothing to pass to.

## What this team fixes

Each of these is a defect in what the sample teams feed their models, addressed here in a
new module rather than by changing shared code the other four teams depend on.

| Problem | Where it was | Here |
|---------|--------------|------|
| Stamina rendered with `%.0f` on a 0-1 float, so 0.95 and 0.65 both print `1` | `state.py` | `tactical_state.py` prints `stam=65%` |
| `distOppGoal` is an x-axis gap, but prompts read it as shooting range | `state.py` | true distance to the goal centre, plus a lateral gate on shots |
| `INTERCEPT` asks the model to predict; ball velocity never reaches it | `state.py` | ball velocity and speed are in the summary |
| "when we lose possession", "after saves" — transitions invisible in a single frame | all prompts | `possession_changed` from tick memory |
| `playMode` printed but never explained; `modeTeamId` dropped | `state.py` | `RESTART` phase, and the set piece's owner is shown |
| left/right never defined, and mirrored wrong for AWAY | all prompts | team-relative `channel` on every player, computed once |
| `RESET` offered to all five agents with no guidance, though it is team-scoped | all prompts | refused for every role, in the prompt *and* in code |
| Commands listed that the position cannot use | all prompts | per-role command reference, enforced by `enforce_role` |
| Every prompt's only example is an on-ball action | all prompts | off-ball example first, three per role |
| Nothing warns the model against Python's `True`/`False` | all prompts | stated in the response contract |
| ~48% of each prompt is duplicated boilerplate | five `main.py` files | `lib/prompt_common.py` |
| Fallbacks identical across teams, so LLM failure silently reverts the tactics | `lib/fallback.py` | `tactical_fallback.py` is phase- and role-aware |
| Score and clock supplied but never used | all prompts | captain sets stance from the scoreline |

One more, found while testing: both teams number their players 0-4 and `possessionAgentId`
carries only that index, so `state.get_possession_info` — which takes the first match —
resolves any AWAY carrier to the HOME player of the same number. `phase.resolve_holder`
disambiguates by distance to the ball. **The shared helper still has this bug and the other
four teams still use it.**

## Local test

No AWS needed:

```bash
python3 ../lib/test_tactical.py      # phase classifier, geometry, fallbacks, whitelist
python3 ../lib/test_telemetry.py     # calibration records and outcome pairing
python3 ai-gk/test_local.py          # per agent
python3 ai-gk/test_local.py --llm    # adds one real Bedrock call
```

Serve all five as local HTTP endpoints on ports 8080-8084:

```bash
python3 ../run_team_local.py ai-team-strands-diamond --smoke
```

Without valid credentials in a `us-*` region every agent answers from its phase fallback —
which is the point of that layer, but check the logs for `agent error` before concluding
the model is working.

## Deploy

```bash
AWS_DEFAULT_REGION=us-east-1 python deploy_all.py
```

Runtime names are suffixed `_diamond_agent`, so this team can coexist with the other sample
teams in one account.

## Calibration

Every tunable number in this team lives in [`lib/calibration.py`](../lib/calibration.py),
and all twelve currently carry their initial guess — shot ranges, the counter window, press
radii, the keeper's sweep radius. `PROVENANCE` in that file records which have been
measured and which have not.

Physics here is a property of the game engine: identical in every match, opponent
independent, and a handful of numbers. That makes it worth learning — but *between*
matches, not during one. An in-match sample is too small to fit anything, an LLM is a poor
numeric aggregator, and a constant belongs in code rather than in a store queried on every
tick. So there is no runtime memory involved.

[`lib/telemetry.py`](../lib/telemetry.py) produces the data, through the logger the agents
already have — the runtime templates set `observability: enabled: true`, so the lines land
in CloudWatch with no new infrastructure. Three record kinds:

| Kind | Emitted | Carries |
|------|---------|---------|
| `kinematics` | every tick | player speed (engine-reported and displacement-derived), sprint flag, stamina and its rate, ball speed and acceleration |
| `decision` | every tick | the command chosen, its source (`llm` / `fallback` / `fallback_after_error`), and the features that should predict success: `distGoal`, `lateral`, `nearestOpp`, `stam`, `distBall` |
| `outcome` | when a tracked decision resolves | `goal` / `lost` / `completed` / `intercepted` / `won_ball_self` / … paired to its decision by `id`, with the decision's features repeated |

Only commands with an observable result open an outcome — shots, passes, distributions,
presses, tackles, interceptions. `MOVE_TO` does not: the kinematics stream already measures
movement, and one outcome per tick would bury the useful records. A maintained command
reissued every tick opens one pending record, not one per tick.

The loop:

1. Play matches with `TELEMETRY_ENABLED=1` (the default; set `0` to silence it).
2. Pull the records out of CloudWatch Logs Insights:

   ```
   fields @message
   | filter @message like /TELEMETRY/
   | parse @message 'TELEMETRY *' as body
   | filter body like /"kind":"outcome"/
   ```

3. Fit the constant — e.g. bucket `SHOOT` outcomes by `distGoal` and `lateral` and find
   where conversion stops beating the value of passing.
4. Update the value in `calibration.py` and replace its `PROVENANCE` entry with the date
   and sample size. Nothing else in the codebase changes.

Telemetry is best-effort by construction: every entry point swallows its own exceptions, so
a malformed payload costs you a record rather than a tick.

## Harnesses

The five agents deploy as **runtimes** — that is the real team, and it is what
runs the phase classifier, the state summariser, the role whitelist, the fallback
and the telemetry.

`build_harnesses.py` additionally generates a **harness** per agent, so each also
appears as a managed `AWS::BedrockAgentCore::Harness` resource:

```bash
python3 build_harnesses.py           # regenerate app/ and agentcore.json
python3 build_harnesses.py --check   # fail if stale (they are generated, not edited)
```

Each harness is `app/<name>/harness.json` plus an auto-discovered
`system-prompt.md`. The prompt is derived from the runtime agent's own
`SYSTEM_PROMPT`, so a tactical change reaches both; what the generator adds is an
input contract, because a harness receives the raw payload rather than the
summary `tactical_state.py` builds.

A harness runs a managed loop: a system prompt, a model, and declared tools. None
of this team's Python executes in one, so relative to the runtime it has

- no turnover detection — `COUNTER` is approximated from a single frame
- no role whitelist — nothing refuses `RESET` or an out-of-role command
- no rule-based fallback — a bad model response is simply lost
- no telemetry, and no clamping of `MOVE_TO` to the pitch

Both sets deploy from the same `deploy_all.py`. If you only want the runtimes,
delete the `harnesses` array from `agentcore/agentcore.json` and redeploy.

## Known limits

- **Turnover memory lives in the container.** `phase.py` keeps the previous tick in a module
  global, which survives between invocations but not a container recycle. After a recycle
  the first tick sees no transition and reads as `POSSESS`/`DEFEND` rather than misfiring a
  counter.
- **`myPlayers` can contradict the prompt.** If the payload assigns a different player than
  the agent was built for, the runtime honours the payload and logs a warning — but the
  system prompt still describes the configured player.
- **Engine semantics are unverified from this repo.** Whether `SET_STANCE` is per-player or
  team-scoped, and how long maintained commands persist, is visible only on the game server.
  The captaincy design assumes stance is worth setting once at a restart.
