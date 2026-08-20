"""Bedrock model IDs for the diamond team, in one place.

The players run Claude Haiku 4.5: a football tick is a reflex, and Haiku is the
strongest model that still answers inside the ~1s budget once time-to-first-token
plus ~60 output tokens are paid for. The captain reviews the match every
CAPTAIN_PERIOD_SECONDS, not every tick, so it can afford a Sonnet-tier model.

Both are cross-region inference profiles. If your account exposes different IDs
(check `aws bedrock list-inference-profiles`), override with the environment
variables rather than editing five main.py files.
"""

from __future__ import annotations

import os

PLAYER_MODEL_ID = os.environ.get(
    "PLAYER_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

CAPTAIN_MODEL_ID = os.environ.get(
    "CAPTAIN_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
