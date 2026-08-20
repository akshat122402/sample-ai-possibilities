"""Shared team blackboard — the one cross-runtime channel this team has.

Each of the five agents runs in its own AgentCore runtime, so module globals
never cross a position boundary. This module is the bridge: one DynamoDB
partition per team holding a handful of tiny items,

    {"teamId": 0, "entry": "plan"}      — the captain's current strategy + stance
    {"teamId": 0, "entry": "stats#3"}   — player 3's cumulative outcome tallies

The captain writes the plan and reads everyone's stats; every player reads the
plan (cached ~2s, so the tick path costs one DynamoDB read every couple of
seconds, not sixty a second) and publishes its own stats (throttled to one
write per ~5s).

Wiring: the CDK stack creates the table and injects TEAM_BLACKBOARD_TABLE into
every runtime. Everything here is best-effort and degrades to None/no-op when
the table is missing or unreachable — without it the team plays exactly the
football it played before the blackboard existed. boto3 arrives transitively
with bedrock-agentcore.
"""

from __future__ import annotations

import json
import os
import time

PLAN_CACHE_SECONDS = 2.0
STATS_PUBLISH_SECONDS = 5.0

# [table | None (unresolved) | False (unavailable)]
_table = [None]
_plan_cache: dict = {}      # team_id -> {"at": monotonic, "plan": dict|None}
_last_stats_pub: dict = {}  # (team_id, player_id) -> monotonic


def reset() -> None:
    """Forget the resolved table and caches — used by tests."""
    _table[0] = None
    _plan_cache.clear()
    _last_stats_pub.clear()


def set_table_for_tests(table) -> None:
    _table[0] = table if table is not None else False


def _resolve_table():
    if _table[0] is not None:
        return _table[0] or None
    name = os.environ.get("TEAM_BLACKBOARD_TABLE")
    if not name:
        _table[0] = False
        return None
    try:
        import boto3
        from botocore.config import Config
        # The tick budget is ~1s; a blackboard hiccup must cost milliseconds,
        # not boto3's default multi-second timeouts and retries.
        cfg = Config(connect_timeout=0.4, read_timeout=0.4, retries={"max_attempts": 1})
        _table[0] = boto3.resource("dynamodb", config=cfg).Table(name)
    except Exception:
        _table[0] = False
    return _table[0] or None


# ── plan: captain writes, players read ───────────────────────────────────────

def publish_plan(team_id: int, strategy: str, stance: int, reason: str, game_time: float) -> None:
    table = _resolve_table()
    plan = {"strategy": strategy, "stance": int(stance), "reason": reason,
            "updatedAt": f"{game_time:.0f}"}
    # The captain's own runtime sees the new plan immediately either way.
    _plan_cache[team_id] = {"at": time.monotonic(), "plan": plan}
    if table is None:
        return
    try:
        table.put_item(Item={"teamId": int(team_id), "entry": "plan", **plan})
    except Exception:
        pass


def read_plan(team_id: int):
    """The captain's current plan, or None. Cached for PLAN_CACHE_SECONDS."""
    cached = _plan_cache.get(team_id)
    if cached is not None and time.monotonic() - cached["at"] < PLAN_CACHE_SECONDS:
        return cached["plan"]
    table = _resolve_table()
    plan = cached["plan"] if cached else None
    if table is not None:
        try:
            item = table.get_item(Key={"teamId": int(team_id), "entry": "plan"}).get("Item")
            if item:
                plan = {"strategy": str(item.get("strategy", "")),
                        "stance": int(item.get("stance", 0)),
                        "reason": str(item.get("reason", "")),
                        "updatedAt": str(item.get("updatedAt", ""))}
        except Exception:
            pass  # keep the stale plan rather than dropping to None on a blip
    _plan_cache[team_id] = {"at": time.monotonic(), "plan": plan}
    return plan


# ── stats: players write, captain reads ──────────────────────────────────────

def publish_stats(team_id: int, player_id: int, tally: dict) -> None:
    """Player's cumulative outcome tallies ({cmd: [attempts, ok]}). Throttled."""
    if not tally:
        return
    table = _resolve_table()
    if table is None:
        return
    key = (team_id, player_id)
    now = time.monotonic()
    if now - _last_stats_pub.get(key, -1e9) < STATS_PUBLISH_SECONDS:
        return
    _last_stats_pub[key] = now
    try:
        table.put_item(Item={
            "teamId": int(team_id), "entry": f"stats#{int(player_id)}",
            "tally": json.dumps(tally),
        })
    except Exception:
        pass


def read_stats(team_id: int) -> dict:
    """All players' tallies, {player_id: {cmd: [attempts, ok]}}. Empty on failure."""
    table = _resolve_table()
    if table is None:
        return {}
    try:
        try:
            from boto3.dynamodb.conditions import Key
            cond = Key("teamId").eq(int(team_id))
        except ImportError:  # test fakes match on the bare team id
            cond = int(team_id)
        items = table.query(KeyConditionExpression=cond).get("Items", [])
    except Exception:
        return {}
    stats = {}
    for item in items:
        entry = str(item.get("entry", ""))
        if not entry.startswith("stats#"):
            continue
        try:
            stats[int(entry.split("#", 1)[1])] = json.loads(item["tally"])
        except (KeyError, ValueError):
            continue
    return stats
