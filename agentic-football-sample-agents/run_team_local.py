"""Run a full 5-agent team locally as HTTP endpoints (one port per agent).

    python run_team_local.py ai-team-strands-balanced          # serve on 8080-8084
    python run_team_local.py ai-team-strands-balanced --smoke  # serve + invoke once + exit

Each agent is a BedrockAgentCoreApp, so it exposes GET /ping and POST /invocations
exactly like the deployed AgentCore runtime:

    curl -s localhost:8081/invocations -H 'Content-Type: application/json' \
         -d '{"prompt": "{\"gameState\": {...}, \"teamId\": 0}"}'

Requires AWS creds + Bedrock access in a us-* region for real LLM output; without
them each agent still answers from its rule-based fallback layer.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

AGENT_ORDER = ["ai-gk", "ai-def", "ai-mid", "ai-ml", "ai-mr", "ai-fwd", "ai-fwd1", "ai-fwd2"]


def discover_agents(team_dir):
    """Agent directories in a team, in a stable back-to-front order."""
    found = [d for d in os.listdir(team_dir)
             if d.startswith("ai-") and os.path.isdir(os.path.join(team_dir, d, "src"))]
    return sorted(found, key=lambda d: (AGENT_ORDER.index(d) if d in AGENT_ORDER else 99, d))

BASE_PORT = int(os.environ.get("BASE_PORT", "8080"))
ROOT = os.path.dirname(os.path.abspath(__file__))

SERVE_SRC = '''
import importlib.util, os, sys
agent_dir, port = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, os.path.join(agent_dir, "src"))
sys.path.insert(0, agent_dir)  # gateway/memory bases staged next to src/
spec = importlib.util.spec_from_file_location(f"agent_{port}", os.path.join(agent_dir, "src", "main.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.app.run(port=port)
'''


def wait_healthy(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def sample_payload():
    sys.path.insert(0, os.path.join(ROOT, "lib"))
    from test_helpers import GAME_STATE, TEAM_ID
    return {"prompt": json.dumps({"gameState": GAME_STATE, "teamId": TEAM_ID})}


def main():
    team = sys.argv[1]
    smoke = "--smoke" in sys.argv
    agents = discover_agents(os.path.join(ROOT, team))
    procs = []
    try:
        for i, agent in enumerate(agents):
            port = BASE_PORT + i
            agent_dir = os.path.join(ROOT, team, agent)
            log = open(os.path.join(ROOT, f".{team}-{agent}.log"), "w")
            procs.append((agent, port, subprocess.Popen(
                [sys.executable, "-c", SERVE_SRC, agent_dir, str(port)],
                stdout=log, stderr=subprocess.STDOUT)))

        for agent, port, _ in procs:
            print(f"{'OK  ' if wait_healthy(port) else 'DEAD'} {agent:8s} http://127.0.0.1:{port}")

        if smoke:
            body = json.dumps(sample_payload()).encode()
            for agent, port, _ in procs:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/invocations", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    print(f"{agent:8s} -> {r.read().decode().strip()[:160]}")
            return

        print("\nCtrl+C to stop.")
        signal.pause()
    finally:
        for _, _, p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
