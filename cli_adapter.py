import json, os, subprocess, shlex, re
from typing import Dict, Any

_JSON_RE = re.compile(r'(\{.*\})', re.DOTALL)

def _parse_stdout(stdout: str) -> Dict[str, Any]:
    matches = _JSON_RE.findall(stdout)
    if matches:
        for m in reversed(matches):
            try:
                obj = json.loads(m)
                return {
                    "text": obj.get("text", "").strip(),
                    "citations": obj.get("citations", []) or [],
                    "trace": {
                        "retrieved": (obj.get("trace", {}) or {}).get("retrieved", []) or [],
                        "latency": (obj.get("trace", {}) or {}).get("latency"),
                        "tokens": (obj.get("trace", {}) or {}).get("tokens"),
                        "tool_calls": (obj.get("trace", {}) or {}).get("tool_calls"),
                    },
                }
            except json.JSONDecodeError:
                continue
    return {
        "text": stdout.strip(),
        "citations": [],
        "trace": {"retrieved": [], "latency": None, "tokens": None, "tool_calls": None},
    }

def run_agent(prompt: str) -> Dict[str, Any]:
    agent_cmd = os.environ.get("AGENT_CMD")
    if not agent_cmd:
        raise RuntimeError("Set AGENT_CMD, e.g. $env:AGENT_CMD='python main.py --oneshot'")

    cmd = f'{agent_cmd} "{prompt}"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    stdout = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return _parse_stdout(stdout)
