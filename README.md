# pi-crsi-sim

Global Pi extension implementing Continual Recursive Self-Improvement (CRSI) with Self-Improvement Memory (SIM), per Section 3 of [arxiv 2611.18856v1](https://arxiv.org/abs/2611.18856).

## What it does

- **Harness optimization loop**: Propose → Evaluate → Select, with strict aggregate improvement promotion.
- **SIM (paper-exact)**: Only accepted edits appended to cause_id history. No accepted edit → SIM unchanged. Full untruncated SIM read by proposer and selector. Predictions before evaluation, measured outcomes after.
- **Source-to-target continuation**: `/crsi source` optimizes on a source benchmark; `/crsi target` continues on a target (shifted model/split) inheriting H + SIM with a fresh baseline.
- **Active by default**: Persisted H is injected into `before_agent_start`. Escape: `/crsi off`.
- **Background learning** (beyond paper): Records observations (tool errors, premature stops) as feedback for the next `/crsi` round. Does NOT auto-promote, does NOT modify SIM, does NOT replay chat externally.

## Setup

1. Install (symlink into Pi's global extensions directory):
   ```bash
   mkdir -p ~/.pi/agent/extensions
   ln -sfn ~/Projects/pi-crsi-sim ~/.pi/agent/extensions/crsi-sim
   ```

2. Start the local runtime model (required for CRSI operations):
   ```bash
   ~/.bin/lfm littlemoe
   ```
   Serves LFM2.5-8B-A1B-Q5_K_M at `http://127.0.0.1:8080/v1` (model ID: `lfm-moe`).

3. Create a benchmark manifest (JSON array):
   ```json
   [
     {"task": "Create a file hello.txt containing 'world'", "verify": "test $(cat hello.txt) = world"},
     {"task": "Write a Python script that prints the sum of 1-10", "verify": "python3 solution.py | grep -q 55"}
   ]
   ```
   Each entry: `task` (required), `verify` (optional shell command; exit 0 = pass), `fixtures` (optional path to a newline-separated file list to copy into the workspace).

4. Run:
   ```
   /crsi source ./manifest.json
   /crsi target ./manifest-target.json
   /crsi inspect
   /crsi off
   ```

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `CRSI_SIM_FILE` | `~/.pi/extensions/crsi-sim/sim.ndjson` | NDJSON persistence path |
| `CRSI_RUNTIME_URL` | `http://127.0.0.1:8080/v1` | Local model endpoint |
| `CRSI_RUNTIME_MODEL` | `lfm-moe` | Model ID |

No API keys required. The runtime model is local.

## H (Harness) format

```json
{
  "systemPrompt": "You are a capable coding agent.",
  "toolRules": "Use tools to solve tasks. Respond with {\"done\":true} when finished.",
  "maxTurns": 8,
  "parserRetries": 2
}
```

## SIM format (NDJSON lines)

```jsonl
{"type":"round","round":1,"role":"source","harness":{...},"candidates":[...]}
{"type":"checkpoint","round":1,"candidateId":0,"taskId":"...","score":1,"passed":true}
{"type":"sim-append","causeId":"premature-completion","record":{"attempt":"...","outcomes":"...","provenance":"..."}}
{"type":"promoted","round":1,"candidateId":0,"score":0.75,"harness":{...}}
{"type":"observation","ts":1234567890,"note":"tool_error: bash ..."}
```

## Evaluation boundary

- `/crsi` command path: direct `fs`/`child_process` execution in isolated `/tmp/crsi-eval-*` workspaces. **Not a security sandbox.** Tasks run with your user permissions.
- `swift_worker` tool path: uses `ctx.executeTool()` which respects Pi's tool guards (e.g. `mcp-bash-guard`).

## Limitations

- Single-machine, single-user. No concurrent writer protection (one Pi session at a time).
- The local 8B model is the runtime for all CRSI inference (policy, proposer, selector). Quality is bounded by model capability.
- Held-out split is the user's responsibility in the manifest design (exclude held-out tasks from the evolve manifest).
- Background learning is observational only; it does not constitute the paper's SIM mechanism.
- Copy isolation is NOT a security sandbox.

## Files

- `sim.ndjson` — append-only persistence (gitignored)
- `.tmp-*` — temporary workspaces (gitignored)
- `node_modules` — not used (gitignored)
