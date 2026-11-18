/**
 * pi-crsi-sim — Continual Recursive Self-Improvement (CRSI) with Self-Improvement Memory (SIM).
 *
 * Implements Section 3 of "Self-Improvement Memory Enables Continual Recursive Self-Improvement"
 * (arxiv 2611.18856v1): harness optimization loop with SIM-conditioned proposer/selector,
 * source-to-target continuation, and paper-exact SIM semantics.
 *
 * SIM rules (paper-exact):
 *   - Only accepted/selected edits are appended to cause_id history.
 *   - No accepted edit → SIM unchanged.
 *   - Full untruncated SIM read by both proposer and selector.
 *   - Predictions recorded before evaluation; measured outcomes after.
 *   - Rejected eval checkpoints are NOT SIM and not supplied as memory.
 *
 * Runtime model: Local Corvid-Magpie-8B (corvid-magpie) at :8080. No remote fallback.
 *   Start with: ~/.bin/corvid start
 *   If unavailable, CRSI operations defer gracefully (no crash).
 *
 * Active by default: persisted H is injected into before_agent_start.
 *   Escape: /crsi off (disables injection for this session).
 *
 * Background learning (beyond paper): agent_settled records observations
 * (errors, parser failures, premature stops) as feedback for the next /crsi round.
 * Does NOT auto-promote, does NOT replay chat externally, does NOT modify SIM.
 *
 * Evaluation boundary: /crsi command uses direct fs/child_process in isolated
 * /tmp workspaces (not a security sandbox). swift_worker tool uses ctx.executeTool()
 * which respects Pi's tool guards (mcp-bash-guard, etc.).
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { appendFileSync, readFileSync, existsSync, mkdirSync, rmSync, copyFileSync, writeFileSync } from "node:fs";
import { join, dirname, isAbsolute } from "node:path";
import { homedir } from "node:os";
import { execSync } from "node:child_process";

// --- Configuration ---
const SIM_FILE = process.env.CRSI_SIM_FILE ?? join(homedir(), ".pi", "extensions", "crsi-sim", "sim.ndjson");
const RT_URL = process.env.CRSI_RUNTIME_URL ?? "http://127.0.0.1:8080/v1";
const RT_MODEL = process.env.CRSI_RUNTIME_MODEL ?? "corvid-magpie";

// --- Types ---
type H = { systemPrompt: string; toolRules: string; maxTurns: number; parserRetries: number };
type Attempt = { attempt: string; outcomes: string; provenance: string };
type SimEntry = { causeId: string; attempts: Attempt[] };
type Executor = (tool: string, args: any, cwd: string, signal?: AbortSignal) => Promise<string>;

// --- NDJSON persistence (append-only, safe torn-line handling) ---
function appendLine(obj: unknown) {
  const d = dirname(SIM_FILE);
  if (!existsSync(d)) mkdirSync(d, { recursive: true });
  appendFileSync(SIM_FILE, JSON.stringify(obj) + "\n");
}
function readLines(): string[] {
  if (!existsSync(SIM_FILE)) return [];
  const raw = readFileSync(SIM_FILE, "utf8").split("\n").filter(Boolean);
  const valid: string[] = [];
  for (let i = 0; i < raw.length; i++) {
    try { JSON.parse(raw[i]); valid.push(raw[i]); }
    catch { if (i === raw.length - 1) break; throw new Error(`NDJSON corruption at line ${i + 1}`); }
  }
  return valid;
}
function replaySim(): { sim: SimEntry[]; activeH: H | null; pendingRound: number; completed: Set<string>; observations: string[] } {
  const sim = new Map<string, Attempt[]>();
  let activeH: H | null = null; let pendingRound = 0;
  const completed = new Set<string>(); const observations: string[] = [];
  for (const line of readLines()) {
    const r = JSON.parse(line);
    if (r.type === "sim-append") { const arr = sim.get(r.causeId) ?? []; arr.push(r.record); sim.set(r.causeId, arr); }
    if (r.type === "promoted") activeH = r.harness;
    if (r.type === "round") pendingRound = r.round;
    if (r.type === "checkpoint") completed.add(`${r.round}:${r.candidateId}:${r.taskId}`);
    if (r.type === "observation") observations.push(r.note);
  }
  return { sim: [...sim.entries()].map(([causeId, attempts]) => ({ causeId, attempts })), activeH, pendingRound, completed, observations };
}
function simToText(entries: SimEntry[]): string {
  if (!entries.length) return "(empty)";
  return entries.map(e => `[${e.causeId}]\n` + e.attempts.map(a => `  attempt: ${a.attempt}\n  outcomes: ${a.outcomes}\n  provenance: ${a.provenance}`).join("\n")).join("\n\n");
}

// --- Local runtime model client (no remote fallback) ---
async function rt(prompt: string, system: string, signal?: AbortSignal): Promise<string> {
  const res = await fetch(`${RT_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: RT_MODEL, messages: [{ role: "system", content: system }, { role: "user", content: prompt }] }),
    signal,
  });
  if (!res.ok) throw new Error(`Runtime model unavailable (HTTP ${res.status}). Start with: ~/.bin/corvid start`);
  const j = await res.json() as any;
  return j.choices?.[0]?.message?.content ?? "";
}

// --- Direct executor (evaluation boundary: /crsi command path, isolated /tmp) ---
const directExec: Executor = async (tool, args, cwd, signal) => {
  const resolve = (p: string) => isAbsolute(p) ? p : join(cwd, p);
  switch (tool) {
    case "bash": return execSync(args.command, { cwd, timeout: 30000, encoding: "utf8", signal }).slice(0, 4000);
    case "read": return readFileSync(resolve(args.path), "utf8").slice(0, 8000);
    case "write": writeFileSync(resolve(args.path), args.content); return "ok";
    case "edit": { const p = resolve(args.path); let c = readFileSync(p, "utf8"); c = c.replace(args.oldText, args.newText); writeFileSync(p, c); return "ok"; }
    case "grep": return execSync(`grep -rn ${JSON.stringify(args.pattern)} ${args.path ?? "."}`, { cwd, encoding: "utf8", signal }).slice(0, 4000);
    case "find": return execSync(`find . -name ${JSON.stringify(args.pattern)}`, { cwd, encoding: "utf8", signal }).slice(0, 4000);
    case "ls": return execSync("ls -la", { cwd, encoding: "utf8", signal }).slice(0, 4000);
    default: throw new Error(`unknown tool: ${tool}`);
  }
};

// --- Isolated workspace with optional fixture copy ---
function makeWorkDir(round: number, cand: number, task: string, fixtures?: string): string {
  const d = `/tmp/crsi-eval-${round}-${cand}-${task.replace(/[^a-z0-9]/gi, "_").slice(0, 20)}`;
  rmSync(d, { recursive: true, force: true }); mkdirSync(d, { recursive: true });
  if (fixtures && existsSync(fixtures)) {
    for (const f of readFileSync(fixtures, "utf8").trim().split("\n").filter(Boolean)) {
      const dest = join(d, f.replace(/^\/\//, "").replace(/^\//, ""));
      mkdirSync(dirname(dest), { recursive: true });
      copyFileSync(f, dest);
    }
  }
  return d;
}

// --- Inner policy tool loop (evaluate one harness on one task) ---
async function evalTask(h: H, task: string, workDir: string, exec: Executor, signal?: AbortSignal): Promise<{ score: number; passed: boolean; trace: string[] }> {
  const trace: string[] = [];
  let msgs: { role: string; content: string }[] = [
    { role: "system", content: h.systemPrompt + "\n\n" + h.toolRules },
    { role: "user", content: task },
  ];
  for (let turn = 0; turn < h.maxTurns; turn++) {
    const resp = await rt(msgs.map(m => `${m.role}: ${m.content}`).join("\n"),
      'Coding agent. Respond with exactly one JSON object: {"tool":"name","args":{...}} or {"done":true,"summary":"..."}.', signal);
    let parsed: any = null;
    for (let retry = 0; retry <= h.parserRetries; retry++) {
      try { parsed = JSON.parse(resp); break; }
      catch {
        if (retry < h.parserRetries) {
          trace.push(`parser_retry_${retry + 1}`);
          try { parsed = JSON.parse(await rt(`Extract the JSON tool call from this text:\n${resp}`, "Output only valid JSON, no explanation.", signal)); break; } catch { continue; }
        }
      }
    }
    if (!parsed) { trace.push("parse_failed"); break; }
    if (parsed.done) { trace.push("completed"); break; }
    if (!parsed.tool) { trace.push("no_tool"); break; }
    try {
      const out = await exec(parsed.tool, parsed.args, workDir, signal);
      msgs.push({ role: "assistant", content: resp }, { role: "user", content: `Result: ${out}` });
    } catch (e: any) {
      trace.push(`error:${e.message?.slice(0, 80)}`);
      msgs.push({ role: "assistant", content: resp }, { role: "user", content: `Error: ${e.message}` });
    }
  }
  // Objective verification: run verify command if specified
  const verifyMatch = task.match(/verify:\s*(.+)$/);
  let passed = false, score = 0;
  if (verifyMatch) {
    try { const out = await exec("bash", { command: verifyMatch[1] }, workDir, signal); passed = out.trim() === "" || !out.includes("FAIL"); score = passed ? 1 : 0; }
    catch { score = 0; }
  } else {
    passed = trace.includes("completed"); score = passed ? 1 : 0;
  }
  return { score, passed, trace };
}

// --- /crsi command: source|target <manifest> | inspect | off ---
let crsiDisabled = false;
function registerCrsi(pi: ExtensionAPI) {
  pi.registerCommand("crsi", {
    description: "CRSI: source|target <manifest.json> | inspect | off",
    handler: async (args, ctx) => {
      const parts = args?.trim().split(/\s+/) ?? []; const sub = parts[0];
      if (sub === "off") { crsiDisabled = true; ctx.ui.notify("CRSI harness injection disabled for this session", "info"); return; }
      if (sub === "inspect") {
        const { sim, activeH, pendingRound, observations } = replaySim();
        const obs = observations.length ? `\nObservations (F_t):\n${observations.slice(-10).join("\n")}` : "";
        ctx.ui.notify(`H: ${activeH ? JSON.stringify(activeH).slice(0, 150) : "none"} | round: ${pendingRound}\nSIM:\n${simToText(sim).slice(0, 2000)}${obs}`, "info");
        return;
      }
      if (sub !== "source" && sub !== "target") { ctx.ui.notify("Usage: /crsi source|target <manifest.json> | inspect | off", "error"); return; }
      // Graceful deferral if runtime model unavailable
      try { await rt("ping", "Reply ok.", new AbortController().signal); } catch (e: any) {
        ctx.ui.notify(`CRSI deferred: ${e.message}`, "error"); return;
      }
      const manifestPath = parts[1];
      if (!manifestPath) { ctx.ui.notify("Missing manifest path", "error"); return; }

      // Manifest: JSON array of {task, verify?, fixtures?}
      const manifest = JSON.parse(readFileSync(isAbsolute(manifestPath) ? manifestPath : join(ctx.cwd, manifestPath), "utf8")) as { task: string; verify?: string; fixtures?: string }[];
      const { sim: simEntries, activeH: prevH, pendingRound, completed, observations } = replaySim();
      const role = sub;
      const h: H = prevH ?? { systemPrompt: "You are a capable coding agent.", toolRules: "Use tools to solve tasks. Respond with {\"done\":true} when finished.", maxTurns: 8, parserRetries: 2 };
      const round = pendingRound + 1;
      const simText = simToText(simEntries);
      const feedback = observations.length ? `Recent observations:\n${observations.slice(-10).join("\n")}` : "(none)";

      // 1. Source baseline evaluated FIRST (before any proposals)
      ctx.ui.setStatus("crsi", `r${round}: baseline...`);
      const baselineScores: number[] = [];
      for (const t of manifest) {
        const wd = makeWorkDir(round, -1, t.task, t.fixtures);
        try { const r = await evalTask(h, t.task + (t.verify ? `\nverify: ${t.verify}` : ""), wd, directExec, ctx.signal); baselineScores.push(r.score); }
        catch { baselineScores.push(0); }
        rmSync(wd, { recursive: true, force: true });
      }
      const baselineAgg = baselineScores.reduce((a, b) => a + b, 0) / Math.max(baselineScores.length, 1);

      // 2. Proposer: K2=2 candidates based on failure root cause + SIM + feedback
      ctx.ui.setStatus("crsi", `r${round}: proposing...`);
      const proposeResp = await rt(
        `Current harness H:\n${JSON.stringify(h, null, 2)}\n\nSIM (full, untruncated):\n${simText}\n\nFeedback (F_t):\n${feedback}\n\nTasks:\n${manifest.map((t, i) => `${i}: ${t.task.slice(0, 80)}`).join("\n")}\n\nPropose exactly 2 candidate harness modifications. Output JSON array:\n[{"causeId":"<root-cause>","attempt":"<harness modification text>","rationale":"<why>","predictedFixes":[<task indices>]}]`,
        "You are a harness optimizer. Propose modifications based on failure root causes, SIM history, and feedback. Each proposal cites an existing SIM causeId or names a new one. Output only valid JSON.",
        ctx.signal
      );
      let candidates: { causeId: string; attempt: string; rationale: string; predictedFixes: number[] }[] = [];
      try { const m = proposeResp.match(/\[[\s\S]*\]/); if (m) candidates = JSON.parse(m[0]); } catch { /* empty */ }
      if (!candidates.length) { ctx.ui.notify("No valid proposals from proposer", "error"); ctx.ui.setStatus("crsi", "idle"); return; }

      // 3. Persist round plan before evaluation (for resume)
      appendLine({ type: "round", round, role, harness: h, candidates: candidates.map(c => ({ causeId: c.causeId, attempt: c.attempt })) });

      // 4. Evaluate each candidate on all evolve tasks (skip completed checkpoints for resume)
      const results: { scores: number[] }[] = candidates.map(() => ({ scores: [] as number[] }));
      for (let ci = 0; ci < candidates.length; ci++) {
        ctx.ui.setStatus("crsi", `r${round}: eval C${ci}...`);
        const candH: H = { ...h, systemPrompt: h.systemPrompt + "\n" + candidates[ci].attempt };
        for (let ti = 0; ti < manifest.length; ti++) {
          const ckKey = `${round}:${ci}:${manifest[ti].task.slice(0, 40)}`;
          if (completed.has(ckKey)) { results[ci].scores.push(0); continue; }
          const wd = makeWorkDir(round, ci, manifest[ti].task, manifest[ti].fixtures);
          try {
            const r = await evalTask(candH, manifest[ti].task + (manifest[ti].verify ? `\nverify: ${manifest[ti].verify}` : ""), wd, directExec, ctx.signal);
            results[ci].scores.push(r.score);
            appendLine({ type: "checkpoint", round, candidateId: ci, taskId: manifest[ti].task.slice(0, 40), score: r.score, passed: r.passed });
          } catch { results[ci].scores.push(0); }
          rmSync(wd, { recursive: true, force: true });
        }
      }

      // 5. Selector: strict aggregate improvement promotion
      ctx.ui.setStatus("crsi", `r${round}: selecting...`);
      const selectResp = await rt(
        `H: ${JSON.stringify(h)}\nSIM:\n${simText}\n\nCandidates and scores:\n${candidates.map((c, i) => `C${i} [${c.causeId}]: ${c.attempt.slice(0, 100)}\n  aggregate: ${(results[i].scores.reduce((a, b) => a + b, 0) / results[i].scores.length).toFixed(3)}\n  baseline: ${baselineAgg.toFixed(3)}`).join("\n")}\n\nSelect the best candidate index (0-${candidates.length - 1}) or -1 to reject all. Rule: accept ONLY if aggregate strictly exceeds baseline. Output only an integer.`,
        "You are a harness selector. Apply strict aggregate improvement. Consider regression risk and domain specificity from SIM. Output only an integer.",
        ctx.signal
      );
      const selIdx = parseInt(selectResp.trim()) ?? -1;

      // 6. Paper-exact SIM update: only accepted edit appended
      if (selIdx >= 0 && selIdx < candidates.length) {
        const selAgg = results[selIdx].scores.reduce((a, b) => a + b, 0) / results[selIdx].scores.length;
        if (selAgg > baselineAgg) {
          const newH: H = { ...h, systemPrompt: h.systemPrompt + "\n" + candidates[selIdx].attempt };
          const record: Attempt = {
            attempt: candidates[selIdx].attempt,
            outcomes: `predicted_fixes:[${(candidates[selIdx].predictedFixes ?? []).join(",")}] measured_agg:${selAgg.toFixed(3)} baseline:${baselineAgg.toFixed(3)}`,
            provenance: `role=${role} round=${round} model=${RT_MODEL} split=evolve`,
          };
          appendLine({ type: "sim-append", causeId: candidates[selIdx].causeId, record });
          appendLine({ type: "promoted", round, candidateId: selIdx, score: selAgg, harness: newH });
          pi.appendEntry("sim-record", { causeId: candidates[selIdx].causeId, record });
          ctx.ui.notify(`CRSI r${round}: PROMOTED C${selIdx} [${candidates[selIdx].causeId}] (${selAgg.toFixed(3)} > ${baselineAgg.toFixed(3)})`, "info");
        } else {
          ctx.ui.notify(`CRSI r${round}: REJECTED (C${selIdx} agg ${selAgg.toFixed(3)} <= baseline ${baselineAgg.toFixed(3)})`, "info");
        }
      } else {
        ctx.ui.notify(`CRSI r${round}: REJECTED (no candidate accepted)`, "info");
      }
      ctx.ui.setStatus("crsi", "idle");
    },
  });
}

// --- swift_worker tool (uses ctx.executeTool for Pi-guarded execution) ---
function registerSwiftWorker(pi: ExtensionAPI) {
  pi.registerTool({
    name: "swift_worker",
    label: "CRSI Worker",
    description: "Evaluate a task using the CRSI runtime model with a harness config. For testing harness modifications.",
    parameters: Type.Object({
      task: Type.String({ description: "Task description for the policy model" }),
      systemPrompt: Type.Optional(Type.String({ description: "Harness system prompt override" })),
      maxTurns: Type.Optional(Type.Number({ description: "Max policy turns (default 8)" })),
    }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      const h: H = {
        systemPrompt: params.systemPrompt ?? "You are a capable coding agent.",
        toolRules: 'Use tools to solve tasks. Respond with {"done":true} when finished.',
        maxTurns: params.maxTurns ?? 8,
        parserRetries: 2,
      };
      // Use Pi's native tool execution (respects guards like mcp-bash-guard)
      const exec: Executor = async (tool, args, cwd, sig) => {
        try {
          const r = await (ctx as any).executeTool(tool, args, { cwd, signal: sig });
          return r?.content?.map((c: any) => c.text ?? "").join("") ?? "";
        } catch (e: any) { throw new Error(e.message ?? tool); }
      };
      const wd = makeWorkDir(0, 0, params.task.slice(0, 16));
      try {
        const r = await evalTask(h, params.task, wd, exec, signal);
        return { content: [{ type: "text" as const, text: `score=${r.score} passed=${r.passed} turns_used=${r.trace.length} trace=[${r.trace.join(",")}]` }], details: r };
      } finally { rmSync(wd, { recursive: true, force: true }); }
    },
  });
}

// --- Tracing + background learning (beyond paper, documented) ---
function registerTracing(pi: ExtensionAPI) {
  let activeH: H | null = null;
  pi.on("session_start", () => { activeH = replaySim().activeH; });

  // Active by default: inject H into system prompt
  pi.on("before_agent_start", (event) => {
    if (!activeH || crsiDisabled) return;
    return { systemPrompt: event.systemPrompt + `\n\n<crsi-harness>\n${activeH.systemPrompt}\n${activeH.toolRules}\nmaxTurns=${activeH.maxTurns} parserRetries=${activeH.parserRetries}\n</crsi-harness>` };
  });

  // Trace tool errors (feeds F_t for next /crsi round)
  pi.on("tool_result", (event) => {
    if (!activeH || crsiDisabled) return;
    if ((event as any).isError) {
      appendLine({ type: "observation", ts: Date.now(), note: `tool_error: ${(event as any).toolName} ${((event as any).content?.[0]?.text ?? "").slice(0, 100)}` });
    }
  });

  // Background learning: record session-level observations on settle (beyond paper)
  pi.on("agent_settled", (event) => {
    if (!activeH || crsiDisabled) return;
    const aborted = (event as any).aborted;
    if (aborted) return; // Don't record aborted runs
    // Check if last assistant message had no tool calls (potential premature stop)
    const entries = (event as any).entries ?? [];
    const lastAssistant = [...entries].reverse().find((e: any) => e.type === "message" && e.message?.role === "assistant");
    if (lastAssistant && !lastAssistant.message?.content?.some((c: any) => c.type === "toolCall")) {
      appendLine({ type: "observation", ts: Date.now(), note: "premature_stop: agent ended without tool call" });
    }
  });
}

// --- Main export ---
export default function (pi: ExtensionAPI) {
  registerCrsi(pi);
  registerSwiftWorker(pi);
  registerTracing(pi);
}
