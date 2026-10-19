"""Guard the Explain trace builder against the "silent throw" class of bug.

If ``buildTrace`` throws, the Explain pane keeps whatever text it had (often the
stale "Waiting for the employee to start…") while Flow renders fine — a failure
that is invisible to Python tests. This runs the real ``buildTrace`` in Node
against a retry trace and asserts it builds two separated attempts.

Skipped when Node isn't available (the Python suite stays offline-capable).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "src/ai_employee/api/static/app.js"

_NODE_SCRIPT = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const start = src.indexOf("function buildTrace(task) {");
const end = src.indexOf("\nconst _jsonBlk", start);
if (start < 0 || end < 0) { console.error("buildTrace not found"); process.exit(2); }
eval(src.slice(start, end));
const task = { events: [
 { event: "agent.context", fields: { step: 1, agent: "coding", retry: 0, task: "ORIG", system: "S" } },
 { event: "brain.request",  fields: { step: 1, retry: 0, kind: "chat", messages: [] } },
 { event: "brain.response", fields: { step: 1, retry: 0, content: "r0" } },
 { event: "tool.execute",   fields: { step: 1, retry: 0, tool: "python_exec", args: { cmd: "x" } } },
 { event: "tool.result",    fields: { step: 1, retry: 0, tool: "python_exec", ok: true, output: "ok0" } },
 { event: "agent.answer",   fields: { step: 1, retry: 0, output: "first attempt" } },
 { event: "verifier.input", fields: { step: 1, retry: 0, task: "ORIG", criteria: "c", output: "first" } },
 { event: "verifier.verdict", fields: { step: 1, retry: 0, passed: false, confidence: 0.3, issues: ["missing test"], suggestion: "add" } },
 { event: "agent.context",  fields: { step: 1, agent: "coding", retry: 1, task: "CONTINUE from prev" } },
 { event: "tool.execute",   fields: { step: 1, retry: 1, tool: "write_file", args: { path: "t.py" } } },
 { event: "tool.result",    fields: { step: 1, retry: 1, tool: "write_file", ok: true, output: "ok1" } },
 { event: "agent.answer",   fields: { step: 1, retry: 1, output: "fixed" } },
 { event: "verifier.verdict", fields: { step: 1, retry: 1, passed: true, confidence: 0.9, issues: [] } }
], result: { steps: [{ id: 1, agent: "coding", retries: 1 }] } };
const comps = buildTrace(task);
const st = comps.find(c => c.kind === "step");
console.log(JSON.stringify({
  retries: st.retries,
  attempts: st.attempts.map(a => ({ retry: a.retry, input: a.input && a.input.task, tools: a.tools.length, passed: a.verify.verdict && a.verify.verdict.passed }))
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_build_trace_handles_retry_without_throwing() -> None:
    proc = subprocess.run(
        ["node", "-e", _NODE_SCRIPT, str(APP_JS)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"buildTrace threw in Node:\n{proc.stderr}"
    data = json.loads(proc.stdout.strip())
    assert data["retries"] == 1
    attempts = {a["retry"]: a for a in data["attempts"]}
    assert set(attempts) == {0, 1}
    # attempt 0 = the original planner input; attempt 1 = the retry continuation.
    assert attempts[0]["input"] == "ORIG"
    assert attempts[0]["passed"] is False
    assert attempts[1]["input"] == "CONTINUE from prev"
    assert attempts[1]["passed"] is True
    assert attempts[0]["tools"] == 1 and attempts[1]["tools"] == 1


_FLOW_NODE_SCRIPT = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const start = src.indexOf("function deriveFlow(task) {");
const end = src.indexOf("\nfunction nodeHtml", start);
if (start < 0 || end < 0) { console.error("deriveFlow not found"); process.exit(2); }
eval(src.slice(start, end));
const task = { events: [
  { event: "agent.start", fields: { step: 1, agent: "coding", node: "execute" } },
  { event: "brain.response", fields: { step: 1, purpose: "agent:coding", usage: { total_tokens: 1234 } } },
  { event: "brain.response", fields: { step: 1, purpose: "agent:coding", usage: { total_tokens: 200 } } },
  { event: "brain.response", fields: { step: 1, purpose: "verifier", usage: { total_tokens: 999 } } }, // excluded
  { event: "workflow.retry", fields: { step: 1, attempt: 2 } },
  { event: "verifier.verdict", fields: { step: 1, passed: true, confidence: 0.9 } },
], result: {} };
const fl = deriveFlow(task);
const s = fl.steps[0];
console.log(JSON.stringify({ agent: s.agent, tokens: s.tokens, retries: s.retries, verified: s.verified }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_derive_flow_accumulates_agent_tokens_and_retries() -> None:
    proc = subprocess.run(
        ["node", "-e", _FLOW_NODE_SCRIPT, str(APP_JS)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"deriveFlow threw in Node:\n{proc.stderr}"
    data = json.loads(proc.stdout.strip())
    assert data["agent"] == "coding"
    # 1234 + 200, but NOT the verifier's 999 (purpose "verifier" is excluded).
    assert data["tokens"] == 1434
    assert data["retries"] == 2
    assert data["verified"] is True

