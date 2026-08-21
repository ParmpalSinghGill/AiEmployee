# AI Employee Framework — Demo Video Script

**Target length:** 6–7 minutes (a 90-second short cut is at the bottom)
**Format:** screen recording + voiceover. No face needed.
**Two columns:** left = what you SAY, right = what you SHOW.

The demo is built around **three real tasks**, chosen because each one proves a
different claim and each one produces something you can *see* on screen:

| # | Task | Proves | Visible payoff |
|---|---|---|---|
| **A** | ML resource scouting + runnable code | Multi-agent delegation, real tool calls | Model/dataset list, VRAM numbers, generated code file |
| **B** | Data analysis with a chart | Code execution, artifacts | A PNG chart rendered inline in the Result tab |
| **C** | Follow-up after a server restart | Memory + persistence | It answers with context from before the restart |

---

## Before you hit record — setup checklist

```bash
conda activate AIEmployee
cd /media/parmpal/Data/Codes/VibeCoding/AIEmployee
make test                       # confirm green
python -m ai_employee serve     # leave running in a 2nd terminal (port 8010)
```

- **Restart the server and hard-refresh the browser** (`Ctrl+Shift+R`) before you
  start. The console is a static SPA, so a browser holding a cached `app.js` will
  silently show an older UI — missing badges, missing flow features. Confirm the
  running server has the current code:
  ```bash
  curl -s localhost:8010/static/app.js | grep -c parallel-badge   # expect 4, not 0
  ```
- Open `http://localhost:8010/ui`. Clear old saved tasks so the left panel starts clean.
- **Run all three tasks once before recording.** You need to know what they produce and how long they take. Task A hits the network (HF, arXiv, Kaggle) — confirm it works today.
- Use a **real provider** (deepseek or local Ollama) for the demo. `mock` is only for the "runs offline" beat in Scene 7.
- **Heads-up on rehearsing Task A:** downloads are now cached workspace-wide. Once
  you've fetched a model/dataset, later runs reuse the local copy and skip the
  download — so a rehearsed Task A will look *faster* on camera than a cold one, and
  the agent will say it reused an existing path. That's a feature worth showing (see
  the bandwidth beat below), but if you specifically want the cold-search footage,
  clear the ledger first: `rm data/workspace/assets.json`.
- **Task B needs no file prep** — it pulls a public CSV over HTTP. Just confirm the
  URL is up on the day you record:
  ```bash
  curl -sI https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv | head -1
  ```
  Expect `HTTP/2 200`. If it's down, use the fallback named in Scene 6.
- Note the asset cache applies here too: after your first rehearsal the CSV is
  already local, so the recorded run reuses it instead of re-downloading. To film a
  cold fetch, clear the ledger first (`rm data/workspace/assets.json`).
- Terminal: ~18pt font, dark theme, ~120 cols, scrollback cleared.
- **Move `www.youtube.com_cookies.txt` out of the repo root** and make sure `.env` with real keys is never on screen.

---

## SCENE 1 — Hook (0:00–0:20)

| SAY | SHOW |
|---|---|
| "Most AI tools give you a chat box. You ask, it answers, and you still do the work. This is different. This is an AI *employee*. You give it a goal — one sentence — and it plans the work, splits it across specialized agents, uses real tools, verifies its own output, and remembers what it did." | Full-screen `/ui`, clean and empty. |
| "Let me give it a real job." | Type **Task A** into the goal box, slowly, but don't press Run yet. |

**Task A — type this exactly:**
```
Find a pretrained model and a dataset for detecting defects on a factory
production line, tell me the GPU I need to fine-tune it, and write me
runnable training code with a README.
```

> This typing shot is your thumbnail. Keep it crisp.

---

## SCENE 2 — The plan (0:20–1:00)

| SAY | SHOW |
|---|---|
| "Watch what happens. It doesn't answer. It *plans*." | Hit **▶ Run**. Stay on the **⬡ Flow** tab. Let the plan nodes populate for 3 seconds with **no talking**. |
| "It decomposed my one sentence into steps, and handed each step to a different specialist. That's a Scout for finding resources, a Hardware Analyst for the GPU question, and a Code Generator for the training script. There are 23 of these specialists in the framework." | Hover slowly over the plan nodes so each assigned agent name is readable — point out `scout`, `hardware_analyst`, `codesmith`. |

---

## SCENE 3 — Real tools, real work (1:00–2:00)

| SAY | SHOW |
|---|---|
| "Now the execution. And this is the part that matters — these aren't the model making things up. These are real tool calls going out to real places." | Flow view running live. Zoom in on the tool-call nodes. |
| "It's searching Hugging Face for models and datasets. Papers With Code. arXiv. Kaggle. Roboflow for vision datasets. Each call shows its status and its latency right on the node." | Zoom on `hf_search`, then `arxiv_search` / `kaggle_search` / `roboflow_search` nodes as they fire. Hold on one showing latency. |
| "Then it estimates the hardware — actual VRAM numbers for inference and for full fine-tuning, checked against the GPU I've got." | Zoom on the `estimate_hardware` node, then its output. |
| "And when a step finishes, it gets verified. The employee reviews its own work before moving on, and reports a confidence score." | Zoom on a **🔎 Verification** badge with its confidence value. Hold 2 seconds. |

> **Editing:** real runs are slower than narration. Speed-ramp the middle to 4×–8×, but keep the *start* and *end* of each node at 1× so it still reads as real.

---

## SCENE 4 — The result you can actually use (2:00–2:50)

| SAY | SHOW |
|---|---|
| "Here's what I got back." | Click **★ Result**. Scroll slowly from the top. |
| "A shortlist of real models and datasets — with real IDs, not invented ones. The VRAM it needs. And the training code, written to the workspace, with pinned dependencies and a step-by-step README telling me exactly how to run it." | Scroll through the model/dataset list → the hardware numbers → the generated code. Pause on the file list the run produced. |

> **Result tab shows results only.** Downloaded dataset images and model files are
> inputs, not output, so they are filtered out of the gallery — the Result tab shows
> charts, evaluation figures, and generated deliverables. You won't get a contact
> sheet of training images on screen. Nothing to avoid here; just don't go hunting
> for dataset thumbnails to show, because there won't be any.
| "Every step's individual output is here too, so you can see which agent produced which piece. It's not a black box handing you a paragraph." | Scroll to the per-step outputs. Pause on one. |
| "And the raw event stream is one click away." | Click **≡ Logs**, scroll fast for 3 seconds, move on. |

---

## SCENE 5 — Explainability (2:50–3:25)

| SAY | SHOW |
|---|---|
| "The question everyone asks about agents is: *why* did it do that? So there's a tab for exactly that." | Click **🔎 Explain**. |
| "This traces the real decision points — what the model was asked, what it chose, and why that step happened at all. When a run goes wrong, this is where you find out whether it was a bad plan, a bad tool call, or just a bad model." | Scroll the Explain view. Stop on one decision entry, zoom in, let the viewer read it. |

---

## SCENE 6 — Task B: it writes and runs code (3:25–4:15)

| SAY | SHOW |
|---|---|
| "Second job. This one's different — I want it to actually *compute* something." | New task. Type **Task B**. |

**Task B — type this exactly:**
```
Download https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
and analyse it: total revenue and average tip percentage by day, then save a bar
chart of revenue by day as a PNG.
```

> Public URL, so there's nothing to prepare before recording and the demo runs on
> any machine. 244 rows, and the columns (`total_bill`, `tip`, `day`, `time`) give
> a real grouping to aggregate. Verified reachable — but **re-check it on the day**
> (`curl -sI <url> | head -1` → `200`), and if it's down, swap in
> `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
> ("fare by passenger class") or keep a local copy as a fallback.

| SAY | SHOW |
|---|---|
| "It fetches the file, previews it, writes Python, executes it in a sandbox, and saves the output." | Flow view: point at the fetch/download node, then `csv_preview`, then `python_exec` firing. |
| "And there's the chart. It ran the code, produced a real PNG, and the console renders it inline — along with every other file the run produced." | **★ Result** tab: the bar chart rendered inline. Scroll to the produced-files list. Click the chart to open it full size. |
| "So it's not describing an analysis. It did one." | Hold on the chart. |

> If the chart doesn't appear, the file wasn't written to the workspace root — check the run's produced-files list before you re-record. Don't narrate over a missing image.

---

## SCENE 7 — Task C: memory that survives a restart (4:15–5:05)

| SAY | SHOW |
|---|---|
| "An employee you have to re-brief every morning isn't much of an employee." | Scroll to the follow-up box on the **Task A** tab. |
| "So — same task, follow-up question." | Type: `Which of those models is the cheapest to fine-tune on a single 8GB GPU?` → **↳ Ask follow-up** |
| "It reuses the session. Conversation and memory carry over, and the follow-up appears as a new run inside the same tab. It doesn't start from zero — notice it's answering about the models *it* found earlier, and I never re-listed them." | New run appears in the tab. Flow lights up briefly → cut to Result. Highlight that it references models from the first run. |
| "Now watch this. I'm going to kill the server." | Cut to the `serve` terminal. Press **Ctrl-C** on camera. Beat of silence. |
| "And bring it back." | `python -m ai_employee serve`. Reload `/ui`. |
| "Every task is on disk — the goal, the full event history, the result. Reopen it, and memory is reloaded with it." | **Saved tasks** panel on the left. Click the Task A entry. Full history renders. |
| "Same question style, after a full restart." | Ask one more follow-up: `Remind me which dataset you recommended and why.` Show it answering **correctly, with context from before the restart**. |

> This restart beat is the most convincing thing in the video. Don't rush it.

---

## SCENE 7b — It doesn't download the same thing twice (optional, 30s)

A strong beat if bandwidth matters to your audience. Run this *after* Task A has
already fetched something.

| SAY | SHOW |
|---|---|
| "One thing that matters when you're working with big models and datasets: it doesn't re-download what it already has." | New task: `Set up an inference script using a defect detection model.` |
| "The workspace keeps an index of everything it's downloaded. Before it searches the internet, it checks local inventory — and if it already has something that fits, it uses that path instead of pulling gigabytes again." | Flow view: point at the `search_local_assets` node firing **before** any hub search. |
| "Same model, no download, no wait." | Result tab: it names the existing local path and says it reused it. |

> To show the ledger itself: `cat data/workspace/assets.json` — one entry per asset,
> with its source URL and where it landed.

---

## SCENE 7c — Multiple agents at once (optional, 40s)

The most visually striking beat in the demo. Turn this on **only for this scene** —
see the caveat below.

**Setup:** click **⚡ Parallel** in the header so it reads **⚡ Parallel: On**.

| SAY | SHOW |
|---|---|
| "So far each specialist has worked one at a time. But most of these steps don't actually depend on each other — finding a model, finding a dataset, and sizing the GPU are three independent jobs." | Header: click **⚡ Parallel**, it flips to **On**. |
| "So let them run at the same time." | Re-run Task A. Cut straight to the **⬡ Flow** tab. |
| "Three agents, live, simultaneously — the scout, the researcher, and the hardware analyst. Each one is a separate specialist working its own step." | The banner: **⚡ 3 agents running in parallel**, with a named chip per agent. Zoom in on it. Hold 3 seconds — this is the shot. |
| "And the step that needs all three waits. The moment its last dependency lands, it starts — no idle time." | Point at the dependent step node sitting idle, then starting the instant the others finish. |
| "It's dependency-aware. Independent work overlaps; dependent work waits exactly as long as it has to." | The "Running now" counter in the summary stats ticking down 3 → 1 as steps land. |

> **Caveat — film this scene last, or on its own.** With several agents live the
> **Logs** tab interleaves events from concurrent steps, which makes the log-reading
> beat in Scene 3 much harder to follow. Keep ⚡ Parallel **off** for Scenes 2–4, turn
> it on here, and the contrast between the two runs becomes part of the story.

> If the banner doesn't appear: the plan had no independent steps (a fully sequential
> plan correctly shows nothing). Check the run's shape first — a goal phrased as
> "find X **and** Y **and** Z" parallelizes; "do X, then use X to do Y" does not.

---

## SCENE 8 — Provider independence (5:05–5:50)

| SAY | SHOW |
|---|---|
| "Here's the architectural claim: no business logic in this framework ever imports an LLM SDK. Everything goes through one `Brain` interface." | Editor: `src/ai_employee/brains/base.py`, scroll the interface. Flash `brains/factory.py`. |
| "So the provider is a config value, not a code change." | Open `.env`, highlight `LLM_PROVIDER=`. |
| "DeepSeek, OpenAI, Gemini, GLM, or a local model through Ollama — one variable." | README provider table. |
| "And I can switch it live, without restarting." | Click the **brain badge**, top-right. Picker opens (it lists local Ollama models). Switch provider/model. Apply. |
| "Same goal, different brain, zero code touched." | Run a short goal on the new brain: `Summarize what you found about defect detection in 3 bullets.` Show it completing. Point at the badge showing the new provider + model. |

---

## SCENE 9 — It runs with nothing (5:50–6:25)

| SAY | SHOW |
|---|---|
| "One more thing if you're evaluating this. It runs with no API keys, no database, no network." | Terminal: `LLM_PROVIDER=mock python -m ai_employee run "Plan a product launch"` |
| "There's a mock brain and an in-memory store, so the whole pipeline — planning, agents, tools, verification — works completely offline. You can explore it in two minutes instead of two hours." | Show the result panel printing. |
| "The test suite is offline too." | `make test` → green. |
| "Full suite, well under a second, no infrastructure." | Freeze/zoom on the green summary line. **Read whatever count is actually on screen.** |

---

## SCENE 10 — CLI + API breadth (6:25–6:55)

Fast montage — type, hold output ~2s, cut.

| SAY | SHOW |
|---|---|
| "It's not UI-only." | `python -m ai_employee agents` → the agent catalog scrolling |
| "Full CLI —" | `python -m ai_employee tools` → tool list with categories |
| "— RAG over your own documents —" | `python -m ai_employee ingest ./README.md` then `python -m ai_employee query "What providers are supported?"` → show the answer **and the citations JSON** |
| "— you can call one agent directly —" | `python -m ai_employee ask scout "Find a small speech-to-text model I can run locally"` |
| "— and check exactly what's active." | `python -m ai_employee config` → brain, model, embeddings, memory backend |
| "Plus a FastAPI server, so everything you just saw is an API too." | `http://localhost:8010/docs` — scroll `/chat`, `/task`, `/agents`, `/tools`, `/knowledge`, `/memory` |

---

## SCENE 11 — Close (6:55–7:15)

| SAY | SHOW |
|---|---|
| "So: give it a goal, and it plans, delegates to a team of specialists — several working in parallel — calls real tools, writes and runs code, verifies itself, explains its reasoning, reuses what it's already downloaded, and remembers across restarts. On whatever model you want, including one on your own machine." | Cut back to the Task A Flow view and let it sit. |
| "Apache-2.0. `docker compose up` gets you the full stack — Postgres with pgvector, Redis, and the API. Link's below." | README Quickstart block → your link card. |
| "Thanks for watching." | End card. |

---

## Optional 15-second beat: the approval gate

Strong if you have time — insert after Scene 4. Sensitive tools are approval-gated by
default (`shell_write`, `email_send`, `git_push`, `db_write`, `deploy`,
`resource_download`, `resource_acquisition`).

| SAY | SHOW |
|---|---|
| "And it doesn't get to do whatever it wants. Anything sensitive — downloads, shell writes, a git push, a deploy — stops and waits for a human." | Run a task that triggers a download, e.g. `Download the model you recommended into the workspace.` The task pauses. |
| "It's blocked until I approve it." | Show the **Approve & continue** button. Click it. Watch it resume. |

**Rehearse this one.** If you don't expect the pause, it looks like a hang.

---

## Things NOT to show

- **Never show dataset images or dataset filename dumps.** Only charts, evaluation
  figures, and reports. The Result gallery now filters downloaded dataset/model
  content automatically, but the same rule applies to anything *you* put on screen:
  don't open the dataset folder in a file manager, don't `ls` a directory of sample
  images, and don't scroll a terminal listing hundreds of `.jpg` names. It reads as
  clutter and makes the output look unfiltered.
- Don't run the main demo on `mock` — the answers are placeholders and the UI literally prints a warning about it. A viewer reads that as "the product doesn't work".
- Don't show `.env` with real keys, or `www.youtube.com_cookies.txt` in the repo root.
- Don't narrate numbers (test counts, VRAM, model names) from this script — read what's on screen in *your* run.

---

## 90-second short cut

Run this one with **⚡ Parallel on** — the multi-agent banner is the strongest
single image in the whole demo and it belongs in a short cut.

1. **0:00–0:10** — Type Task A, hit Run. *"You give it a goal, not a prompt."*
2. **0:10–0:30** — Flow populating; the **⚡ 3 agents running in parallel** banner with named chips; `hf_search` / `estimate_hardware` firing; a verification badge. *"It plans, then puts a team of specialists on it — several working at the same time, calling real tools."*
3. **0:30–0:50** — Result: model + dataset list, VRAM numbers, generated code. *"Real model IDs, the GPU you need, and runnable training code."*
4. **0:50–1:05** — Task B chart rendered inline. *"It writes and runs the code too."*
5. **1:05–1:20** — Ctrl-C the server, restart, reopen the saved task, follow-up lands with context. *"And it remembers across restarts."*
6. **1:20–1:30** — `.env` provider line + live brain switch. *"Any model. One variable. Apache-2.0, link below."*

---

## Recording notes

- **Record voice separately** from screen. Doing both at once makes both worse.
- 1080p minimum. The Flow view has small text — zoom in post, don't shrink it live.
- Keep the mouse **still** when you're not pointing. Idle jitter is the #1 amateur tell.
- Leave 2 seconds of silence at each scene boundary for cutting room.
- Expect ~25 minutes of raw footage for a 7-minute cut. That's normal.
