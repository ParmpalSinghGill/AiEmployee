# Running Tasks with the AI Employee — Operator SOP

> **SOP-AIE-001 · v1.0 · Effective 21 Aug 2026**
> Applies to: AI Employee Web Console · Audience: all operators · Owner: Engineering (admin)
>
> How to give the AI Employee a goal, watch it work, approve sensitive steps, and collect
> the results — written for team members with no technical background.
> A printable, styled copy of this document lives at [operator-sop.html](operator-sop.html)
> (download it and open it in a browser).

---

## 1. What the AI Employee is

The AI Employee is not a chatbot. You give it **one goal in plain English** — for example,
*"Research our three top competitors and write a one-page comparison"* — and it does the
whole job: it breaks the goal into steps, assigns each step to a specialist (a researcher,
an analyst, a writer, and so on), runs them, checks its own work, and hands you the
finished output.

Your job as the operator is to: write a clear goal, keep an eye on progress, **approve or
decline** anything sensitive it asks permission for, and collect the result. This document
covers exactly that — nothing here requires installing software or writing code.

## 2. Before you start

- **Access:** open the console in your web browser at the address your administrator gave
  you (it looks like `http://<server>:8010/ui`). No login software or installation is
  needed.
- **If the page won't load:** the server is probably not running. Contact the
  administrator (Section 9) — do not try to restart anything yourself.
- **One task, one tab:** each task you start gets its own tab inside the console. Old
  tasks stay in the left-hand list, so you can come back to them any time.

## 3. Starting a task

1. In the box labelled **Give the employee a goal…**, type what you want done — one
   complete sentence, as if briefing a new colleague.
2. Leave the toggles as you find them unless this SOP or your administrator says
   otherwise. In particular, leave **auto-approve off** (see Section 5).
3. Press **▶ Run**. A new tab opens for the task and work starts immediately.

### Writing a goal that works

The single biggest factor in getting a good result is the goal you type. Include **what**
you want, **what the finished output should be**, and any limits.

| | Example | Why |
|---|---|---|
| ❌ Too vague | *"Look into customer churn."* | No deliverable named — the employee has to guess what "look into" means, and you may get something you didn't want. |
| ✅ Good | *"Analyse the attached churn spreadsheet, chart churn by month, and write a half-page summary of the top three drivers."* | Names the input, the outputs (a chart and a summary), and the size. |

> [!NOTE]
> You don't need to tell it *how* to do the work or which specialist to use — planning is
> its job. Describe the destination, not the route.

## 4. While it runs — the four views

Each task tab has four views. You will mostly use the first and last.

| View | What it shows | When to use it |
|---|---|---|
| **⬡ Flow** | A live diagram of which specialists and tools are working right now. | Default view. Glance here to see progress and spot a pause that's waiting on you. |
| **🔎 Explain** | A plain-language account of each step: what was asked, what was decided, and why. | When a result surprises you and you want to see the reasoning behind it. |
| **≡ Logs** | The raw technical event stream. | Only when the administrator asks you to check something specific. |
| **★ Result** | The finished deliverables — reports, charts, files. | When the task finishes. This is where you collect the output. |

You do not need to keep the page open. A running task continues on the server; close the
browser, come back later, and open the task from the left-hand list to see where it got to.

## 5. Approvals — the part that needs your judgement

Some operations are deliberately **gated**: before the employee downloads a large file,
runs a system command, or does anything else sensitive, the task **pauses** and asks you.
This is a safety feature, not an error.

When a task pauses for approval:

1. Read the request text. It says exactly what the employee wants to do (for example,
   download a specific dataset).
2. Ask yourself: *does this clearly serve the goal I typed?* A task about competitor
   research asking to fetch a competitor's public web page — fine. The same task asking to
   run an unfamiliar system command — decline and ask the administrator.
3. If it makes sense, press **Approve & continue**. The task resumes exactly where it
   paused.
4. If you're unsure, **leave it paused** and ask the administrator. A paused task loses
   nothing by waiting.

> [!CAUTION]
> **Policy:** leave the auto-approve toggle (*"Never pause for approval"*) **off**. It
> grants every gated operation automatically and is reserved for administrators running
> pre-vetted jobs. If you find it switched on, switch it off and mention it to the
> administrator.

## 6. Stopping and resuming

- **To stop a running task,** press **Stop** in its tab. Progress up to the last completed
  step is saved automatically.
- **To continue later,** open the task and press **Resume from last checkpoint**. It
  carries on from where it stopped — it does not start over.
- This also covers accidents: if the server restarts or your browser crashes mid-run, the
  task shows up as interrupted and offers the same resume button. Nothing is lost.

## 7. Collecting results and asking follow-ups

When the flow view shows the task is finished, open **★ Result**. You'll find the written
output plus any files it produced — charts, reports, spreadsheets, code. Download what you
need from there; files also remain saved on the server with the task.

If the result is close but not quite right, don't start a new task. Type into the box at
the bottom of the same tab — **↳ Ask follow-up** — for example *"Shorten the summary to
three bullet points"*. The employee remembers everything from the run and revises with
full context.

> [!TIP]
> Follow-ups in the same tab are almost always faster and better than re-running from
> scratch, because the research and files from the first run are reused.

## 8. Housekeeping — what's safe and what isn't

| Control | What it does | Safe for operators? |
|---|---|---|
| **🗑 Delete task** | Permanently removes one saved task and its history. | Yes — but only for your own finished tasks you no longer need. |
| **🧹 Clear memories** | Erases everything the employee has learned across *all* tasks. | **No.** Administrator only. (There is an **↩** restore button, but don't rely on it.) |
| **🔑 Keys** | Manages the service credentials the employee uses. | **No.** Administrator only — never edit or share keys. |
| **📊 Usage** | Shows how much the AI service has been used, by day. | Yes — read-only; useful if asked to report usage. |

## 9. Troubleshooting

| Symptom | What it usually means | What to do |
|---|---|---|
| Page won't load | The server is down. | Contact the administrator. Running tasks will resume once it's back. |
| Task seems frozen | Most often it's paused, waiting for an approval. | Check the **⬡ Flow** view for a pending approval prompt (Section 5). If there is none and nothing changes for 15+ minutes, press **Stop**, then **Resume from last checkpoint**. |
| Result is wrong or incomplete | Usually a goal that was too vague. | Use **↳ Ask follow-up** to correct it, and check **🔎 Explain** to see where it went off course. Refine the goal next time (Section 3). |
| A step shows as failed | The employee retries failed steps on its own within limits. | Wait first. If the whole task ends in failure, note the task name and time and report it (Section 10). |
| Asked to approve something odd | Possibly a planning mistake. | Don't approve. Leave it paused and escalate. |

## 10. Escalation

Contact the administrator — don't work around the problem — when any of these happen:

- The console is unreachable, or every new task fails immediately.
- An approval request asks for something you can't confidently connect to your goal.
- A task produced output that looks sensitive, wrong, or outward-facing (e.g. it drafted
  an email you didn't expect) — **do not send or publish it**; flag it first.
- Anything involving **🔑 Keys**, **🧹 Clear memories**, or the auto-approve toggle.

When you report a problem, include: the **task name**, the **time** it happened, and a
screenshot of the **⬡ Flow** view. That's enough for the administrator to find the full
record.

## 11. Quick reference

| Control | The whole workflow in six lines |
|---|---|
| **▶ Run** | Type a specific goal with a named deliverable, then run it. |
| **⬡ Flow** | Watch progress; this is where approval pauses appear. |
| **Approve & continue** | Approve only what clearly serves your goal; unsure = leave paused & ask. |
| **★ Result** | Collect the finished report, charts, and files. |
| **↳ Ask follow-up** | Refine the result in the same tab instead of re-running. |
| **Stop** / **Resume** | Pause any time; it always continues from the last saved step. |

---

*SOP-AIE-001 · v1.0 · Review due Feb 2027. Questions and change requests go to the
document owner (Engineering). This procedure covers operator use of the AI Employee Web
Console only; installation and configuration are covered in the administrator runbook.*
