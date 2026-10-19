# Resource isolation & task ↔ source links

The resource catalog now keeps a **bidirectional link between every task and every
source it owns or uses**, uses that link for **reference-counted deletion**, and
supports **isolating tasks** so no source stays shared.

Two kinds of source exist:

| Kind | Meaning | Who may read it |
|---|---|---|
| **shared** | a reusable model / dataset / file | any task in the workspace |
| **private** | owned by exactly one task | only the owning task |

Everything below is enforced in `src/ai_employee/resources/` — outside the model —
so no prompt can opt out of it.

---

## 1. Task ↔ source links (both directions)

Every resource record carries a list of `ResourceLink` edges
(`resources/models.py`):

```python
ResourceLink(
    task_id="<run id>",      # the concrete TaskRecord.id, when known
    task_key="<session key>",  # stable ownership identity (safe slug)
    session_id="<session id>", # groups the runs of one session
    role="owner" | "consumer",
    user_id=..., ts=...,
)
```

- **task → sources**: `manager.resources_for(task)` — everything a task links.
- **source → tasks**: `manager.tasks_for(resource_id)` / `links_for(...)` — every
  task linked to a source, so the console can show **“this same source is used by
  N tasks.”**
- `manager.reference_count(resource_id)` — the number of distinct tasks.

Links are attached automatically:

- when a download is indexed (`tools/builtins/resources.py::index_download`), and
- when an agent reads a resource (`get_resource`, consumer edge), and
- whenever `register`/`put` is called with a task.

Links are **idempotent** (a repeated read by the same task does not add duplicates)
and records created before links existed (only `owner_task`) are treated as having
an implicit owner edge, so counting stays correct.

Example:

```
Source                       Visibility  #tasks  Linked tasks
demo_shared_model.bin        shared      2       demo-task-A, demo-task-B
```

---

## 2. Reference-counted deletion

Deleting a task (or a run, or a whole session) removes that task's **links**; a
source is removed only when the deleted task was its **last** link
(`ResourceManager.delete_for_task` / `delete_for_session`):

```
2 tasks share one source
  delete(task A) -> source KEPT   (task B still links it)
  delete(task B) -> source DELETED (last link gone; bytes purged)
```

- Private sources are isolated to one task, so deleting that task removes them.
- Shared sources survive as long as any other task links them.
- Bytes are purged only if **no other record** still references the same content
  hash (the content-addressed object store is shared, so this check is essential).

The result reports `{unlinked, deleted, kept, bytes_freed}`.

This is wired into the console: `DELETE /ui/api/sessions/{id}` and
`.../runs/{task_id}` now run the cascade too, so deleting a task from the UI cleans
its sources — and only its sources.

---

## 3. Isolating tasks

`isolate` turns every **shared** source a task uses into a **task-private copy**:

1. **Copy** — the shared bytes are copied into
   `tasks/<task>/<YYYY>/<MM>/<DD>/resources/` as a **real, independent** copy
   (never a hard-link, so it cannot be shared through the object store).
2. **Re-download** — if the shared bytes are gone but the source has a URL, the
   URL is re-fetched from the internet into the private area
   (`resources/isolation.py`). Sources with neither bytes nor URL are reported
   `unrecoverable`.
3. **Purge** — a shared record left with **no task links** after isolation is
   deleted, so nothing shared remains.

```
isolate(all) -> isolated=1 purged=['d71ceb2c…'] pending=0
=> 0 shared · 1 private
```

- One task:  `manager.isolate_task(task_ref)`
- All tasks: `manager.isolate_all()` → every task, then purge orphaned shared records.

The async orchestrator (`resources/isolation.py`) adds the network step and is what
the API/CLI/UI call; the manager itself stays filesystem-only.

---

## 4. Where to use it

| Surface | What |
|---|---|
| **Console** | header badge **🗂 Resources** → table of sources with their linked tasks; **🔒 Isolate all tasks**, and **🗑 Delete all** (wipe every resource for a fresh start) |
| **API** | `GET /ui/api/resources` · `POST /ui/api/resources/isolate` `{all\|task_id\|session_id, allow_redownload}` · `POST /ui/api/resources/purge` `{purge_bytes}` · `DELETE /ui/api/resources/{id}` |
| **CLI** | `python -m ai_employee resources links` · `resources isolate --all` (or `--task <id>`) [--no-redownload] · `resources purge` [--keep-bytes] |
| **Python** | `employee.resources_overview()` · `employee.isolate_resources(all=True)` · `employee.purge_resources()` · `employee.delete_task_resources(session_id=…)` |

### Delete everything (fresh start)

`🗑 Delete all` in the Resources panel (or `POST /ui/api/resources/purge`, or
`python -m ai_employee resources purge`) wipes **every** resource:

- all catalog entries,
- the files/objects each record points at, the managed `shared/` and `tasks/`
  trees, and the content-addressed object store,
- the legacy `assets.json` download cache (so future runs re-acquire instead of
  reusing old files).

`purge_bytes: false` (or `--keep-bytes`) empties the catalog but leaves files on
disk. The action is **irreversible** — the console asks for confirmation first.

---

## 5. What changed in the data model

- `ResourceRecord.links: list[ResourceLink]` (new) and `ResourceLink` in
  `resources/models.py`.
- Catalog schema bumped to **3** (`resources/catalog.py`); older catalogs load fine
  — a record without `links` falls back to its `owner_task`.
- `register` / `put` / `_add` accept `task_id`, and shared records link the
  downloading task as a `consumer` (they stay global, `owner_task=None`, so many
  tasks share **one** record).

---

## 6. Guarantees

| Requirement | How it holds |
|---|---|
| Each task has an id linked to its sources | `ResourceLink.task_id`/`task_key` + `resources_for` |
| Source knows its tasks | `tasks_for` / `links_for` |
| Multiple tasks on one source are known | `reference_count`, the console `#tasks` column |
| Deleting a task keeps a shared source | cascade removes only the deleted task's links |
| Deleting the last task removes the source | cascade deletes a record with no remaining links (+ purges bytes when unreferenced) |
| Two kinds of source | `Visibility.private` (task) vs `Visibility.shared` (reusable) |
| “Isolate” leaves nothing shared | copy → (re-download if needed) → purge orphaned shared records |
| No prompt can bypass it | all logic lives in the resource manager / API, not in agent prompts |

Covered by `tests/test_resource_links.py` (links, reference-counted delete,
real-copy isolation, re-download path, API, CLI, UI) and the existing
`tests/test_resources.py`.
