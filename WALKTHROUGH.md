# Scenario walkthrough

[DESIGN.md](DESIGN.md) explains each mechanism and why it exists.
[EVALUATION.md](EVALUATION.md) explains how it was measured.

This document is a third view: following one request through the code,
scenario by scenario. Every step names the file and line that runs, and every
scenario names the test that covers it, so any claim here can be checked
directly.

---

## The pipeline in one picture

```
user message
     │
     ▼
InteractionAgentRuntime.execute                    runtime.py:65
     │
     ├─ build prompt ──► _render_active_agents     agent.py:48      ◄── scenario 8
     │
     ▼
LLM decides to delegate ──► send_message_to_agent  tools.py:142
     │
     ├─ (1) exact name match? ─────────────────────────────────────► scenario 1
     ├─ (2) embed + rank ──► find_candidates       agent_matcher.py:139
     │        └─ embedding failed? lexical fallback ───────────────► scenario 7
     ├─ (2b) top two within margin? ───────────────────────────────► scenario 3
     ├─ (3) similarity floor ──────────────────────────────────────► scenario 2
     ├─ (4) decide_routing (the judge)             agent_matcher.py:275
     └─ create_or_link                             roster.py:386    ◄── scenario 9
              │
              ▼
     _dispatch_to_agent (fire and forget)          tools.py:268
              │
              ▼
     ExecutionBatchManager.execute_agent           batch_manager.py
              ├─ resolve_name (follow merges) ─────────────────────► scenario 11
              ├─ per-agent lock ───────────────────────────────────► scenario 12
              ├─ runtime.execute  (does the Gmail work)
              └─ reconcile_around                  agent_matcher.py:493
                       ├─ forward link ────────────────────────────► scenario 4
                       └─ reverse link ────────────────────────────► scenario 5
```

---

## Scenario 1 — The same agent, named identically

> "Follow up with Keith" → the model calls `send_message_to_agent("Email to Keith", ...)`
> and that agent already exists.

| Step | Code | What happens |
|---|---|---|
| 1 | `tools.py:151` | Load roster from disk |
| 2 | `tools.py:157` | `get_record(agent_name)` hits — exact match |
| 3 | `tools.py:158` | `resolve_name()` in case it was merged away |
| 4 | `tools.py:159` | `mark_active()` — refresh recency |
| 5 | `tools.py:160` | Dispatch, `action="reused"` |

No extra embedding call, no extra LLM call. This is deliberately the first
branch: it's the common case, and the change shouldn't make it slower than the
code it replaced. "No extra" is the precise claim, not "no inference" — the
interaction agent is already an LLM invoked once per turn regardless of which
branch fires, and it's that same call that has to recognise "Keith" refers to
the existing `agent_name` and reproduce it exactly. What this branch actually
saves is a second, separate call: no `embed_text()` request and no judgment
call get added on top of the interaction agent's own turn, which is what a
retrieval or judgment branch would cost instead.

Covered by `test_exact_name_match_reuses_without_any_network_call`, which
asserts the embedding call count is unchanged, so a regression here fails the
suite.

---

## Scenario 2 — A paraphrase of an existing thread

> "did keith ever get back to me about lunch?" when `Email to Keith about lunch` exists.

| Step | Code | What happens |
|---|---|---|
| 1 | `tools.py:157` | No exact match — fall through |
| 2 | `tools.py:168` | Embed `"{agent_name}. {instructions}"` |
| 3 | `tools.py:171` | `find_candidates` ranks the whole roster by cosine |
| 4 | `tools.py:178` | Drop anything below `agent_min_candidate_similarity` (0.20) |
| 5 | `tools.py:189` | Top two far apart → not ambiguous, continue |
| 6 | `agent_matcher.py:275` | `decide_routing` — the judge sees only the shortlist |
| 7 | `tools.py` | Judge names a duplicate → build a `DuplicateLink` |
| 8 | `roster.py:386` | `create_or_link` — a new agent is created, carrying the link |

The important part: no merge happens here. A separate agent exists, both are
independently addressable, and the link is only a hypothesis. Text similarity
alone doesn't fuse histories — that's the central design decision behind this
whole change.

Covered by `test_similar_task_stages_a_link_but_does_not_merge`.

---

## Scenario 3 — Two people named Keith

> "follow up on lunch w/ Keith" with both `Keith Rivera lunch` and `Keith Chen lunch`
> in the roster. Measured live, they score 0.0089 apart.

| Step | Code | What happens |
|---|---|---|
| 1–4 | as above | Both Keiths clear the floor |
| 5 | `tools.py:189-191` | Top-two margin ≤ `agent_ambiguity_margin` (0.05) → ambiguous |
| 6 | `tools.py:192` | Collect every tied candidate |
| 7 | `tools.py:210` | Return immediately — `status: needs_clarification` |

Nothing is created. No link. No execution agent is started. The tool result
carries `ambiguous_with` and a note; `system_prompt.md` instructs the model to
ask which Keith.

Why no work starts: dispatching while simultaneously asking "which one?" would
be a bit theatrical — the agent could email the wrong Keith before the answer
arrives, and asking after an irreversible action doesn't accomplish much.

How it resumes: the user says "Chen," the model calls
`send_message_to_agent("Keith Chen lunch", ...)`, and that takes scenario 1.
The conversation loop is the resume mechanism; no pending-state machine exists
or needs to.

Covered by `test_tied_candidates_start_no_work`,
`test_clarified_followup_routes_by_exact_match`.

---

## Scenario 4 — Proof arrives, and the merge commits

> The linked agent from scenario 2 does Gmail work and touches the same thread.

| Step | Code | What happens |
|---|---|---|
| 1 | `batch_manager.py` | Execution finishes; logs now contain tool calls |
| 2 | `agent_matcher.py:493` | `reconcile_around` runs |
| 3 | `agent_matcher.py:440` | `reconcile_link` — this agent holds a link |
| 4 | `agent_matcher.py:406` | `score_structural_evidence` compares both logs |
| 5 | — | Shared `thread_id` → confidence 1.0 |
| 6 | `roster.py:448` | 1.0 ≥ 0.9 threshold → `merge_agent` |

Confidence comes from structural evidence alone, never combined with the
judge's text score. An earlier version took `max(text, structural)`, which let
a confident LLM plus a weak signal clear the bar. That's fixed, with a
regression test parameterised to 0.99.

Covered by `test_shared_thread_id_promotes_a_staged_link_to_a_merge`,
`test_llm_confidence_can_never_drive_a_merge`.

---

## Scenario 5 — Proof arrives on the other side

> The new agent ran days ago. The older agent finally runs — via a trigger — and
> logs the same thread.

| Step | Code | What happens |
|---|---|---|
| 1 | `agent_matcher.py:493` | `reconcile_around(older_agent)` |
| 2 | `agent_matcher.py:440` | Forward check: older agent has no link → nothing |
| 3 | `roster.py:249` | Reverse check: `links_pointing_at(older_agent)` |
| 4 | `agent_matcher.py:440` | Reconcile each source found → evidence now exists → merge |

Why this branch exists: evidence is symmetric — a shared thread id proves
identity regardless of whose log it landed in — but a link is stored only on
the source agent. Reconciling just the agent that ran would miss the more
common ordering: the newer agent carries the link and runs immediately, while
the older target may sit idle for days. Found by review; without this, a
provable duplicate could have stayed unmerged indefinitely.

Covered by `test_evidence_arriving_on_the_target_side_still_merges`.

---

## Scenario 6 — Two agents read the same email (should not merge)

> A search agent and a summarise agent both surface `message_id: msg998877`.

| Step | Code | What happens |
|---|---|---|
| 1 | `agent_matcher.py:406` | Compare structural evidence |
| 2 | — | No shared thread, but a shared message |
| 3 | — | Confidence 0.6 — below the 0.9 threshold |
| 4 | `agent_matcher.py` | Link updated with the evidence; no merge |

Why this distinction matters: an earlier version pooled thread, message, and
draft ids into one set and scored any intersection 1.0. Two unrelated searches
surfacing one email would then have merged irreversibly. A thread is the
conversation; a message is just an object both agents happened to see.

Covered by `test_shared_message_id_is_not_proof_of_same_thread`,
`test_shared_draft_id_is_not_proof_either`.

---

## Scenario 7 — The embedding provider is down

| Step | Code | What happens |
|---|---|---|
| 1 | `agent_matcher.py` | `embed_text` catches the error, returns `None` |
| 2 | `agent_matcher.py:139` | `find_candidates` sees no vector → lexical overlap |
| 3 | `tools.py` | Routing continues; no `embedding_model` tag is stored |

The turn completes. Match quality degrades; nothing breaks outright. This is
meant to be safe by construction — weak evidence couldn't merge anything
anyway, so "no evidence available" is treated as a normal state, not an error.

Covered by `test_embedding_failure_still_creates_the_agent`,
`test_embedding_failure_leaves_no_stale_model_tag`.

---

## Scenario 8 — The roster grows past the prompt budget

> Every turn, before the model sees anything.

| Step | Code | What happens |
|---|---|---|
| 1 | `agent.py:48` | `_render_active_agents(query_text)` |
| 2 | — | `len(records) ≤ agent_prompt_top_k`? → render all, no embedding call |
| 3 | — | Otherwise: embed the turn, rank by relevance |
| 4 | — | Recent-agent slots filled first, remaining budget filled by rank — a capped fill, not a true union |
| 5 | — | Render `name` + `description` |

Two things this is meant to guarantee: small installs pay no added latency, and
prompt size stays capped regardless of roster size — the cap is on the total
rendered, not on each component separately (see DESIGN.md's "Bounded prompts"
for why that distinction matters). Archived and merged agents are excluded.

Covered by `test_small_roster_makes_no_embedding_call`,
`test_large_roster_is_capped_at_top_k`,
`test_recent_agents_survive_even_when_semantically_distant`.

---

## Scenario 9 — Two agents whose names collide as filenames

> The model proposes `Keith/Lunch` while `Keith Lunch` exists. Both slugify to
> `keith-lunch`.

| Step | Code | What happens |
|---|---|---|
| 1 | `roster.py:338` | `resolve_available_name` compares slugs, not names |
| 2 | — | Collision with a different agent → append a suffix |
| 3 | — | Stored as `Keith/Lunch (2)` |

Without this, the log store — which derives filenames by slugifying — would
give two deliberately distinct agents one shared history file. Fixing it at
the roster level means the log store, triggers table, and runtime need no
changes at all.

Covered by `test_slug_colliding_names_get_distinct_records`.

---

## Scenario 10 — An agent goes quiet, then is needed again

| Step | Code | What happens |
|---|---|---|
| 1 | `roster.py:297` | `is_archived` computed from `last_active` age — nothing stored |
| 2 | — | Owns a non-completed trigger? → exempt, however idle |
| 3 | `agent.py:48` | Archived agents excluded from the default prompt |
| 4 | `tools.py:301` | `search_agents` searches the full index, archived included |
| 5 | `roster.py:416` | Any reuse calls `mark_active` → back in the default view |

Archived isn't meant to mean unreachable. Retirement affects only what's shown
by default.

Covered by `test_idle_agent_is_archived`,
`test_live_trigger_exempts_an_idle_agent_from_archival`,
`test_search_finds_an_agent_too_idle_for_the_prompt`.

---

## Scenario 11 — A trigger fires on an agent that was merged away

| Step | Code | What happens |
|---|---|---|
| 1 | `trigger_scheduler.py` | Trigger stores its owner by name — a stale one |
| 2 | `batch_manager.py` | `resolve_name` follows `merged_into` to the survivor |
| 3 | — | Survivor executes; `mark_active` is not called |

Two separate things here. Merge pointers keep stale trigger rows working. And
a trigger firing isn't treated as user engagement — if it refreshed recency, a
recurring reminder would look permanently active despite zero human contact,
which would defeat the point of retirement.

Covered by `test_exact_match_on_a_merged_agent_redirects_to_the_survivor`,
`test_trigger_execution_does_not_refresh_last_active`.

---

## Scenario 12 — Two turns hit the same agent at once

| Step | Code | What happens |
|---|---|---|
| 1 | `batch_manager.py` | Acquire the roster's per-agent `asyncio.Lock` |
| 2 | — | Second turn waits; unrelated agents run in parallel |
| 3 | — | Load transcript → execute → append, without interleaving |

The pre-existing `_batch_lock` only ever guarded batch counters, not the
execution itself. Better dedup makes this collision more frequent, so it
needed fixing alongside the rest. Roster writes are guarded separately in
`create_or_link`.

Covered by `test_same_agent_executions_serialize`,
`test_concurrent_creates_do_not_duplicate_a_record`.

---

## Scenario 13 — A merge chain

> `A` merges into `B`. Later `B` merges into `C`.

| Step | Code | What happens |
|---|---|---|
| 1 | `roster.py:233` | `resolve_name(A)` walks the chain → `C` |
| 2 | `roster.py:268` | `merged_sources(C)` walks the whole graph → `[A, B]` |
| 3 | `execution_agent/agent.py` | C's prompt inherits both transcripts |

Previously `merged_sources` matched only direct children, so C loaded B's
history and silently dropped A's — breaking the guarantee that merging
preserves context, in the one direction that wouldn't have been noticed easily.
Both functions carry a cycle guard.

Covered by `test_chained_merges_preserve_every_ancestor`,
`test_chained_merge_history_reaches_the_final_survivor`.

---

## What no scenario covers

- **A real Gmail merge.** Scenarios 4, 5, and 6 are exercised with synthetic log
  entries. Committing a merge against a live mailbox has never been run.
- **Multi-process roster writes.** Scenario 12 covers concurrency within one
  process. See NOTES.md finding 7 for the inherited file-locking bug.
- **Whether the model actually asks in scenario 3.** The tie is surfaced and
  the system prompt instructs clarification, but nothing enforces it.
