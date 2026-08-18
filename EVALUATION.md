# How this was evaluated

The take-home asks for test cases and evaluators, and for an explanation of how
correctness was assessed. This document is that explanation: what each
measurement was trying to establish, the code that does it, and what actually
came out, including results that didn't match the original expectations.

Implementation rationale lives in [DESIGN.md](DESIGN.md); additional failure
points found along the way are in [NOTES.md](NOTES.md).

```bash
python -m pytest server/tests -v                    # 111 tests, offline, <1s
python server/tests/eval_degradation.py             # does the problem exist?
python server/tests/eval_ablations.py               # does each layer earn its place?
python server/tests/eval_agent_matching.py          # is routing correct?   (needs API key)

python server/tests/eval_degradation.py --live      # adds the real-LLM curve
python server/tests/eval_ablations.py --live        # real embeddings + real judge
```

---

## Terminology

Plain definitions for terms used throughout. Several are standard
information-retrieval vocabulary; a couple are specific to this problem.

| Term | Meaning |
|---|---|
| Roster | The list of execution agents that exist. `N` is its size. |
| Distractor | A wrong agent added to the roster purely to make matching harder — the padding that takes a test from 5 agents to 200. |
| Confusable distractor | A distractor deliberately similar to the right answer (same person, different topic). Random distractors make a test trivially easy; these make it closer to real. |
| recall@k | Whether the correct agent is somewhere in the top *k* shown to the model. Pass/fail, and fairly decisive — if it isn't in the shortlist, the model can't pick it. |
| MRR (mean reciprocal rank) | How high the right answer ranked. #1 → 1.0, #2 → 0.5, #4 → 0.25, averaged over cases. |
| Margin | Gap between the right agent's score and the best wrong agent's score. Winning by 10 seconds and by 0.1 seconds are both wins; one is more fragile than the other. |
| Ablation | From medicine — remove one part to see what it did. Switch off one pipeline stage, hold everything else fixed, measure what breaks. |
| Staged link | A recorded hypothesis that two agents might be the same thread. Reversible, invisible to the user. |
| Committed merge | An actual merge. Two agents' histories become one. Effectively irreversible. |
| False merge | Wrongly merging two distinct threads — the expensive error here, since it's silent, permanent, and corrupts context going forward. |
| Correct abstention | Declining to link when nothing fits, and creating a new agent instead. Roughly the behaviour that separates a usable system from one that always force-matches. |
| Flip rate | Share of runs where the model answered differently on identical input. Measures instability, which a single run hides entirely. |
| Rule of three | Statistics: observing zero failures in *n* trials bounds the true rate at roughly `3/n` at 95% confidence. Zero in 40 trials means "≤8%," not "zero." |

---

## The organising idea: three layers, three strategies

A single testing approach doesn't really fit, because the system has three
layers that behave differently across repeated runs:

| Layer | Deterministic? | Strategy |
|---|---|---|
| Pure logic — cosine, slug collisions, roster state | Always | Ordinary unit tests |
| Embedding retrieval | Yes for a fixed model, but needs network | Real embeddings live; a synthetic stand-in offline |
| LLM judgment | No — same input, different answers | Sample N times, measure the distribution |

That middle row is roughly what makes most of this runnable offline for free.

---

## Step 1 — Prove the problem exists

**What this step was trying to establish.** A fix for roster bloat is hard to
justify without first showing the problem actually degrades something. One
variable changes — roster size — and the rest is watched for what breaks.

**The code** (`eval_degradation.py`). Distractors are deliberately confusable,
because an earlier version wasn't:

```python
def build_distractors(count, targets):
    """Plausible competing agents, deliberately confusable rather than random.

    A first version filled the roster with agents sharing no person and no topic
    with any target, which made every correct agent rank first at every N - a flat
    100% recall that measured nothing. Synthetic data that clean flatters the eval.

      ~40% share a target's PERSON (different topic)  <- the false-merge shape
      ~40% share a target's TOPIC  (different person)
      ~20% unrelated filler
    """
```

Three curves are measured across N ∈ {11, 15, 25, 50, 100, 200}:

**(a) Prompt cost** — tokens in the `<active_agents>` block, paid every single turn:

```
      N    old (all agents)     new (top_k)
     11                  99             320
     50                 400             403
    200               1,576             403     <- old still climbing, new capped
```

**(b) Retrieval margin** — the error budget:

```
      N    recall@k    MRR    mean margin    worst margin
     11        100%   1.00          0.971           0.870
    200        100%   1.00          0.344           0.000
```

**(c) Selection accuracy** — can a real model actually pick, with 95% intervals:

```
      N       old accuracy (95% CI)       new accuracy (95% CI)
     11     100% [82%-100%]             100% [82%-100%]
     25     100% [82%-100%]             100% [82%-100%]
    200      94% [74%-99%]              100% [82%-100%]
```

(As of the latest live run, retrieval on both sides of this comparison uses real
OpenRouter embeddings rather than a synthetic stand-in — `eval_degradation.py
--live` previously kept the shortlist synthetic and only made the judge call
real, which understated what "live" was meant to mean here. That's fixed; see
DESIGN.md's degradation table for the full N-sweep with the correction applied.)

### Outcome

- **Cost degrades badly, and the fix caps it.** 16× growth, unbounded, versus a
  fixed 403-token ceiling. This half of the thesis holds up.
- **Margin erodes 73%** (0.870 → 0.235). Rank never flips, but the safety gap
  narrows a lot — that gap is roughly how much noise a correct match can absorb
  before a confusable neighbour overtakes it.
- **Accuracy shows no resolvable degradation.** Every interval overlaps every
  other. Sonnet handled 200 unranked agents about as well as a small roster in
  this test. The accuracy justification for prompt pruning didn't hold up and
  was withdrawn; only the cost justification survives.
- **Unexpected finding:** margin stops eroding around N≈10. Degradation seems to
  track confusable density rather than roster size — five agents about the same
  person hurt far more than a hundred about strangers. "You have 300 agents" on
  its own isn't the right alarm to raise.

### What wasn't done

The reference spec suggests testing to N=500; this stops at 200.

---

## Step 2 — Build a labeled set

**What this step was trying to establish.** `(request → correct agent)` pairs
with known ground truth, along with an honest account of the biases
hand-authoring introduces.

**The code** (`eval_agent_matching.py`). Each case is a request, a roster, and
the answer:

```python
EvalCase(
    "reuse-typo",
    "true-reuse",
    "Emial to Kieth about lunhc",
    "Send Kieth another note about lunhc next Tuesday.",
    _KEITH_LUNCH.name,          # <- ground truth: should link here
    _BASE_ROSTER,
),
EvalCase(
    "adversarial-same-person-different-topic",
    "adversarial",
    "Keith birthday gift",
    "Order a birthday gift for Keith Rivera and have it delivered Friday.",
    None,                       # <- ground truth: should NOT link to anything
    _BASE_ROSTER,
),
```

### Outcome: the dataset was rebuilt three times

Each rebuild happened because the set scored 100% while measuring close to
nothing:

1. **Distractors too different.** Roughly like finding a friend in a crowd of
   giraffes.
2. **Requests were near-perfect descriptions of their target.** Closer to
   searching with an exact photograph than a real request. Real requests tend
   to be vague: `"Keith lunch f/u"`.
3. **Every adversarial case fell below the similarity threshold**, so the LLM
   judge was never consulted on anything hard. The score mostly measured the
   threshold.

Fix for the third one — the harness now reports this permanently, so the same
failure shouldn't be able to recur silently:

```
Judge consulted on 14/17 cases. Cases the floor filters out never test it,
so a score dominated by those is really a score for the threshold, not the judge.
```

### Documented biases

- Hand-authored, so likely biased toward failure modes already imagined —
  probably the set least likely to contain the surprising ones.
- The offline tier uses a synthetic embedding model whose semantic relationships
  were invented. It shows the logic works, not that retrieval works.
- No real traffic anywhere in this dataset.

---

## Step 3 — Metrics that match the failure mode

**What this step was trying to establish.** Accuracy alone looks like the wrong
headline, because the two errors aren't equally bad. A missed link leaves a
harmless duplicate. A false merge silently and permanently fuses two unrelated
threads.

**The code.** Five buckets, not four:

```python
tally = {"tp": 0, "tn": 0, "false_merge": 0, "missed": 0, "clarify": 0}
for kind, linked_to in self.decisions:
    expected = self.case.expected_link
    if kind == CLARIFY:
        tally["clarify"] += 1              # asked the user - neither right nor wrong
    elif linked_to is not None and linked_to == expected:
        tally["tp"] += 1                   # linked, correctly
    elif linked_to is not None:
        tally["false_merge"] += 1          # THE EXPENSIVE ERROR
    elif expected is None:
        tally["tn"] += 1                   # created new, correctly = ABSTENTION
    else:
        tally["missed"] += 1               # should have linked, did not
```

Correctly creating a new agent is a true negative, not a true positive.
Collapsing the two would inflate recall — a mistake this harness caught in
itself early on (it once reported 220% recall).

**Staged links and committed merges are scored separately**, because the
pipeline is defence in depth, and whichever layer gets measured tends to get
credit for all the layers before it:

```
Links staged by the judge: 60   |   merges committed after the evidence gate: 0
```

### Outcome

| Metric | Result |
|---|---|
| recall@k | 100% at every roster size |
| Accuracy / precision / recall / F1 | 100% / 100% / 100% / 1.00 |
| Committed false merges | 0 across every live run |
| Correct abstention | Decomposed by mechanism — floor vs. judge vs. evidence gate vs. ambiguity |
| Exact-match baseline | 0% recall on the same cases |

**The judge isn't perfect.** Across roughly 340 live decisions, one batch of 85
produced 3 bad links — "Vercel contractor invoice" linked to the Vercel *offer*
thread, and a dental billing question to the dental *appointment*. The other
~255 produced none. All three were stopped by the evidence gate.

### Not covered

**Latency isn't measured.** Token counts and API-call budgets are asserted as
regression tests, but there's no wall-clock timing anywhere. The source
material names latency as a constraint, so this looks like a real gap.

---

## Step 4 — Ablations

**What this step was trying to establish.** Whether every stage of the pipeline
earns its place. Anything that can't show a delta is a candidate for deletion.

**The code** (`eval_ablations.py`). Each row flips exactly one switch:

```python
CONFIGS = [
    Config("SHIPPED (all layers)"),
    Config("A5  no evidence gate",          use_evidence_gate=False),
    Config("A2  no LLM judgment",           use_judgment=False),
    Config("A3  lexical, no embeddings",    use_embeddings=False),
    Config("A1  no retrieval (full roster)", show_full_roster=True),
    Config("A4  no similarity floor",       use_floor=False),
    Config("A6  top_k=1",                   top_k=1),
    Config("A6  top_k=10",                  top_k=10),
]
```

Run both offline (synthetic) and `--live` (real embeddings + real Sonnet).

### Outcome

| Ablation | Live result | Reading |
|---|---|---|
| A3 lexical instead of embeddings | recall 100% → 33% | Embeddings look justified |
| A5 remove evidence gate | run 1: caught 3 real bad links. Rerun: 0 — the judge staged nothing bad that time, so the gate had nothing to catch | Central design decision, functioning as insurance — not yet shown necessary on this scale of live traffic; see below |
| A2 remove LLM judgment | bad links staged 0 → 2 | Judge looks justified |
| A1 no retrieval, show everything | recall 100% → 100% | Retrieval isn't justified on recall alone — mainly cost |
| A4 no similarity floor | recall 33% → 100% | Floor looked harmful here — changed a shipped default |
| A6 top_k sweep (now measured, not just theorized: top_k=1 and top_k=10) | recall 100%, 0 false merges at both | Inconclusive — roster too small to say much |

**On the A5 rerun.** A second live pass of the same 8-scenario ablation,
run months apart, produced a different judge outcome: zero bad links staged, so
removing the evidence gate changed nothing that time — there was nothing to
catch. That doesn't contradict the first result; it looks like the same "small
samples of judge behavior are noisy" pattern DESIGN.md's routing-quality section
also runs into (a 3-run sample once showed zero bad links and was, at the time,
read as evidence the gate was unproven, when it was probably just
underpowered). Read both runs together rather than either alone: on this model,
judgment errors that need the gate appear to be real but infrequent enough that
an 8-scenario set won't reliably surface one, which is roughly why the gate is
best thought of as insurance rather than something expected to fire on every
small evaluation. The rule-of-three bound on the false-merge rate (~8% at 95%
confidence from the 40-decision adversarial set) is still the honest ceiling; a
clean rerun lowers confidence that the true rate sits near that ceiling, but
doesn't move the bound itself.

### A4 changed the product

The threshold had been calibrated from a clean sweep on one dataset: 0.60 kept
5/5 true reuse and blocked every adversarial. It looked like data-driven tuning
at the time. Then the ablations, on a different set, showed the same 0.60 floor
cutting recall to 33%.

Pooling every real similarity measured across both sets:

```
TRUE REUSE   0.246 ────────────────────────────── 0.765
ADVERSARIAL          0.505 ─────────── 0.665
UNRELATED    0.172
```

The classes interleave almost completely. No threshold cleanly separates them;
the apparent separation looks like an artifact of which cases happened to be
picked — something close to textbook overfitting to the calibration set,
noticed only because a second dataset disagreed.

Consequence: the floor doesn't look like it can serve as a safety mechanism at
any value. It dropped to 0.20, closer to a noise filter than a decision
boundary, and discrimination moved mostly to the judge, which live testing
suggests handles it reasonably. Cases reaching the judge went from 9/17 to
14/17.

### On reranking

A reranker was considered and set aside based on this evidence: `recall@k` is
already 100%, so there's not much for it to reorder; the no-retrieval ablation
scored 100%, so the shortlist doesn't look like where quality is lost; and the
LLM judge already reads the request and candidates jointly, which is close to
what a cross-encoder reranker would do. The trigger to revisit this would be
`recall@k` falling as the roster grows, which `eval_degradation.py` measures and
which is currently flat.

### Not covered

The reference spec suggests ablating a recency/frequency prior against a
larger, purpose-built dataset. What exists instead is a single live trial
(`eval_degradation.py --live`, section (d)): a just-touched agent, given a vague
pronoun-style follow-up, survived prompt rendering both with and without the
recency guarantee. One trial doesn't settle it either way — see DESIGN.md's
"Known limitations" for the fuller read — but "never measured at all" is no
longer accurate; it's closer to "measured once, inconclusively."

That same live check also gave `_render_active_agents` its first real coverage:
recall@15 was 100% (18/18) at N=51 with real embeddings on both sides, closing a
gap where every unit test for that function stubs embeddings and none of its
test-fixture agents ever carry a stored vector at all.

---

## Step 5 — Adversarial cases

**What this step was trying to establish.** The more interesting behaviour
tends to live where two things look alike but aren't. These cases were written
deliberately, and in the hardest ones the similarity was measured first to make
sure they'd reach the judge rather than get filtered out early.

```python
# Two different people sharing a first name AND a topic. Retrieval scores them
# 0.0089 apart - indistinguishable. Picking is a coin flip; abstaining is the only
# right answer.
EvalCase(
    "adversarial-two-people-same-name-same-topic",
    "adversarial",
    "Lunch with Keith",
    "Follow up on lunch with Keith.",
    None,
    _TWO_KEITHS_ROSTER,
),
# Same roster, but the request names WHICH Keith - so linking is correct. Paired
# with the case above to check the judge discriminates on the distinguishing
# detail rather than freezing whenever two names collide.
EvalCase(
    "reuse-disambiguated-by-surname",
    "true-reuse",
    "Lunch with Keith Chen",
    "Follow up with Keith Chen about our lunch.",
    _KEITH_CHEN_LUNCH.name,
    _TWO_KEITHS_ROSTER,
),
```

| Case type | Outcome |
|---|---|
| Two people, same name, same topic (0.0089 apart) | Ambiguity check fires; system asks the user and starts no work |
| Same request, surname supplied | Links correctly 5/5 — discriminates rather than freezing |
| Same person, adjacent topic (dinner vs lunch, 0.67) | Where the 3 bad links appeared |
| Same company, different matter (invoice vs offer, 0.60) | Unstable across runs — sometimes links |
| Requests matching nothing | Correctly creates new |
| Long-idle agents | Unit-tested: archival plus trigger exemption |

---

## Step 6 — What this evaluation does not cover

Stating limits plainly feels like part of the deliverable. An eval that
oversells its own coverage is arguably worse than a smaller, more honest one.

- **The merge path has never run against real Gmail.** Committing a merge needs
  two agents' logs to reference the same real thread ID. Unit-tested with
  mocked logs only, which is a gap worth noting since this is arguably the most
  important safety mechanism in the design.
- **No latency measurement**, despite it being named as a constraint.
- **Small samples.** Around 340 live judgment decisions. Rule of three bounds
  the false-merge rate at ≤8%, not at zero.
- **The evaluation is itself unstable at n=5.** Two identical 5-run invocations
  returned 3 bad links and 0. The aggregate is more informative than any single
  run.
- **One embedding model, one judge model.** Results likely don't transfer: the
  free `nvidia/nemotron-3-embed-1b` model inverts the classes (−0.19
  separation) and would probably break the system silently if swapped in.
- **Hand-authored cases only.** No production distribution behind them.
- **`recall@k` doesn't capture whether the conversation went well** — only
  whether routing was correct.
- **Anaphora is structurally unsolved.** "Ask them to raise the equity portion"
  scores 0.246; a pronoun carries no entity signal, and no threshold rescues it.

---

## Summary

> Three questions, three harnesses. Does the problem exist? For cost, fairly
> decisively — 16× unbounded prompt growth, capped by the fix. For accuracy,
> not really: the predicted degradation didn't appear, and that claim was
> withdrawn. Does each layer earn its place? Embeddings and the evidence gate
> look justified, with numbers behind them; the similarity floor turned out to
> be actively harmful and overfit to its own calibration set, which changed a
> shipped default; the top_k ablation is inconclusive and labelled as such.
> Is routing correct? 100% accuracy with zero committed false merges in these
> runs — though the judge stages a bad link roughly 1% of the time in one
> reading of the data, and the evidence gate is what stops those from becoming
> merges. The dataset needed rebuilding three times because it kept scoring
> 100% while measuring close to nothing.
