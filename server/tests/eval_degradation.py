"""Does agent selection actually degrade as the roster grows, and does the fix stop it?

    python server/tests/eval_degradation.py            # offline, deterministic, free
    python server/tests/eval_degradation.py --live     # adds the real-LLM curve

This is the "prove the problem exists" experiment. Improvement claims are meaningless
without it: a fix that makes selection better has to be measured against a baseline
that was getting worse.

Three curves, swept over roster size N:

  (a) PROMPT COST      tokens in the <active_agents> block.
                       Old: every agent, every turn -> grows linearly in N.
                       New: capped at agent_prompt_top_k -> flat.
                       Deterministic, no model involved.

  (b) RETRIEVAL RANK   where the correct agent lands once N-1 distractors compete.
                       recall@k and MRR. Deterministic given a fixed embedding model.
                       This is the honest measure of "competing for attention": as N
                       grows, more wrong agents can outrank the right one.

  (c) SELECTION        can the model actually pick correctly from what it is shown?
                       Old: choose 1 of N from an unranked list.
                       New: choose 1 of <=top_k, pre-ranked.
                       Requires a real LLM (--live). Offline, the run reports the
                       structural proxy - how many candidates the model must
                       discriminate among - and says so rather than simulating an
                       answer, because a mocked "the LLM gets worse with N" would
                       just be assuming the conclusion.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for path in (str(_REPO_ROOT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from server.config import get_settings  # noqa: E402
from server.services.execution.agent_matcher import (  # noqa: E402
    agent_embedding_text,
    cosine_similarity,
)
from synthetic_embeddings import Concepts, SyntheticEmbeddingModel  # noqa: E402


# Starts at 11 because that is how many target agents the labeled cases need - a
# smaller "roster size" would silently be clamped to 11 and mislabel the axis.
ROSTER_SIZES = [11, 15, 25, 50, 100, 200]

# Distractor vocabulary. Combined person x topic, this yields a few hundred
# plausible, distinct agents - enough to fill the largest roster without repeats.
_PEOPLE = [
    "alice", "bob", "carla", "dan", "elena", "farid", "grace", "hugo", "iris",
    "jonas", "kira", "liam", "maya", "noah", "olive", "pedro", "quinn", "rosa",
    "sam", "tara", "umar", "vera", "wes", "xena", "yuri", "zoe",
]
_TOPICS = [
    "invoice", "renewal", "onboarding", "scheduling", "feedback", "shipping",
    "refund", "contract", "interview", "proposal", "budget", "訪問", "handoff",
]


@dataclass
class LabeledCase:
    """A request whose correct agent is known."""

    case_id: str
    query_text: str
    query_concepts: Concepts
    correct_agent: str


@dataclass
class SyntheticAgent:
    name: str
    description: str
    concepts: Concepts

    @property
    def embedding_text(self) -> str:
        return agent_embedding_text(self.name, self.description)


def build_target_agents() -> List[SyntheticAgent]:
    """The agents the labeled requests are actually about."""
    return [
        SyntheticAgent(
            "Email to Keith about lunch",
            "Planning a lunch meeting with Keith Rivera, coordinating dates and venue.",
            Concepts(person="keith", topic="lunch"),
        ),
        SyntheticAgent(
            "Vercel job offer",
            "Negotiating compensation on the Vercel senior engineer offer.",
            Concepts(person="vercel", topic="offer"),
        ),
        SyntheticAgent(
            "Dentist appointment",
            "Rescheduling a dental cleaning with Bright Smile Clinic.",
            Concepts(person="dentist", topic="appointment"),
        ),
        SyntheticAgent(
            "Landlord heater repair",
            "Chasing the landlord about the broken heater in the apartment.",
            Concepts(person="landlord", topic="repair"),
        ),
        SyntheticAgent(
            "Q3 board deck",
            "Assembling the Q3 board deck and collecting metrics from each team.",
            Concepts(person="board", topic="deck"),
        ),
        SyntheticAgent(
            "Insurance renewal",
            "Comparing quotes and renewing the car insurance policy before it lapses.",
            Concepts(person="insurer", topic="renewal"),
        ),
        SyntheticAgent(
            "Conference travel",
            "Booking flights and a hotel for the Lisbon conference in September.",
            Concepts(person="travel", topic="booking"),
        ),
        SyntheticAgent(
            "Nadia design review",
            "Collecting feedback from Nadia on the new dashboard designs.",
            Concepts(person="nadia", topic="design"),
        ),
        SyntheticAgent(
            "Payroll correction",
            "Getting HR to fix the missing overtime hours on last month's payslip.",
            Concepts(person="hr", topic="payroll"),
        ),
        SyntheticAgent(
            "Gym membership cancel",
            "Cancelling the gym membership and confirming the final billing date.",
            Concepts(person="gym", topic="cancellation"),
        ),
        SyntheticAgent(
            "Visa paperwork",
            "Assembling supporting documents for the work visa application.",
            Concepts(person="immigration", topic="visa"),
        ),
    ]


def build_cases() -> List[LabeledCase]:
    """Requests worded differently from the agent that should handle them.

    Signal strengths encode how much each phrasing degrades the match. These are
    modelling assumptions, not measurements - see the caveat printed by the report.
    """
    return [
        LabeledCase(
            # Full sentence, both facets clear.
            "paraphrase",
            "Check whether Keith ever replied about getting lunch.",
            Concepts(person="keith", topic="lunch", person_strength=1.0, topic_strength=0.9),
            "Email to Keith about lunch",
        ),
        LabeledCase(
            # Terse shorthand: person survives, topic is barely there.
            "abbreviated",
            "Keith lunch f/u",
            Concepts(person="keith", topic="lunch", person_strength=1.0, topic_strength=0.45),
            "Email to Keith about lunch",
        ),
        LabeledCase(
            # Pronoun reference: no person named at all, topic carries it.
            "indirect",
            "Ask them to raise the equity portion of the offer.",
            Concepts(person="vercel", topic="offer", person_strength=0.3, topic_strength=1.0),
            "Vercel job offer",
        ),
        LabeledCase(
            # Different words for the same thing: both facets weakened.
            "different-vocabulary",
            "Move my teeth cleaning to a morning slot instead.",
            Concepts(person="dentist", topic="appointment", person_strength=0.6, topic_strength=0.6),
            "Dentist appointment",
        ),
        LabeledCase(
            "long-gap",
            "Did the landlord ever send someone about the heating?",
            Concepts(person="landlord", topic="repair", person_strength=0.9, topic_strength=0.7),
            "Landlord heater repair",
        ),
        LabeledCase(
            "deck-followup",
            "Chase the teams that still owe me numbers for the board deck.",
            Concepts(person="board", topic="deck", person_strength=0.8, topic_strength=0.9),
            "Q3 board deck",
        ),
        # --- second batch: added purely for statistical power ---------------
        # Six cases could not distinguish "the fix is fine" from "the fix is badly
        # broken" - one failure moved the number by 17%, and the 95% interval on
        # 5/6 runs from roughly 36% to 99.6%. These extend the set across the same
        # phrasing families so each point on the curve rests on more than a handful
        # of trials.
        LabeledCase(
            "renewal-paraphrase",
            "Did we ever sort out renewing the car policy?",
            Concepts(person="insurer", topic="renewal", person_strength=0.7, topic_strength=0.9),
            "Insurance renewal",
        ),
        LabeledCase(
            "renewal-abbreviated",
            "car ins renewal f/u",
            Concepts(person="insurer", topic="renewal", person_strength=0.8, topic_strength=0.5),
            "Insurance renewal",
        ),
        LabeledCase(
            "travel-indirect",
            "Has the hotel been confirmed for that trip yet?",
            Concepts(person="travel", topic="booking", person_strength=0.4, topic_strength=0.9),
            "Conference travel",
        ),
        LabeledCase(
            "travel-different-vocabulary",
            "Sort out where I'm staying in Lisbon.",
            Concepts(person="travel", topic="booking", person_strength=0.6, topic_strength=0.6),
            "Conference travel",
        ),
        LabeledCase(
            "design-paraphrase",
            "Ask Nadia what she thought of the dashboard mockups.",
            Concepts(person="nadia", topic="design", person_strength=1.0, topic_strength=0.9),
            "Nadia design review",
        ),
        LabeledCase(
            "design-typo",
            "Chase Nadya about the dashbord feedback.",
            Concepts(person="nadia", topic="design", person_strength=0.7, topic_strength=0.6),
            "Nadia design review",
        ),
        LabeledCase(
            "payroll-indirect",
            "Did they ever fix the missing hours?",
            Concepts(person="hr", topic="payroll", person_strength=0.3, topic_strength=0.9),
            "Payroll correction",
        ),
        LabeledCase(
            "payroll-abbreviated",
            "overtime payslip f/u",
            Concepts(person="hr", topic="payroll", person_strength=0.6, topic_strength=0.7),
            "Payroll correction",
        ),
        LabeledCase(
            "gym-paraphrase",
            "Follow up on cancelling my gym membership.",
            Concepts(person="gym", topic="cancellation", person_strength=1.0, topic_strength=0.9),
            "Gym membership cancel",
        ),
        LabeledCase(
            "gym-different-vocabulary",
            "When does the fitness subscription actually stop billing me?",
            Concepts(person="gym", topic="cancellation", person_strength=0.6, topic_strength=0.6),
            "Gym membership cancel",
        ),
        LabeledCase(
            "visa-paraphrase",
            "Where are we with the work visa documents?",
            Concepts(person="immigration", topic="visa", person_strength=0.8, topic_strength=0.9),
            "Visa paperwork",
        ),
        LabeledCase(
            "visa-indirect",
            "Do they still need anything else from me for the application?",
            Concepts(person="immigration", topic="visa", person_strength=0.3, topic_strength=0.8),
            "Visa paperwork",
        ),
    ]


def build_distractors(
    count: int, targets: List[SyntheticAgent]
) -> List[SyntheticAgent]:
    """Plausible competing agents, deterministically generated.

    Crucially these are **confusable**, not random. A first version filled the
    roster with agents sharing no person and no topic with any target, which made
    every correct agent rank first at every N - a flat 100% recall that measured
    nothing except that orthogonal vectors are far apart. Synthetic data that
    clean flatters the evaluation.

    So the pool is deliberately weighted toward near-misses:
      ~40% share a target's PERSON (different topic) - the false-merge shape
      ~40% share a target's TOPIC (different person)
      ~20% unrelated filler
    """
    exclude = {agent.name for agent in targets}
    target_people = [agent.concepts.person for agent in targets if agent.concepts.person]
    target_topics = [agent.concepts.topic for agent in targets if agent.concepts.topic]

    same_person: List[SyntheticAgent] = []
    same_topic: List[SyntheticAgent] = []
    unrelated: List[SyntheticAgent] = []

    for topic in _TOPICS:
        for person in target_people:
            name = f"{person.capitalize()} {topic}"
            if name not in exclude:
                same_person.append(
                    SyntheticAgent(
                        name,
                        f"Handling the {topic} thread with {person.capitalize()}.",
                        Concepts(person=person, topic=topic),
                    )
                )

    for person in _PEOPLE:
        for topic in target_topics:
            name = f"{person.capitalize()} {topic}"
            if name not in exclude:
                same_topic.append(
                    SyntheticAgent(
                        name,
                        f"Handling the {topic} thread with {person.capitalize()}.",
                        Concepts(person=person, topic=topic),
                    )
                )

    for topic in _TOPICS:
        for person in _PEOPLE:
            name = f"{person.capitalize()} {topic}"
            if name not in exclude:
                unrelated.append(
                    SyntheticAgent(
                        name,
                        f"Handling the {topic} thread with {person.capitalize()}.",
                        Concepts(person=person, topic=topic),
                    )
                )

    # Interleave 2 person-confusable : 2 topic-confusable : 1 unrelated.
    mixed: List[SyntheticAgent] = []
    seen: set[str] = set()
    pools = [same_person, same_person, same_topic, same_topic, unrelated]
    cursors = [0] * len(pools)
    while len(mixed) < count:
        progressed = False
        for slot, pool in enumerate(pools):
            while cursors[slot] < len(pool):
                candidate = pool[cursors[slot]]
                cursors[slot] += 1
                if candidate.name not in seen:
                    seen.add(candidate.name)
                    mixed.append(candidate)
                    progressed = True
                    break
            if len(mixed) >= count:
                break
        if not progressed:
            break

    return mixed


def build_model(agents: List[SyntheticAgent], cases: List[LabeledCase]) -> SyntheticEmbeddingModel:
    model = SyntheticEmbeddingModel()
    model.register_many({agent.embedding_text: agent.concepts for agent in agents})
    model.register_many({case.query_text: case.query_concepts for case in cases})
    return model


# ----------------------------------------------------------------------
# (a) Prompt cost
# ----------------------------------------------------------------------


def render_old_roster(agents: List[SyntheticAgent]) -> str:
    """The behaviour being replaced: every agent, bare name, every turn."""
    return "\n".join(f'<agent name="{agent.name}" />' for agent in agents)


def render_new_roster(agents: List[SyntheticAgent], top_k: int) -> str:
    """The bounded version: at most top_k, with descriptions."""
    shown = agents[:top_k]
    return "\n".join(
        f'<agent name="{agent.name}" description="{agent.description}" />' for agent in shown
    )


def approx_tokens(text: str) -> int:
    """~4 chars per token. Exact tokenizer is irrelevant to the shape of the curve."""
    return max(1, len(text) // 4)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score).

    Reported alongside every accuracy figure because a bare percentage hides how
    little it may be saying. At 6 trials, 5 successes reads as a confident-looking
    "83%" whose true value sits somewhere between roughly 36% and 99% - an interval
    wide enough to contain both "working perfectly" and "seriously broken". Printing
    the interval makes the difference between a measurement and a number visible.
    """
    if trials == 0:
        return (0.0, 0.0)
    phat = successes / trials
    denom = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    spread = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ----------------------------------------------------------------------
# (b) Retrieval rank
# ----------------------------------------------------------------------


def score_case(
    case: LabeledCase,
    agents: List[SyntheticAgent],
    model: SyntheticEmbeddingModel,
) -> Tuple[Optional[int], float]:
    """Return (rank of correct agent, margin over the best wrong agent).

    The margin is the load-bearing measurement here. Rank alone is too coarse for
    this synthetic geometry: an agent matching both facets strictly dominates one
    matching a single facet, so the correct answer cannot actually lose. The gap
    to the runner-up, however, does erode as confusable neighbours accumulate -
    and that shrinking safety margin is what turns into real errors once vectors
    carry noise.
    """
    query_vector = model.embed(case.query_text)
    scored = sorted(
        (
            (agent.name, cosine_similarity(query_vector, model.embed(agent.embedding_text)))
            for agent in agents
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    rank = None
    correct_score = 0.0
    for index, (name, score) in enumerate(scored):
        if name == case.correct_agent:
            rank, correct_score = index, score
            break

    best_wrong = max(
        (score for name, score in scored if name != case.correct_agent), default=0.0
    )
    return rank, correct_score - best_wrong


# ----------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------


@dataclass
class SizeResult:
    n: int
    old_tokens: int
    new_tokens: int
    recall_at_k: float
    mrr: float
    worst_rank: int
    mean_margin: float
    min_margin: float


def run_sweep(top_k: int) -> List[SizeResult]:
    targets = build_target_agents()
    cases = build_cases()
    distractors = build_distractors(max(ROSTER_SIZES), targets)

    results: List[SizeResult] = []
    for n in ROSTER_SIZES:
        padding = max(0, n - len(targets))
        roster = targets + distractors[:padding]
        model = build_model(roster, cases)

        scored = [score_case(case, roster, model) for case in cases]
        found = [rank for rank, _margin in scored if rank is not None]
        margins = [margin for _rank, margin in scored]
        recall_at_k = sum(1 for rank in found if rank < top_k) / len(cases) if cases else 0.0
        mrr = sum(1.0 / (rank + 1) for rank in found) / len(cases) if cases else 0.0

        results.append(
            SizeResult(
                n=len(roster),
                old_tokens=approx_tokens(render_old_roster(roster)),
                new_tokens=approx_tokens(render_new_roster(roster, top_k)),
                recall_at_k=recall_at_k,
                mrr=mrr,
                worst_rank=max(found) if found else -1,
                mean_margin=sum(margins) / len(margins) if margins else 0.0,
                min_margin=min(margins) if margins else 0.0,
            )
        )

    return results


def print_report(
    results: List[SizeResult],
    top_k: int,
    live_selection: Optional[Dict] = None,
    live_render: Optional[Dict] = None,
):
    print("=" * 78)
    print("(a) PROMPT COST - tokens in the <active_agents> block, every turn")
    print("=" * 78)
    print(f"  {'N':>5}  {'old (all agents)':>18}  {'new (top_k)':>14}  {'delta':>10}")
    crossover = None
    for row in results:
        delta = row.new_tokens - row.old_tokens
        marker = f"+{delta:,}" if delta > 0 else f"{delta:,}"
        if crossover is None and delta < 0:
            crossover = row.n
        print(f"  {row.n:>5}  {row.old_tokens:>18,}  {row.new_tokens:>14,}  {marker:>10}")

    first, last = results[0], results[-1]
    growth_old = last.old_tokens / first.old_tokens if first.old_tokens else 0
    growth_new = last.new_tokens / first.new_tokens if first.new_tokens else 0
    print(f"\n  N grew {last.n / first.n:.0f}x -> old prompt grew {growth_old:.0f}x, "
          f"new grew {growth_new:.1f}x and is capped at top_k={top_k}.")

    if crossover:
        print(f"\n  HONEST TRADEOFF: the new rendering is *more* expensive below "
              f"N~{crossover},")
        print("  because it adds a description per agent. That is the cost of making the")
        print("  roster matchable at all - bare names are what made reuse fail. The trade")
        print(f"  is a fixed ceiling (~{last.new_tokens:,} tokens) for unbounded growth;")
        print(f"  at N={last.n} the old rendering costs {last.old_tokens / last.new_tokens:.1f}x more,")
        print("  and unlike the cap it keeps climbing.")
    print("\n  Either way the old cost is paid on EVERY turn, relevant agents or not.")

    print("\n" + "=" * 78)
    print(f"(b) RETRIEVAL - does the right agent survive N-1 distractors? (k={top_k})")
    print("=" * 78)
    print(f"  {'N':>5}  {'recall@k':>10}  {'MRR':>8}  {'mean margin':>13}  {'worst margin':>14}")
    for row in results:
        print(
            f"  {row.n:>5}  {row.recall_at_k:>9.0%}  {row.mrr:>8.2f}  "
            f"{row.mean_margin:>13.3f}  {row.min_margin:>14.3f}"
        )

    first, last = results[0], results[-1]
    erosion = 1 - (last.min_margin / first.min_margin) if first.min_margin else 0.0
    print(f"\n  Worst-case margin eroded {erosion:.0%} "
          f"({first.min_margin:.3f} -> {last.min_margin:.3f}) as N grew {first.n} -> {last.n}.")
    print("  That gap is the system's error budget: it is how much noise a correct match")
    print("  can absorb before a confusable neighbour overtakes it.")

    saturation = next(
        (
            row.n
            for row, nxt in zip(results, results[1:])
            if abs(row.min_margin - nxt.min_margin) < 1e-9
        ),
        None,
    )
    if saturation:
        print(f"\n  Note the margin stops moving at N~{saturation}. Degradation is driven by")
        print("  CONFUSABLE DENSITY, not roster size: once every near-neighbour of a thread")
        print("  exists, further unrelated agents cost nothing. Five agents about the same")
        print("  person hurt far more than a hundred about strangers - so raw agent count is")
        print("  the wrong thing to alarm on, and 'how crowded is this neighbourhood' is the")
        print("  metric worth tracking in production.")

    print("\n  WHY RANK ITSELF DOES NOT MOVE HERE - and why that is reported, not hidden:")
    print("  under this synthetic geometry an agent matching both facets strictly dominates")
    print("  one matching a single facet, so the correct answer cannot be overtaken however")
    print("  many distractors are added. Rank is therefore uninformative offline, while the")
    print("  margin is not. Injecting tuned noise until the curve bent downward would have")
    print("  produced a more dramatic chart by assuming the conclusion; the honest position")
    print("  is that recall@k degradation requires --live and real embeddings to measure.")

    print("\n" + "=" * 78)
    print("(c) SELECTION - can the model pick correctly from what it is shown?")
    print("=" * 78)
    print(f"  {'N':>5}  {'old: candidates shown':>23}  {'new: candidates shown':>23}")
    for row in results:
        print(f"  {row.n:>5}  {row.n:>23}  {min(row.n, top_k):>23}")

    if live_selection:
        print(f"\n  {'N':>5}  {'old accuracy (95% CI)':>26}  {'new accuracy (95% CI)':>26}")
        for n in ROSTER_SIZES:
            entry = live_selection.get(n)
            if not entry:
                continue
            trials = entry["trials"]
            olo, ohi = wilson_interval(entry["old_hits"], trials)
            nlo, nhi = wilson_interval(entry["new_hits"], trials)
            print(
                f"  {n:>5}  {entry['old']:>7.0%} [{olo:.0%}-{ohi:.0%}]{'':>8}"
                f"  {entry['new']:>7.0%} [{nlo:.0%}-{nhi:.0%}]"
            )
        widest = max(
            (wilson_interval(e["new_hits"], e["trials"]) for e in live_selection.values()),
            key=lambda ci: ci[1] - ci[0],
        )
        print(
            f"\n  Widest interval spans {widest[1] - widest[0]:.0%}. Overlapping intervals"
            " between old and new mean\n  the difference is not resolvable at this sample"
            " size - read the intervals, not the point estimates."
        )
    else:
        print("\n  Accuracy not measured: needs a real model (--live).")
        print("  Deliberately NOT simulated - a mock whose accuracy decays with N would")
        print("  be assuming the very conclusion this experiment is meant to test.")
        print("  Offline, (a) and (b) carry the argument: the old design pays unbounded")
        print("  prompt cost and asks the model to discriminate among N unranked agents,")
        print("  while the new design shows at most top_k, pre-ranked.")

    print("\n" + "=" * 78)
    print("(d) PROMPT RENDER - does _render_active_agents surface the right agent")
    print("    with real embeddings, and does the recency guarantee earn its place?")
    print("=" * 78)
    if live_render:
        print(
            f"  Recall@{live_render['top_k']} at N={live_render['n']}, real embeddings: "
            f"{live_render['recall']:.0%} ({live_render['hits']}/{live_render['trials']})"
        )
        print("\n  Recency ablation - vague follow-up for a just-touched agent:")
        with_label = "survives" if live_render["recency_with"] else "DROPPED"
        without_label = "survives" if live_render["recency_without"] else "DROPPED"
        print(f"    WITH recency guarantee (shipped):    {with_label}")
        print(f"    WITHOUT recency guarantee:            {without_label}")
        if live_render["recency_with"] and not live_render["recency_without"]:
            print("  The guarantee is load-bearing for this shape of query: the agent is")
            print("  dropped without it and survives with it.")
        elif live_render["recency_with"] and live_render["recency_without"]:
            print("  The agent survived even without the guarantee on this query - embedding")
            print("  similarity alone was enough here. Read this as one data point, not proof")
            print("  the guarantee is unnecessary; see the reasoning it was added for.")
        else:
            print("  Unexpected: the agent did not survive even with the guarantee active.")
    else:
        print("\n  Not measured: needs a real model and real embeddings (--live).")
        print("  This is the first live check against the actual _render_active_agents")
        print("  code path - every existing unit test stubs a uniform embedding vector.")


async def _embed_cache(texts: List[str]) -> Dict[str, List[float]]:
    """Real OpenRouter embeddings for a fixed text set, one call per unique text.

    Embeddings are deterministic for a fixed model, so every text here is
    embedded exactly once regardless of how many roster sizes reuse it.
    """
    from server.services.execution.agent_matcher import embed_texts

    unique = list(dict.fromkeys(texts))
    vectors = await embed_texts(unique)
    if vectors is None:
        raise RuntimeError("embedding request failed - check OPENROUTER_API_KEY / network")
    return dict(zip(unique, vectors))


async def run_live_selection(top_k: int) -> Dict:
    """Ask a real model to pick the right agent, old vs new, at each N.

    Retrieval uses real embeddings now, not the synthetic model - a real judge
    reading a synthetic shortlist was measuring a hybrid nothing in production
    ever runs. All embeddings are fetched once upfront and cached across every N.
    """
    from server.openrouter_client import request_chat_completion

    settings = get_settings()
    targets = build_target_agents()
    cases = build_cases()
    distractors = build_distractors(max(ROSTER_SIZES), targets)

    print("  Embedding all agents and queries once (real, cached across every N)...")
    vectors = await _embed_cache(
        [a.embedding_text for a in targets + distractors] + [c.query_text for c in cases]
    )

    async def ask(case: LabeledCase, roster: List[SyntheticAgent]) -> bool:
        listing = "\n".join(f'- "{a.name}": {a.description}' for a in roster)
        prompt = (
            "Pick the single existing agent that should handle this request. "
            "Reply with the agent name exactly, or NONE.\n\n"
            f"Request: {case.query_text}\n\nAgents:\n{listing}"
        )
        try:
            response = await request_chat_completion(
                model=settings.interaction_agent_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=settings.openrouter_api_key,
            )
        except Exception as exc:
            print(f"    live call failed: {exc}")
            return False
        text = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return case.correct_agent.lower() in text.lower()

    out: Dict = {}
    for n in ROSTER_SIZES:
        roster = targets + distractors[: max(0, n - len(targets))]

        old_hits = 0
        new_hits = 0
        for case in cases:
            old_hits += await ask(case, roster)

            query_vector = vectors[case.query_text]
            shortlist = sorted(
                roster,
                key=lambda a: cosine_similarity(query_vector, vectors[a.embedding_text]),
                reverse=True,
            )[:top_k]
            new_hits += await ask(case, shortlist)

        out[n] = {
            "old": old_hits / len(cases),
            "new": new_hits / len(cases),
            "old_hits": old_hits,
            "new_hits": new_hits,
            "trials": len(cases),
        }
        print(f"  N={n:<4} old={out[n]['old']:.0%}  new={out[n]['new']:.0%}  (n={len(cases)})")

    return out


async def run_live_render_check() -> Dict:
    """Does the real `_render_active_agents` surface the right agent, with real
    embeddings, once the roster exceeds `agent_prompt_top_k`?

    Nothing before this called that function with a real embedding: every unit
    test in test_prompt_rendering.py stubs `request_embeddings`, and it defaults
    to a uniform vector - which, combined with roster records never being given a
    stored embedding either, means those tests actually exercise the lexical
    fallback, not cosine ranking, regardless of the stub's value. `run_live_selection`
    above tests the routing/dedup shortlist, a different call site with its own
    `agent_dedup_top_k`. This is the first check against the actual prompt-render
    path with real embeddings on both sides.

    It also runs the recency guarantee's first real ablation. EVALUATION.md's own
    "not covered" list names this: recency is in the code but was never measured
    against removing it. The scenario is the one the guarantee is meant for - a
    vague, pronoun-heavy follow-up with almost no semantic signal to rank on, the
    same shape as the anaphora case in DESIGN.md's known limitations.
    """
    import tempfile
    from pathlib import Path

    from server.agents.interaction_agent import agent as interaction_agent
    from server.services.execution import roster as roster_module
    from server.services.execution.agent_matcher import embed_texts
    from server.services.execution.roster import AgentRoster

    settings = get_settings()
    targets = build_target_agents()
    distractors = build_distractors(40, targets)
    cases = build_cases()
    roster_agents = targets + distractors

    print(f"  Embedding a {len(roster_agents)}-agent roster with real embeddings...")
    vectors = await embed_texts([a.embedding_text for a in roster_agents])
    if vectors is None:
        print("  embedding request failed; skipping render check.")
        return {}

    tmp_dir = tempfile.mkdtemp(prefix="eval_render_")
    roster = AgentRoster(Path(tmp_dir) / "roster.json")
    original_roster = roster_module._agent_roster
    original_has_trigger = AgentRoster._has_live_trigger
    roster_module._agent_roster = roster
    AgentRoster._has_live_trigger = lambda self, name: False

    try:
        for agent, vector in zip(roster_agents, vectors):
            roster.add_agent(
                agent.name,
                description=agent.description,
                embedding=vector,
                embedding_model=settings.embedding_model,
            )

        hits = 0
        for case in cases:
            rendered = await interaction_agent._render_active_agents(case.query_text)
            if f'name="{case.correct_agent}"' in rendered:
                hits += 1
        recall = hits / len(cases) if cases else 0.0
        print(
            f"  Recall@{settings.agent_prompt_top_k} with real embeddings, "
            f"N={len(roster_agents)}: {recall:.0%} ({hits}/{len(cases)})"
        )

        # --- Recency ablation: does the guarantee actually earn its place? ----
        vague_query = "did they ever get back to me on that"
        target = roster_agents[0]  # "Email to Keith about lunch" - unrelated wording to the query
        roster.mark_active(target.name)

        with_recency = await interaction_agent._render_active_agents(vague_query)
        survives_with = f'name="{target.name}"' in with_recency

        original_recent_count = settings.agent_prompt_recent_count
        settings.agent_prompt_recent_count = 0
        try:
            without_recency = await interaction_agent._render_active_agents(vague_query)
        finally:
            settings.agent_prompt_recent_count = original_recent_count
        survives_without = f'name="{target.name}"' in without_recency

        print("  Recency ablation - vague follow-up for a just-touched agent:")
        print(f"    WITH recency guarantee:    {'survives' if survives_with else 'DROPPED'}")
        print(f"    WITHOUT recency guarantee: {'survives' if survives_without else 'DROPPED'}")
    finally:
        roster_module._agent_roster = original_roster
        AgentRoster._has_live_trigger = original_has_trigger

    return {
        "recall": recall,
        "hits": hits,
        "trials": len(cases),
        "n": len(roster_agents),
        "top_k": settings.agent_prompt_top_k,
        "recency_with": survives_with,
        "recency_without": survives_without,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="add the real-LLM selection curve")
    args = parser.parse_args()

    top_k = get_settings().agent_prompt_top_k
    results = run_sweep(top_k)

    live = None
    live_render = None
    if args.live:
        if not get_settings().openrouter_api_key:
            print("--live needs OPENROUTER_API_KEY; running offline curves only.\n")
        else:
            print("Running live selection curve (real embeddings + real judge)...\n")
            live = await run_live_selection(top_k)
            print()
            print("Running live prompt-render check (real embeddings, no judge)...\n")
            live_render = await run_live_render_check()
            print()

    print_report(results, top_k, live, live_render)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
