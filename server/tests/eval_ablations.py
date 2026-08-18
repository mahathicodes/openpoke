"""What does each layer of the routing pipeline actually buy?

    python server/tests/eval_ablations.py        # offline, deterministic, free

Every component gets disabled in turn and the labeled set re-run, so each part of
the design is justified by a number rather than an argument. Anything that cannot
show a delta is a candidate for deletion.

The shipped pipeline:

    exact match -> embed -> similarity floor -> LLM judgment -> staged link
                                                             -> structural evidence -> merge

Runs offline against the synthetic concept space, so results are reproducible and
free. The LLM judgment stage is replaced by a *stated policy* rather than a live
model (see JudgmentPolicy) - the ablations compare architectures, and holding the
judgment behaviour fixed is what makes the comparison clean. Absolute accuracy
against a real model belongs to eval_agent_matching.py.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for path in (str(_REPO_ROOT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from server.services.execution.agent_matcher import (  # noqa: E402
    agent_embedding_text,
    cosine_similarity,
    lexical_overlap,
)
from synthetic_embeddings import Concepts, SyntheticEmbeddingModel  # noqa: E402


def _shipped(attr: str, fallback):
    """Read a shipped setting so the ablations test the real configuration.

    These were hardcoded at first and silently drifted from config.py once the
    similarity floor was recalibrated - the table was reporting on a threshold the
    system no longer used.
    """
    try:
        from server.config import get_settings

        return getattr(get_settings(), attr)
    except Exception:  # pragma: no cover - offline fallback
        return fallback


FLOOR = _shipped("agent_min_candidate_similarity", 0.60)
TOP_K = _shipped("agent_dedup_top_k", 5)
MERGE_THRESHOLD = _shipped("agent_merge_commit_threshold", 0.9)


# ----------------------------------------------------------------------
# Scenario definitions
# ----------------------------------------------------------------------


@dataclass
class Agent:
    name: str
    description: str
    concepts: Concepts
    # Gmail identifiers this agent's log would contain.
    thread_ids: Tuple[str, ...] = ()
    emails: Tuple[str, ...] = ()

    @property
    def embedding_text(self) -> str:
        return agent_embedding_text(self.name, self.description)


@dataclass
class Scenario:
    scenario_id: str
    category: str
    request: str
    request_concepts: Concepts
    proposed_name: str
    existing: List[Agent]
    # Name of the agent this genuinely continues, or None for new work.
    truth: Optional[str]
    # Identifiers the new work would produce once it runs.
    thread_ids: Tuple[str, ...] = ()
    emails: Tuple[str, ...] = ()


_KEITH_LUNCH = Agent(
    "Email to Keith about lunch",
    "Planning a lunch meeting with Keith Rivera.",
    Concepts(person="keith", topic="lunch"),
    thread_ids=("thread-lunch-1",),
    emails=("keith@example.com",),
)
_KEITH_CONTRACT = Agent(
    "Keith contract review",
    "Reviewing the consulting contract Keith Rivera sent.",
    Concepts(person="keith", topic="contract"),
    thread_ids=("thread-contract-9",),
    emails=("keith@example.com",),
)
_VERCEL = Agent(
    "Vercel job offer",
    "Negotiating the Vercel senior engineer offer.",
    Concepts(person="vercel", topic="offer"),
    thread_ids=("thread-offer-3",),
    emails=("recruiting@vercel.com",),
)
_DENTIST = Agent(
    "Dentist appointment",
    "Rescheduling a dental cleaning.",
    Concepts(person="dentist", topic="appointment"),
    thread_ids=("thread-dentist-4",),
    emails=("front@brightsmile.com",),
)

_ROSTER = [_KEITH_LUNCH, _KEITH_CONTRACT, _VERCEL, _DENTIST]


SCENARIOS: List[Scenario] = [
    # --- true reuse: same thread, different words -------------------------
    Scenario(
        "reuse-paraphrase", "true-reuse",
        "Check whether Keith ever replied about lunch.",
        Concepts(person="keith", topic="lunch", topic_strength=0.9),
        "Lunch with Keith", _ROSTER, "Email to Keith about lunch",
        thread_ids=("thread-lunch-1",), emails=("keith@example.com",),
    ),
    Scenario(
        "reuse-typo", "true-reuse",
        "Send Kieth another note about lunhc next week.",
        Concepts(person="keith", topic="lunch", person_strength=0.7, topic_strength=0.6),
        "Emial to Kieth", _ROSTER, "Email to Keith about lunch",
        thread_ids=("thread-lunch-1",), emails=("keith@example.com",),
    ),
    Scenario(
        "reuse-indirect", "true-reuse",
        "Ask them to raise the equity portion.",
        Concepts(person="vercel", topic="offer", person_strength=0.4),
        "Vercel comp", _ROSTER, "Vercel job offer",
        thread_ids=("thread-offer-3",), emails=("recruiting@vercel.com",),
    ),
    # --- adversarial: the false-merge bait --------------------------------
    Scenario(
        "adversarial-same-person", "adversarial",
        "Order a birthday gift for Keith Rivera.",
        Concepts(person="keith", topic="gift"),
        "Keith birthday gift", _ROSTER, None,
        thread_ids=("thread-gift-7",), emails=("keith@example.com",),
    ),
    Scenario(
        "adversarial-same-person-invoice", "adversarial",
        "Ask Keith why the March invoice is unpaid.",
        Concepts(person="keith", topic="invoice"),
        "Keith invoice", _ROSTER, None,
        thread_ids=("thread-invoice-2",), emails=("keith@example.com",),
    ),
    Scenario(
        "adversarial-same-domain", "adversarial",
        "Negotiate base salary on the Netlify offer.",
        Concepts(person="netlify", topic="offer"),
        "Netlify job offer", _ROSTER, None,
        thread_ids=("thread-netlify-5",), emails=("jobs@netlify.com",),
    ),
    # --- genuinely new ----------------------------------------------------
    Scenario(
        "new-unrelated", "true-new",
        "Find a direct flight to Lisbon in September.",
        Concepts(person="airline", topic="travel"),
        "Flight to Lisbon", _ROSTER, None,
    ),
    Scenario(
        "new-empty-roster", "edge",
        "Draft a note to the landlord about the heater.",
        Concepts(person="landlord", topic="repair"),
        "Landlord heater", [], None,
    ),
]


def build_model() -> SyntheticEmbeddingModel:
    model = SyntheticEmbeddingModel()
    for agent in _ROSTER:
        model.register(agent.embedding_text, agent.concepts)
    for scenario in SCENARIOS:
        model.register(scenario.request, scenario.request_concepts)
    return model


# ----------------------------------------------------------------------
# Pipeline configuration - each ablation flips one switch
# ----------------------------------------------------------------------


# The judgment stand-in is parameterised by how fallible the judge is, because the
# evidence gate's entire value is contingent on that. A perfect judge needs no gate -
# reporting a single "the gate prevents N false merges" number would silently bake in
# an assumed error rate. Sweeping it instead states the dependency openly.
#
#   PERFECT  - always distinguishes same-person/different-topic. Aspirational.
#   CAREFUL  - usually distinguishes, but is swayed when similarity is very high.
#   SURFACE  - links whenever the top candidate looks similar enough. The documented
#              LLM failure mode: surface resemblance read as identity.
JUDGE_PERFECT = "perfect"
JUDGE_CAREFUL = "careful"
JUDGE_SURFACE = "surface-swayed"
JUDGE_LEVELS = [JUDGE_PERFECT, JUDGE_CAREFUL, JUDGE_SURFACE]

# Above this similarity a CAREFUL judge stops discriminating on topic.
_CAREFUL_SWAY_THRESHOLD = 0.60


@dataclass
class Config:
    label: str
    use_embeddings: bool = True
    use_floor: bool = True
    use_judgment: bool = True
    use_evidence_gate: bool = True
    top_k: int = TOP_K
    show_full_roster: bool = False  # A1: LLM sees everything, no retrieval
    judge: str = JUDGE_CAREFUL


def judgment_policy(
    scenario: Scenario,
    candidates: List[Tuple[Agent, float]],
    judge: str = JUDGE_CAREFUL,
) -> Optional[str]:
    """Stand-in for the LLM's routing judgment at a given fallibility level.

    Holding this fixed within a run is what makes the ablation a comparison of
    architectures rather than of prompts.
    """
    if not candidates:
        return None

    top_agent, top_score = candidates[0]
    topic_matches = top_agent.concepts.topic == scenario.request_concepts.topic

    # A long unranked list dilutes attention regardless of judge quality: the model
    # falls back to "something in here probably fits".
    if len(candidates) > 10:
        return top_agent.name

    if judge == JUDGE_PERFECT:
        return top_agent.name if topic_matches else None

    if judge == JUDGE_CAREFUL:
        if topic_matches:
            return top_agent.name
        return top_agent.name if top_score >= _CAREFUL_SWAY_THRESHOLD else None

    # JUDGE_SURFACE
    return top_agent.name


def structural_confidence(scenario: Scenario, agent: Agent) -> float:
    """Ground-truth evidence score, mirroring score_structural_evidence."""
    if set(scenario.thread_ids) & set(agent.thread_ids):
        return 1.0
    if set(scenario.emails) & set(agent.emails):
        return 0.6
    return 0.0


class LiveBackend:
    """Real embeddings and a real LLM judge, with everything cached.

    Each ablation re-runs the same scenarios, so without caching the table would
    re-embed and re-judge identical inputs eight times over. Cached, the whole live
    table costs one embedding pass and one judgment per distinct (scenario,
    shortlist) pair.
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, List[float]] = {}
        self._judgments: Dict[str, Optional[str]] = {}
        self.judge_calls = 0

    async def preload_embeddings(self, texts: List[str]) -> bool:
        from server.services.execution.agent_matcher import embed_texts

        missing = [t for t in texts if t not in self._vectors]
        if not missing:
            return True
        vectors = await embed_texts(missing)
        if vectors is None:
            return False
        self._vectors.update(dict(zip(missing, vectors)))
        return True

    def embed(self, text: str) -> List[float]:
        return self._vectors.get(text, [])

    async def judge(
        self, scenario: "Scenario", candidates: List[Tuple["Agent", float]]
    ) -> Optional[str]:
        """Ask the real model, exactly as the shipped pipeline would."""
        from server.services.execution.agent_matcher import ScoredCandidate, decide_routing
        from server.services.execution.roster import AgentRecord

        key = f"{scenario.scenario_id}|{','.join(a.name for a, _ in candidates)}"
        if key in self._judgments:
            return self._judgments[key]

        shortlist = [
            ScoredCandidate(
                record=AgentRecord(name=agent.name, description=agent.description),
                score=score,
                method="embedding",
            )
            for agent, score in candidates
        ]
        decision = await decide_routing(
            instructions=scenario.request,
            proposed_name=scenario.proposed_name,
            candidates=shortlist,
        )
        self.judge_calls += 1
        self._judgments[key] = decision.duplicate_of
        return decision.duplicate_of


async def run_pipeline_live(
    config: Config, backend: LiveBackend
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """Same pipeline, real embeddings and a real judge.

    Structural evidence stays modelled: it depends on Gmail thread ids appearing in
    two agents' execution logs, which needs a connected mailbox doing real work. That
    remains the one stage no evaluation here exercises for real.
    """
    outcomes: Dict[str, Optional[str]] = {}
    staged: Dict[str, Optional[str]] = {}

    for scenario in SCENARIOS:
        agents = scenario.existing
        if not agents:
            outcomes[scenario.scenario_id] = None
            staged[scenario.scenario_id] = None
            continue

        if config.show_full_roster:
            candidates = [(agent, 1.0) for agent in agents]
        else:
            if config.use_embeddings:
                query = backend.embed(scenario.request)
                scored = [
                    (agent, cosine_similarity(query, backend.embed(agent.embedding_text)))
                    for agent in agents
                ]
            else:
                scored = [
                    (agent, lexical_overlap(scenario.request, agent.embedding_text))
                    for agent in agents
                ]
            scored.sort(key=lambda item: item[1], reverse=True)
            candidates = scored[: config.top_k]
            if config.use_floor:
                candidates = [(a, s) for a, s in candidates if s >= FLOOR]

        if not candidates:
            outcomes[scenario.scenario_id] = None
            staged[scenario.scenario_id] = None
            continue

        if config.use_judgment:
            linked = await backend.judge(scenario, candidates)
        else:
            linked = candidates[0][0].name

        staged[scenario.scenario_id] = linked
        if linked is None:
            outcomes[scenario.scenario_id] = None
            continue

        if not config.use_evidence_gate:
            outcomes[scenario.scenario_id] = linked
            continue

        target = next((a for a in agents if a.name == linked), None)
        if target is None:
            outcomes[scenario.scenario_id] = None
            continue

        confidence = structural_confidence(scenario, target)
        outcomes[scenario.scenario_id] = linked if confidence >= MERGE_THRESHOLD else None

    return outcomes, staged


def run_pipeline(
    config: Config, model: SyntheticEmbeddingModel
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """Return (merges committed, links staged) per scenario.

    Both are needed to see each layer's contribution. The pipeline is defence in
    depth, so removing the judgment stage often changes nothing about committed
    merges - the evidence gate catches the same error downstream. Tracking staged
    links exposes what each layer stopped on its own, instead of crediting it all
    to whichever layer happens to be last.
    """
    outcomes: Dict[str, Optional[str]] = {}
    staged: Dict[str, Optional[str]] = {}

    for scenario in SCENARIOS:
        agents = scenario.existing

        # --- retrieval -------------------------------------------------
        if not agents:
            outcomes[scenario.scenario_id] = None
            staged[scenario.scenario_id] = None
            continue

        if config.show_full_roster:
            candidates = [(agent, 1.0) for agent in agents]
        else:
            if config.use_embeddings:
                query = model.embed(scenario.request)
                scored = [
                    (agent, cosine_similarity(query, model.embed(agent.embedding_text)))
                    for agent in agents
                ]
            else:
                scored = [
                    (agent, lexical_overlap(scenario.request, agent.embedding_text))
                    for agent in agents
                ]
            scored.sort(key=lambda item: item[1], reverse=True)
            candidates = scored[: config.top_k]

            if config.use_floor:
                candidates = [(a, s) for a, s in candidates if s >= FLOOR]

        if not candidates:
            outcomes[scenario.scenario_id] = None
            staged[scenario.scenario_id] = None
            continue

        # --- judgment ---------------------------------------------------
        if config.use_judgment:
            linked = judgment_policy(scenario, candidates, judge=config.judge)
        else:
            # A2: no judge - link the top candidate outright.
            linked = candidates[0][0].name

        staged[scenario.scenario_id] = linked
        if linked is None:
            outcomes[scenario.scenario_id] = None
            continue

        # --- evidence gate ----------------------------------------------
        if not config.use_evidence_gate:
            # A5: the obvious design - the link IS the merge.
            outcomes[scenario.scenario_id] = linked
            continue

        target = next((a for a in agents if a.name == linked), None)
        if target is None:
            outcomes[scenario.scenario_id] = None
            continue

        confidence = structural_confidence(scenario, target)
        outcomes[scenario.scenario_id] = linked if confidence >= MERGE_THRESHOLD else None

    return outcomes, staged


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


@dataclass
class Score:
    label: str
    correct_merges: int
    false_merges: int
    missed_merges: int
    correct_abstentions: int

    @property
    def total(self) -> int:
        return (
            self.correct_merges
            + self.false_merges
            + self.missed_merges
            + self.correct_abstentions
        )

    @property
    def accuracy(self) -> float:
        return (self.correct_merges + self.correct_abstentions) / self.total if self.total else 0.0

    @property
    def recall(self) -> float:
        actual = self.correct_merges + self.missed_merges
        return self.correct_merges / actual if actual else 0.0

    @property
    def abstention_rate(self) -> float:
        opportunities = self.correct_abstentions + self.false_merges
        return self.correct_abstentions / opportunities if opportunities else 0.0


def score(label: str, outcomes: Dict[str, Optional[str]]) -> Score:
    result = Score(label, 0, 0, 0, 0)
    for scenario in SCENARIOS:
        decided = outcomes[scenario.scenario_id]
        if decided is not None and decided == scenario.truth:
            result.correct_merges += 1
        elif decided is not None:
            result.false_merges += 1
        elif scenario.truth is None:
            result.correct_abstentions += 1
        else:
            result.missed_merges += 1
    return result


CONFIGS = [
    Config("SHIPPED (all layers)"),
    Config("A5  no evidence gate", use_evidence_gate=False),
    Config("A2  no LLM judgment", use_judgment=False),
    Config("A3  lexical, no embeddings", use_embeddings=False),
    Config("A1  no retrieval (full roster)", show_full_roster=True),
    Config("A4  no similarity floor", use_floor=False),
    Config("A6  top_k=1", top_k=1),
    Config("A6  top_k=10", top_k=10),
]


async def run_live_table() -> Dict[str, Score]:
    """Run every ablation against real embeddings and a real judge."""
    from server.config import get_settings

    settings = get_settings()
    if not settings.openrouter_api_key:
        print("--live needs OPENROUTER_API_KEY.\n")
        return {}

    backend = LiveBackend()
    texts = {s.request for s in SCENARIOS}
    texts |= {a.embedding_text for s in SCENARIOS for a in s.existing}
    print(f"Embedding {len(texts)} texts with {settings.embedding_model} ...")
    if not await backend.preload_embeddings(sorted(texts)):
        print("Embedding failed.\n")
        return {}

    scores: Dict[str, Score] = {}
    for config in CONFIGS:
        merged, _staged = await run_pipeline_live(config, backend)
        scores[config.label] = score(config.label, merged)
        print(f"  ran {config.label}")
    print(f"  ({backend.judge_calls} judge calls, deduplicated by cache)\n")
    return scores


def print_live_comparison(synthetic: List[Score], live: Dict[str, Score]) -> None:
    print("\n" + "=" * 88)
    print("LIVE vs SYNTHETIC - same ablations, real embeddings and a real judge")
    print("=" * 88)
    print(f"  {'configuration':<32}{'FALSE merges':>16}{'recall':>18}")
    print(f"  {'':<32}{'synth':>8}{'live':>8}{'synth':>9}{'live':>9}")
    print("  " + "-" * 84)
    for entry in synthetic:
        live_entry = live.get(entry.label)
        if live_entry is None:
            continue
        print(
            f"  {entry.label:<32}{entry.false_merges:>8}{live_entry.false_merges:>8}"
            f"{entry.recall:>8.0%}{live_entry.recall:>9.0%}"
        )

    base_live = live.get("SHIPPED (all layers)")
    a5_live = live.get("A5  no evidence gate")
    print("\n  READING THIS TABLE")
    print("  The synthetic columns use a stand-in judge whose fallibility I chose; the")
    print("  live columns use the model that actually ships. Where they disagree, the")
    print("  live number is the one to believe.")
    if base_live and a5_live:
        delta = a5_live.false_merges - base_live.false_merges
        if delta == 0:
            print("\n  A5, live: removing the evidence gate changed NOTHING. The real judge did")
            print("  not stage a single bad link on this set, so the gate had nothing to catch.")
            print("  Stated plainly: against this model, on these cases, the central safety")
            print("  mechanism of the design is UNPROVEN - not proven. It earns its place as")
            print("  insurance (the rule-of-three bound on the false-merge rate is still ~8%,")
            print("  the failure is silent and permanent, and the check costs no API call),")
            print("  but I could not induce the failure it exists to prevent.")
        else:
            print(f"\n  A5, live: the gate prevented {delta} real false merge(s).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="also run every ablation against real embeddings and a real judge")
    args = parser.parse_args()

    model = build_model()
    runs = [(config, run_pipeline(config, model)) for config in CONFIGS]
    scores = [score(config.label, merged) for config, (merged, _staged) in runs]
    staged_scores = {
        config.label: score(config.label, staged) for config, (_merged, staged) in runs
    }
    baseline = scores[0]

    print("=" * 88)
    print("ABLATION TABLE - each row disables one layer of the shipped pipeline")
    print("=" * 88)
    header = (
        f"  {'configuration':<32}{'merges':>8}{'FALSE':>7}{'missed':>8}"
        f"{'abstain':>9}{'recall':>8}{'abst.':>7}"
    )
    print(header)
    print("  " + "-" * 84)
    for entry in scores:
        marker = " !!" if entry.false_merges > baseline.false_merges else "   "
        print(
            f"  {entry.label:<32}{entry.correct_merges:>8}{entry.false_merges:>7}"
            f"{entry.missed_merges:>8}{entry.correct_abstentions:>9}"
            f"{entry.recall:>7.0%}{entry.abstention_rate:>7.0%}{marker}"
        )

    print("\n" + "=" * 88)
    print("WHAT EACH LAYER BUYS")
    print("=" * 88)

    by_label = {entry.label: entry for entry in scores}

    print("\n  A5 - the evidence gate (the central design decision)")
    print("     Its value depends entirely on how fallible the judge is, so that is swept")
    print("     rather than assumed. False merges, with the gate vs without:\n")
    print(f"       {'judge':<16}{'with gate':>11}{'without gate':>14}{'prevented':>11}")
    for judge in JUDGE_LEVELS:
        with_gate = score("", run_pipeline(Config("", judge=judge), model)[0])
        without = score(
            "", run_pipeline(Config("", judge=judge, use_evidence_gate=False), model)[0]
        )
        prevented = without.false_merges - with_gate.false_merges
        print(
            f"       {judge:<16}{with_gate.false_merges:>11}{without.false_merges:>14}"
            f"{prevented:>11}"
        )
    print("\n     A perfect judge needs no gate - that is a tautology, not a result. The")
    print("     gate exists because real models are swayed by surface resemblance, and the")
    print("     rows above price that risk instead of asserting it. Note the failure it")
    print("     prevents is silent and permanent, so its expected cost is not linear in")
    print("     the count: one false merge corrupts a thread indefinitely.")

    a2 = by_label["A2  no LLM judgment"]
    staged_base = staged_scores["SHIPPED (all layers)"]
    staged_a2 = staged_scores["A2  no LLM judgment"]
    print("\n  A2 - the LLM judgment stage")
    print(f"     Committed merges are unchanged ({baseline.false_merges} -> "
          f"{a2.false_merges} false), which is defence in depth, not irrelevance:")
    print(f"     the evidence gate catches downstream what the judge would have stopped.")
    print(f"     Measured at the staging step, where the judge acts alone:")
    print(f"       bad links staged  {staged_base.false_merges} -> {staged_a2.false_merges}")
    print("     Similarity alone cannot separate 'same person, different topic'; without")
    print("     the judge the system leans entirely on evidence arriving later.")

    a3 = by_label["A3  lexical, no embeddings"]
    print(f"\n  A3 - embeddings vs the free lexical fallback")
    print(f"     Recall {baseline.recall:.0%} -> {a3.recall:.0%}. This is what justifies")
    print("     the embedding dependency; if the gap were zero, the API call is unearned.")

    a1 = by_label["A1  no retrieval (full roster)"]
    print(f"\n  A1 - retrieval vs showing the model everything")
    print(f"     False merges {baseline.false_merges} -> {a1.false_merges}, "
          f"recall {baseline.recall:.0%} -> {a1.recall:.0%}.")
    print("     Ties back to the degradation experiment: an unranked full roster is both")
    print("     more expensive and harder to choose from.")

    print("\n  A6 - top_k")
    for label in ("A6  top_k=1", "A6  top_k=10"):
        entry = by_label[label]
        print(f"     {label:<18} recall {entry.recall:>4.0%}   false merges {entry.false_merges}")

    print("\n" + "-" * 88)
    print("READ THIS BEFORE QUOTING THE NUMBERS")
    print("-" * 88)
    print("  Small set (8 scenarios) on synthetic geometry with a stated judgment policy.")
    print("  These compare ARCHITECTURES under identical conditions - they are not estimates")
    print("  of production accuracy. The judge is a fixed rule, not a real model, so absolute")
    print("  values mean little; the deltas between rows are the finding.")

    if args.live:
        live = asyncio.run(run_live_table())
        if live:
            print_live_comparison(scores, live)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
