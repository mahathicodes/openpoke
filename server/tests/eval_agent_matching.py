"""Evaluator for agent-routing quality. Requires a real OPENROUTER_API_KEY.

    python server/tests/eval_agent_matching.py

Why this is split into two phases:

The pipeline has one deterministic stage (embedding retrieval) and one
non-deterministic stage (the LLM judgment call). Measuring them together would
be both expensive and misleading, so:

  Phase 1 - retrieval. Embeddings are deterministic for a fixed model, so every
  text is embedded exactly once, cached, and then the similarity threshold is
  swept offline across many values. Zero extra API calls, zero run-to-run noise,
  and the shipped threshold is chosen from the resulting curve rather than by feel.

  Phase 2 - judgment. The LLM can answer differently on identical input, so each
  case runs N times and the report shows the per-case flip rate. A single pass
  would hide exactly the instability worth knowing about.

The headline metric is the false-merge rate, tracked separately from ordinary
accuracy: wrongly linking two distinct threads silently corrupts an unrelated
agent's context, whereas a missed link merely leaves a duplicate around.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.config import get_settings  # noqa: E402
from server.services.execution.agent_matcher import (  # noqa: E402
    ScoredCandidate,
    reconcile_link,
    agent_embedding_text,
    cosine_similarity,
    decide_routing,
    embed_text,
    embed_texts,
)
from server.services.execution.roster import AgentRecord  # noqa: E402


# ----------------------------------------------------------------------
# Labeled dataset
# ----------------------------------------------------------------------


@dataclass
class ExistingAgent:
    name: str
    description: str


@dataclass
class EvalCase:
    case_id: str
    category: str
    proposed_name: str
    instructions: str
    # Name of the agent this task genuinely continues, or None when it is new work.
    expected_link: Optional[str]
    existing: List[ExistingAgent] = field(default_factory=list)


_KEITH_LUNCH = ExistingAgent(
    "Email to Keith about lunch",
    "Planning a lunch meeting with Keith Rivera, coordinating dates and venue.",
)
_KEITH_CONTRACT = ExistingAgent(
    "Keith contract review",
    "Reviewing the consulting contract redlines Keith Rivera sent over.",
)
_VERCEL_OFFER = ExistingAgent(
    "Vercel job offer",
    "Negotiating compensation on the Vercel senior engineer offer.",
)
_DENTIST = ExistingAgent(
    "Dentist appointment",
    "Rescheduling a dental cleaning with Bright Smile Clinic.",
)
_STANDUP = ExistingAgent(
    "Daily standup reminder",
    "Recurring 9am reminder to post a written standup update.",
)

_BASE_ROSTER = [_KEITH_LUNCH, _KEITH_CONTRACT, _VERCEL_OFFER, _DENTIST, _STANDUP]

# Two different people sharing a first name AND a topic. Deliberately constructed to
# clear any similarity floor that still admits genuine reuse: retrieval cannot
# separate them, so the judgment stage has to. Added after a live run scored 100%
# with every adversarial case filtered below the floor - meaning the judge was never
# tested on anything hard and the score flattered a well-tuned threshold.
_KEITH_RIVERA_LUNCH = ExistingAgent(
    "Keith Rivera lunch",
    "Planning a lunch meeting with Keith Rivera, coordinating dates and venue.",
)
_KEITH_CHEN_LUNCH = ExistingAgent(
    "Keith Chen lunch",
    "Planning a lunch meeting with Keith Chen, coordinating dates and venue.",
)
_TWO_KEITHS_ROSTER = [_KEITH_RIVERA_LUNCH, _KEITH_CHEN_LUNCH, _VERCEL_OFFER, _DENTIST]


CASES: List[EvalCase] = [
    # --- true reuse: the same thread, worded differently -------------------
    EvalCase(
        "reuse-paraphrase",
        "true-reuse",
        "Lunch with Keith",
        "Check whether Keith ever replied about getting lunch.",
        _KEITH_LUNCH.name,
        _BASE_ROSTER,
    ),
    EvalCase(
        "reuse-abbreviated",
        "true-reuse",
        "Keith lunch f/u",
        "Follow up on the lunch invite I sent Keith last week.",
        _KEITH_LUNCH.name,
        _BASE_ROSTER,
    ),
    EvalCase(
        "reuse-typo",
        "true-reuse",
        "Emial to Kieth about lunhc",
        "Send Kieth another note about lunhc next Tuesday.",
        _KEITH_LUNCH.name,
        _BASE_ROSTER,
    ),
    EvalCase(
        "reuse-indirect",
        "true-reuse",
        "Vercel comp",
        "Ask them to raise the equity portion of the offer.",
        _VERCEL_OFFER.name,
        _BASE_ROSTER,
    ),
    EvalCase(
        "reuse-different-vocabulary",
        "true-reuse",
        "Teeth cleaning slot",
        "Move my cleaning appointment to a morning slot instead.",
        _DENTIST.name,
        _BASE_ROSTER,
    ),
    # --- adversarial near-misses: the costly false-merge cases -------------
    EvalCase(
        "adversarial-same-person-different-topic",
        "adversarial",
        "Keith birthday gift",
        "Order a birthday gift for Keith Rivera and have it delivered Friday.",
        None,
        _BASE_ROSTER,
    ),
    EvalCase(
        "adversarial-same-person-two-threads",
        "adversarial",
        "Keith invoice question",
        "Ask Keith Rivera why the March invoice has not been paid yet.",
        None,
        _BASE_ROSTER,
    ),
    EvalCase(
        "adversarial-same-domain-different-company",
        "adversarial",
        "Netlify job offer",
        "Negotiate the base salary on the Netlify staff engineer offer.",
        None,
        _BASE_ROSTER,
    ),
    EvalCase(
        "adversarial-similar-wording-unrelated",
        "adversarial",
        "Doctor appointment",
        "Reschedule my annual physical with Dr. Patel to next month.",
        None,
        _BASE_ROSTER,
    ),
    # --- genuinely new work ------------------------------------------------
    EvalCase(
        "new-unrelated",
        "true-new",
        "Flight to Lisbon",
        "Find me a direct flight to Lisbon in early September.",
        None,
        _BASE_ROSTER,
    ),
    EvalCase(
        "new-unrelated-2",
        "true-new",
        "Renew car insurance",
        "Compare quotes and renew the car insurance policy before it lapses.",
        None,
        _BASE_ROSTER,
    ),
    # --- cases that reach the judge and must be REFUSED --------------------
    # Selected by measuring real similarity first: each scores above the 0.60 floor
    # with a wide gap to second place, so it arrives at the judgment stage alone and
    # the judge has to decide. Without these the judge is only ever tested on cases
    # where it should link - its abstention, which is the safety-critical half, would
    # go entirely unmeasured while the score still read 100%.
    EvalCase(
        "adversarial-adjacent-topic-same-person",  # ~0.67 similarity
        "adversarial",
        "Keith dinner Friday",
        "Set up dinner with Keith Rivera on Friday evening.",
        None,  # a meal with the same person, but not the lunch thread
        _BASE_ROSTER,
    ),
    EvalCase(
        "adversarial-same-company-different-matter",  # ~0.60 similarity
        "adversarial",
        "Vercel contractor invoice",
        "Ask Vercel accounts payable about my outstanding contractor invoice.",
        None,  # same company as the offer thread, unrelated business
        _BASE_ROSTER,
    ),
    EvalCase(
        "adversarial-same-provider-different-matter",  # ~0.66 similarity
        "adversarial",
        "Dentist billing question",
        "Ask Bright Smile Clinic about a charge on my dental bill.",
        None,  # same clinic as the appointment thread, different issue
        _BASE_ROSTER,
    ),
    # --- the hard case: ambiguous between two equally good matches ---------
    EvalCase(
        "adversarial-two-people-same-name-same-topic",
        "adversarial",
        "Lunch with Keith",
        "Follow up on lunch with Keith.",
        # Neither is correct. One of them may well be what the user meant, but the
        # request does not say which, and picking is a coin flip. Abstaining is the
        # only right answer; a link here is a false merge.
        None,
        _TWO_KEITHS_ROSTER,
    ),
    EvalCase(
        "reuse-disambiguated-by-surname",
        "true-reuse",
        "Lunch with Keith Chen",
        "Follow up with Keith Chen about our lunch.",
        # Same roster, but the request names which Keith - so linking is correct.
        # Paired with the case above to check the judge discriminates on the
        # distinguishing detail rather than just refusing whenever two names collide.
        _KEITH_CHEN_LUNCH.name,
        _TWO_KEITHS_ROSTER,
    ),
    # --- empty roster: nothing to compare against --------------------------
    EvalCase(
        "empty-roster-first-agent",
        "edge-case",
        "First ever task",
        "Draft a note to the landlord about the broken heater.",
        None,
        [],
    ),
]


# ----------------------------------------------------------------------
# Phase 1: deterministic retrieval evaluation
# ----------------------------------------------------------------------


@dataclass
class RetrievalResult:
    case: EvalCase
    ranked: List[Tuple[str, float]]  # (agent name, similarity), best first

    def rank_of_expected(self) -> Optional[int]:
        if self.case.expected_link is None:
            return None
        for index, (name, _score) in enumerate(self.ranked):
            if name == self.case.expected_link:
                return index
        return None

    def top_score(self) -> float:
        return self.ranked[0][1] if self.ranked else 0.0


async def run_retrieval_phase() -> List[RetrievalResult]:
    """Embed every text exactly once, then score all cases from the cache."""
    settings = get_settings()

    unique_agent_texts: Dict[str, str] = {}
    for case in CASES:
        for agent in case.existing:
            unique_agent_texts[agent.name] = agent_embedding_text(agent.name, agent.description)

    agent_names = list(unique_agent_texts)
    query_texts = [f"{case.proposed_name}. {case.instructions}" for case in CASES]

    print(
        f"Embedding {len(agent_names)} agents + {len(query_texts)} queries "
        f"with {settings.embedding_model} ..."
    )
    vectors = await embed_texts([unique_agent_texts[name] for name in agent_names] + query_texts)
    if vectors is None:
        raise SystemExit("Embedding request failed - check OPENROUTER_API_KEY and connectivity.")

    agent_vectors = dict(zip(agent_names, vectors[: len(agent_names)]))
    query_vectors = vectors[len(agent_names) :]

    results: List[RetrievalResult] = []
    for case, query_vector in zip(CASES, query_vectors):
        ranked = sorted(
            (
                (agent.name, cosine_similarity(query_vector, agent_vectors[agent.name]))
                for agent in case.existing
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        results.append(RetrievalResult(case=case, ranked=ranked))

    return results


def sweep_thresholds(results: List[RetrievalResult], top_k: int) -> None:
    """Print recall@k and floor behaviour across candidate similarity thresholds."""
    reuse_results = [r for r in results if r.case.expected_link is not None]

    within_k = sum(
        1
        for r in reuse_results
        if (rank := r.rank_of_expected()) is not None and rank < top_k
    )
    recall_at_k = within_k / len(reuse_results) if reuse_results else 0.0

    print(f"\nRetrieval recall@{top_k}: {recall_at_k:.0%} "
          f"({within_k}/{len(reuse_results)} true-reuse cases had the right agent shortlisted)")
    print("  (an agent missing from the shortlist can never be linked, whatever the LLM decides)")

    print("\nSimilarity-floor sweep")
    print(f"  {'floor':>6}  {'reuse kept':>11}  {'non-reuse considered':>21}")
    for floor in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        kept = sum(1 for r in reuse_results if r.top_score() >= floor)
        considered = sum(
            1 for r in results if r.case.expected_link is None and r.top_score() >= floor
        )
        print(f"  {floor:>6.2f}  {kept:>4}/{len(reuse_results):<6}  {considered:>21}")
    print("  'reuse kept'  - true-reuse cases still reaching the LLM (higher is better)")
    print("  'non-reuse considered' - new-work cases reaching it (lower means fewer chances to err)")


# ----------------------------------------------------------------------
# Phase 2: non-deterministic judgment evaluation
# ----------------------------------------------------------------------


# Outcomes the *shipped* pipeline can produce, as opposed to the judge in isolation.
LINKED = "linked"  # created, with a possible_duplicate_of pointing somewhere
CREATED = "created"  # created cleanly, no link
CLARIFY = "needs_clarification"  # tie detected, nothing started
REUSED = "reused"  # exact-name match, straight reuse


@dataclass
class JudgmentOutcome:
    case: EvalCase
    # (outcome kind, linked-to agent name or None) per run
    decisions: List[Tuple[str, Optional[str]]]

    @property
    def flip_rate(self) -> float:
        """Share of runs disagreeing with this case's own most common answer."""
        if not self.decisions:
            return 0.0
        modal = max(set(self.decisions), key=self.decisions.count)
        return sum(1 for d in self.decisions if d != modal) / len(self.decisions)

    def counts(self) -> Dict[str, int]:
        """Confusion-matrix counts across runs.

        Five buckets rather than four: asking the user is a distinct outcome from
        both linking and creating. Scoring a clarification as a plain "created"
        would hide the behaviour entirely, and scoring it as a miss would punish
        the system for correctly refusing to guess.
        """
        tally = {"tp": 0, "tn": 0, "false_merge": 0, "missed": 0, "clarify": 0}
        for kind, linked_to in self.decisions:
            expected = self.case.expected_link

            if kind == CLARIFY:
                tally["clarify"] += 1
            elif linked_to is not None and linked_to == expected:
                tally["tp"] += 1  # linked, correctly
            elif linked_to is not None:
                # Linked when it should not have, or linked to the wrong agent.
                tally["false_merge"] += 1
            elif expected is None:
                tally["tn"] += 1  # created new, correctly
            else:
                tally["missed"] += 1  # should have linked, did not
        return tally


async def run_judgment_phase(
    results: List[RetrievalResult], runs: int, floor: float, top_k: int
) -> List[JudgmentOutcome]:
    """Drive the SHIPPED routing path, not the judge in isolation.

    An earlier version called `decide_routing` directly. That measured the judge but
    skipped everything wrapping it - the exact-match fast path, the similarity floor,
    and the ambiguity check - so it reported on a code path the real system would
    never take. The two-Keiths case is the clearest example: live, the ambiguity check
    catches that tie *before* the judge is ever consulted, so testing the judge on it
    answered a question the system does not ask.

    This version builds a real roster with real embeddings and calls
    `send_message_to_agent`, reading the outcome back off the roster.
    """
    import tempfile

    from server.agents.interaction_agent import tools as interaction_tools
    from server.services.execution import log_store as log_store_module
    from server.services.execution import roster as roster_module
    from server.services.execution.log_store import ExecutionAgentLogStore
    from server.services.execution.roster import AgentRoster

    outcomes: List[JudgmentOutcome] = []
    judged_cases: List[bool] = []
    staged_links: List[str] = []
    committed_merges: List[str] = []
    settings = get_settings()

    for result in results:
        case = result.case
        decisions: List[Tuple[str, Optional[str]]] = []
        llm_consulted: List[bool] = []

        for _ in range(runs):
            # Fresh roster per run so runs cannot contaminate each other.
            tmp = Path(tempfile.mkdtemp())
            roster = AgentRoster(tmp / "roster.json")
            roster_module._agent_roster = roster
            log_store_module._execution_agent_logs = ExecutionAgentLogStore(tmp / "logs")

            for agent in case.existing:
                await roster.create_or_link(agent.name, description=agent.description)
                vector = await embed_text(agent_embedding_text(agent.name, agent.description))
                if vector:
                    roster.update_embedding(agent.name, vector, settings.embedding_model)

            dispatched: List[str] = []
            original_dispatch = interaction_tools._dispatch_to_agent
            original_decide = interaction_tools.decide_routing

            def _capture(agent_name, instructions, *, action):
                dispatched.append(action)
                return interaction_tools.ToolResult(
                    success=True, payload={"agent_name": agent_name, "status": "submitted"}
                )

            # Count judgment calls. Whether the LLM was consulted at all is a
            # first-class diagnostic: a case the floor filters out never tests the
            # judge, and an evaluation where that happens to *every* hard case
            # reports a score that really only measures the threshold.
            async def _counting_decide(**kwargs):
                judged.append(True)
                return await original_decide(**kwargs)

            judged: List[bool] = []
            interaction_tools._dispatch_to_agent = _capture
            interaction_tools.decide_routing = _counting_decide
            try:
                tool_result = await interaction_tools.send_message_to_agent(
                    case.proposed_name, case.instructions
                )
            finally:
                interaction_tools._dispatch_to_agent = original_dispatch
                interaction_tools.decide_routing = original_decide

            judged_this_run = bool(judged)
            llm_consulted.append(judged_this_run)

            if tool_result.payload.get("status") == "needs_clarification":
                decisions.append((CLARIFY, None))
                continue

            if dispatched and dispatched[-1] == "reused":
                decisions.append((REUSED, case.proposed_name))
                continue

            record = roster.get_record(roster.resolve_name(case.proposed_name))
            link = record.possible_duplicate_of.name if record and record.possible_duplicate_of else None

            # A staged link is a hypothesis, not a merge. Run the reconciliation the
            # shipped system runs after execution, so the two are scored separately:
            # the judge's error rate is what it stages, and the system's error rate
            # is what actually survives the evidence gate. Conflating them either
            # credits the gate for the judge's caution or blames the judge for a
            # failure that never reached the user.
            merged_into = None
            if link:
                try:
                    merged_into = reconcile_link(record.name)
                except Exception:  # pragma: no cover - defensive
                    merged_into = None
                staged_links.append(link)
                if merged_into:
                    committed_merges.append(merged_into)

            decisions.append((LINKED if link else CREATED, link))

        outcome = JudgmentOutcome(case=case, decisions=decisions)
        kinds = {kind for kind, _ in outcome.decisions}
        judged_any = any(llm_consulted)
        note = f"flip={outcome.flip_rate:.0%}"
        if kinds == {CLARIFY}:
            note = "asked the user (tie detected)"
        elif not result.ranked:
            note = "empty roster - no LLM call"
        elif not judged_any:
            note = "below floor - no LLM call"
        judged_cases.append(judged_any)

        outcomes.append(outcome)
        tally = outcome.counts()
        marker = "!!" if tally["false_merge"] else "  "
        print(
            f"  {marker} {case.case_id:<42} tp={tally['tp']} tn={tally['tn']} "
            f"false_merge={tally['false_merge']} missed={tally['missed']}  {note}"
        )

    reached = sum(judged_cases)
    print(
        f"\n  Judge consulted on {reached}/{len(judged_cases)} cases. Cases the floor"
        f" filters out never test it,\n  so a score dominated by those is really a score"
        f" for the threshold, not the judge."
    )
    print(
        f"\n  Links staged by the judge: {len(staged_links)}"
        f"   |   merges committed after the evidence gate: {len(committed_merges)}"
    )
    if staged_links and not committed_merges:
        print("  Every staged link was stopped by the gate - which is the gate doing")
        print("  exactly the job it exists for, on real model output.")

    return outcomes


# ----------------------------------------------------------------------
# Baseline + reporting
# ----------------------------------------------------------------------


def exact_match_baseline() -> Tuple[int, int]:
    """The behaviour being replaced: reuse only on an exact name match.

    Included so the improvement is measured rather than asserted. Note this
    baseline is structurally incapable of a false merge - it is a recall
    comparator, not a safety one.
    """
    reuse_cases = [case for case in CASES if case.expected_link is not None]
    hits = sum(
        1
        for case in reuse_cases
        if any(agent.name == case.proposed_name for agent in case.existing)
    )
    return hits, len(reuse_cases)


def report(outcomes: List[JudgmentOutcome], runs: int) -> bool:
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    totals = {"tp": 0, "tn": 0, "false_merge": 0, "missed": 0, "clarify": 0}
    adversarial_false_merges = 0

    for outcome in outcomes:
        tally = outcome.counts()
        for key, value in tally.items():
            totals[key] += value
        if outcome.case.category == "adversarial":
            adversarial_false_merges += tally["false_merge"]

    total_decisions = sum(totals.values())

    # Precision/recall are scoped to the *link* decision specifically: a correctly
    # created new agent is a true negative, not a true positive.
    predicted_links = totals["tp"] + totals["false_merge"]
    actual_links = totals["tp"] + totals["missed"]
    precision = totals["tp"] / predicted_links if predicted_links else 0.0
    recall = totals["tp"] / actual_links if actual_links else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Denominator is every decision where linking would have been wrong.
    non_link_decisions = totals["tn"] + totals["false_merge"]
    false_merge_rate = totals["false_merge"] / non_link_decisions if non_link_decisions else 0.0
    # Clarifications are excluded from accuracy: refusing to guess is neither a
    # correct nor an incorrect answer, and counting it either way distorts the number.
    decided = total_decisions - totals["clarify"]
    accuracy = (totals["tp"] + totals["tn"]) / decided if decided else 0.0

    print(f"\nDecisions evaluated: {total_decisions} ({len(outcomes)} cases x {runs} runs)")
    print(f"  true links: {totals['tp']}   correct new agents: {totals['tn']}   "
          f"false merges: {totals['false_merge']}   missed links: {totals['missed']}   "
          f"asked user: {totals['clarify']}")
    print(f"\n  overall accuracy:                         {accuracy:.0%}")
    print(f"  precision (of links made, share correct): {precision:.0%}")
    print(f"  recall    (of links needed, share made):  {recall:.0%}")
    print(f"  F1:                                       {f1:.2f}")
    print(f"\n  FALSE-MERGE RATE: {false_merge_rate:.1%} "
          f"({totals['false_merge']}/{non_link_decisions} decisions where linking would be wrong)")
    print("    the asymmetric error: silently mixes two unrelated threads' context")

    flip_rates = [o.flip_rate for o in outcomes]
    unstable = [o.case.case_id for o in outcomes if o.flip_rate > 0]
    print(f"\nDecision stability across {runs} runs:")
    print(f"  mean flip rate: {statistics.mean(flip_rates):.1%}")
    if unstable:
        print(f"  unstable cases: {', '.join(unstable)}")
    else:
        print("  every case answered identically on every run")

    baseline_hits, baseline_total = exact_match_baseline()
    baseline_recall = baseline_hits / baseline_total if baseline_total else 0.0
    print(f"\nBaseline (exact name match, current behaviour):")
    print(f"  recall: {baseline_recall:.0%} ({baseline_hits}/{baseline_total} true-reuse cases)")
    print(f"  new system recall: {recall:.0%}")

    # ---- acceptance thresholds ----
    print("\n" + "-" * 72)
    print("ACCEPTANCE")
    print("-" * 72)

    checks = [
        (
            "zero false merges on the adversarial set",
            adversarial_false_merges == 0,
            f"{adversarial_false_merges} observed",
        ),
        (
            "recall strictly better than the exact-match baseline",
            recall > baseline_recall,
            f"{recall:.0%} vs {baseline_recall:.0%}",
        ),
    ]

    passed = True
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} ({detail})")
        passed = passed and ok

    adversarial_decisions = sum(
        runs for o in outcomes if o.case.category == "adversarial"
    )
    print(
        f"\n  Statistical caveat: {adversarial_decisions} adversarial decisions is a small "
        f"sample.\n  Zero observed false merges bounds the true rate at roughly "
        f"<={3 / adversarial_decisions:.0%} with 95% confidence (rule of three) - enough to "
        f"catch\n  a badly calibrated threshold, not enough to certify rarity in production."
    )

    return passed


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="LLM runs per case (default 5)")
    parser.add_argument("--floor", type=float, default=None, help="similarity floor override")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set; this evaluator needs real API access.")
        return 2

    floor = args.floor if args.floor is not None else settings.agent_min_candidate_similarity
    top_k = settings.agent_dedup_top_k

    print("=" * 72)
    print("PHASE 1 - retrieval (deterministic, embeddings cached)")
    print("=" * 72)
    results = await run_retrieval_phase()
    sweep_thresholds(results, top_k)

    print("\n" + "=" * 72)
    print(f"PHASE 2 - judgment (non-deterministic, {args.runs} runs/case, floor={floor})")
    print("=" * 72)
    outcomes = await run_judgment_phase(results, runs=args.runs, floor=floor, top_k=top_k)

    return 0 if report(outcomes, args.runs) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
