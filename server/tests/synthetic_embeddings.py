"""A deterministic stand-in embedding model with *designed* semantic geometry.

Why this exists
---------------
Evaluating agent selection needs embeddings, and real ones need an API key, cost
money, and vary by provider. That would push the entire evaluation into the "live"
tier, which is the wrong place for it: most of what needs testing is decision logic,
not the embedding provider.

This module builds vectors from an explicit concept basis instead, so similarity
between any two items is a *known quantity* rather than something to be measured.
That buys two things real embeddings cannot:

1. **Offline, deterministic, free evaluation** - the whole labeled set, degradation
   sweep, and ablation table run with no key and identical results every time.
2. **Controlled probing of thresholds** - pairs can be constructed at a target
   cosine similarity, so "at 0.75, a same-person/different-topic pair must not
   merge" becomes a directly testable claim.

What it does NOT do
-------------------
It cannot validate that *real* embeddings place "lunch w/ Keith" near "Email to
Keith about lunch". Those semantic relationships are ones this file invents, so
grading the system against them proves the logic works, not that the retrieval
works in the real world. That claim belongs to the live tier, which checks the
ordering assumption against a real model.

Geometry
--------
Each concept (a person, a topic) gets its own orthogonal axis. An item's vector is
the normalized sum of its weighted concepts, so cosine similarity falls out of
concept overlap:

    same person + same topic      -> ~1.0   (true reuse)
    same person + different topic -> ~0.5   (the adversarial case)
    different person + same topic -> ~0.5
    nothing shared                -> ~0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Weights control how much each facet contributes to the vector. Person is
# weighted slightly higher than topic because in this domain "who is this about"
# is the stronger identity signal - matching how a shared correspondent is
# suggestive while a shared topic alone is weak.
PERSON_WEIGHT = 1.0
TOPIC_WEIGHT = 0.9


class ConceptSpace:
    """Assigns each named concept its own orthogonal axis."""

    def __init__(self) -> None:
        self._axes: Dict[str, int] = {}

    def axis(self, concept: str) -> int:
        if concept not in self._axes:
            self._axes[concept] = len(self._axes)
        return self._axes[concept]

    @property
    def dimensions(self) -> int:
        return len(self._axes)

    def vector(self, weighted_concepts: Iterable[Tuple[str, float]], size: int) -> List[float]:
        """Build a normalized vector from (concept, weight) pairs."""
        vector = [0.0] * size
        for concept, weight in weighted_concepts:
            index = self.axis(concept)
            if index < size:
                vector[index] += weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


@dataclass
class Concepts:
    """The facets an agent or request is *about*.

    `person_strength` / `topic_strength` scale how clearly each facet comes
    through. Agents describing a thread default to 1.0 on both. Requests often
    should not: a terse "Keith lunch f/u" carries the person strongly and the
    topic weakly, and an abbreviation or typo degrades the signal further.

    Modelling that loss matters. With every request tagged identically to its
    target agent, the correct agent scores a perfect 1.0 while any distractor
    sharing one facet scores ~0.55 - retrieval is never contested and the
    evaluation reports a meaningless flat 100%. Real requests are lossy, and the
    interesting behaviour lives in the region where a confusable neighbour can
    plausibly outrank the right answer.
    """

    person: Optional[str] = None
    topic: Optional[str] = None
    extra: List[str] = field(default_factory=list)
    person_strength: float = 1.0
    topic_strength: float = 1.0

    def weighted(self) -> List[Tuple[str, float]]:
        pairs: List[Tuple[str, float]] = []
        if self.person:
            pairs.append((f"person:{self.person}", PERSON_WEIGHT * self.person_strength))
        if self.topic:
            pairs.append((f"topic:{self.topic}", TOPIC_WEIGHT * self.topic_strength))
        pairs.extend((f"extra:{item}", 0.4) for item in self.extra)
        return pairs


class SyntheticEmbeddingModel:
    """Maps text to a vector via a registered concept tagging.

    Text is not parsed - callers register what each string is *about*. Inferring
    concepts from the words would just be a worse embedding model, and would
    reintroduce the guessing this file exists to remove.
    """

    # Fixed size so all vectors are comparable; larger than any test's concept count.
    DIMENSIONS = 256

    def __init__(self) -> None:
        self._space = ConceptSpace()
        self._tags: Dict[str, Concepts] = {}

    def register(self, text: str, concepts: Concepts) -> None:
        self._tags[text] = concepts

    def register_many(self, tagged: Dict[str, Concepts]) -> None:
        for text, concepts in tagged.items():
            self.register(text, concepts)

    def embed(self, text: str) -> List[float]:
        """Return the vector for a registered text.

        Unregistered text embeds to a near-zero-similarity vector on its own axis,
        so an unexpected string behaves like unrelated content rather than
        silently matching everything.
        """
        concepts = self._tags.get(text)
        if concepts is None:
            return self._space.vector([(f"unknown:{text}", 1.0)], self.DIMENSIONS)
        return self._space.vector(concepts.weighted(), self.DIMENSIONS)

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two registered texts."""
        a, b = self.embed(text_a), self.embed(text_b)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def as_request_embeddings(self):
        """An async stand-in matching the signature of `request_embeddings`.

        Lets the real code paths run unmodified against synthetic geometry:
            monkeypatch.setattr(matcher, "request_embeddings", model.as_request_embeddings())
        """

        async def _request_embeddings(*, model, texts, api_key=None, timeout=None, **kwargs):
            return self.embed_many(list(texts))

        return _request_embeddings


def two_vectors_at_similarity(target: float, size: int = 256) -> Tuple[List[float], List[float]]:
    """Construct two unit vectors with an exact cosine similarity.

    Used to probe threshold behaviour directly: place a pair at 0.75 and assert
    what the system does, without depending on any text or model.
    """
    target = max(-1.0, min(1.0, target))
    a = [0.0] * size
    b = [0.0] * size
    a[0] = 1.0
    b[0] = target
    b[1] = math.sqrt(max(0.0, 1.0 - target * target))
    return a, b


__all__ = [
    "Concepts",
    "ConceptSpace",
    "SyntheticEmbeddingModel",
    "two_vectors_at_similarity",
]
