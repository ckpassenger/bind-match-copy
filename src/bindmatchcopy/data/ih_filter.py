"""Induction-head (IH) paragraph scoring via the ``freq_rel`` two-rule filter.

The paper's v20 IH data uses the ``freq_rel`` variant (the "standard Aoyama"
scoring), which combines two *type-level* statistics over a tokenised
paragraph:

- ``F``  (``bigram_freq``): the fraction of bigram *types* that repeat
  (a repeated bigram type is one observed at least twice).
- ``R2`` (``l2_reliability``): among bigram types with at least two successor
  observations, the fraction whose successor has a *strict majority*
  (the most frequent successor token appears in more than half of that
  bigram's successor observations).

Both statistics are computed over **types**, not token-frequency weighted, and
``R2`` does not require all successors to be identical.

A paragraph is kept when::

    composite = w_freq * F + w_rel * R2 > threshold

with ``w_freq = w_rel = 0.5`` and ``threshold = 0.10`` for the v20 ``freq_rel``
dataset. Empty denominators are treated as zero (never kept).

Paragraph preprocessing (kept from the original implementation):

- Paragraphs are split on blank lines; after ``strip()``, a paragraph shorter
  than 100 characters is discarded before scoring.
- The GPT-2 scoring window is at most 384 tokens; a paragraph shorter than
  30 tokens is not scored.
- The full kept paragraph text is written out — the 384-token scoring window
  is *not* applied to the training text.
- The final bigram (which has no successor) contributes to ``F`` but not to
  ``R2``'s successor observations.

This module only implements the *scoring* logic on an already-tokenised
sequence of hashable tokens; tokenisation is left to the caller so the core
can be unit-tested without a tokenizer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Hashable, Iterable


def bigram_frequency_ratio(tokens: list[Hashable]) -> float:
    """Compute ``F``: the fraction of bigram *types* that repeat.

    A "type" is a distinct adjacent token pair ``(a, b)``. A type repeats if it
    is observed at least twice. ``F = n_repeated_types / n_total_types``.

    Returns 0.0 when the sequence has fewer than two tokens (no bigrams).
    """
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    type_counts = Counter(bigrams)
    n_total = len(type_counts)
    n_repeated = sum(1 for c in type_counts.values() if c >= 2)
    return n_repeated / n_total


def reliability_ratio(tokens: list[Hashable]) -> float:
    """Compute ``R2``: fraction of (multi-observation) bigram types with a
    strict-majority successor.

    For each bigram type, its successor list is the token following every
    occurrence. The final bigram (position ``len(tokens) - 2``) has no successor
    and therefore contributes no successor observation.

    Only bigram types with at least two successor observations enter the
    denominator. A type has a strict-majority successor when its most frequent
    successor appears in more than half of its successor observations.

    Returns 0.0 when the denominator is empty.
    """
    if len(tokens) < 3:
        return 0.0
    successors: dict[tuple, list] = defaultdict(list)
    # The final bigram has no successor, so stop at index len(tokens) - 3.
    for i in range(len(tokens) - 2):
        successors[(tokens[i], tokens[i + 1])].append(tokens[i + 2])

    eligible = 0
    majority = 0
    for succs in successors.values():
        if len(succs) < 2:
            continue
        eligible += 1
        counts = Counter(succs).values()
        if max(counts) > len(succs) / 2:
            majority += 1
    if eligible == 0:
        return 0.0
    return majority / eligible


def ih_composite_score(
    tokens: list[Hashable], w_freq: float = 0.5, w_rel: float = 0.5
) -> float:
    """Return the ``freq_rel`` composite score ``w_freq * F + w_rel * R2``."""
    return w_freq * bigram_frequency_ratio(tokens) + w_rel * reliability_ratio(tokens)


def keep_paragraph(
    tokens: list[Hashable],
    w_freq: float = 0.5,
    w_rel: float = 0.5,
    threshold: float = 0.10,
    min_scoring_tokens: int = 30,
) -> bool:
    """Decide whether a tokenised paragraph passes the IH filter.

    Args:
        tokens: token ids of the paragraph (scoring window; the caller should
            truncate to at most 384 tokens before calling).
        w_freq / w_rel: rule weights (default 0.5 / 0.5).
        threshold: keep when composite score is strictly greater than this.
        min_scoring_tokens: paragraphs shorter than this are not scored (rejected).

    Returns:
        True if the paragraph should be kept for the IH training mixture.
    """
    if len(tokens) < min_scoring_tokens:
        return False
    return ih_composite_score(tokens, w_freq, w_rel) > threshold


def filter_paragraphs(
    paragraphs: Iterable[list[Hashable]],
    w_freq: float = 0.5,
    w_rel: float = 0.5,
    threshold: float = 0.10,
    min_scoring_tokens: int = 30,
) -> list[list[Hashable]]:
    """Filter an iterable of tokenised paragraphs, keeping the passing ones."""
    kept = []
    for tokens in paragraphs:
        if keep_paragraph(tokens[:384], w_freq, w_rel, threshold, min_scoring_tokens):
            kept.append(tokens)
    return kept


def filter_text(text, tokenizer):
    """Score a 384-token prefix; retain full stripped paragraphs with provenance."""
    import re
    import hashlib

    records = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if len(paragraph) < 100:
            continue
        tokens = tokenizer.encode(paragraph, add_special_tokens=False)[:384]
        if keep_paragraph(tokens):
            records.append(
                dict(
                    text=paragraph,
                    sha256=hashlib.sha256(paragraph.encode()).hexdigest(),
                    score=ih_composite_score(tokens),
                    scoring_tokens=len(tokens),
                )
            )
    return records
