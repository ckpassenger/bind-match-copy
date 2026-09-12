"""Fine-grained token bucketing for the paper's eval metrics.

Protocol ID: recall_buckets_v31_1. Domain string heuristics are inherited and
require validation against the server classifier; structural buckets are tested.

This module classifies every token position of a sequence into a hierarchy of
buckets. The per-bucket perplexity reported by :mod:`bindmatchcopy.evaluation`
is computed over these labels, so this classifier *is* the definition of the
evaluation metrics (``B3d``, ``code_var_reuse``, ``distance_far``, ...).

Label hierarchy
---------------

Level 1 (base):
  - ``B0`` — token never seen before in the context window.
  - ``B1`` — the *A* token appeared before, but not as the same ``(A, B)`` bigram.
  - ``B3`` — induction pattern: the ``(A, B)`` bigram appeared before.

Level 2 (B3 sub-type, needs a tokenizer to decode):
  - ``B3a`` — function-word / short-token induction.
  - ``B3b`` — identifier induction (code variable names).

Level 3 (stability):
  - ``B3c`` — stable: A is always followed by the same B (unique-B count == 1).
  - ``B3d`` — ambiguous: A is followed by multiple distinct B tokens.

Level 4 (distance × stability cross, in window units):
  - ``B3c_near / B3d_near``   (1-20), ``B3c_mid / B3d_mid`` (21-100),
    ``B3c_far / B3d_far``     (101-512), ``B3c_vfar / B3d_vfar`` (>512).
  - Plain distance buckets ``distance_near/mid/far/vfar`` are emitted too.

Level 5 (consistency, continuous):
  - ``consistency_perfect / high / medium / low / very_low``.

Level 6 (domain-specific):
  - code: ``code_var_reuse``, ``code_scope_shadow``, ``code_long_id``,
    ``code_kw_repeat``, ``code_attr_chain``.
  - math: ``math_var_bind``, ``math_ambig_symbol``, ``math_long_proof``,
    ``math_op_pattern``.
  - wiki: ``wiki_ngram_common``, ``wiki_ngram_rare``, ``wiki_entity_repeat``,
    ``wiki_syntax_pattern``.

The core ``B0/B1/B3/B3c/B3d`` and distance/consistency buckets depend only on
token ids. The ``B3a/B3b`` and domain-specific buckets additionally decode
tokens to strings and therefore require a tokenizer; if none is available they
are simply skipped, leaving the structural buckets intact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Function-word / keyword / symbol sets
# ----------------------------------------------------------------------
FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "about",
    "and",
    "or",
    "but",
    "if",
    "that",
    "this",
    "it",
    "not",
    "as",
    "he",
    "she",
    "they",
    "we",
    "you",
    "I",
    "me",
    "him",
    "her",
    "us",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "may",
    "might",
    "has",
    "have",
    "had",
    "been",
    "its",
    "his",
    "her",
    "their",
    # code keywords
    "def",
    "return",
    "import",
    "class",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "try",
    "except",
    "with",
    "pass",
    "break",
    "continue",
    "yield",
    "raise",
    "in",
    "as",
    "None",
    "True",
    "False",
    "self",
    "lambda",
    "global",
    "nonlocal",
    # common math tokens
    "x",
    "y",
    "z",
    "n",
    "m",
    "k",
    "i",
    "j",
    "t",
    "a",
    "b",
    "c",
    "d",
    "f",
    "g",
    "=",
    "+",
    "-",
    "*",
    "/",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
}

# Distance bins: (label, lo, hi), all inclusive.
DISTANCE_BINS = [
    ("near", 1, 20),
    ("mid", 21, 100),
    ("far", 101, 512),
    ("very_far", 513, 999999),
]

# Consistency bins: (label, lo, hi), all inclusive.
CONSISTENCY_BINS = [
    ("perfect", 1.0, 1.0),
    ("high", 0.8, 1.0),
    ("medium", 0.5, 0.8),
    ("low", 0.2, 0.5),
    ("very_low", 0.0, 0.2),
]

# Default window size for the bigram index.
DEFAULT_WINDOW_SIZE = 512


def _domain_sub_labels(domain: str) -> List[str]:
    if domain == "code":
        return [
            "code_var_reuse",
            "code_scope_shadow",
            "code_long_id",
            "code_kw_repeat",
            "code_attr_chain",
        ]
    if domain == "math":
        return [
            "math_var_bind",
            "math_ambig_symbol",
            "math_long_proof",
            "math_op_pattern",
        ]
    if domain == "wiki":
        return [
            "wiki_ngram_common",
            "wiki_ngram_rare",
            "wiki_entity_repeat",
            "wiki_syntax_pattern",
        ]
    return []


def state_tracking_labels() -> List[str]:
    """Labels for the synthetic state-tracking task (kept for completeness)."""
    return [
        "state_latest_value",
        "state_overwrite_distractor",
        "state_multi_key_stress",
        "state_write_dilution",
    ]


def bucket_labels() -> List[str]:
    """Return the full registry of bucket label names.

    Downstream evaluators use this to know every label that may appear in a
    classified sequence.
    """
    labels = ["B0", "B1", "B3", "B3a", "B3b", "B3c", "B3d"]
    for dist_label, _, _ in DISTANCE_BINS:
        labels.append(f"distance_{dist_label}")
        labels.append(f"B3c_{dist_label}")
        labels.append(f"B3d_{dist_label}")
    for con_label, _, _ in CONSISTENCY_BINS:
        labels.append(f"consistency_{con_label}")
    for d in ("code", "math", "wiki"):
        labels.extend(_domain_sub_labels(d))
    labels.extend(state_tracking_labels())
    labels.append("total")
    return labels


def _is_identifier(s: str) -> bool:
    """Heuristic: alphanumeric + underscore, not purely numeric, not empty."""
    if not s:
        return False
    if s.isnumeric():
        return False
    for c in s:
        if not (c.isalnum() or c == "_"):
            return False
    return True


class BucketClassifier:
    """Streaming token bucket classifier for one evaluation session.

    Args:
        window_size: size of the sliding bigram index window (default 512).
        tokenizer: optional tokenizer used to decode token ids into strings for
            the ``B3a/B3b`` and domain-specific buckets. May be ``None``; in
            that case those string-dependent buckets are skipped, but the core
            structural buckets (``B0/B1/B3/B3c/B3d``, distance, consistency)
            are still produced.
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE, tokenizer=None):
        self.window_size = window_size
        self.tok = tokenizer
        self.stats: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Clear per-session bucket statistics."""
        self.stats = defaultdict(int)

    def classify_sequence(self, token_ids: List[int], domain: str) -> List[List[str]]:
        """Classify every position of a token id sequence.

        Args:
            token_ids: list of token ids (GPT-2 BPE tokens).
            domain: one of "wiki", "python", "math", "ood_python", "ood_math".

        Returns:
            A list (same length as ``token_ids``) where each element is a list
            of bucket label strings. ``labels[0]`` is always ``["B0"]`` (the
            first token has no preceding context).
        """
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        n = len(token_ids)
        labels = [[] for _ in token_ids]
        # History contains only completed bigrams before the scored target.
        history = []
        for pos in range(n):
            if pos == 0:
                labels[pos] = ["B0"]
            else:
                a, b = token_ids[pos - 1], token_ids[pos]
                history = [
                    (ap, aa, bb)
                    for ap, aa, bb in history
                    if pos - ap - 1 <= self.window_size
                ]
                previous = [(ap, bb) for ap, aa, bb in history if aa == a]
                successors = {bb for _, bb in previous}
                if b not in successors:
                    labels[pos] = ["B1" if previous else "B0"]
                else:
                    labels[pos] = ["B3"]
                    stable = "B3c" if len(successors) == 1 else "B3d"
                    labels[pos].append(stable)
                    # Nearest previous A, even when its successor was NOT B.
                    distance = pos - max(ap for ap, _ in previous) - 1
                    for name, lo, hi in DISTANCE_BINS:
                        if lo <= distance <= hi:
                            labels[pos].extend([f"distance_{name}", f"{stable}_{name}"])
                            break
                    consistency = 1 / len(successors)
                    for name, lo, hi in CONSISTENCY_BINS:
                        if lo <= consistency <= hi:
                            labels[pos].append(f"consistency_{name}")
                            break
                    if self.tok is not None:
                        token = self.tok.decode([b]).strip().lower()
                        if token in FUNCTION_WORDS or len(token) <= 2:
                            labels[pos].append("B3a")
                        elif domain in ("python", "ood_python") and _is_identifier(
                            token
                        ):
                            labels[pos].append("B3b")
                        self._classify_domain_specific(
                            token_ids, pos, distance, domain, labels
                        )
                history.append((pos - 1, a, b))
            for label in set(labels[pos]):
                if not label.startswith(("code_", "math_", "wiki_")):
                    self.stats[label] += 1
        self.stats["total"] += n
        return labels

    def _classify_domain_specific(self, token_ids, pos, dist, domain, labels) -> None:
        """Append domain-specific fine-grained bucket labels."""
        tok = self.tok
        token_str = tok.decode([token_ids[pos]]).strip()

        if domain in ("python", "ood_python"):
            if _is_identifier(token_str):
                # scope shadow: same identifier with a different preceding token.
                shadow = False
                for lookback in range(max(0, pos - 200), pos - 1):
                    if token_ids[lookback] == token_ids[pos]:
                        prev_A = token_ids[lookback - 1] if lookback >= 1 else -1
                        curr_A = token_ids[pos - 1]
                        if prev_A != curr_A:
                            shadow = True
                            break

                # attr chain: a "." preceding the token (or within a short span).
                is_chain = False
                if tok.decode([token_ids[pos - 1]]).strip() == ".":
                    is_chain = True
                else:
                    for scan in range(max(0, pos - 6), pos - 1):
                        if tok.decode([token_ids[scan]]).strip() == ".":
                            is_chain = True
                            break

                if is_chain:
                    labels[pos].append("code_attr_chain")
                    self.stats["code_attr_chain"] += 1
                elif shadow:
                    labels[pos].append("code_scope_shadow")
                    self.stats["code_scope_shadow"] += 1
                elif dist is not None and dist > 100:
                    labels[pos].append("code_long_id")
                    self.stats["code_long_id"] += 1
                else:
                    labels[pos].append("code_var_reuse")
                    self.stats["code_var_reuse"] += 1
            elif token_str.lower() in FUNCTION_WORDS:
                labels[pos].append("code_kw_repeat")
                self.stats["code_kw_repeat"] += 1

        elif domain in ("math", "ood_math"):
            if token_str.isalpha() and len(token_str) <= 3:
                if "B3c" in labels[pos]:
                    labels[pos].append("math_var_bind")
                    self.stats["math_var_bind"] += 1
                elif "B3d" in labels[pos]:
                    labels[pos].append("math_ambig_symbol")
                    self.stats["math_ambig_symbol"] += 1
                if dist is not None and dist > 100:
                    labels[pos].append("math_long_proof")
                    self.stats["math_long_proof"] += 1

            math_ops = {
                "+",
                "-",
                "*",
                "/",
                "=",
                ">",
                "<",
                "≤",
                "≥",
                "^",
                "_",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
                ",",
                ".",
                "×",
                "÷",
            }
            if token_str in math_ops or (
                len(token_str) <= 1 and not token_str.isalpha()
            ):
                near_ops = 0
                for scan in range(max(0, pos - 5), pos):
                    if scan == pos:
                        continue
                    s = tok.decode([token_ids[scan]]).strip()
                    if s in math_ops or (len(s) <= 1 and not s.isalpha()):
                        near_ops += 1
                if near_ops >= 2:
                    labels[pos].append("math_op_pattern")
                    self.stats["math_op_pattern"] += 1

        elif domain == "wiki":
            if "B3c" in labels[pos]:
                labels[pos].append("wiki_ngram_common")
                self.stats["wiki_ngram_common"] += 1
            elif "B3d" in labels[pos]:
                labels[pos].append("wiki_ngram_rare")
                self.stats["wiki_ngram_rare"] += 1
            if token_str and len(token_str) > 2 and token_str[0].isupper():
                labels[pos].append("wiki_entity_repeat")
                self.stats["wiki_entity_repeat"] += 1
            if token_str.lower() in (
                "the",
                "a",
                "an",
                "of",
                "in",
                "to",
                "for",
                "with",
                "and",
                "or",
            ):
                labels[pos].append("wiki_syntax_pattern")
                self.stats["wiki_syntax_pattern"] += 1
