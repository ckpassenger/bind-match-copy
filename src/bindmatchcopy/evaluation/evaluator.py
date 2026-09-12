"""Bucket perplexity evaluator.

The paper's headline functional metric is the per-bucket perplexity over a
*fixed* eval set whose token positions have been pre-labelled by
:class:`~bindmatchcopy.evaluation.bucket.BucketClassifier`. This module computes
that metric: for every sequence, run the model once, obtain per-token NLL, then
accumulate NLL per bucket label and report ``exp(mean NLL)`` per label.

The evaluation set format is ``{domain: [(token_ids, labels), ...]}`` where
``labels`` is the per-position list of bucket strings (as produced by
``BucketClassifier.classify_sequence``). Position ``i`` of the sequence is
predicted by the logit at position ``i-1``; the label list is indexed by the
*target* position ``i`` (matching ``labels[pos + 1]`` in the original loop).
"""

from __future__ import annotations

from ..protocol import EVAL_SAMPLES

import math
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F

from .bucket import bucket_labels


# The canonical 5-domain eval split used by the fixed eval set.
DEFAULT_DOMAINS = ["wiki", "python", "math", "ood_python", "ood_math"]

# Training maximum sequence length (sequences are truncated to this).
DEFAULT_MAX_SEQ_LEN = 1024


def _extract_logits(out) -> torch.Tensor:
    """Extract the logits tensor from a model output (dict or bare tensor)."""
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out.logits if hasattr(out, "logits") else out


@torch.no_grad()
def eval_bucket_ppl_fixed(
    model,
    evalset: Dict[str, list],
    domains: List[str] | None = None,
    device: str | torch.device = "cuda:0",
    max_seqs: int = EVAL_SAMPLES,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> Dict[str, dict]:
    """Compute per-bucket perplexity over a fixed, pre-labelled eval set.

    Args:
        model: a model whose ``model(x)`` returns logits (dict with ``"logits"``
            or a bare tensor), already placed on ``device`` and in eval mode.
        evalset: ``{domain: [(token_ids, labels), ...]}`` as produced by the
            fixed-evalset generator. ``labels[i]`` labels the target position
            ``i``.
        domains: domains to evaluate (default: the canonical 5-domain split).
        device: device to run the model on.
        max_seqs: maximum sequences to use per domain.
        max_seq_len: truncate each sequence to this many tokens.

    Returns:
        A dict mapping label -> ``{"ppl", "nll", "n"}``. The ``"total"`` label
        aggregates every predicted token regardless of bucket.
    """
    if domains is None:
        domains = DEFAULT_DOMAINS

    if model.training:
        raise ValueError("Evaluation requires model.eval()")
    if max_seqs < 1 or max_seq_len < 2 or len(set(domains)) != len(domains):
        raise ValueError("Invalid evaluation counts or duplicate domains")
    for domain in domains:
        if not evalset.get(domain):
            raise ValueError(f"Missing or empty evaluation domain: {domain}")
    device = torch.device(device)
    bucket_nll: Dict[str, float] = defaultdict(float)
    bucket_n: Dict[str, int] = defaultdict(int)

    for domain in domains:
        seqs = evalset.get(domain, [])
        n_seqs = min(len(seqs), max_seqs)
        for i in range(n_seqs):
            token_ids, labels = seqs[i]
            if len(token_ids) < 2 or len(labels) != len(token_ids):
                raise ValueError(
                    "Each sequence needs >= 2 tokens and one label list per token"
                )

            x = torch.tensor([token_ids[:max_seq_len]], device=device)
            logits = _extract_logits(model(x))

            per_token_nll = F.cross_entropy(
                logits[0, :-1].float(), x[0, 1:], reduction="none"
            )

            for pos_idx in range(min(len(token_ids), max_seq_len) - 1):
                nll_val = per_token_nll[pos_idx].item()
                for lbl in set(labels[pos_idx + 1]) - {"total"}:
                    bucket_nll[lbl] += nll_val
                    bucket_n[lbl] += 1
                bucket_nll["total"] += nll_val
                bucket_n["total"] += 1

    results: Dict[str, dict] = {}
    for label in sorted(set(bucket_labels()) | set(bucket_n)):
        nll = bucket_nll[label]
        n = bucket_n[label]
        ppl = math.exp(nll / n) if n > 0 and nll / n < 709 else None
        results[label] = {
            "ppl": ppl,
            "nll": nll,
            "n": n,
            "status": "measured" if n else "empty_bucket",
        }
    return results


def summarize_buckets(results: Dict[str, dict], keys: List[str]) -> Dict[str, dict]:
    """Extract a subset of bucket results for concise reporting.

    Args:
        results: output of :func:`eval_bucket_ppl_fixed`.
        keys: label names to report (e.g. ``["total", "B3d", "B3d_far"]``).

    Returns:
        A dict with only the requested labels (missing labels are omitted).
    """
    return {k: results[k] for k in keys if k in results}
