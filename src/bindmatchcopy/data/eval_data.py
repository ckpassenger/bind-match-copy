"""Fixed evaluation-set generator (with pre-computed bucket labels).

The bucket evaluator requires a *fixed* eval set: a set of tokenised sequences
whose token positions have been classified once into bucket labels (see
:class:`~bindmatchcopy.evaluation.bucket.BucketClassifier`). Pre-computing the
labels offline keeps evaluation deterministic and fast; the evaluator then only
has to run the model and accumulate per-bucket NLL.

Eval set content
----------------

The eval set spans five domains, each read from JSONL shards under ``data_root``:

- ``wiki``       — Wikipedia text (natural language long-range references).
- ``python``     — Python source code (identifier reuse, scope shadowing, ...),
                   from ``bigcode/the-stack-dedup`` (``data/python``).
- ``math``       — mathematical text (variable binding, symbol ambiguity, ...),
                   from ``open-web-math/open-web-math``.
- ``ood_python`` — out-of-distribution Python (held-out style, ~100M tokens).
- ``ood_math``   — out-of-distribution math (~100M tokens).

Legacy compatibility loader: directory names alone do not establish disjoint
training/evaluation data. Use data.eval_manifest for a declared split and hash
registry. Historical packing is retained here only for old server workflows;
it must not be represented as the corrected v31 evaluation protocol.

Each shard file is JSONL with one document per line and a ``"text"`` field.
Sequences are tokenised with the GPT-2 BPE tokenizer, packed to
``max_seq_len`` tokens with a half-window overlap between consecutive chunks,
and read across *all* shards in round-robin order (so no single shard dominates
the tail of the set). Each packed sequence is then labelled position-wise by
the bucket classifier.

Output
------

``generate_fixed_evalset`` writes (to ``output_dir``):

- ``eval_seqs_with_labels.pt`` — ``{domain: [(token_ids, labels), ...]}``.
- ``eval_metadata.json`` — per-domain sequence/token counts plus config.

The ``.pt`` file is the input consumed by
:func:`bindmatchcopy.evaluation.evaluator.eval_bucket_ppl_fixed`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..evaluation.bucket import BucketClassifier

# Canonical domain -> shard-directory / shard-pattern mapping.
# ``wiki`` reads the tokenised WikiText shards (``wikitext_shards/eval_shard_*.jsonl``),
# produced by the wiki eval-shard generator from the raw WikiText-103 download
# (``download_wikitext103`` writes raw ``wikitext103/``). The code/math domains
# are produced by ``download_code_python`` / ``download_math``.
DOMAIN_DIRS = {
    "wiki": "wikitext_shards",
    "python": "train_python",
    "math": "train_math",
    "ood_python": "ood_python",
    "ood_math": "ood_math",
}

DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_MAX_SEQS = 200
DEFAULT_DOMAINS = ["wiki", "python", "math", "ood_python", "ood_math"]


def _load_tokenizer(tokenizer_path: str | None):
    """Load the GPT-2 BPE tokenizer from a local dir or a HuggingFace id."""
    from transformers import GPT2TokenizerFast

    path = tokenizer_path or "gpt2"
    tok = GPT2TokenizerFast.from_pretrained(path)
    tok.pad_token = tok.eos_token
    return tok


def _read_jsonl_text(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get("text") if isinstance(rec, dict) else None
            if t:
                yield t


def _token_stream_from_shard(spath: str, tok, max_seq_len: int):
    """Pack a shard's documents into overlapping fixed-length token chunks."""
    buf: List[int] = []
    eos = tok.eos_token_id
    for text in _read_jsonl_text(spath):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 5:
            continue
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= max_seq_len:
            yield buf[:max_seq_len]
            buf = buf[max_seq_len // 2 :]


def generate_fixed_evalset(
    data_root: str,
    domains: List[str] | None = None,
    max_seqs: int = DEFAULT_MAX_SEQS,
    output_dir: str | None = None,
    tokenizer_path: str | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    window_size: int = 512,
) -> Dict[str, List[Tuple[List[int], List[List[str]]]]]:
    """Generate a fixed, pre-labelled tokenised eval set.

    Args:
        data_root: directory containing the domain shard directories.
        domains: domains to include (default: the canonical 5-domain split).
        max_seqs: maximum sequences per domain.
        output_dir: directory to write the ``.pt`` and metadata files. If
            ``None``, the set is returned without writing to disk.
        tokenizer_path: GPT-2 tokenizer path (local dir or HF id; default "gpt2").
        max_seq_len: packed sequence length.
        window_size: bucket-classifier sliding-window size.

    Returns:
        ``{domain: [(token_ids, labels), ...]}`` — the same structure written
        to ``eval_seqs_with_labels.pt``.
    """
    import warnings

    warnings.warn(
        "Legacy shard loader does not verify data splits; use prepare-eval",
        DeprecationWarning,
    )
    domains = domains or DEFAULT_DOMAINS
    tok = _load_tokenizer(tokenizer_path)
    bc = BucketClassifier(window_size=window_size, tokenizer=tok)

    all_sequences: Dict[str, List[Tuple[List[int], List[List[str]]]]] = {}

    for domain in domains:
        ddir = os.path.join(data_root, DOMAIN_DIRS[domain])
        pattern = "eval_shard_*.jsonl" if domain == "wiki" else "shard_*.jsonl"
        shard_paths = sorted(str(s) for s in Path(ddir).glob(pattern))
        if not shard_paths:
            raise FileNotFoundError(f"No shards for {domain}: {ddir}")

        streams = [_token_stream_from_shard(sp, tok, max_seq_len) for sp in shard_paths]
        active = {i: True for i in range(len(streams))}

        seqs_with_labels: List[Tuple[List[int], List[List[str]]]] = []
        round_idx = 0
        t0 = time.time()
        while len(seqs_with_labels) < max_seqs and any(active.values()):
            i = round_idx % len(streams)
            if active[i]:
                try:
                    chunk = next(streams[i])
                    labels = bc.classify_sequence(chunk, domain)
                    seqs_with_labels.append((chunk, labels))
                except StopIteration:
                    active[i] = False
            round_idx += 1
            if len(seqs_with_labels) % 500 == 0:
                print(
                    f"    {len(seqs_with_labels)}/{max_seqs} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

        all_sequences[domain] = seqs_with_labels[:max_seqs]
        total_tokens = sum(len(s[0]) for s in all_sequences[domain])
        print(
            f"  [{domain}] {len(all_sequences[domain])} seqs, {total_tokens:,} tokens "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        seqs_path = os.path.join(output_dir, "eval_seqs_with_labels.pt")
        torch.save(all_sequences, seqs_path)

        meta = {
            "max_seqs": max_seqs,
            "domains": domains,
            "max_seq_len": max_seq_len,
            "window_size": window_size,
            "method": "all shards, interleaved round-robin, pre-computed labels",
        }
        for domain in domains:
            meta[f"{domain}_n_seqs"] = len(all_sequences.get(domain, []))
            meta[f"{domain}_n_tokens"] = sum(
                len(s[0]) for s in all_sequences.get(domain, [])
            )
        meta_path = os.path.join(output_dir, "eval_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return all_sequences


def load_fixed_evalset(path: str) -> Dict[str, List[Tuple[List[int], List[List[str]]]]]:
    """Load a previously generated fixed eval set from disk."""
    return torch.load(path, map_location="cpu", weights_only=True)
