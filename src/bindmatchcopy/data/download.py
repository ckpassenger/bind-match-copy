"""Legacy fetch helpers: downloading two subsets does not establish disjointness.
Use explicit source-document identities and eval manifests for new evaluations.

Training and evaluation data downloaders.

This module downloads the raw text corpora consumed by the project's training
and evaluation pipelines. It is intended to be run once on a machine with
network access; the resulting directories are then pointed to by the training
and eval generators.

Data contents
-------------

**Training corpus** — ``openwebtext_1b/`` (``download_openwebtext``):

- Source: ``Skylion007/openwebtext`` on HuggingFace.
- A ~1B-token subset (by default) written as JSONL shards, one document per
  line with a ``"text"`` key, in chunks of ``shard_size`` documents.
- This is the *natural* (non-IH) stream of the co-training mixture. The IH
  stream is produced by scoring paragraphs of this corpus with
  :mod:`bindmatchcopy.data.ih_filter` (the ``freq_rel`` two-rule filter).
- Also writes ``meta.json`` recording source, document/token counts, and format.

**OOD / natural-language eval corpus** — ``wikitext103/`` (``download_wikitext103``):

- Source: ``wikitext`` / ``wikitext-103-raw-v1`` on HuggingFace.
- Written as ``train.txt`` / ``validation.txt`` / ``test.txt`` (one line per
  document). The ``validation`` split is the standard OOD perplexity target.

**Code / math eval domains** — ``train_python`` / ``train_math`` /
``ood_python`` / ``ood_math`` (``download_code_python`` / ``download_math``):

- Python code: ``bigcode/the-stack-dedup`` (``data/python`` subset; requires
  HuggingFace login + The Stack license acceptance). Fallback that needs no
  login: ``codeparrot/github-code-clean`` (``Python-all``).
- Math: ``open-web-math/open-web-math`` (no login required).

Both are written as JSONL shards (``shard_XXXX.jsonl``, one document per line
with a ``"text"`` key) so they share the same on-disk format as
:mod:`bindmatchcopy.data.eval_data` expects. The ``train_*`` directories are
the multi-domain pretraining subsets (``~500M`` tokens); the ``ood_*``
directories are the held-out ~100M-token subsets. The first ~100M tokens of
``train_*`` overlap ``ood_*`` by streaming determinism, so the eval set
generator skips ``shard_0000`` of ``ood_*`` to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def download_openwebtext(
    output_dir: str, target_tokens: int = 1_000_000_000, shard_size: int = 50_000
) -> str:
    """Download a ~1B-token subset of OpenWebText as JSONL shards.

    Returns the path to the created ``openwebtext_1b`` directory.
    """
    from datasets import load_dataset

    save_dir = os.path.join(output_dir, "openwebtext_1b")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(
        f"[1/2] Downloading OpenWebText subset (~{target_tokens / 1e9:.0f}B tokens)..."
    )
    print("      Source: Skylion007/openwebtext on HuggingFace")
    print("=" * 60)

    ds = load_dataset("Skylion007/openwebtext", split="train")
    print(f"      Full dataset: {len(ds)} documents")

    total_tokens = 0
    total_docs = 0
    shard_idx = 0
    shard_docs: list[str] = []

    t0 = time.time()
    for example in ds:
        text = example.get("text", "")
        if not text.strip():
            continue
        est_tokens = len(text) // 4
        total_tokens += est_tokens
        total_docs += 1
        shard_docs.append(text)

        if len(shard_docs) >= shard_size:
            _write_shard(save_dir, shard_idx, shard_docs)
            print(
                f"      shard {shard_idx}: {total_docs} docs, "
                f"~{total_tokens / 1e9:.2f}B tokens ({time.time() - t0:.0f}s)"
            )
            shard_docs = []
            shard_idx += 1

        if total_tokens >= target_tokens:
            break

    if shard_docs:
        _write_shard(save_dir, shard_idx, shard_docs)
        shard_idx += 1

    meta = {
        "source": "Skylion007/openwebtext",
        "total_docs": total_docs,
        "est_total_tokens": total_tokens,
        "n_shards": shard_idx,
        "shard_size": shard_size,
        "format": "jsonl, one doc per line, key='text'",
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size_gb = (
        sum(os.path.getsize(os.path.join(save_dir, fn)) for fn in os.listdir(save_dir))
        / 1e9
    )
    print(
        f"      Done: {total_docs} docs, ~{total_tokens / 1e9:.2f}B tokens, "
        f"{shard_idx} shards, {size_gb:.1f}GB ({time.time() - t0:.0f}s)"
    )
    return save_dir


def _write_shard(save_dir: str, shard_idx: int, docs: list[str]) -> None:
    path = os.path.join(save_dir, f"shard_{shard_idx:04d}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def download_wikitext103(output_dir: str) -> str:
    """Download WikiText-103 (train/validation/test) as plain-text files.

    Returns the path to the created ``wikitext103`` directory.
    """
    from datasets import load_dataset

    save_dir = os.path.join(output_dir, "wikitext103")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("[2/2] Downloading WikiText-103 (~0.5GB)...")
    print("=" * 60)

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        path = os.path.join(save_dir, f"{split}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for ex in ds[split]:
                text = ex.get("text", "")
                if text.strip():
                    f.write(text + "\n")
        print(f"      {split}: {len(ds[split])} lines")
    return save_dir


def download_code_python(
    output_dir: str,
    target_tokens: int = 100_000_000,
    shard_size: int = 50_000,
    source: str = "stack-dedup",
) -> str:
    """Download a ~100M-token Python code eval corpus as JSONL shards.

    Args:
        output_dir: parent directory; writes to ``output_dir/ood_python``.
        target_tokens: target token count (estimated as ``len(text) // 4``).
        shard_size: documents per JSONL shard.
        source: ``"stack-dedup"`` (``bigcode/the-stack-dedup`` ``data/python``,
            requires HF login) or ``"codeparrot"`` (``codeparrot/github-code-clean``
            ``Python-all``, no login).

    Returns the path to the created ``ood_python`` directory.
    """
    from datasets import load_dataset

    save_dir = os.path.join(output_dir, "ood_python")
    os.makedirs(save_dir, exist_ok=True)

    if source == "stack-dedup":
        print(
            "[code/python] Loading bigcode/the-stack-dedup (data/python) via streaming..."
        )
        print("             Requires HuggingFace login + The Stack license acceptance.")
        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
        )
        text_key = "content"
        source_name = "bigcode/the-stack-dedup"
    else:
        print(
            "[code/python] Loading codeparrot/github-code-clean (Python-all) via streaming..."
        )
        ds = load_dataset(
            "codeparrot/github-code-clean",
            "Python-all",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        text_key = "code"
        source_name = "codeparrot/github-code-clean"

    total_tokens, total_docs, shard_idx = 0, 0, 0
    shard_docs: list[str] = []
    t0 = time.time()
    for example in ds:
        text = example.get(text_key, "")
        if not text or not text.strip():
            continue
        text = text[:100_000]
        total_tokens += len(text) // 4
        total_docs += 1
        shard_docs.append(text)
        if len(shard_docs) >= shard_size:
            _write_shard(save_dir, shard_idx, shard_docs)
            shard_docs = []
            shard_idx += 1
        if total_tokens >= target_tokens:
            break
    if shard_docs:
        _write_shard(save_dir, shard_idx, shard_docs)
        shard_idx += 1

    meta = {
        "source": source_name,
        "subset": "python",
        "split": "train",
        "total_docs": total_docs,
        "est_total_tokens": total_tokens,
        "n_shards": shard_idx,
        "note": "OOD-severe (code), eval-only",
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"      Done: {total_docs} docs, ~{total_tokens / 1e6:.1f}M tokens, "
        f"{shard_idx} shards ({time.time() - t0:.0f}s)"
    )
    return save_dir


def download_math(
    output_dir: str,
    target_tokens: int = 100_000_000,
    shard_size: int = 50_000,
) -> str:
    """Download a ~100M-token math eval corpus as JSONL shards.

    Source: ``open-web-math/open-web-math`` (no login required).

    Returns the path to the created ``ood_math`` directory.
    """
    from datasets import load_dataset

    save_dir = os.path.join(output_dir, "ood_math")
    os.makedirs(save_dir, exist_ok=True)

    print("[math] Loading open-web-math/open-web-math via streaming...")
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

    total_tokens, total_docs, shard_idx = 0, 0, 0
    shard_docs: list[str] = []
    t0 = time.time()
    for example in ds:
        text = example.get("text", "")
        if not text or not text.strip():
            continue
        text = text[:100_000]
        total_tokens += len(text) // 4
        total_docs += 1
        shard_docs.append(text)
        if len(shard_docs) >= shard_size:
            _write_shard(save_dir, shard_idx, shard_docs)
            shard_docs = []
            shard_idx += 1
        if total_tokens >= target_tokens:
            break
    if shard_docs:
        _write_shard(save_dir, shard_idx, shard_docs)
        shard_idx += 1

    meta = {
        "source": "open-web-math/open-web-math",
        "split": "train",
        "total_docs": total_docs,
        "est_total_tokens": total_tokens,
        "n_shards": shard_idx,
        "note": "OOD-symbolic (math), eval-only",
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"      Done: {total_docs} docs, ~{total_tokens / 1e6:.1f}M tokens, "
        f"{shard_idx} shards ({time.time() - t0:.0f}s)"
    )
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download training (OpenWebText) + eval (WikiText-103) corpora"
    )
    parser.add_argument(
        "--output_dir", default="./data", help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--skip_openwebtext",
        action="store_true",
        help="Skip the (largest) OpenWebText download",
    )
    parser.add_argument(
        "--owt_tokens",
        type=int,
        default=1_000_000_000,
        help="Target token count for the OpenWebText subset",
    )
    parser.add_argument(
        "--skip_code", action="store_true", help="Skip the Python code eval download"
    )
    parser.add_argument(
        "--skip_math", action="store_true", help="Skip the math eval download"
    )
    parser.add_argument(
        "--code_source",
        choices=["stack-dedup", "codeparrot"],
        default="stack-dedup",
        help="Python code source (stack-dedup needs HF login)",
    )
    parser.add_argument(
        "--code_tokens",
        type=int,
        default=100_000_000,
        help="Target token count for the code eval subset",
    )
    parser.add_argument(
        "--math_tokens",
        type=int,
        default=100_000_000,
        help="Target token count for the math eval subset",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Data download -> {args.output_dir}\n")

    if not args.skip_openwebtext:
        download_openwebtext(args.output_dir, target_tokens=args.owt_tokens)
    else:
        print("[SKIP] OpenWebText\n")

    download_wikitext103(args.output_dir)

    if not args.skip_code:
        download_code_python(
            args.output_dir, target_tokens=args.code_tokens, source=args.code_source
        )
    else:
        print("[SKIP] Python code\n")

    if not args.skip_math:
        download_math(args.output_dir, target_tokens=args.math_tokens)
    else:
        print("[SKIP] math\n")


if __name__ == "__main__":
    main()
