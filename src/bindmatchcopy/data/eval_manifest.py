"""Build fixed evaluations from explicitly split, tokenized document records.

Manifest paths are relative to the manifest. Each JSONL record contains id,
source_document_sha256, token_ids. A training hash registry is required; this
verifies declared document identity, not semantic or near-duplicate overlap.
"""

import json
from pathlib import Path
from ..protocol import EVAL_SAMPLES, condition_counts
from ..evaluation.bucket import BucketClassifier
from ..artifacts.io import sha256, write_json


def prepare_eval(
    manifest_path, output, max_seqs=EVAL_SAMPLES, max_seq_len=1024, tokenizer=None
):
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if max_seqs < 1 or max_seq_len < 2:
        raise ValueError("Invalid evaluation counts")
    train_path = manifest_path.parent / m["training_document_hashes"]
    train = set(train_path.read_text().splitlines())
    if not train:
        raise ValueError("Training document hash registry must not be empty")
    seen = set()
    seen_ids = set()
    evalset = {}
    sources = []
    sequences = []
    counts = condition_counts(list(m["domains"]), max_seqs)
    bc = BucketClassifier(tokenizer=tokenizer)
    for domain, relative in m["domains"].items():
        path = manifest_path.parent / relative
        sources.append(dict(domain=domain, file=relative, sha256=sha256(path)))
        seqs = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            digest = rec["source_document_sha256"]
            identity = str(rec["id"])
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid source_document_sha256")
            if digest in train or digest in seen or identity in seen_ids:
                raise ValueError(f"Duplicate/overlapping document: {identity}")
            seen.add(digest)
            seen_ids.add(identity)
            ids = rec["token_ids"]
            if (
                not isinstance(ids, list)
                or len(ids) < 2
                or any(type(t) != int or t < 0 for t in ids)
            ):
                raise ValueError("Invalid token_ids")
            # Do not add a cross-document bigram or overlap chunks.
            for start in range(0, len(ids) - 1, max_seq_len):
                chunk = ids[start : start + max_seq_len]
                if len(chunk) < 2 or len(seqs) >= counts[domain]:
                    continue
                seqs.append((chunk, bc.classify_sequence(chunk, domain)))
                sequences.append(
                    dict(
                        domain=domain,
                        index=len(seqs) - 1,
                        document_id=identity,
                        source_document_sha256=digest,
                        start=start,
                        length=len(chunk),
                    )
                )
        if len(seqs) < counts[domain]:
            raise ValueError(
                f"{domain}: requested {counts[domain]} sequences, available {len(seqs)}"
            )
        evalset[domain] = seqs
    if not evalset:
        raise ValueError("No evaluation domains")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(evalset, output / "evalset.pt")
    write_json(
        output / "manifest.json",
        dict(
            protocol_id="recall_buckets_v31_1",
            packing="within_document_nonoverlapping",
            tokenizer_id=m["tokenizer_id"],
            tokenizer_revision=m["tokenizer_revision"],
            domain_heuristics=tokenizer is not None,
            requested_sequences_per_condition=max_seqs,
            allocation="balanced across listed domains",
            actual_sequences={d: len(s) for d, s in evalset.items()},
            training_registry_sha256=sha256(train_path),
            input_manifest_sha256=sha256(manifest_path),
            evalset_sha256=sha256(output / "evalset.pt"),
            sources=sources,
            sequences=sequences,
        ),
    )
    return dict(evalset=str(output / "evalset.pt"), domains=list(evalset))
