# Data and result formats

## Training inputs

`--tokens` and `--ih-tokens` are local integer `.npy` files. A 1D stream is sampled into fixed-length windows; a 2D file must already have `seq_len` columns. IDs are validated against the model vocabulary before each update. The sampler stores its RNG state and input-file SHA256. Effective batch size is `batch_size × grad_accum`; logged loss is the average microbatch loss.

The single-device sampler's packing/order is **not claimed identical** to the historical multi-GPU data loader. Server acceptance must record tokenizer files/revision, corpus version, split identity, document deduplication, packing/EOS policy and original sampling order.

IH scoring weights repeated bigram **types** equally. Reliability uses repeated bigram types with at least two successor observations and a **strict majority** successor. `0.5 F + 0.5 R2 > 0.10`; empty denominators are zero. `filter_text` splits blank-line paragraphs, rejects stripped text shorter than 100 characters, scores the first 384 tokens only if at least 30 tokens exist, and retains the complete accepted paragraph. The tokenizer is supplied by the caller. Natural-text arms use IH probability zero; enriched arms use .70 through update 5000 and .50 thereafter. A random-retention control is a separate dataset and must keep its own identity.

## Fixed evaluation set

Example `eval_input.json` (paths relative to that file):

```json
{
  "tokenizer_id": "gpt2",
  "tokenizer_revision": "REPLACE_WITH_PINNED_REVISION",
  "training_document_hashes": "training_document_sha256.txt",
  "domains": {
    "wiki": "eval_wiki.jsonl",
    "python": "eval_python_primary.jsonl",
    "math": "eval_math_primary.jsonl",
    "ood_python": "eval_python_secondary.jsonl",
    "ood_math": "eval_math_secondary.jsonl"
  }
}
```

Each JSONL document has `id`, `source_document_sha256` (SHA256 of the original normalized source text), and integer `token_ids`. The nonempty training registry contains one SHA256 per line. Duplicate document IDs/hashes within the evaluation input or hashes overlapping training are rejected. These checks verify declared identity; they do not establish semantic deduplication or validate that the source hash was computed honestly. “Secondary” partitions are not automatically a newly sourced OOD corpus.

```bash
bmc prepare-eval --manifest datasets/eval_input.json --tokenizer datasets/gpt2_pinned --n-samples 200 --output datasets/fixed_eval
bmc evaluate --checkpoint checkpoints/run42_step30000.pt --evalset datasets/fixed_eval/evalset.pt --n-samples 200 --output runs/recall_run42.json
```

Preparation builds the requested count **in total per condition**, balanced over domains, with exact source document/chunk records; insufficient data is an error. Evaluation also selects **200 sequences in total per condition**, balanced over the listed domains in order, and records every selected sequence index. For five domains this is 40 each; for three it is 67/67/66. This is an explicit new allocation; replace it with the original sample allocation when server manifests are recovered. Do not claim historical equivalence merely because total count is 200.

The new packer uses nonoverlapping chunks within each document and does not create cross-document bigrams. The legacy generator remains available for migration and is explicitly labeled as not verifying splits. It is not used by `prepare-eval`.

Bucket labels refer to the **target token**: `logits[t-1]` predicts `tokens[t]`. B3 requires a past completed `(A,B)` pair. B3c/B3d count distinct historical continuations of A within the window. Recency uses the most recent A, even if that occurrence continued to a different token: `t - previous_A_position - 1`. Structural history uses a 512-token distance limit. Domain string heuristics are retained with a new protocol version and need a pinned tokenizer plus server comparison; failures are not silently discarded. PPL is `exp(sum_nll/n_targets)` within a run; then compute mean and sample SD across runs. Overlapping buckets do not multiply the `total` count.

## New probe artifacts

Each result directory has four files:

- `manifest.json`: schema/protocol, paper version, checkpoint SHA256, actual model configuration, training identity when known, environment, requested count, runtime masks and hashes of the other files.
- `prompts.jsonl`: one record per generated prompt, exact clean/donor IDs, seed, target/distractor, source/query/off-site coordinates and prompt hash.
- `rows.jsonl`: one measured coordinate per prompt, with metric, signed value, units, conditions, head space and status.
- `summary.json`: within-checkpoint prompt mean/sample SD; factorial results additionally contain paired contrast summaries.

The checkpoint SHA256 is the model identity; a path or informal alias alone is insufficient. Run means from different training seeds should be aggregated separately from prompt uncertainty. Classification helpers reject duplicate runs/cells. Raw values are saved before display rounding. JSON disallows NaN/Infinity; missing metrics use null with a status.

## Archived plotting data

The compact snapshot contains previously exported measurements, not a new evaluation. Original numerical values and recorded counts are preserved. New measurements use 200 samples per condition and write a separate result directory. The plotting command reaggregates exported per-run PPL; newly measured PPL is computed from full-precision NLL sums/counts.
