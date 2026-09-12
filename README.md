# BindMatchCopy

Code for **The Token Before the Value Is the Key: How Hybrid Architectures Organize Induction Circuits**.

The project implements controlled Transformer, Gated DeltaNet and sliding-window hybrid models; Bind/Match/Copy probes; source-path interventions; formation-time training controls; and recall-conditioned evaluation.

[中文说明](README_ZH.md) · [Measurement protocols](docs/PROTOCOLS.md) · [Data formats](docs/DATA_AND_RESULTS.md)

## Installation

Python 3.10 or later:

```bash
pip install -e '.[cpu,analysis]'
```

Optional dependencies: `.[gdn]` for Gated DeltaNet on CUDA, `.[qwen]` for Qwen, and `.[dev]` for the short smoke test. The FLA distribution is `flash-linear-attention`; its Python import is `fla`.

## Layout

```text
configs/                  Model, training and evaluation settings
src/bindmatchcopy/
  models/                 Native mixers and checkpoint loading
  data/                   Paired prompts, IH filter and token streams
  interventions/          Update/attention patches and training controls
  probes/                 Bind, Match, Copy, paths and Qwen components
  training/               Resumable single-device training
  evaluation/             Recall buckets and perplexity
  analysis/               Aggregation and plotting
  artifacts/              Result serialization
  paper_data/v31/         Archived figure-data tables
examples/                 CLI examples
scripts/smoke.py           Small CPU smoke check
```

## Evaluation protocol

**Every formal condition uses 200 samples.** Mechanistic probes use 200 paired prompts per model/run/checkpoint/condition, reusing the same prompt set across layers and interventions. Recall evaluation uses 200 sequences in total per condition. Across multiple domains, samples are balanced in the declared domain order and the exact allocation is recorded.

`EVAL_SAMPLES = 200` in `src/bindmatchcopy/protocol.py` is the shared default. The CLI rejects another count for formal evaluations; smaller runs require the explicit `--demo` flag. A clean-positive subset is selected from its 200-prompt parent set and can be smaller than 200.

## Checkpoint probes

```bash
bmc probe --checkpoint checkpoints/model.pt --mode bind --layers 0 1 2 3 4 5 6 7 --output runs/bind
bmc probe --checkpoint checkpoints/model.pt --mode match --layers 3 7 --output runs/match
bmc probe --checkpoint checkpoints/model.pt --mode copy --layers 3 7 --output runs/copy
bmc probe --checkpoint checkpoints/model.pt --mode selection --layers 3 7 --output runs/selection
bmc probe --checkpoint checkpoints/model.pt --mode path --sender 2 --receiver 3 --output runs/path
bmc probe --checkpoint checkpoints/model.pt --mode factorial --sender 2 --receiver 3 --output runs/factorial
```

Each run records exact paired prompts, signed effects, sample counts, checkpoint identity and summaries. Checkpoints carry model configuration and active local-head masks. Legacy weights require an explicit matching `--config`; parameters load strictly.

## Training

```bash
bmc train --config configs/experiments/gdn_early_lr.yaml \
  --tokens datasets/natural.npy --ih-tokens datasets/ih.npy \
  --device cuda:0 --output runs/gdn_early_lr
```

Inputs are a one-dimensional integer token stream or two-dimensional packed `.npy` rows. Resume with the same settings and input files using `--resume CHECKPOINT`. Checkpoints include optimizer, RNG, sampling and mask state. The included trainer is a single-device implementation; trained paper weights and the original distributed data pipeline are not bundled.

A small CPU demonstration:

```bash
bmc train --config configs/experiments/demo.yaml --demo --output runs/demo
bmc probe --checkpoint runs/demo/checkpoint_final.pt --demo --mode bind \
  --n-samples 3 --prompt-config configs/probes/demo.json --output runs/demo_bind
python scripts/smoke.py
```

## Recall and released models

```bash
bmc prepare-eval --manifest datasets/eval_input.json --tokenizer datasets/gpt2_pinned --output datasets/fixed_eval
bmc evaluate --checkpoint checkpoints/model.pt --evalset datasets/fixed_eval/evalset.pt --output runs/recall.json
bmc qwen-components --model checkpoints/qwen35 --revision PINNED_REVISION --mode components --output runs/qwen_components.json
```

Preparation and evaluation both use a total of 200 sequences. See the [data contract](docs/DATA_AND_RESULTS.md) for document identity, domain allocation and tokenizer requirements. The version-specific Qwen adapter checks receiver identity reconstruction before measurement; GPU/checkpoint compatibility still depends on the supplied environment.

## Paper data and parameter counts

```bash
bmc reproduce --output runs/paper_plots
bmc count-params --config configs/models/transformer_8l.yaml
bmc count-params --config configs/models/swa4_hybrid_8l.yaml
```

The supplied eight-layer Transformer and SWA hybrid implementations each contain **77,258,833 parameters (77.26M)**. The accompanying manuscript revision uses these code-derived counts. Other architecture counts should be obtained from their actual configurations.

The compact data snapshot contains previously exported measurements for plotting; this release does not rerun those historical experiments. Original exported values and actual recorded counts are preserved. Fresh evaluation uses the uniform 200-sample protocol and writes new results separately. `bmc reproduce` regenerates review plots and the recall table from those saved values; it does not run model inference or recreate every appendix panel.

## License

Apache-2.0, following the license declaration in the supplied code. Model weights and datasets retain their respective terms and are not redistributed here.
