# Supplementary material index

[Read the experimental supplement](README.md) for the methods, results, and interpretation of all figures and tables.

## Figures

- [Query-association controls](query-controls.pdf)
- [Sender and source-edge dose interaction](sender-edge-dose.pdf)
- [Selective lag-mask trajectories](selective-trajectories.pdf)
- [Shifted global-layer layout](shifted-layout.pdf)
- [Transformer learning-rate interventions](transformer-lr.pdf)
- [318M allocation sweep: part 1](scale-allocation-1.pdf) and [part 2](scale-allocation-2.pdf)
- [GDN formation](formation-gdn.pdf) and [Transformer formation](formation-transformer.pdf)
- [SWA window 2](formation-swa2.pdf), [window 8](formation-swa8.pdf), and [window 16](formation-swa16.pdf)
- [Complete formation trajectories](formation-complete.pdf)

Each figure also has a PNG preview in the same directory. The experimental supplement links every numerical table to its CSV file.

## Measurement conventions

Archived experiments retain their original sample counts: 200 paired prompts for standard probes and factorial conditions, 60 prompts per cell for the GDN sender-by-edge dose sweep, 10,000 sequences for natural-text PPL, and 200 sequences for head ablation. Training-run summaries use sample SD; prompt-bootstrap intervals are within-checkpoint intervals. The code release uses a uniform 200-sample default for fresh evaluations.

Layers and heads are zero-indexed. Carrying corresponds to Bind and Matching to Match in legacy data and CLI names. Separate evaluation batches retain their own matched baselines.
