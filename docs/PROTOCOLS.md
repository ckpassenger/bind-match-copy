# Measurement protocols

All layer/head/position indices are zero-based. The margin is always `logit(B) - logit(D)` with **the original B/D labels**, including historical-key-exchange donors. Every new run records its actual IDs; seeds alone do not identify an old prompt batch because generators and draw order can differ.

## Controlled prompts

Formal conditions use exactly **200 paired prompts**, reused across compared interventions. Default length 1020, query 1019, four distinct needles sampled from `[3000,3200)`, filler from `[500,3000)`, replacement from `[100,3000)`. Distance is from the second historical key to the query, sampled from `[300,700)` (exclusive upper bound). Clean structure is `AB … CD … A_query`. Bind donor changes only the first A. Match donor changes historical keys to `CB … AD … A_query`. All remaining tokens are identical.

Lag uses a separate single-source construction: it places A at `source-lag`, B at source, and A at query. It is not obtained by moving the first A in the two-record prompt. The tiny demo has its own explicit vocabulary/length configuration.

## Interventions and signs

| Readout | Replacement / observation | Reported quantity |
|---|---|---|
| Bind | At historical B, replace full block update `output-input` from predecessor-changed donor | `m_clean - m_patched` |
| Match | At query, replace full block update from historical-key-exchange donor | `m_clean - m_patched` |
| Self-control | Inject clean block update into clean recipient | Same signed difference; should approach zero |
| P1 | Query→historical B probability, physical query head | Probability; never rescaled into a margin |
| Copy | At historical B, replace V with clean distractor V; freeze clean P | Signed margin damage, whole layer and each physical head |
| Fixed-V selection | Replace query head output with `P_d V_c`, `P_c V_d`, or `P_d V_d` | Three separately labeled signed effects; native output projection/FFN remain |
| Head-output Match | Replace one query head's attention output from key-exchanged donor | Separate `match_head_output_margin`; not asserted equal to archived headwise assay |
| Edge | Zero source probability post-softmax and renormalize within head | Source and off-site suppression effects; distinct from Match |
| Path propagation | Sender Bind patch; transfer receiver source K, V or KV into clean run | Damage from the clean margin |
| Source restoration | Clean source K into sender-patched recipient | `m_restored - m_perturbed` |
| Off-site restoration | Add the identical clean-minus-perturbed K repair vector at another historical position | Restoration gain, with location and repair norm recorded |
| Factorial | Bind donor × source-edge closure × source-V replacement, all eight cells | Raw cell margins plus paired conditional effects |

Block updates include attention/recurrent mixer, FFN and normalization effects; they are not attention-only deltas. FP32 RMS normalization uses the **captured clean update** as reference with denominator epsilon `1e-8`, then casts back. Updates are patched only at the specified position in a one-prompt batch. Hooks are always removed after success or failure.

Controlled-model K/V interventions act at the **source slice after RoPE**. Qwen component restoration replaces **whole receiver tensors**, after Q/K normalization and RoPE for Q/K. These scopes are different and must not be pooled under an unspecified “K restoration” label. Qwen query/KV head spaces are also different; never align raw KV indices to query heads by numerical index alone.

The native attention path applies intervention callbacks before the normal output projection, residual additions and FFN. Identity reconstruction tests compare it with the unmodified path. Source-edge deletion is post-softmax, not pre-softmax logit subtraction. Factorial contrast statistics use paired prompt IDs, not independently resampled condition means.

## Training controls

Early LR reduction multiplies scheduled LR for **every parameter of explicitly selected blocks** by 0.1, including FFN and normalization, while preserving optimizer moments. Principal GDN window: updates 1–3000 inclusive; SWA: 1–4000 inclusive. At the next update, the common schedule resumes. Base schedule: linear warmup ending at 1000, cosine to `3e-5` on update 30000.

SWA windows include self. Window 2 allows self and lag 1. A local head with lag 2 allows **self and exactly lag 2**, excluding lag 1. Persistent local-head masks must be stored with the checkpoint and replayed for evaluation. The v31 Transformer local-head series trains on natural text; H0 at L0–L5 is constrained in the persistent arm. Transient arms release after update 3000; all-layer early uses L0–L7.

Historical gate and short-conv ablation controller APIs are retained, but the new trainer rejects them pending GPU parity/optimizer masking validation. They are not silently substituted for LR reduction or local-head masking.

## Formation and uncertainty

Choose the endpoint layer once per independent training run and readout, then follow that layer through time. A changing layerwise maximum is a different statistic. SWA windows: 1K–10K each 1K, then 15K/20K/30K. Architecture×text: 0.5K,1K,1.5K,2K,2.5K,3K,4K,5K,8K,10K,15K,20K,30K.

Prompt-level effects within a checkpoint and independent training-run variation are distinct. Paper line bands and run summaries use sample SD (`ddof=1`) across runs. Missing runs yield unavailable statistics, never invented zero or an implied three-run result. Paired bootstrap uses within-prompt contrasts and retains its seed/sample count. The released Qwen clean-positive summaries use 2,000 resamples; all-prompt dose summaries use 10,000.

## Qwen

Restricted token IDs define an association condition; ID rank alone does not prove rarity or novelty. Record exact tokenizer revision, needles and filler pool. Component defaults are Qwen3.5 sender 18, receiver 19; token cutoff 180000. The clean-positive subset is selected once by `m_clean > 0` and held fixed across interventions/doses. There is no required 50% success gate. Doses 0,.25,.5,.75,1 interpolate clean/donor updates **before** clean-RMS normalization; record the resulting perturbation L2 norm.

Before measurement, the CLI verifies receiver identity reconstruction on a controlled prompt with recorded tolerances. This check does not replace full GDN/kernel/cache/long-sequence parity against the original server implementation.
