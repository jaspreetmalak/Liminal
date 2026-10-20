# FineWeb data integration — 2026-10-18

Historical scalar `f64`, single-window training run. Liminal now uses one Burn
tensor implementation with batched `f32` WGPU training. These original metrics
describe the prior run, not the current training defaults; see the
[Burn transition record](2026-10-18-burn-gpu-training.md).

## Question and protocol

Can Liminal prepare a bounded slice of the original FineWeb in Rust and train
on it while keeping source provenance, whole-document validation separation,
and causal document boundaries intact?

Acceptance checks, defined before judging model results:

- Fetch at most 128 source rows from `HuggingFaceFW/fineweb`, `sample-10BT`,
  upstream split `train`, offset 0; retain at most 200,000 Unicode scalars in
  complete documents. No full shards or Python tooling.
- Observe one consistent `X-Revision`, reject partial pages/truncated records,
  and record per-page/document/corpus SHA-256 and original provenance.
- Deduplicate exact IDs/texts before a seed-42 document shuffle; hold out
  `ceil(N/10)` accepted documents, with both splits nonempty.
- Load independent JSONL documents. Fit the vocabulary and unigram control
  only on training documents; reset state and avoid cross-document targets.
- Rust tests, formatting, and Clippy pass. Run the fixed 500-update baseline and
  report finite initial/final train/validation metrics and the unigram control,
  without adjusting the budget to select a favorable result.

The model remains one LIF layer, 48 neurons, leak 0.9, threshold/width 1, `f64`
CPU arithmetic, max sequence length 32, Adam learning rate 0.003, betas
0.9/0.999, epsilon 1e-8, global gradient clip 1, and initialization/window seed
42. Its forward and backward equations are unchanged. Document selection
changes the sampling population and evaluation boundaries.

This bounded prefix is a pipeline experiment, not representative FineWeb quality
evaluation. Exact-text deduplication does not exclude near duplicates or shared
web domains. The earlier toy-corpus metrics concern a different population.

## Commands

From the Liminal project root:

```text
cargo run --release -- prepare-fineweb --rows 128 --max-chars 200000 --seed 42
cargo run --release -- train
cargo run --release -- evaluate
cargo run --release -- generate --prompt "The " --tokens 160 --temperature 0.8 --seed 42
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

Default paths are `data/fineweb/train.jsonl`, `data/fineweb/validation.jsonl`,
`data/fineweb/manifest.json`, and the current model `artifacts/liminal.bin`.
Preparation refuses to replace a nonempty corpus directory. The dataset viewer
cannot request a historical revision; retain local files and hashes for replay.

## Execution evidence

The first live Rust import fetched page 0 successfully but received HTTP 502 at
offset 100. No corpus was published. Bounded transient-request retries were added
before retrying the same selection protocol.

The retry of the same 128-row request completed successfully. Both pages reported
source revision `9bb295ddab0e05d785b879661af7260fed5140fc`.

| Selection measurement | Observed |
| --- | ---: |
| Source rows inspected | 128 |
| Complete documents retained | 78 |
| Training documents / characters | 70 / 185,628 |
| Validation documents / characters | 8 / 14,348 |
| Total retained characters | 199,976 |
| Duplicates / malformed or truncated rows | 0 / 0 |
| Complete rows skipped for the character budget | 50 |
| Training vocabulary entries, including unknown | 148 |
| Unknown validation characters | 4 / 14,348 |
| Trainable model parameters | 16,708 |

No text was cut to fill the character budget. The two local splits have
185,558 and 14,340 within-document prediction targets respectively. The training
budget was 500 randomly sampled windows of up to 32 targets; it was not an epoch
over the prepared corpus.

| Evaluation set/control | Cross-entropy (nats/character) | Character perplexity |
| --- | ---: | ---: |
| Training, untrained | 5.006986 | 149.453611 |
| Training, after 500 updates | 3.158109 | 23.526071 |
| Validation, untrained | 5.010968 | 150.049900 |
| Validation, train-frequency unigram | 3.213943 | 24.876995 |
| Validation, after 500 updates | 3.053037 | 21.179566 |
| Validation, reloaded checkpoint | 3.053037 | 21.179566 |

Validation improved over the untrained model and was 0.160906 nats/character
below the smoothed frequency control. Validation loss being below training loss
on this run reflects the different small document populations and is not a
generalization guarantee. Unknown characters are scored as unknown tokens.

Environment: `rustc 1.98.1`, `cargo 1.98.1`, Windows `10.0.26300`, Ryzen 7
7800X3D, single-threaded model computation. One optimized run took about 6.070
seconds including metrics and checkpoint I/O; no performance benchmark is claimed.
The HTTP, JSON, and hash crates are reqwest 0.13.5, serde_json 1.0.154, and sha2
0.11.1. The lockfile records transitive dependencies.

SHA-256 of exact local files:

| File | SHA-256 |
| --- | --- |
| `data/fineweb/train.jsonl` | `ad0b069c93c70dc6fc52000edefbd272a574ae77dd7706ab763eff702ee2633a` |
| `data/fineweb/validation.jsonl` | `3ca429b3909a50a591286cb27f9b650527e2bd2846ffb22e5c460b20ad352866` |
| `artifacts/liminal.bin` | `329f2d0e8cb2891d2c58d6308090a7d0517189dea26e09cef0f76e366934f4c2` |

The manifest stores response hashes, source row indices and provenance, exact
document hashes and split membership. Captured outputs are
`artifacts/fineweb-prepare.log`, `fineweb-train.log`, `fineweb-evaluate.log`, and
`fineweb-generation.txt` under the same artifacts directory. The current model
checkpoint replaces the previous smoke checkpoint.

The seeded generated sample remains mostly fragmented text. No fluency claim
follows from beating a unigram control on eight held-out web pages.

## Verification and decision

All 37 Rust tests passed (33 library, 4 integration). Tests cover whole-document
retention/budgets, source revision changes, malformed/truncated pages, exact
deduplication, deterministic disjoint splits, output nonoverwrite, response
limits, vocabulary isolation, multi-document target/state boundaries, weighted
evaluation, checkpoint equivalence, and the existing math checks. Clippy passed
with warnings denied; formatting passed after applying Rustfmt.

FineWeb is now the default corpus and is recorded in the project rules. Keep this
bounded preparation for reproducible development. Scaling requires a larger and
more representative selection, additional overlap controls, and a streaming Rust
Parquet pipeline; the public rows API and in-memory corpus loader are not a
full-dataset training system.

Source and preparation details: [FineWeb data documentation](../data.md),
[official dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb), and
[official rows API](https://huggingface.co/docs/dataset-viewer/en/rows).
