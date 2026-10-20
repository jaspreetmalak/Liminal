# Burn batched GPU transition — 2026-10-18

## Objective and checks fixed before measurement

Replace the scalar implementation in place with one Burn model and training
implementation. Use the available AMD Radeon RX 9060 XT through WGPU; execute
the same Burn model on Flex CPU for numerical checks. Keep the LIF recurrence,
binary threshold forward, triangular surrogate derivative, attached subtractive
reset, causal document boundaries, global norm clip, and Adam settings.

Acceptance checks:

- Remove production scalar forward/backward/optimizer code; keep only the Burn
  model across devices. Small independent scalar math oracles belong in tests.
- Verify hard binary spikes at threshold, surrogate center/support endpoints,
  leak and reset, both temporal paths, causality, and all trainable gradients
  against a frozen-operating-point finite-difference oracle.
- Verify masked minibatch losses and all parameter gradients equal appropriately
  target-weighted independent windows, including shorter padded rows.
- Verify a single global gradient norm across parameter groups before Burn Adam;
  explicitly set betas 0.9/0.999, epsilon 1e-8, learning rate 0.003, clip 1.
- Run CPU tests and an explicit actual-GPU check comparing binary forward,
  masked loss, and every parameter gradient. Initial CPU/GPU comparison tolerances:
  logits/loss absolute 2e-5; all gradients absolute 3e-5; surrogate slope 2e-6.
- Run batched GPU training on the existing local FineWeb selection, save/load the
  resulting current checkpoint, and execute held-out evaluation and generation.
  Record actual budget/metrics; compilation alone is not GPU execution evidence.
- Pass Rust formatting, tests, and Clippy with warnings denied.

## Mathematical preservation

The forward equations remain `v=beta*u_prev+E[x]+R*s_prev+b`,
`s=1[v>=theta]`, `u=v-theta*s`, and `logits=O*s+c`.

For a stable primitive surrogate, set `z=clamp((v-theta)/width,-1,1)` and
`F=0.5+z-0.5*z*abs(z)`. Compute spikes as `hard+(F-F.detach())`, subtracting
equal smooth values before adding the hard result. The forward stays exactly
binary while autodiff sees `ds/dv=max(0,1-|v-theta|/width)/width`, including
the peak and support endpoints. The reset and recurrent state remain attached.

Minibatch loss is the sum of valid next-character negative log likelihoods
divided by valid target count; right-padding is masked. Each row begins from
zero state and stays inside one document. This extends the objective to batches;
different row lengths are weighted by actual target counts.

GPU tensors use `f32`, a deliberate precision change from the historical scalar
`f64` implementation. Validation must distinguish precision, batching, and training
budget changes from an SNN equation change. No speedup or quality claim is assumed.

## Data and planned run

Reuse the current FineWeb selection: 70 training documents / 185,628 characters,
8 validation documents / 14,348 characters, vocabulary fitted only on training.
Source revision is `9bb295ddab0e05d785b879661af7260fed5140fc`.
The manifest in `data/fineweb/manifest.json` records exact file/document hashes.
The fixed development run uses hidden size 48, 500 updates, batch size 16,
sequence length 32, seed 42, and the Adam settings above; report the actual
non-padding target count rather than assuming every row is full length.

```text
cargo test
cargo test --test gpu -- --ignored --nocapture
cargo run --release -- train --device gpu --batch-size 16 --steps 500 --sequence-length 32 --hidden 48 --seed 42
cargo run --release -- evaluate --device gpu --batch-size 16
cargo run --release -- generate --device gpu --prompt "The " --tokens 160 --seed 42
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

The current model checkpoint remains `artifacts/liminal.bin`. Its existing f64
wire envelope stores the Burn model's f32 values exactly; no old training
implementation or separate model version is retained. Optimizer state is not
saved for exact training resume.

## Execution evidence

Burn 0.22.0 compiled with `std`, `optim`, `wgpu`, `flex`, and `fusion`, with its
default features disabled. No LibTorch backend or Python bridge is selected.
The hardware test executed on the discrete **AMD Radeon RX 9060 XT**, reported
by Burn as fingerprint `spirv_4098_30096`; the installed driver is
`32.0.31041.1004`. The host is Windows `10.0.26300` with Ryzen 7 7800X3D,
`rustc 1.98.1`, and `cargo 1.98.1`.

Executed numerical results:

- All 27 library tests and all 4 integration tests passed on Burn Flex CPU.
- The separately invoked GPU test passed on the Radeon. Binary forward spikes
  matched exactly across devices, including threshold/support checks.
- Masked CPU/GPU fixture loss was **1.12073290 / 1.12073278**. Maximum absolute
  difference across all parameter gradients was **1.4901161e-8**, below the
  predeclared 3e-5 tolerance.
- The independent frozen-tangent oracle checked all 21 parameters of a two-neuron
  fixture across two unequal lanes, with eight valid targets, nonzero initial
  states, active and inactive surrogate slopes. Its central difference step was
  1e-5; absolute gradient tolerance 3e-6 and forward-loss tolerance 3e-7.
- The batched alternating-character test learned the exact greedy continuation
  with reproducible CPU parameters; its 160 updates of batch 4 × 8 targets
  processed 5,120 targets.
- Clippy passed with warnings denied. Rustfmt was applied.

## Optimized FineWeb GPU run

The optimized run completed all **500 updates**, batch size **16**, sequence
length **32**, processing **256,000 non-padding targets**. Burn reported the
Radeon RX 9060 XT throughout. The source corpus remained unchanged; the file
hashes still match the FineWeb preparation manifest:

- Training: `ad0b069c93c70dc6fc52000edefbd272a574ae77dd7706ab763eff702ee2633a`.
- Validation: `3ca429b3909a50a591286cb27f9b650527e2bd2846ffb22e5c460b20ad352866`.

The vocabulary has 148 entries, the model has 16,708 parameters, and validation
has 4 unknown characters out of 14,348. The initial and final metrics use the
same zero-state, document-contained 32-target windows, batched by 16.

| Evaluation set/control | Cross-entropy (nats/character) | Character perplexity |
| --- | ---: | ---: |
| Training, initial | 5.006986 | 149.453587 |
| Training, after 500 batched GPU updates | 2.791621 | 16.307428 |
| Validation, initial | 5.010968 | 150.049882 |
| Validation, train-frequency unigram | 3.213943 | 24.876995 |
| Validation, trained model | 2.747231 | 15.599385 |
| Validation, reloaded checkpoint on GPU | 2.747231 | 15.599385 |
| Validation, same checkpoint on Flex CPU | 2.747232 | 15.599386 |

Reloaded GPU scores match the reported training-run evaluation. The CPU result
differs at the displayed last decimal, consistent with float32 execution order.
The GPU training log records finite losses and global gradient norms, including
a norm above the clip bound at step 400, which was scaled before Adam.

The current checkpoint SHA-256 is
`4eb56bc9fff8709931313d12e49c1e59d784eccdc1487ff1a21e52a6023f6fca`.
Artifacts are `artifacts/burn-gpu-train.log`, `burn-gpu-evaluate.log`,
`burn-cpu-evaluate.log`, and `burn-gpu-generation.txt` under `artifacts/`, plus
the single current `artifacts/liminal.bin` checkpoint.

Autoregressive generation from the loaded GPU checkpoint completed for prompt
`The `, seed 42, temperature 0.8, and 160 new characters. The output is still
fragmented text; the migration establishes correct batched execution and learning,
not fluent natural language generation.

## Decision and limits

The transition is complete: one Burn model and one Burn training path now replace
the scalar production forward/backward/optimizer. The same tensors execute on
WGPU or Flex. There are **31 passing CPU tests plus the separately passing GPU
test**; formatting and Clippy pass. Test-only scalar tangent oracles remain to
check the mathematics independently.

This run uses 16 times the target budget of the previous 500-update, batch-one
record, along with float32 arithmetic. Its loss is therefore not an isolated
framework or GPU quality comparison. The first optimized build took 6m02s;
training and evaluation time were not measured separately, so no speedup is
claimed. The bounded FineWeb slice, character tokenizer, one recurrent layer,
short windows, and absence of optimizer-resume state remain development limits.
