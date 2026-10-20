# One-bit memory learning diagnostic

## Question and protocol before training

The paired correction experiment failed to learn its training set. This experiment
isolates basic delayed memory before making further architectural changes.
Can the continuous-memory path, spiking path, and full loop each learn one bit,
and how do activity and credit assignment change as the delay increases?

The model equations, weights, initialization, triangular surrogate, attached
reset, and checkpoint format are unchanged. New diagnostic path controls apply
zero multipliers within the same forward graph:

- `memory-only`: no feedback connections or spike readout; continuous memory
  receives the token embedding and its own recurrent state.
- `spikes-only`: no feedback connections or memory readout; output comes from
  spiking dynamics.
- `full`: normal bidirectional feedback and both output projections.

All paths retain the same 11 parameter groups. Disabled paths have present zero
gradients. Each condition starts a fresh Adam optimizer; optimizer state never
crosses path conditions. Production `step` and `loss` delegate to the full path.

## Data and curriculum

- Data seed 42, delays 1, 2, 4, 8, 16 distractor characters.
- Prompt: two irrelevant prefix characters from x/y/z, one payload bit 0 or 1,
  D distractors from x/y/z, then `?`. Predict the payload bit after the query.
  The payload appears exactly once in the input; the answer is a target only.
- Each history produces both bit values with opposite answers. Ignoring the
  payload cannot beat 50% on these balanced paired examples.
- Eight training histories per delay give 16 examples. Histories are disjoint
  between splits. Held-out histories number min(32, remaining space): 19 at
  delay 1 (38 examples), 32 at later delays (64 examples each).
- Fit one fixed seven-token vocabulary on the union of training-stage prompts
  only. No held-out characters or labels fit vocabulary or model parameters.
- Next-character cross-entropy over the full vocabulary is masked to the final
  answer position. All prompt state paths remain attached.

Model seeds 42, 43, 44; H=32; f32 WGPU on RX 9060 XT; Adam .9/.999, epsilon 1e-8,
learning rate .003, global norm clip 1. Each update uses all 16 training examples.
Model weights and Adam moments carry between delays within the same path only.
The three paths start with identical parameter values within each seed.

Each stage has at most 300 updates. Check training metrics every 25 updates and
at the cap. Advance only when training accuracy is 100% and training answer NLL
is at most .1 nats. The first passing training check ends that stage; validation
does not select weights, stopping time, or progression. A failed stage stops that
path's curriculum; report later stages as not reached.

The practical milestone is delay-16 training accuracy >=99% and held-out accuracy
>=95%, achieved by at least two of three seeds for each path. This is a diagnostic
milestone, not a matched-compute architecture ranking: training-based early
stopping can produce different budgets between paths. Report every actual budget.

## Instrumentation

Before curriculum training and after it finishes, hold each model fixed and sweep
all five delays using the same diagnostic pair: `xy0` or `xy1`, D copies of `x`,
then `?`. These traces are diagnostic inputs, not held-out accuracy measurements.

Record actual pre-reset voltage ranges, firing fractions/per-neuron rates,
occupancy inside triangular-surrogate support, surrogate slopes, state RMS, and
bit-pair separation in membrane, spike trace, and continuous memory over time.
Separately differentiate the final-answer loss and record gradient norms/maxima
for every parameter group, plus the two payload embedding rows. The payload token
is used once, so those rows measure credit reaching its input. Compare frozen
models across delays; smaller gradients after successful learning alone do not
establish vanishing gradients.

Also differentiate the mean correct-bit logit minus the other-bit logit, using a
fresh forward graph and no optimizer update. Report that margin and gradients to
both payload embedding rows. This removes the cross-entropy confidence multiplier;
it remains local sensitivity at the measured operating points, not an isolated
temporal Jacobian. The balanced-pair margin has zero output-bias gradient by
construction. Inactive component gradients are zero by design. Full-path payload
gradients combine both routes. Resolve payload rows through tokenizer IDs, not
literal numeric rows (index 0 is unknown).

## Commands and evidence

Checks before measurement:

- Nine model tests pass, including the unchanged 45-parameter independent
  surrogate-gradient oracle, hard spikes/reset, traced voltage, and component
  isolation with present zero gradients for excluded paths.
- Nine training tests pass, including a fresh-Adam component update proving
  inactive parameters stay byte-exact while active groups change.
- Five memory-probe tests pass: paired disjoint histories, vocabulary/masking,
  metric scoring, bounded arguments/outputs, and frozen CE/margin sweeps with
  unchanged parameters and correctly resolved payload embedding rows.
- Explicit RX 9060 XT CPU/GPU parity passes for all three paths, using the same
  45-parameter fixture: Full loss 1.12197852/1.12197840, max gradient difference
  1.4901161e-8; memory-only 1.10121250/1.10121238, max 3.7252903e-8;
  spikes-only 1.12073290/1.12073278, max 1.4901161e-8. Tolerance is 3e-5;
  binary spikes agree exactly and inactive parameter gradients are exactly zero.

```powershell
cargo run --release --bin memory_probe -- --device gpu --steps 300 --seeds 42,43,44 --output artifacts/memory-diagnostic.json
```

The probe uses the single model and shared Burn update function. Synthetic data
is restricted to this focused diagnostic; FineWeb remains the language corpus.
The probe writes a diagnostic report, not a language checkpoint. The following
measurements come from that completed command; the protocol above is retained.

## Measured outcome

All nine seed/path curricula reached delay 16. All **45 reached stages** achieved
100% training and held-out answer accuracy, and every training stopping gate
passed. Each of the three paths passed the final-delay milestone in **3/3 seeds**,
exceeding its predefined two-of-three requirement. The full-vocabulary argmax,
not binary forced choice, defines these accuracies. The constant or history-only
control scores exactly 50% on each paired split; reading the payload is the 100%
oracle.

Each model has H=32, V=7 and **5,863 parameters**. The fixed vocabulary is
`[U+FFFD, 0, 1, ?, x, y, z]`; payload IDs are 1 and 2. Every update uses the full
16-example training batch. Updates below are ordered by delays **1/2/4/8/16**.
Final validation NLL is nats per supervised answer at delay 16.

| Path | Seed | Updates by delay | Total updates | Answer targets | Input tokens | Final validation NLL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| memory-only | 42 | 175/25/25/25/25 | 275 | 4,400 | 32,400 | 0.00000997318 |
| memory-only | 43 | 150/25/25/25/25 | 250 | 4,000 | 30,400 | 0.0000312557 |
| memory-only | 44 | 175/25/25/25/25 | 275 | 4,400 | 32,400 | 0.0000200090 |
| spikes-only | 42 | 75/25/25/25/25 | 175 | 2,800 | 24,400 | 0.00848694 |
| spikes-only | 43 | 75/25/25/25/25 | 175 | 2,800 | 24,400 | 0.00918434 |
| spikes-only | 44 | 75/25/25/25/25 | 175 | 2,800 | 24,400 | 0.00823093 |
| full | 42 | 50/25/25/25/25 | 150 | 2,400 | 22,400 | 0.0000723546 |
| full | 43 | 50/25/25/25/25 | 150 | 2,400 | 22,400 | 0.0000435912 |
| full | 44 | 50/25/25/25/25 | 150 | 2,400 | 22,400 | 0.0000480047 |

Total across the nine curricula: **1,775 updates, 28,400 supervised answer
targets, and 235,600 input tokens**. Subsequent delays always passed at their
first 25-update check; that does not measure the exact minimum number of updates
they needed. Adam moments and learned weights carried forward between stages.
Training-based stopping produced unequal budgets, so neither the shorter Full
curriculum nor the differing final NLL establishes a feedback advantage.

### Retained data, device, and hashes

The report retains all prompts, answers, split hashes, initial/final parameter
hashes, stopping checks, and diagnostic traces. At delay 1, example training pair
`yy0x?`/`yy1x?` has answers `0`/`1`; held-out pair `zx0y?`/`zx1y?` has the same
opposite answers. At delay 16, a training prompt is `xz0xyyxxzyxxzyzxxxx?`, and a
held-out prompt is `zx0xyzxyyxzzyyzxzzx?`. Their paired versions replace only the
single payload `0` by `1`. Each delay has eight training histories; held-out
histories number 19 at delay 1 and 32 at every later delay. Split membership is
fixed across paths and model seeds, so their repeated evaluations are not new
independent datasets.

Execution used Burn 0.22, f32 WGPU, discrete **AMD Radeon RX 9060 XT**, with device
fingerprint `spirv_4098_30096`. The log reports a 1m 56s release build; this is
compilation time, not a separately measured training duration. No throughput,
energy, or real-time deadline result is inferred from this run.

- Raw report: [memory-diagnostic.json](../../artifacts/memory-diagnostic.json),
  SHA256 `d45ea8b3845c74365f3dbaef562d41bf7ddbf76e9a27918e792ba9b8493022da`.
- Command log: [memory-diagnostic.log](../../artifacts/memory-diagnostic.log),
  SHA256 `7f6b3bdc709b3916dc434b3c6058aac9e02093dca15f6da4034b6b73a92b54de`.
- Combined data manifest SHA256:
  `a2a57ba4660a9f749bd37b6a266d8e46b71e7f32c88d807e0609992e848be571`.
- Full-batch index schedule SHA256:
  `f23d672bb9b341f9afa8498423b75deb80e726145969391d4b9392464c2298ee`.

Initial parameter SHA256 values, identical across all three paths within a seed:

| Seed | Initial row-major f32 little-endian parameter SHA256 |
| ---: | --- |
| 42 | `e350c108b6bdc70dbb6c615dee407a64f8ba1586f6a1c4ba0faa70f73039bc20` |
| 43 | `1b589837f014248f8fd00a26a48893ecd495dae6723255241d41a5386b96bae1` |
| 44 | `50112de1d229967dfa23cfffbb85d1572be9a7c5bee86030073d4aa2b5ad8e95` |

## What the activity and credit measurements show

The following representative measurements are seed 42 on the fixed diagnostic
pair, before training and after that path completes its curriculum. Each row is
the **final query step**, not an average over all prompt positions. Support is
the fraction of actual pre-reset voltages strictly inside |v-theta|<width. Slope
is the mean triangular surrogate slope over both lanes and all 32 neurons.

| Path | Weights | Delay | Voltage min/max | Firing % | Support % | Mean slope |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| memory-only | initial | 1 | -0.2274 / 1.5301 | 17.1875 | 93.7500 | 0.54026 |
| memory-only | initial | 16 | -1.3087 / 1.4756 | 18.7500 | 79.6875 | 0.52404 |
| memory-only | final | 1 | -0.7622 / 1.6013 | 28.1250 | 81.2500 | 0.43890 |
| memory-only | final | 16 | -2.3958 / 1.6210 | 21.8750 | 75.0000 | 0.45279 |
| spikes-only | initial | 1 | -0.2274 / 1.5301 | 17.1875 | 93.7500 | 0.54026 |
| spikes-only | initial | 16 | -1.3087 / 1.4756 | 18.7500 | 79.6875 | 0.52404 |
| spikes-only | final | 1 | -2.0530 / 3.8842 | 60.9375 | 28.1250 | 0.07807 |
| spikes-only | final | 16 | -9.5046 / 12.7205 | 60.9375 | 4.6875 | 0.01013 |
| full | initial | 1 | -0.1891 / 1.5763 | 12.5000 | 93.7500 | 0.53860 |
| full | initial | 16 | -1.8344 / 1.4541 | 20.3125 | 81.2500 | 0.49771 |
| full | final | 1 | -1.8097 / 4.1024 | 57.8125 | 20.3125 | 0.06203 |
| full | final | 16 | -20.7400 / 22.4798 | 56.2500 | 4.6875 | 0.00869 |

Over the entire delay-16 diagnostic prompt, initial spikes-only and Full firing
fractions were 29.21875% and 29.765625%; four of 32 neurons never fired in each
case. Final fractions were 55.390625% and 52.109375%, with three and four silent
neurons respectively. This is not a globally silent network. It also does not
establish that every neuron is useful. The final large voltage ranges and low
support occupancy in these two paths show that many query-step neurons sit
outside the surrogate support. Successful recall does not establish uniformly
healthy gradients, bounded long-rollout behavior, or efficient sparse execution.
Memory-only still computes spike diagnostics, but its output and learning are
disconnected from them.

State values below use **RMS / bit-pair L2 separation**. RMS is over both lanes;
separation is ||state(bit0)-state(bit1)||2 at the final query step.

| Path | Weights | Delay | Membrane RMS / separation | Trace RMS / separation | Memory RMS / separation |
| --- | --- | ---: | --- | --- | --- |
| memory-only | initial | 1 | 0.5518 / 2.1290 | 0.1634 / 0.5646 | 0.04563 / 0.12070 |
| memory-only | initial | 16 | 0.6444 / 2.1447 | 0.3512 / 0.4262 | 0.14525 / 0.03409 |
| memory-only | final | 1 | 0.4480 / 2.4847 | 0.2216 / 0.5623 | 0.29751 / 1.76427 |
| memory-only | final | 16 | 0.8094 / 1.9275 | 0.4035 / 0.3910 | 0.83741 / 6.43609 |
| spikes-only | initial | 1 | 0.5518 / 2.1290 | 0.1634 / 0.5646 | 0.04563 / 0.12070 |
| spikes-only | initial | 16 | 0.6444 / 2.1447 | 0.3512 / 0.4262 | 0.14525 / 0.03409 |
| spikes-only | final | 1 | 1.2319 / 6.7798 | 0.3727 / 1.8594 | 0.04829 / 0.19664 |
| spikes-only | final | 16 | 5.5020 / 32.8964 | 0.7661 / 4.2597 | 0.14649 / 0.05760 |
| full | initial | 1 | 0.5801 / 2.0987 | 0.1536 / 0.5623 | 0.05169 / 0.14553 |
| full | initial | 16 | 0.6836 / 2.1122 | 0.3791 / 0.4276 | 0.20634 / 0.06291 |
| full | final | 1 | 1.2957 / 7.5176 | 0.3645 / 1.6699 | 0.21788 / 1.71976 |
| full | final | 16 | 12.6418 / 125.8182 | 0.7349 / 4.1596 | 0.79560 / 7.63618 |

For the final delay-16 models, both lanes have exactly zero state separation
before payload position 2. Memory-only memory separation then grows from 1.0110
at the payload to 6.2849 after the last distractor and 6.4361 at the query.
Spikes-only membrane separation grows from 2.9780 to 32.1813 and 32.8964 at those
same positions; its trace separation reaches 4.2597 at the query. Full memory
separation grows from 0.8369 to 7.4330 and 7.6362, while membrane separation
reaches 125.8182. Thus the learned states preserve a clear payload distinction
through 16 distractors on this diagnostic pair, and the separately held-out
accuracy shows that recall extends to the recorded unseen noise histories.
Separation alone is not a measure of symbolic representations or language skill.

### Input credit, including the confidence control

These are L2 norms of the two payload embedding-row gradients, written
**bit0 / bit1**. Each payload occurs exactly once in its own lane, so these rows
receive credit from that input position rather than repeated later occurrences.
CE differentiates the final-answer cross-entropy; margin differentiates the mean
correct-bit logit minus the other-bit logit on a fresh graph. Both objectives
average the two lanes. Diagnostics make no optimizer updates and check that the
model's parameter hash stays unchanged.

| Path | Weights | Delay | Payload CE gradient norms | Payload margin gradient norms |
| --- | --- | ---: | --- | --- |
| memory-only | initial | 1 | 0.0145203 / 0.0101098 | 0.0205448 / 0.0208401 |
| memory-only | initial | 16 | 0.00583162 / 0.00399631 | 0.00895777 / 0.00901643 |
| memory-only | final | 1 | 0.00196208 / 0.000766000 | 0.0417206 / 0.0498299 |
| memory-only | final | 16 | 3.27479e-8 / 2.08172e-7 | 0.0112014 / 0.0174838 |
| spikes-only | initial | 1 | 0.0423327 / 0.0240007 | 0.0421653 / 0.0395292 |
| spikes-only | initial | 16 | 6.25783e-11 / 5.82123e-11 | 4.56131e-10 / 3.57906e-10 |
| spikes-only | final | 1 | 0.000983578 / 0.00176655 | 0.142356 / 0.190767 |
| spikes-only | final | 16 | 2.90613e-7 / 1.09433e-11 | 7.91970e-5 / 1.28441e-9 |
| full | initial | 1 | 0.0419940 / 0.0313686 | 0.0445191 / 0.0440212 |
| full | initial | 16 | 0.00742102 / 0.00688429 | 0.0126029 / 0.0125697 |
| full | final | 1 | 0.00107896 / 0.00232583 | 0.128436 / 0.246363 |
| full | final | 16 | 1.26905e-6 / 1.37686e-6 | 0.0407499 / 0.0823600 |

All initial payload gradients are nonzero in these traces, but magnitude matters:
the initial spikes-only delay-16 credit is extremely weak even after removing
the CE confidence multiplier. Its margin norms fall from roughly 0.04 at delay 1
to roughly 4e-10 at delay 16 for the same frozen weights. Neurons can fire while
credit from a distant input is weak. The memory-only and Full paths retain much
larger local input sensitivity at initialization in this example. Curriculum
training nevertheless teaches all three paths the task; a nonzero gradient by
itself did not predict how readily they would learn it.

At delay 16 after training, memory-only CE is 1.01327e-5 on this pair while its
margin is 11.9854; Full CE is 6.96751e-5 and its margin is 10.7194. Their very
small CE gradients coexist with margin sensitivity, consistent with confident
correct predictions reducing the CE multiplier. Spikes-only has CE 0.00848701
and margin 4.97499; even its margin gradient is tiny for the bit-1 input at this
operating point. A correct learned response can therefore coexist with weak
local surrogate credit. These measurements do not justify reading every small
gradient as a present recall failure.

The full report preserves norms and maxima for all 11 parameter groups. For
example, initial Full delay-16 CE gradients reach both interfaces: B has L2
0.49745/max 0.06492 and T has L2 0.27205/max 0.04774. Component-disabled groups
have exactly zero gradients by design; those zeros are not learning failures.
The balanced-pair margin also cancels output-bias gradients by construction.
Full-path embedding gradients sum the spike-input and memory-input routes.
Neither objective isolates a temporal Jacobian, and operating points differ
across delays. Before/after comparisons also change learned weights, prediction
confidence, and the curriculum exposure. The repeated-x pair is an activity
diagnostic, not a separate held-out generalization test.

## Decision and remaining uncertainty

The unchanged equations and shared Burn training path can learn delayed one-bit
recall under this curriculum. No architecture change was needed to reach the
predefined milestone. This rules out a blanket claim that these implementations
cannot learn or retain any delayed information. It does not prove why the earlier
paired correction task failed: that task requires different associations,
training exposure, and optimization. Globally dead neurons are not supported by
these traces, while weak long-delay input credit and post-training surrogate
saturation remain measured concerns in particular paths and operating points.

Do not rank feedback from this curriculum: its stopping rule gives unequal
budgets, all paths solve the simpler task, and there is no matched cold-start
control here. The next fixed comparison is documented in
[memory feedback comparison](2026-10-18-memory-feedback-comparison.md). The
separate [FineWeb passage diagnostic](2026-10-18-fineweb-overfit.md) tests genuine
text fitting with independent validation documents. Dense next-character
supervision complements this sparse delayed-answer objective; memorization of a
passage would not establish language generalization or replace a feedback test.
Neither this memory result nor either follow-up establishes emotional reasoning,
complex symbolic semantics, coherent conversation, or real-time performance.

The final combined `cargo test` run passed all 48 library and four integration
tests (52 CPU tests total), including seven memory-probe and four language-probe
tests. `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` passed.
The explicitly run GPU parity check passed as reported above; ordinary
`cargo test` leaves that hardware-dependent test ignored.
