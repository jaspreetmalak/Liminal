# Architecture proposal: multi-timescale, stacked spiking layers

Status: **stage 1 implemented; stage 1 experiment fixed before training.**
The user delegated the decision ("do whatever you want to make it work").
The work is staged to limit risk. Stage 1 is changes 1 and 3 below (learned
per-neuron decays and the spike-trace readout). They add four parameter groups
and leave the state shape unchanged. Stage 2, the second spiking layer, is
implemented only if stage 1 passes, because it changes the state every probe
reads.

Stage 1 experiment, fixed before its training run: arms A (current model H256)
and B (current model H328) as below, run with the pre-change binary. Arm C1 is
the stage 1 model at H256, run with the same data, budget and seed. With the
1,868-character vocabulary, C1 has 4VH + 5H^2 + 5H + V = 2,243,660 parameters,
A has 1,764,684, and B has 2,378,556, so B is still the larger,
parameter-matched control. Acceptance: C1 validation CE at least 0.10
nats/character below B. If C1 fails, stage 1 is removed rather than kept as an
alternate path.

## Problem being addressed

Diagnostics on the current single-layer model
([text diagnostic](2026-10-18-text-diagnostic.md), [width scaling](2026-10-19-width-scaling.md))
show that it trails a five-gram count table by 16.5 accuracy points, which
means it does not capture local character structure. Measured causes:

- Every neuron shares one fixed membrane decay (beta .9). The membrane acts as
  both input integrator and memory, and subtractive reset removes only one
  threshold per step. A neuron with mean input I > theta settles near
  u = (I - theta)/(1 - beta) and fires every step; with I < 0 it never fires.
  Training drove voltages to [-74, 55], leaving 34 of 64 neurons nearly constant.
- Trace decay (.8) and memory decay (.9) are also fixed, and memory takes only
  10% of each new candidate. Nothing can learn to keep exact recent characters
  for some neurons and long-range context for others.
- There is one spiking layer, so combinations of characters must be built by
  a single thresholded recurrence.
- Each step, the readout sees only the current binary spikes and the slow
  memory, with no direct view of recent spike history.

The voltage penalty removed the constant neurons but gained only 2.9 accuracy
points, so operating point alone is not the bottleneck.

## Proposed changes

Hard binary forward spikes, the triangular surrogate, the attached subtractive
reset and causal ordering are all unchanged. Each change can be disabled in
an ablation.

### 1. Learned per-neuron time constants

For each layer l and neuron i, replace the fixed decays with learned logits:

```text
beta[l,i]  = beta_max * sigmoid(kappa[l,i])     membrane
rho[l,i]   = beta_max * sigmoid(lambda[l,i])    spike trace
gamma[i]   = beta_max * sigmoid(mu[i])          continuous memory
beta_max   = .995
```

Initialization spreads time constants log-uniformly over tau in [2, 32] steps
(decay = exp(-1/tau)), so the population starts with both fast and slow neurons.
Reference: learnable membrane time constants (Fang et al., 2021,
[arXiv:2007.05785](https://arxiv.org/abs/2007.05785)). Gradients flow through the
sigmoid, e.g. dv/dkappa = u_prev * beta * (1 - sigmoid(kappa)).

### 2. Two stacked spiking layers

```text
v1_t = beta1 ⊙ u1_(t-1) + E[x_t] + R1 s1_(t-1) + b1 + f_down T c_(t-1)
s1_t = H(v1_t - theta);  u1_t = v1_t - theta s1_t;  r1_t = rho1 ⊙ r1_(t-1) + (1-rho1) ⊙ s1_t
v2_t = beta2 ⊙ u2_(t-1) + W2 s1_t + R2 s2_(t-1) + b2
s2_t = H(v2_t - theta);  u2_t = v2_t - theta s2_t;  r2_t = rho2 ⊙ r2_(t-1) + (1-rho2) ⊙ s2_t
h_t  = A c_(t-1) + C e_t + f_up B r2_t + a
c_t  = gamma ⊙ c_(t-1) + (1-gamma) ⊙ tanh(h_t)
```

Layer 2 sees layer 1's spikes in the same step, which is causal because both
depend only on x_t and earlier inputs. Memory reads the top layer's trace, and
top-down feedback still enters layer 1 from c_(t-1).

### 3. Spike-trace readout

```text
z_t = O1 s1_t + O2 s2_t + P1 r1_t + P2 r2_t + D c_t + o
```

Traces are low-pass spike history, like a synaptic current in a non-spiking
output neuron. They give the readout several steps of recent spiking activity
without exposing membrane voltage. Reading both layers is a skip connection
that also shortens the gradient path to layer 1.

### Deliberately not proposed

An input-gated (GRU-style) continuous memory would probably lower loss fastest,
but it would let the non-spiking path do most of the work. Liminal is meant to
be a spiking network. This proposal keeps the memory ungated (learned decay
only) and puts the new capacity in the spiking path. Please say if you would
rather allow the gated memory as a separately tested option.

## Parameters

Current: 3VH + 5H^2 + 2H + V. Proposed (L=2):
VH (E) + 2VH (O1,O2) + 2VH (P1,P2) + VH (D) = 6VH;
H^2 (R1, W2, R2, C, A, B, T) = 7H^2; biases and decay logits 8H; plus V.
At H=256, V=148: proposed ≈ 688k, current 442k. A current-model width of
H=328 gives ≈ 685k, so it is the parameter-matched baseline.

## Stability

With every decay at most beta_max < 1, the existing infinity-norm bound holds
per layer: ||u_t|| <= beta_max ||u_(t-1)|| + K_l, where K_2 includes ||W2||
(inputs are binary spikes). The geometric bound is finite, but it loosens as
beta approaches beta_max, which is why the cap is .995. As before, this does
not guarantee well-conditioned surrogate gradients.

## Compatibility and tests

- Old checkpoints remain loadable exactly. L=1, P=0, and decay logits set to
  reproduce .9/.8/.9 make the new model equal to the current one. Loading
  maps old files this way, as the existing loader already does for the memory
  groups.
- Extend the cfg(test) finite-difference surrogate oracle to every new
  parameter group: decay logits, W2, R2, b2, O2, P1, P2. Also test layer-to-
  layer causality, decay bounds, and old-checkpoint equivalence.
- `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets -- -D warnings`,
  and the ignored GPU/CPU parity test.

## Matched experiment (to be fixed before training)

Data: `data/fineweb-large` from the data-scale experiment, which avoids the
small corpus's overfitting regime. Budget per arm: batch 64, sequence 64,
5,000 updates (20.5M targets), learning rate .003, clip 1, seed 42.

| Arm | Model |
| --- | --- |
| A | current model, H256 |
| B | current model, H328 (parameter-matched) |
| C | proposed, H256, all three changes |
| C-1, C-2, C-3 | proposed with one change removed, run only if C passes |

Acceptance: C's validation CE is at least 0.10 nats/character below B's. Report
generation for the same six prompts. If C fails, the changes are removed, not
kept as an alternate path.

## Stage 1 results

Implementation: four parameter groups appended after the original eleven
(`membrane_decay`, `trace_decay`, `semantic_decay`, `trace_readout`). The fixed
`ModelConfig.decay` and the trace/memory constants were removed. The checkpoint
format is now `LIMINAL\x02`, and both earlier formats are still imported (see
[architecture](../architecture.md)). Checks, all passing: `cargo fmt --check`;
`cargo test` (75 unit + 4 CLI tests), where the frozen-tangent finite-difference
oracle now covers all 57 fixture parameters, including distinct per-neuron
decays and P; `cargo clippy --all-targets -- -D warnings`; and the ignored WGPU
parity test, with a maximum CPU/GPU gradient difference of 3.73e-8 across all
paths. The previous `fineweb-large-h256.bin` scores 1.935936 after import versus
1.935937 before.

Each arm used the same command with a different `--hidden` and checkpoint. A and
B ran on the pre-change binary, C1 on the stage 1 binary:

```text
cargo run --release -- train --device gpu --data data/fineweb-large/train.jsonl --validation-data data/fineweb-large/validation.jsonl --hidden 256 --sequence-length 64 --batch-size 64 --steps 5000 --learning-rate .003 --gradient-clip 1 --seed 42 --feedback both --checkpoint artifacts/arch-c1-h256.bin
```

| Arm | Model | Parameters | Train CE | Validation CE |
| --- | --- | ---: | ---: | ---: |
| A | previous, H256 | 1,764,684 | 2.268480 | 2.260186 |
| B | previous, H328 | 2,378,556 | 2.217423 | 2.208939 |
| C1 | stage 1, H256 | 2,243,660 | 1.800633 | **1.793523** |

Mean batch CE over steps 1,001-2,000, all on unseen text: A 2.452, B 2.475,
C1 2.058. **Acceptance passes**: C1 is 0.415 nats/character below B, against a
required 0.10, with fewer parameters. With a quarter of the updates, C1 also
beats the 20,000-update previous-model run (1.935937). Checkpoint SHA256:
A `833d9728f574ed7ca30866d2f2c7604a1934a116444bb8343a37ec53d0fe82b4`,
B `41359452cf1ec08e7d6391800125ed26e7f22a65283a73d1f93d41438f741178`,
C1 `529d421cf982825840085f16c6eee6682a305164f5682371a0e83f21053366ce`.

Generation (`artifacts/arch-c1-h256-generation.log`, same six prompts) is more
English-like than the previous model's. Sampling yields longer runs of real
words and some phrase structure (`and people and`, `What it were`). Dialogue
prompts begin first-person replies (`I was the ...`). Greedy decoding still
loops (`of the start of the start`). It is not coherent.

Limitations: one seed, and both changes were tested together. Which change
(learned decays or trace readout) contributes how much is untested; that needs
ablations with one change disabled. A and B ran on a pre-change binary that is
not retained as code; it is equivalent to this model with uniform decays .9/.8/.9
and P=0 frozen.

Decision: keep stage 1 as the single implementation. Next steps are a 20,000-
update stage 1 run on the same corpus, then stage 2 (second spiking layer).

## Stage 1 at 20,000 updates

Same corpus and settings as arm C1, with 20,000 updates at a constant rate
(`artifacts/stage1-large-h256.bin`, SHA256
`0f5e5681f6df8f6fc81a12688a24847d9565a1652b469fb73460bded00130db6`). Train CE is
1.670938 and validation CE is 1.668781 (reset 64) and 1.651477 (reset 128). The
previous model with the identical budget reached 1.935937, so stage 1 is 0.267
better at matched compute. There is no train/validation gap, so the model is
not overfitting at this budget. Follow-ups are in the
[schedule experiment](2026-10-19-schedule-and-context.md) and the
[subword experiment](2026-10-19-subword-tokens.md). Stage 2 (second spiking
layer) has not been implemented.
