# Liminal architecture and learning equations

Liminal is an original recurrent spiking language model written in Rust with
Burn 0.22 tensors, automatic differentiation, optimization, and WGPU execution.
Flex CPU checks run the same model. Its task is causal next-character prediction.
The current model couples binary spiking dynamics to a slower, bounded continuous
memory. Calling this memory semantic describes its intended responsibility;
semantic competence and emotional reasoning require separate measured evidence.

## Representation, dimensions, and parameters

By default one token is one Unicode scalar value (`char`), not a byte,
grapheme, or word. Fit the vocabulary on training documents only. Index 0 is
U+FFFD/unknown; other characters are unique and sorted. Unknown evaluation and
prompt characters map to 0. There is no normalization, start token, end token, or
conversation-role protocol.

`train --merges N` adds up to N subword tokens learned from the same training
documents. Text is first split into chunks: runs of one character class
(letters, digits, single spaces, other whitespace, other characters), with the
last space before a letter or digit run joining it (" word"). The learner
repeatedly merges the most frequent adjacent token pair inside chunks, breaking
ties toward the smallest token pair, and never crosses chunks. Merge k becomes
token (character count + k). Encoding applies merges in rank order; decoding
concatenates token text and is lossless apart from unknown characters. The
model equations treat every token the same, so V counts characters plus
merges. Losses are reported per character: total NLL divided by the characters
of the scored tokens, excluding each document's first token, which is context.

Let V be vocabulary size and H the spiking and memory width. Equations below
use column vectors; implementation states are row tensors [batch,H], with
matrix transposes preserving the receiving-neuron convention.

| Stored field / symbol | Shape | Responsibility |
| --- | --- | --- |
| `embedding`, E | V × H | Token input currents e = E[x,:] |
| `recurrent`, R | H × H | Previous spikes to current voltage |
| `hidden_bias`, b | H | Neuron current bias |
| `readout`, O | V × H | Spikes to next-character logits |
| `output_bias`, o | V | Next-character bias |
| `semantic_input`, C | H × H | Current token embedding to memory candidate |
| `semantic_recurrent`, A | H × H | Previous memory to memory candidate |
| `spike_to_semantic`, B | H × H | Filtered spikes to memory candidate |
| `semantic_to_spike`, T | H × H | Previous memory to neuron current |
| `semantic_bias`, a | H | Memory candidate bias |
| `semantic_readout`, D | V × H | Current memory to next-character logits |
| `membrane_decay`, κ | H | Per-neuron membrane decay logit, β = β_max σ(κ) |
| `trace_decay`, λ | H | Per-neuron spike-trace decay logit, ρ = β_max σ(λ) |
| `semantic_decay`, μ | H | Per-unit memory decay logit, γ = β_max σ(μ) |
| `trace_readout`, P | V × H | Current spike trace to next-character logits |

Bias and decay-logit tensors have a leading singleton dimension. Parameter order
is table order. The count is P = 4VH + 5H² + 5H + V; H=2,V=3 has 57 parameters.
All parameters and Adam moments execute as f32. Configuration, reporting, and
independent numerical test calculations may use f64.

Initialization uses seeded SplitMix64: E uniform [-.5,.5], R uniform
[-.2/√H,.2/√H], b uniform [.2,.3], O uniform [-1/√H,1/√H], o zero. Then C,A,B,T
are independently uniform [-1/√H,1/√H], a zero, and D independently uniform
[-1/√H,1/√H]. Each decay logit then draws a time constant τ log-uniformly in
[2, 32] steps and stores logit((e^(-1/τ))/β_max), separately for κ, λ, and μ,
so fresh neurons span fast and slow dynamics. P is last, uniform [-1/√H,1/√H].
The first eleven groups draw the same values as before these additions. This
generator provides repeatability, not cryptography.

## Causal state update and feedback

An independent window starts with four zero states: post-reset voltage u, binary
spikes s, filtered spike trace r, and continuous memory c. For each input x_t:

```text
β = β_max σ(κ),  ρ = β_max σ(λ),  γ = β_max σ(μ),  β_max = .995   (per neuron)
e_t = E[x_t,:]
v_t = β ⊙ u_(t-1) + e_t + R s_(t-1) + b + f_down T c_(t-1)
s_t[i] = 1[v_t[i] >= θ]
u_t = v_t - θ s_t
r_t = ρ ⊙ r_(t-1) + (1-ρ) ⊙ s_t
h_t = A c_(t-1) + C e_t + f_up B r_t + a
q_t = tanh(h_t)
c_t = γ ⊙ c_(t-1) + (1-γ) ⊙ q_t
z_t = O s_t + P r_t + D c_t + o
p_t = softmax(z_t)
```

P r_t reads recent spiking history: the trace is a low-pass filter of binary
spikes, like a synaptic current in a non-spiking output neuron. The readout does
not see membrane voltage.

z_t predicts x_(t+1). Top-down modulation uses previous memory; bottom-up
feedback uses the just-completed spike trace. There is no instantaneous circular
dependency or future token. Memory updates every token with a slower mixing time
constant. It has no independent wall clock, biological duration, or demonstrated
real-time deadline.

θ and surrogate width w are fixed ModelConfig values; the standard CLI uses θ=1,
w=1. The decays β, ρ, and γ are learned per neuron through κ, λ, and μ. The
sigmoid keeps each decay in (0, β_max), so the stability bound below still
applies. The local derivative is, for example, ∂β/∂κ = β (1 - σ(κ)). Before
this change all neurons shared fixed β=.9 (CLI), ρ=.8, and γ=.9. The
[architecture experiment](experiments/2026-10-19-architecture-proposal.md)
measured the change. State paths remain attached inside each window.
Threshold equality emits one spike. Reset subtracts one threshold even above two
thresholds; negative voltage is allowed. There is no refractory period, adaptive
threshold, or extra simulation cycle. Hidden spikes are binary; membrane,
memory, traces, currents, and probabilities are continuous. Burn executes dense
tensors. Spikes alone do not establish sparse execution or an energy advantage.

The four modes share parameters and operation graph; only the multipliers vary:

| FeedbackMode | f_down | f_up |
| --- | ---: | ---: |
| both | 1 | 1 |
| top-down | 1 | 0 |
| bottom-up | 0 | 1 |
| neither | 0 | 0 |

Disabled terms stay in the graph multiplied by zero, yielding present but zero
gradients. Every mode retains memory recurrence, token-to-memory input, and both
output projections. This comparison isolates coupling terms; `neither` is not
the previous smaller model. Retrained conditions differ from disabling a
connection after training `both`.

Diagnostic `ModelPath` controls additionally isolate complete components.
`memory-only` disables both interfaces and the spike output projection;
`spikes-only` disables both interfaces and the memory output projection.
`full` preserves the normal feedback-mode behavior. These are zero gates inside
one shared state update, not alternative forward implementations. Production
`step`/`loss` delegate to `full`. Paths are explicit call arguments used by probes,
not saved checkpoint settings. A component-trained diagnostic requires its
specified path when evaluated. Each independently trained path needs fresh Adam
state so momentum from another condition cannot move an inactive parameter.

The shared traced step can expose the actual pre-reset voltage for measurement.
It remains attached to the same graph. Activity and credit-assignment diagnostics
do not modify model parameters or insert extra recurrent-state detach operations.

## Hard spikes and complete surrogate BPTT

Keep hard binary spikes forward and use this chosen local derivative backward:

```text
g(v) = max(0, 1 - abs(v-θ)/w) / w
ds/dv := g(v)
```

Its support is |v-θ|<w, peak 1/w, integral 1, and endpoint value zero. This is a
surrogate rule, not the ordinary derivative of a discontinuous hard function or
a guarantee of hard-loss descent. Background:
[Neftci et al., 2019](https://arxiv.org/abs/1901.09948).

Burn constructs the desired graph with the symmetric integrated triangle:

```text
j = clamp((v-θ)/w, -1, 1)
F = 1/2 + j - j abs(j)/2
s = float(v >= θ) + (F - stop_gradient(F))
```

Subtract equal smooth values before adding the hard spike so forward f32 stays
exactly binary. The derivative has the intended peak and zero endpoint values
without depending on clamp/abs boundary conventions. Keep u=v-θs attached,
including its -θg reset contribution. Membrane, spike, trace, and memory paths
all remain attached during BPTT.

The local reverse equations below specify Burn's cross-entropy graph. The optional
training-only voltage objective adds the direct voltage adjoint defined below.
Adjoints are
column vectors; parameter sums accumulate over time and independent batch lanes.
du, ds_future, dr_future, dc_future arrive from later steps; dz is the masked
logit adjoint defined below.

```text
dc = dc_future + Dᵀ dz
dh = (1-γ) ⊙ dc ⊙ (1-q_t²)
dr = dr_future + f_up Bᵀ dh + Pᵀ dz
ds = ds_future + Oᵀ dz + (1-ρ) ⊙ dr
dv = du + g(v_t) ⊙ (ds - θ du)
de = dv + Cᵀ dh

dE[x_t,:] += de             dR += dv s_(t-1)ᵀ
db += dv                   dO += dz s_tᵀ
do += dz                   dC += dh e_tᵀ
dA += dh c_(t-1)ᵀ          dB += f_up dh r_tᵀ
dT += f_down dv c_(t-1)ᵀ   da += dh
dD += dz c_tᵀ              dP += dz r_tᵀ
dκ += dv ⊙ u_(t-1) ⊙ β ⊙ (1-σ(κ))
dλ += dr ⊙ (r_(t-1) - s_t) ⊙ ρ ⊙ (1-σ(λ))
dμ += dc ⊙ (c_(t-1) - q_t) ⊙ γ ⊙ (1-σ(μ))

du_previous = β ⊙ dv
ds_previous = Rᵀ dv
dr_previous = ρ ⊙ dr
dc_previous = γ ⊙ dc + Aᵀ dh + f_down Tᵀ dv
```

There is no production manual backward implementation. An independent scalar
oracle exists only under cfg(test): record the hard trajectory, freeze its
local slopes, then replace perturbed spikes by s_bar+g_bar*(v_perturbed-v_bar).
Central finite differences of this local tangent recurrence check every
parameter. Finite differences of hard spike loss itself do not validate a
surrogate. Tests exercise nonzero initial states, both memory directions,
attached reset, unequal lanes, masking, and direct center/support checks.

## Loss, document boundaries, and optimization

For target y_t=x_(t+1), ell_t=logsumexp(z_t)-z_t[y_t], with stable log-softmax.
For mask M[j,t] and N=sum(M)>0:

```text
L_CE = sum_(j,t) M[j,t] ell[j,t] / N
dz[j,t,k] = M[j,t] (p[j,t,k] - 1[k=y[j,t]]) / N
```

Natural-text training masks valid adjacent pairs as 1 and right padding as 0.
The focused probe masks prompt losses as 0 and only the answer target as 1;
prompt computation remains attached and receives gradients from the answer.
This uses the same masked-loss implementation. A target is never consumed
before its prediction. Do not average per-row means with unequal valid counts.

The optional training-only voltage regularizer measures actual pre-reset v from
that same forward execution. Let k be the allowed distance from threshold in
surrogate-width units, and lambda the nonnegative coefficient:

```text
q[j,t,i] = max(0, abs((v[j,t,i]-θ)/w) - k)
P_voltage = sum_(j,t,i) M[j,t] q[j,t,i]^2 / (N*H)
L_train = L_CE + lambda*P_voltage
```

`P_voltage` is the raw P in training logs, distinct from the parameter count P
above. It is dimensionless and averages across valid targets and hidden neurons,
not all padded time positions or per-row means. Masked prompt steps receive no
direct cost but retain temporal dependencies when used by a configured objective.
The standard optional setting k=3, θ=w=1 allows voltages in [-2,4] without cost,
wider than the triangular surrogate's direct support (0,2). Squared excess has
zero derivative inside and on the band; outside, its direct addition to the
local `dv` adjoint is

```text
dvoltage_cost[j,t,i] =
    lambda * 2*M[j,t]*q[j,t,i]*sign(v[j,t,i]-θ) / (N*H*w)
dv = du + g(v_t) ⊙ (ds - θ du) + dvoltage_cost
```

Future adjoints already include costs at later steps. The voltage penalty uses
ordinary autodiff through voltage and its squared excess; it does not replace
the hard forward spike, triangular spike derivative, or attached reset derivative.
It adds an objective and no voltage clamp, extra neuron state, or inference step.

The API is `Model::training_loss_for_path(batch, path, VoltagePenalty)`, returning
`LossTerms { cross_entropy, voltage_penalty, total }`. Its `voltage_penalty` field
is raw normalized P, before coefficient multiplication. `VoltagePenalty` defaults
to `{ coefficient: 0.0, band_widths: 3.0 }` and is supplied by
`TrainConfig.voltage_penalty`. Coefficients must be finite and nonnegative, with
positive values remaining positive finite f32 values; band widths must remain
positive finite f32 values. Both CLI options, `--voltage-penalty` and
`--voltage-band-widths`, are training options with the same defaults.

When lambda is zero, the single loss loop takes the original CE graph without
computing voltage excess or an auxiliary backward graph. `total` is an exact
clone of CE; raw P is a constant zero indicating disabled measurement, not a
claim that measured voltages were in band. Existing focused-probe update wrappers
use that disabled configuration. A nonzero penalty on a component diagnostic
would also differentiate voltage even where a readout path is disabled; the
completed voltage trial used `ModelPath::Full` with both feedback directions.

`loss` and `loss_for_path` always return pure CE. Initial/final corpus evaluation,
held-out NLL, and perplexity use CE regardless of the training setting. Progress
at update 1, every 50 updates, and the final update separates CE, raw P,
lambda*P, total, and the global norm of the combined gradient before clipping.
The completed [fixed matched trial](experiments/2026-10-18-voltage-regularization.md)
compared lambda .001 with zero at k=3, using one paired seed (42), H64/B16/T64,
and 3,000 GPU updates / 3,072,000 targets per condition. Validation CE at reset
32/64 improved from 2.509806/2.508364 to 2.408202/2.408742. On the matched
4,096-target validation sample, accuracy improved from 31.1035% (1,274 correct)
to 34.0088% (1,393), a 2.9053-point gain that **failed** the predeclared minimum
3-point gain despite passing both CE thresholds. Nearly always-firing neurons
(>95%) fell from 20 to zero, nearly silent neurons (<5%) from 14 to zero, and
strict surrogate support increased from 35.4263% to 73.7808%, passing the
operating-point gate. Across eight stochastic real-prefix continuations,
lexical hits declined from 17/34 to 16/37;
both conditions produced 0/12 coherent continuations and 0/4 relevant dialogue
replies. The combined operating-point/prediction criterion was not met. These
results demonstrate activity and CE changes for this pair, not a validated
repair of natural-language generation. The coefficient remains optional,
default zero; inference equations and the maintained language baseline remain
unchanged.

FineWeb preparation splits disjoint whole documents before vocabulary fitting.
Training draws B independent windows uniformly with replacement from eligible
within-document starts; short documents use all available adjacent pairs. No
state or target pair crosses documents. All four states reset for each window;
gradients stop at its beginning. This is truncated BPTT.

Standard evaluation scores every within-document adjacent pair once in ordered
windows, resetting each window/document. Divide total negative log likelihood by
actual target count. Report reset length and unknown rate. Continuous-state
evaluation uses a different context policy. The additive-one unigram control
fits training target frequencies only, skipping each document's first character.
Validation never enters fitting or updates. Training-split scores reported by
`train` are exact up to 4,000,000 training tokens. Above that they use an evenly
spaced sample of every k-th document, k = ceil(tokens / 4,000,000), and the run
log states the sample size; validation is always scored in full.

Reported CE is nats per character, character perplexity exp(L_CE), bits per
character L_CE/ln(2). The regularized total is not a perplexity. These are not
word metrics. Improved loss alone does not establish
conversational relevance, symbolic reasoning, or emotional understanding.

The shared optimizer differentiates one complete masked batch objective (CE or
CE plus the optional voltage cost), gathers all 15
parameter groups, and clips their combined Euclidean gradient norm to C:
G *= min(1,C/sqrt(sum_i G_i²)); zero gradients stay unchanged. Burn's per-tensor
norm clipping is not substituted. Nonfinite norms abort the update. Adam uses
betas .9/.999, epsilon 1e-8 outside the square root, bias correction, no weight
decay, and zero initial moments. The learning rate is constant by default.
`--lr-schedule cosine` sets update t of T to
η_t = η_min + (η - η_min)(1 + cos(π(t-1)/(T-1)))/2, with η_min = 0.1η.
Language training and the probe share this update function.
Evaluation/generation use valid() without autodiff.

## Generation and persistence

Generation starts all states at zero and consumes each prompt token once.
It samples from the final logits, feeds each sampled token once, and
preserves all four states. Before choosing, a repetition penalty ρ ≥ 1
(CTRL-style; default 1, disabled) rescales the logit of every distinct token
among the last 64 context tokens, prompt included: z/ρ if z > 0, else zρ.
Temperature >0 samples stable softmax(z/temperature); zero chooses an argmax.
With top-p < 1, sampling is restricted to the smallest set of most probable
tokens whose probability reaches p, ties broken by token id. With top-p 1 and
ρ = 1 the draw is identical to plain temperature sampling. Sampling uses a
separate seed and one uniform draw per token. Empty prompts are rejected; there
is no learned stopping token.

Checkpoints (format `LIMINAL\x03`) store the vocabulary size (characters plus
merges), hidden size, threshold, surrogate width, feedback mode, merge count,
the base characters, the ranked merges as token pairs, and all 15 parameter
groups, with corruption checks. `LIMINAL\x02` files are the same without merges
and still load. f64 wire fields store f32 values exactly; this
does not change execution precision. Adam moments, update number, sampler, and
recurrent state are not stored, so exact training resume is unavailable.
Training-only voltage settings are recorded in run logs rather than adding
inference configuration or checkpoint fields. Generation and evaluation use
the same state equations and artifact format for either training objective.

The loader imports two earlier formats into this same model. Both store a fixed
membrane decay in the header. Eleven-group fixed-decay artifacts
(`LIMINAL\x01`) gain uniform decay logits reproducing their β, ρ=.8, and γ=.9,
plus a zero trace readout. Five-group artifacts (`LIMINAL\0`) also gain zero
memory parameters, so they have no trained semantic memory. Imported predictions
match up to f32 rounding of σ(logit). For example, `fineweb-large-h256.bin`
scores 1.935936 nats/character on its validation split after import versus
1.935937 before. Zeroing the entire memory subsystem would be a dead learning
initialization. Fresh training uses seeded nonzero Model::new parameters. No
previous forward/backward implementation is retained.

## Stability, cost, and evidence

From zero, |r_i|<=1 and |c_i|<=1, as convex mixtures of bounded spikes/tanh. For
fixed finite weights, every learned β_i < β_max = .995, so
||u_t||∞ <= β_max||u_(t-1)||∞+K, with
K=max_x||E[x,:]||∞+||R||∞+||b||∞+f_down||T||∞+θ. Its geometric-series bound is
finite. This does not establish contractive surrogate gradients or floating-point
safety under arbitrary parameter updates.

Large weights or narrow surrogate support can amplify or starve gradients.
Measure firing rates, support occupancy, membrane ranges, gradient norms,
clipping, learning curves, and long-rollout behavior when diagnosing failures.
Dense work remains O(BT(H²+VH)), with larger constants for memory projections;
weights/gradients/Adam storage O(P), recurrent activations O(BTH), output storage
O(BTV) unless recomputed. Inference state is O(BH).

Required evidence includes dynamics/causality, every-parameter surrogate checks,
masking and independent lanes, clipping/Adam, checkpoint equivalence, CPU/GPU
parity, reproducible learning, held-out text, and budget-matched feedback
ablations. Record actual commands, settings, precision, seeds, tolerances,
hardware, metrics, and limits under docs/experiments/. Synthetic cases are for
focused tests; FineWeb remains the primary language-training corpus.
