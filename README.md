# Liminal

Liminal is a spiking neural network for **natural language generation**, built
from scratch in Rust, including training and data preparation. The current model
uses character or learned subword tokens with **Burn 0.22** tensors, automatic
differentiation, and batched WGPU training. The same model runs on Burn's Flex
CPU device for tests. Rust crates handle HTTP, JSON, and SHA-256 for FineWeb.

It has recurrent leaky integrate-and-fire neurons, binary hidden spikes, a slower
continuous memory, and learned connections in both directions between memory and
spiking activity. Every neuron learns its own membrane, spike-trace, and memory
decay. Next-token logits read the current spikes, the filtered spike trace,
and the memory through an explicit surrogate-gradient tensor graph. This memory
is a research component; semantic competence and emotional reasoning remain
unverified.
Training uses truncated backpropagation through time, global gradient clipping,
and Adam. The neuron reset participates in the backward calculation.

The equations and their limitations are derived in
[docs/architecture.md](docs/architecture.md). Focused tests demonstrate one-bit
recall through 16 distractor characters and exact reproduction of one genuine
FineWeb sentence. All four feedback settings solve that recall test; the
comparison does not demonstrate a feedback accuracy benefit. Sentence-only
training worsens held-out language loss. Fluent language generation, long
context, and energy savings are not established; detailed evidence is linked below.

The [diverse-document FineWeb run](docs/experiments/2026-10-18-fineweb-language.md)
trained the historical H64 model for 3,000 GPU updates using the 70 prepared training
documents. At identical 32-character evaluation resets, held-out NLL improved
from the earlier checkpoint's 2.747231 to 2.506498 nats/character. Its fixed output
suite still produced no coherent continuations or relevant dialogue replies.

The [frozen text diagnostic](docs/experiments/2026-10-18-text-diagnostic.md)
measures this gap: sampled validation next-character accuracy is 31.03%, versus
47.53% for a training-only five-gram control. On identical text, training changes
the neuron population from zero to 24 of 64 neurons firing above 95% of the time.
This identifies weak local prediction and a training-associated activity concern;
a causal repair has not yet been validated.

Follow-ups on 2026-10-19 measured held-out improvements:

- [Width](docs/experiments/2026-10-19-width-scaling.md): H256 instead of H64 on
  the small corpus gave validation CE 2.341 vs 2.512. This is a modest gain.
- [Data](docs/experiments/2026-10-19-fineweb-data-scale.md): a 44.4M-character
  FineWeb sample (`data/fineweb-large`) plus 20,000 batch-64 updates gave 1.936
  vs 2.326 for the small-corpus H256 on the same new validation split. Output
  shifted from invented words to real words.
- [Architecture](docs/experiments/2026-10-19-architecture-proposal.md): learned
  per-neuron decays plus the spike-trace readout gave 1.794 vs 2.209 for a
  parameter-matched H328 of the previous model (5,000 updates each), and 1.669
  vs 1.936 at 20,000 updates.
- [Schedule and windows](docs/experiments/2026-10-19-schedule-and-context.md):
  cosine decay gave 1.639 vs 1.669, just under its 0.03 threshold. 128-character
  windows at matched compute were worse.
- [Subword tokens](docs/experiments/2026-10-19-subword-tokens.md): 6,000 merges
  (about 3.6 characters/token) gave **1.335** nats/character vs 1.619 for the
  best character model, with the same training characters. This is the largest
  gain so far. Sentences become mostly grammatical, but generation is still not
  coherent beyond a sentence or two, and greedy decoding loops.
- [Larger corpus and decoding](docs/experiments/2026-10-19-data-scale-and-decoding.md):
  277M training characters from disjoint FineWeb rows and 16,000 updates
  gave 1.284 vs 1.344 for the previous subword model on the same new split.
  The train/validation gap fell to 0.012. A 1.3 repetition penalty removes
  greedy loops; the text is fluent generic web prose but still not coherent.

The earlier 44M-character subword run, about one hour on an RX 9060 XT:

```text
cargo run --release -- prepare-fineweb --output data/fineweb-large --rows 16000 --max-chars 60000000
cargo run --release -- train --device gpu --data data/fineweb-large/train.jsonl --validation-data data/fineweb-large/validation.jsonl --merges 6000 --hidden 256 --sequence-length 64 --batch-size 64 --steps 5650 --learning-rate .003 --lr-schedule cosine --gradient-clip 1 --seed 42 --checkpoint artifacts/subword-6000-h256.bin
cargo run --release -- generate --checkpoint artifacts/subword-6000-h256.bin --prompt "The " --tokens 72 --temperature 0.8
```

The latest completed checkpoint is `artifacts/subword-xl-h256.bin`: H256,
10,351,410 parameters, trained on 277M FineWeb characters. Select it explicitly;
the ordinary CLI defaults to the older `artifacts/liminal.bin`.

```text
cargo run --release -- generate --checkpoint artifacts/subword-xl-h256.bin --prompt "The " --tokens 72 --temperature 0 --repetition-penalty 1.3
```

The [current XL generation audit](docs/experiments/2026-10-20-generation-failure-audit.md)
inspects 60 frozen continuations and 128 balanced fact contexts. Fact-consistent
candidate choice is 50%, with both opposite-fact answers correct in only 2/64
pairs. None of 15 library/basil continuations sustains two further on-topic
sentences; 30/30 free fact responses are unrelated nonanswers. Readable local
phrases and lower language loss have not yet established reliable prompt-grounded
generation. The audit preserves weights and records all outputs and limitations.
To repeat it, choose an absent report filename:

```text
cargo run -j 1 --bin generation_probe -- --device gpu --output artifacts/generation-diagnostic-new-run.json
```

## Run

The repository includes prepared datasets, provenance manifests, trained
checkpoints, and raw experiment evidence. Datasets and checkpoints use Git LFS;
install Git LFS before cloning, or run `git lfs pull` in an existing checkout.
Cargo's `target/` build cache is excluded.

Use a Rust toolchain supporting the 2024 edition. Run commands from this
directory. The first command needs an internet connection and prepares a bounded
sample of the original [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb):

```text
cargo run --release -- prepare-fineweb
cargo run --release -- train
cargo run --release -- evaluate
cargo run --release -- generate --prompt "The " --tokens 160 --temperature 0.8
cargo run --release -- --help
```

To generate with the diverse-document language baseline, select it explicitly:

```text
cargo run --release -- generate --checkpoint artifacts/fineweb-language.bin --prompt "The " --tokens 256 --temperature 0.8 --seed 42
```

`artifacts/liminal.bin` retains the earlier checkpoint for comparison. Both use
the single current model and inference implementation.

Training, evaluation, and generation default to the first discrete GPU through
WGPU. Initialization fails explicitly if that GPU is unavailable. `--device cpu`
uses the same Burn model on Flex; there is no alternate scalar implementation.
Training and evaluation accept `--batch-size`, default 16. For example:

```text
cargo run --release -- train --device gpu --batch-size 16
cargo run --release -- evaluate --device gpu --batch-size 16
```

`train --lr-schedule cosine` decays the learning rate to 10% by the last
update; the default is `constant`. `train --merges N` learns up to N subword
tokens from the training documents only. Characters remain the base vocabulary,
and the default of 0 keeps character tokens. With merges, `generate --tokens`
counts subword tokens. Reported losses are always nats per character, so
character and subword runs are directly comparable.

`generate --top-p P` samples only from the most probable tokens that together
reach probability P. `--repetition-penalty R` (R ≥ 1) discourages tokens used
in the last 64 tokens. Both default to 1, which leaves generation unchanged.
For the current subword model, `--temperature 0 --repetition-penalty 1.3`
reads best: it removes greedy loops without the invented words of plain
sampling. See the
[decoding comparison](docs/experiments/2026-10-19-data-scale-and-decoding.md).

Fresh training enables both feedback directions. `train --feedback` accepts
`both`, `top-down`, `bottom-up`, or `neither` for controlled ablations of the same
model; every setting retains the same parameters and continuous memory.

An optional training-only voltage objective is controlled by
`--voltage-penalty` (coefficient lambda, default **0**) and
`--voltage-band-widths` (k, default **3**). For actual pre-reset voltage v,
threshold theta, surrogate width w, valid-target mask M, N=sum(M), and H neurons:

```text
P = sum_(batch,time,neuron) M * max(0, abs((v-theta)/w)-k)^2 / (N*H)
training objective = next-character cross-entropy + lambda*P
```

Both terms exclude padding. The same forward loop supplies logits and voltage;
one backward pass, global clip, and Adam update use the combined objective.
Coefficient zero disables penalty computation and logs raw/weighted penalty as
zero because activity was not measured. Initial/final evaluation and perplexity
always use plain cross-entropy. The optional cost changes training pressure;
hard spikes, the triangular surrogate, attached reset, inference equations,
and checkpoint format stay unchanged. The library exposes this through
`TrainConfig.voltage_penalty`, `VoltagePenalty`, and
`Model::training_loss_for_path`, whose `LossTerms` separates CE, raw P, and total.

The completed [voltage-regularization trial](docs/experiments/2026-10-18-voltage-regularization.md)
compared fresh seed-42 models at lambda 0 and .001, k=3, H64/B16/T64, with
3,000 GPU updates / 3,072,000 targets each. Plain validation CE at resets 32/64
improved from 2.509806/2.508364 to 2.408202/2.408742. On 4,096 matched targets,
accuracy rose from 1,274/4,096 (31.1035%) to 1,393/4,096 (34.0088%):
**+2.9053 points, below the predeclared +3-point requirement**.

Activity improved: neurons firing above 95% fell from 20 to zero, those below 5%
from 14 to zero, and strict surrogate support rose from 35.4263% to 73.7808%.
Across eight stochastic real-prefix continuations, lexical hits fell from
17/34 (50%) to 16/37 (43.2432%).
Both conditions still scored 0/12 coherent continuations and 0/4 relevant
dialogue replies. The combined mechanism criterion was not met; this one paired
seed supports a measured activity/loss improvement without establishing a fix
for garbled generation. The coefficient remains optional and defaults to zero.

For another experiment, enable it explicitly and choose a fresh checkpoint path:

```text
cargo run --release -- train --device gpu --hidden 64 --sequence-length 64 --batch-size 16 --steps 3000 --learning-rate .003 --gradient-clip 1 --seed 42 --feedback both --voltage-penalty .001 --voltage-band-widths 3 --checkpoint artifacts/fineweb-voltage-new-run.bin
```

Burn 0.22 requires Rust 1.95 or newer. Production tensors use `f32`; the previous
scalar `f64` trainer and its handwritten optimizer/backpropagation are removed.

Preparation inspects at most 128 rows from offset 0 of FineWeb's `sample-10BT`
subset and retains at most 200,000 Unicode scalar values, without cutting
documents. It removes exact duplicate IDs/texts before creating separate
`data/fineweb/train.jsonl` and `validation.jsonl` files with split seed 42.
`manifest.json` records selection, source revision, document provenance, and
hashes. The output directory must be absent or empty.

This small prefix sample is for initial experiments; it is not the whole FineWeb
dataset or a representative benchmark. See [docs/data.md](docs/data.md) for
selection rules, attribution, revision limitations, and preparation options.

Training uses the prepared training file by default, fits the vocabulary only
on training documents, and evaluates against the separate validation file.
Documents stay independent: state and next-character target pairs never cross
their boundaries. Each batch row is an independently sampled window with zero
initial state. Short rows are right-padded and masked; loss is the mean over
actual predicted characters, so padding contributes no loss or gradient.
The default budget is 500 updates of 16 windows of up to 32 predicted characters
each, with 48 hidden neurons and seed 42. The CLI reports
initial/final training and validation cross-entropy/perplexity and a
training-frequency unigram control. A completed import or training command alone
does not demonstrate language quality; measured runs belong in
`docs/experiments/`.

At update 1, every 50 updates, and the final update, progress reports batch CE,
raw P, lambda*P, total, the combined gradient norm before clipping, and cumulative
target count. Regularized totals are not used to calculate perplexity.

The language checkpoint is saved to `artifacts/liminal.bin`. It contains the model
configuration, vocabulary, feedback mode, and weights for generation and evaluation. It does
not contain optimizer state for resuming training. Re-running `train` initializes
a fresh model and replaces that checkpoint.

Earlier five-group checkpoints load into the current model with zero memory
weights to preserve their inference predictions. They have no trained semantic
memory. Fresh training initializes the new memory with nonzero seeded weights.

## Feedback experiment

A focused synthetic contextual-memory probe uses the same model, masked loss,
global clipping, and Adam step. It compares both feedback directions against
either direction and neither, with identical initialization and minibatches per
seed. Its data stays separate from the FineWeb language corpus, and it does not
overwrite the language checkpoint.

```text
cargo run --release --bin feedback_probe -- --device gpu --steps 300 --seeds 42,43,44 --output artifacts/feedback-counterfactual.json
```

This measures delayed query-conditioned correction on a paired symbolic task.
Queries for the same history require opposite answers, so query-blind controls
score 50%. The task does not establish memory for two independent entity histories.
It does not
establish language fluency, emotional reasoning, or real-time performance. The
predeclared protocol and measurements are in
[the counterfactual experiment](docs/experiments/2026-10-18-counterfactual-feedback.md).
The [first pilot](docs/experiments/2026-10-18-semantic-feedback.md) retains its
original results and the subsequently discovered shortcut limitation.

For separate custom UTF-8 or JSONL corpora:

```text
cargo run --release -- train --data data/corpus.txt --validation-data data/heldout.txt --steps 2000 --hidden 64 --sequence-length 64
cargo run --release -- evaluate --data data/heldout.txt --sequence-length 64
```

Provide both files and keep evaluation documents out of training. A JSONL file
contains one object with a string `text` field per line and requires a separate
validation corpus for training. Plain UTF-8 input without `--validation-data`
retains a 90/10 character split for focused smoke checks.

Evaluation resets model state at every document and independent sequence window,
then weights each window by its number of targets. Report the reset length with
results. Perplexity is per character, not per word. No Python or external ML
runtime is used.

Unseen characters map to a visible replacement token (`�`), and the CLI reports
their count. The model has no end-of-text token; generation emits the requested
number of additional tokens (characters when merges are zero). Temperature zero
chooses greedy output.

## Verify

The one-bit memory diagnostic isolates the continuous, spiking, and full paths,
increases delays from 1 to 16 characters, and records firing, surrogate support,
state separation, and parameter gradients. It uses paired synthetic fixtures
and never overwrites the FineWeb checkpoint:

```text
cargo run --release --bin memory_probe -- --device gpu --steps 300 --seeds 42,43,44 --output artifacts/memory-diagnostic.json
```

Its training-based progression rules and actual measurements are in
[the memory diagnostic](docs/experiments/2026-10-18-memory-diagnostic.md).

For a fixed-budget comparison of the four feedback settings, train directly on
delay 16 with fresh weights and no curriculum:

```text
cargo run --release --bin memory_probe -- --compare-feedback --device gpu --steps 300 --seeds 42,43,44 --output artifacts/memory-feedback-comparison.json
```

The [matched comparison](docs/experiments/2026-10-18-memory-feedback-comparison.md)
records the protocol and results. A separate genuine FineWeb sentence check tests
teacher-forced fitting and exact greedy continuation, then measures held-out loss:

```text
cargo run --release --bin language_probe -- --device gpu --steps 800 --output artifacts/fineweb-overfit.json
```

It saves a separate diagnostic checkpoint and preserves `artifacts/liminal.bin`.
See [the passage experiment](docs/experiments/2026-10-18-fineweb-overfit.md) for
source provenance, measurements, and limits. The language diagnostic requires
absent output files. Use fresh filenames for repeated experiments to retain evidence.

The frozen text probe compares learned conditional predictions, a training-only
character-count control, real versus self-generated history, and neuron/readout
activity. It reads the fixed `fineweb-language.bin` diagnostic subject without
updating weights or saving checkpoints:

```text
cargo run --bin text_probe -- --device gpu --output artifacts/text-diagnostic-new-run.json
```

It verifies the subject's recorded checkpoint hash and requires a new report
filename. The experiment log distinguishes the original run and the additional
fresh-initialization/activity controls.

The frozen understanding probe scores balanced fact-consistent continuations
for binding, negation, temporal order, and semantic roles, then measures how
ordered context changes genuine FineWeb target likelihoods. It compares the
fixed CE control and voltage-regularized checkpoints with a fresh seed-42 model,
performs no training, and verifies source/checkpoint hashes:

```text
cargo run --bin understanding_probe -- --device gpu --output artifacts/understanding-diagnostic.json
```

Choose an absent report filename for each run. The
[understanding diagnostic](docs/experiments/2026-10-18-understanding-diagnostic.md)
records the fixed fixtures, scoring boundaries, and actual results:

| Frozen subject | Correct cases | Both members of a pair correct |
| --- | --- | --- |
| CE control | 65/128 (50.78%) | 4/64 |
| Voltage penalty | 62/128 (48.44%) | 5/64 |
| Fresh seed 42 | 63/128 (49.22%) | 11/64 |

Every family fails the predefined 75% case / 50% strict-pair criterion. On 512
genuine validation targets with 32-character prefixes, NLL is 2.330152 for the
CE control, 2.226824 for the penalty, and 4.974252 for fresh initialization.
Shuffling the first 16 prefix characters while preserving the final 16 raises
the penalty model's NLL to 2.273979; the CE control changes slightly to 2.327620.
Ordered context can affect prediction without yielding consistent answers to
changed facts. These chosen templates are a focused diagnostic, not a general
comprehension benchmark or evidence of emotional reasoning. No weights changed.

```text
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --test gpu -- --ignored --nocapture
```

The last command explicitly requires a discrete GPU and compares its forward
spikes, masked loss, and all parameter gradients with the same model on CPU.
Ordinary tests run on Flex CPU without requiring GPU hardware.

Tests include hand-computed neuron/state dynamics, causality, stable cross
entropy, every parameter's surrogate gradient against an independent local
finite-difference oracle, batched/masked gradient identities, global clipping,
Burn Adam updates, deterministic sequence learning,
checkpoint equivalence, tokenizer behavior, and command errors. Hard spikes are
kept in the forward pass; the gradient oracle checks the chosen surrogate graph,
not the ordinary derivative of the discontinuous hard-spike objective.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/model.rs` | Neuron recurrence, masked loss, surrogate autodiff graph |
| `src/training.rs` | Adam, clipping, training, evaluation, sampling |
| `src/runtime.rs` | Explicit WGPU/Flex device selection |
| `src/data.rs` | Document loading and training-fitted character/subword vocabulary |
| `src/fineweb.rs` | Bounded FineWeb import, deduplication, splitting, and provenance |
| `src/checkpoint.rs` | Model/vocabulary persistence and integrity checks |
| `src/memory_probe.rs` | Component recall curriculum, activity/credit diagnostics, matched feedback comparison |
| `src/language_probe.rs` | FineWeb passage fitting, continuation, and held-out measurements |
| `src/text_probe.rs` | Frozen text prediction, character-count controls, history and lexical diagnostics |
| `src/text_activity.rs` | Shared-model voltage, spike, gradient, and output-component measurements |
| `src/understanding_probe.rs` | Frozen fact consistency, corpus concentration, and ordered-context diagnostics |
| `src/understanding_cases.rs` | Balanced counterfactual language fixtures and independent label checks |
| `src/continuation_score.rs` | Shared masked continuation likelihoods through production model steps |
| `src/generation_probe.rs` | Frozen current-checkpoint generation and fact-consistency audit |
| `src/probe.rs` | Paired query-conditioned correction experiment |
| `src/rng.rs` | Deterministic initialization, windows, sampling |
| `src/main.rs` | Prepare, train, generate, evaluate command interface |
| `data/` | Prepared FineWeb corpora and the historical smoke fixture |
| `docs/` | Equations, design decisions, and experiment evidence |
| `artifacts/` | Experiment reports/logs in Git; trained checkpoints in Git LFS |

Maintain one current implementation. Experiment findings belong in
`docs/experiments/`; superseded source is removed. See [AGENTS.md](AGENTS.md).
