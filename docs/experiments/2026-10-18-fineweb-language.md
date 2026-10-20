# Diverse-document FineWeb language training

## Protocol fixed before training

Question: does the current full Liminal model improve held-out next-character
prediction when trained on all prepared FineWeb training documents, rather than
repeating one sentence? The user authorized this next training experiment.
The model equations and Rust/Burn optimizer implementation stay unchanged.

Use the existing bounded sample of HuggingFaceFW/fineweb, sample-10BT, revision
`9bb295ddab0e05d785b879661af7260fed5140fc`: 70 training documents / 185,628
characters and eight disjoint validation documents / 14,348 characters. Training
has 185,558 within-document adjacent targets; validation has 14,340. Vocabulary
is fitted on training documents only (V148); validation has four unknown
characters. Source hashes match the retained preparation manifest:

- Training JSONL: `ad0b069c93c70dc6fc52000edefbd272a574ae77dd7706ab763eff702ee2633a`.
- Validation JSONL: `3ca429b3909a50a591286cb27f9b650527e2bd2846ffb22e5c460b20ad352866`.
- Manifest: `4b0011a6961e86e6301e0ad849c4ef096bf74dfaeff23023bffce9172c182722`.

Fresh model, H64, Full path, both feedback directions, seed 42, f32 WGPU on the
AMD Radeon RX 9060 XT. This gives 49,172 parameters. Keep beta=.9, threshold=1,
triangular surrogate width=1, attached subtractive reset, trace decay=.8, and
memory decay=.9. Adam uses betas .9/.999, epsilon 1e-8, fixed learning rate .003,
global gradient norm clip 1, and no weight decay.

Train exactly **3,000 updates**, batch 16, sequence length 64. Draw independent
document-contained windows with replacement using the existing seeded sampler.
Reset all four recurrent states at each window; right padding has zero loss.
Nominal budget is 3,072,000 targets, approximately 16.6 times the training
corpus's distinct adjacent pairs. Record actual non-padding targets. No learning
rate tuning, early stopping, validation-selected checkpoint, synthetic language
data, or extra training is planned after seeing results. Validation is evaluated
only for measurement. The global gradient norm is checked every update; scalar
loss is checked at step 1, every 50 steps, and the final step. Detected nonfinite
values abort the run and are reported.

## Baselines and acceptance

Preserve `artifacts/liminal.bin`, the earlier H48 spiking checkpoint, SHA256
`4eb56bc9fff8709931313d12e49c1e59d784eccdc1487ff1a21e52a6023f6fca`, and the
separate sentence-memorization checkpoint. Save this run to
`artifacts/fineweb-language.bin`; this is a purpose-named artifact of the same
implementation, not another code/model implementation.
The baseline originally trained 16,708 parameters. The current loader imports
it into the same H48 model with zero memory parameters (33,076 allocated total),
preserving its predictions without providing trained semantic memory.

Before training, evaluate the earlier checkpoint on both complete splits at
reset lengths **32 and 64**, batch 16. Evaluate the trained checkpoint at those
same lengths, and record the new model's initial/final full-split metrics at its
training length 64. Compare with the additive-one unigram fitted only on training
targets (previously 3.213943 validation nats/character).

The primary prediction milestone is final held-out NLL at least **0.05 nats per
character below the earlier checkpoint at both reset lengths**, below its own
initial length-64 score, and below the train-fitted unigram. Lower is better.
Report character perplexity and train/validation gap. This is a development
validation set used in earlier experiments, not an untouched final test set.
One seed cannot establish variance or broad corpus generalization. Changes in
width, context, and training budget prevent attributing any improvement to
feedback alone. No architecture-selection claim follows from this run.

For qualitative before/after comparison, generate exactly 256 new characters
for every prompt below, at temperature **0 and .8**, sampling seed 42. Reset
state per invocation and retain all outputs; do not select the best sample.
Assess only newly generated characters, not the echoed prompt. Record spelling,
sentence coherence, repetition, and relevance separately; readable fragments
alone are not coherent replies. These prompts are an evaluation fixture, never
training examples added to FineWeb.

1. `The `
2. `Language allows us to `
3. `When I feel uncertain, `
4. `Once upon a time, `
5. `User: Hi Liminal! How are you today? Assistant: `
6. `User: I had a long day. How was yours? Assistant: `

Each checkpoint produces 12 continuations / 3,072 new characters. Generation
carries state beyond 64 characters, whereas training truncates it; this exposes
long-rollout behavior without proving long-context competence. There is no EOS,
conversation fine-tuning, external language model, or claim of fluent dialogue.

## Commands and artifacts

Run from the project root. Rebuild the current release entry point, then use the
same Rust CLI for training, evaluation, and generation:

```powershell
cargo build --release --bin liminal
./target/release/liminal.exe train --device gpu --data data/fineweb/train.jsonl --validation-data data/fineweb/validation.jsonl --checkpoint artifacts/fineweb-language.bin --hidden 64 --sequence-length 64 --batch-size 16 --steps 3000 --learning-rate 0.003 --gradient-clip 1 --seed 42 --feedback both
./target/release/liminal.exe evaluate --device gpu --checkpoint artifacts/fineweb-language.bin --data data/fineweb/validation.jsonl --sequence-length 32 --batch-size 16
./target/release/liminal.exe evaluate --device gpu --checkpoint artifacts/fineweb-language.bin --data data/fineweb/validation.jsonl --sequence-length 64 --batch-size 16
./target/release/liminal.exe generate --device gpu --checkpoint artifacts/fineweb-language.bin --prompt "The " --tokens 256 --temperature 0.8 --seed 42
```

Capture training logs in `artifacts/fineweb-language-train.log`, evaluations in
`fineweb-language-baseline-evaluate.log` / `fineweb-language-evaluate.log`, and
the complete fixed output suite in `fineweb-language-baseline-generation.log` /
`fineweb-language-generation.log`. Record actual commands, timings (including
which phases are timed), metrics, source/checkpoint/log hashes, and limitations
after the run. Check the saved model through the existing load/inference path.
No further model-code changes or alternate trainer are required for this run.

## Outcome

### Baselines measured before the run

The current debug executable evaluated the earlier checkpoint on all four
split/reset-length combinations. Each command completed with exit status zero,
on the recorded RX 9060 XT, and reproduced the earlier length-32 scores.
The subsequent fixed generation suite used the newly rebuilt release executable.
No source or model equations changed between these builds.

| Reset length | Baseline training NLL | Baseline validation NLL | Validation perplexity |
| ---: | ---: | ---: | ---: |
| 32 | 2.791621 | 2.747231 | 15.599385 |
| 64 | 2.788916 | 2.743646 | 15.543556 |

The corresponding primary final-score ceilings are 2.697231 at length 32 and
2.693646 at length 64. These are based on the printed six-decimal scores;
borderline differences at the final decimal would require finer precision.
Baseline logs are retained in the two `fineweb-language-baseline-*` files named
above. The source hash list is `artifacts/fineweb-language-source-hashes.txt`.
Toolchain: rustc 1.98.1 and Cargo 1.98.1. The current release entry-point build
took 1m46s, which is compilation time, not training time. Its SHA256 is
`36a2a763b4f21098b52159576e561eb77ed864258c5d4df42d778a46a79e80f2`.

The complete baseline prompt suite produced **0/12 coherent connected
continuations** and **0/4 relevant dialogue replies** in an independent informal
review, excluding echoed prompt text. All six greedy samples fall into frequent
function-word cycles such as `the`, `an`, and `ant`. All six temperature-.8 samples
reduce exact repetition but contain mostly malformed words and disconnected
fragments. Both sampled dialogue continuations begin `od aat the...` and do not
address the user. No numerical spelling percentage or calibrated benchmark
score is inferred from this qualitative review.

### Completed training and prediction milestone

The fixed run completed with exit status zero after **3,000 updates and
3,072,000 non-padding prediction targets**. The predefined held-out prediction
milestone **passed** at both reset lengths. These are character-level metrics
on the retained development split, rather than a fluency result.

| Reset length | Final training NLL | Final validation NLL | Final validation perplexity | Validation NLL improvement over baseline | Validation minus training NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2.421532 | 2.506498 | 12.261909 | 0.240733 | 0.084966 |
| 64 | 2.411103 | 2.511597 | 12.324597 | 0.232049 | 0.100494 |

The required margins were at least .05 nats/character. Improvements of .240733
and .232049 exceed those margins without depending on the final printed
decimal. Validation perplexity decreased from 15.599385 to 12.261909 at length
32 and from 15.543556 to 12.324597 at length 64. These comparisons use identical
data, vocabulary, reset lengths, batch sizes, and the current inference path.

For the fresh H64 model at its training reset length 64:

| Full-split metric | Before updates | After 3,000 updates |
| --- | ---: | ---: |
| Training NLL, nats/character | 4.990149 | 2.411103 |
| Training perplexity | 146.958249 | 11.146247 |
| Validation NLL, nats/character | 4.984031 | 2.511597 |
| Validation perplexity | 146.062032 | 12.324597 |

Final length-64 validation NLL is **2.472434 nats below its own initial score**
and **.702346 nats below the full-training-target unigram** (3.213943 nats,
perplexity 24.876995). Thus all components of the primary milestone passed.
The checkpoint was the final fixed-budget model, with no early stopping or
validation-selected weights. Validation was measured before/after training;
there is no intermediate validation learning curve.

### Sampled training-loss observations

The trainer retained batch loss, preclip global gradient norm, and cumulative
targets at update 1 and every 50 updates through 3,000. The table gives a
summary at fixed milestones; the complete 61 logged observations remain in the
training log. These are **sampled pre-update batch losses**, not full-training
or validation scores. They fluctuate because the sampler draws new windows.

| Update | Cumulative targets | Batch NLL | Preclip global gradient norm |
| ---: | ---: | ---: | ---: |
| 1 | 1,024 | 4.984723 | 0.603178 |
| 50 | 51,200 | 3.418523 | 0.434713 |
| 100 | 102,400 | 3.069081 | 0.662560 |
| 250 | 256,000 | 2.761226 | 0.850921 |
| 500 | 512,000 | 2.561502 | 0.870354 |
| 1,000 | 1,024,000 | 2.555277 | 0.815358 |
| 1,500 | 1,536,000 | 2.370774 | 0.868226 |
| 2,000 | 2,048,000 | 2.499344 | 0.672010 |
| 2,500 | 2,560,000 | 2.498018 | 0.654349 |
| 3,000 | 3,072,000 | 2.378032 | 0.628525 |

Several logged preclip norms exceeded 1, including 1.943750 at update 650;
the shared trainer applied its specified global clipping before Adam. The
logged values do not establish the total clipping frequency between log points.
There was no detected nonfinite-value failure. This does not imply a separate
scalar loss readback at every update; the checking schedule is recorded above.

### Actual budget, duration, and checkpoint reload

The training split contained **70 eligible original documents**, comprising
185,558 distinct within-document adjacent positions. No sampled-document coverage
counter was recorded, so this does not claim measured unique-document coverage.
Their minimum length was 321 characters,
so every sampled batch row supplied all 64 targets: 16 x 64 x 3,000 =
**3,072,000 target presentations**. This is about 16.56 times the number of
distinct training pairs, but it is not 16.56 complete epochs: sampling is with
replacement over eligible window starts, and window edges affect exposure.
Validation retained its eight documents and 14,340 targets, with four unknown
characters among 14,348 original characters. No new data was downloaded or
synthetic language example added.

The **whole training command took 328.928357 seconds** (about 5m 29s), including
GPU initialization, initial/final full-split evaluation, and checkpoint I/O.
This is not an isolated optimizer-update time or a training-throughput
benchmark. Execution used Burn 0.22, f32 WGPU, RX 9060 XT, fingerprint
`spirv_4098_30096`. The preceding 1m 46s release build was separate compilation
time.

The actual training command, with its captured log, was:

```powershell
./target/release/liminal.exe train --device gpu --data data/fineweb/train.jsonl --validation-data data/fineweb/validation.jsonl --checkpoint artifacts/fineweb-language.bin --hidden 64 --sequence-length 64 --batch-size 16 --steps 3000 --learning-rate 0.003 --gradient-clip 1 --seed 42 --feedback both 2>&1 | Tee-Object -FilePath artifacts/fineweb-language-train.log
```

The final checkpoint was separately reloaded through ordinary release CLI
evaluation on both complete splits at both reset lengths. All four commands
completed with exit status zero. Reloaded length-64 NLL matches the training
command's final metrics to the printed six decimals. Subsequent generation
commands also load this saved checkpoint, rather than an in-memory model.
The artifact contains weights, vocabulary, configuration, and feedback mode;
it does not contain Adam moments or sampler state for exact training resume.

### Retained artifacts and prior implementation checks

The data/manifest hashes recorded before the run remain the source provenance.
The retained source-hash list records model, trainer, CLI, data/checkpoint/runtime
and RNG sources plus Cargo.toml/Cargo.lock. All nine entries were rehashed after
the run and matched exactly. Source and model equations did not change for this
experiment. Artifact SHA256 values:

- [Final language checkpoint](../../artifacts/fineweb-language.bin): `41aca0870018d462344f2a128d4aadd0d4feda7de156d48f220094adc798572c`.
- [Training log](../../artifacts/fineweb-language-train.log): `e6751668866c4aed9b3f6920051925136378c87b257fd79194b9393f1a032ced`.
- [Baseline evaluations](../../artifacts/fineweb-language-baseline-evaluate.log): `9233f138ce8192dc0a09525a77be453e3150ec2b5fe68ae8db9221b597a9c90a`.
- [Reloaded final evaluations](../../artifacts/fineweb-language-evaluate.log): `97e04e6ff3fc4059670a31b28dac33ca2b83fd0b782222c85811c0e8489ac1e2`.
- [Complete baseline output suite](../../artifacts/fineweb-language-baseline-generation.log): `13fb7589c865a6b4e73638c37c6270dfb25003cac7d69994999d7713654279a9`.
- [Complete final output suite](../../artifacts/fineweb-language-generation.log): `86a83b91d3355dc3835ed767adde8243ba192891141883e9ba6286ea9d9ec282`.
- [Source/Cargo hash list](../../artifacts/fineweb-language-source-hashes.txt): `4ac610d64d77b1a47b9019a7712d3a2fe213b6e28f828f6f30de698cfe58074d`.

Both earlier checkpoints remain unchanged: `liminal.bin` retains SHA256
`4eb56bc9fff8709931313d12e49c1e59d784eccdc1487ff1a21e52a6023f6fca`, and
`fineweb-overfit.bin` retains
`6b115e1cceb2934c9a6581c0ae5bacfe30e7f210617fc6eade5e5e1568c1f630`.

The unchanged implementation previously passed **52 CPU tests**, formatting,
Clippy with warnings denied, the 45-parameter numerical surrogate-gradient
fixture, and explicit RX 9060 XT CPU/GPU parity for all three diagnostic paths.
Those are **prior checks**, documented in the
[memory diagnostic](2026-10-18-memory-diagnostic.md) and
[passage diagnostic](2026-10-18-fineweb-overfit.md); they were not rerun for this
zero-source-change training experiment. This run's new evidence is successful
fixed-budget optimization, finite reported metrics, checkpoint reload, and the
retained prediction/generation measurements.

### Qualitative continuation assessment

All 12 final continuations were retained and independently reviewed, excluding
the echoed prompts. The informal counts were unchanged from the baseline:

| Qualitative observation | Earlier checkpoint | Final checkpoint |
| --- | ---: | ---: |
| Coherent connected continuations | 0/12 | 0/12 |
| Relevant replies to the two dialogue prompts at both temperatures | 0/4 | 0/4 |
| Greedy continuations with severe frequent-function-word loops | 6/6 | 6/6 |
| Temperature-.8 continuations dominated by malformed/disconnected text | 6/6 | 6/6 |

All six final greedy outputs cycle around `the`, `and`, and `are`, with occasional
fragments rather than connected sentences. For example, the new characters
following `The ` begin `the the the the the the the and are...`. Temperature .8
reduces exact cycles but produces mostly malformed words, ungrammatical clauses,
and disconnected content. There are more plausible character fragments and
common words in places; no measured spelling percentage or calibrated language
benchmark supports a quantified spelling improvement.

Neither dialogue prompt receives a relevant greeting, an answer to the question,
or acknowledgment of the user's day. The sampled greeting prompt begins
`ed acter's lingag art in asstith, ha et eaiss...`; recognizable fragments do not
make it a conversational reply. Thus **conversation did not improve on this
fixed suite**, despite the lower held-out next-character loss. This is an
informal review of the specified outputs, not a validated dialogue benchmark.

### Quantitative decision and limitations

The trained current model improves next-character prediction on this retained
FineWeb development split by the predeclared margins. The result supports the
bounded language-training step, without identifying which of increased model
capacity, context, target exposure, or their interaction produced the gain.
It does not isolate an advantage from feedback or data diversity alone.

The corpus remains a small, source-prefix sample with reused development
validation documents. Domain isolation and near-duplicate exclusion were not
established, and one seed gives no estimate of initialization variance. The
64-character state-reset training horizon also differs from 256-character
autoregressive rollout. Lower teacher-forced loss is therefore a prediction
improvement in this setting, not demonstrated fluency, conversational ability,
long-context reasoning, or emotional/symbolic semantics.
