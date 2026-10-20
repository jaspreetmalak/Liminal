# Debugging garbled natural-language output

## Question and protocol before measurement

The H64 full Liminal checkpoint improves held-out character loss but generates
malformed words and repetitive greedy continuations. Diagnose the failure before
changing neuron equations, tokenization, optimization, or training budgets.

Use `artifacts/fineweb-language.bin`, SHA256
`41aca0870018d462344f2a128d4aadd0d4feda7de156d48f220094adc798572c`.
It has V148, H64, 49,172 parameters, both feedback directions, and was trained
for 3,000 updates / 3,072,000 character targets. Every diagnostic records its
loaded path/hash/configuration. The ordinary CLI's default checkpoint is still
the earlier `artifacts/liminal.bin`; that default must not select the model for
this investigation. The prior malformed candidate outputs explicitly loaded
the newer checkpoint, so the default does not explain those outputs.

All executable checks are Rust. The diagnostic calls the production Burn model,
state update, loss, and generation; there is no replacement SNN, optimizer,
training run, parameter update, or checkpoint write. Run f32 WGPU on the existing
AMD Radeon RX 9060 XT. Readbacks and summary arithmetic may use f64. Preserve
hard spikes, triangular surrogate, and attached subtractive reset.

## Discriminating checks

1. Verify prepared FineWeb file/manifest hashes, whole-document split isolation,
   saved vocabulary equality with the training-fitted vocabulary, and exact
   training tokenization roundtrip. Account for the four known validation
   unknown characters explicitly. Inspect actual text rather than JSON metadata.
   Audit input/target shift and prompt/sample feeding against the existing code.

2. Use a fixed deterministic sample of 64 document-contained 64-target windows
   per split: 4,096 targets each. Retain exact document/start selections, hashes,
   and sampled document coverage. Windows may overlap; they are not independent
   observations. Do not use sampled validation to fit any parameters.

3. On identical targets, compare the trained model with all recurrent history,
   the same weights with all state reset before every input token, and a fresh
   same-configuration seed-42 model. Measure NLL, argmax accuracy, prediction
   frequencies, and bins of window positions 1–8, 9–16, 17–32, and 33–64.
   Resetting states is an inference intervention and changes operating points;
   it is not a retrained capacity-matched control.

4. Fit one statistical character n-gram control on all training documents only.
   Its maximum order is five (at most four preceding characters), without
   crossing document boundaries. Root probabilities use additive-one smoothing.
   Higher contexts use the fixed interpolation rule
   `p(c|h) = (count(h,c) + 10*p(c|suffix(h))) / (count(h) + 10)`.
   Evaluate full available window context and current-token-only context on the
   identical sampled targets. This controls local orthography; it is neither a
   replacement production model nor a matched-parameter comparison. Report its
   context/count budget. Do not tune order or smoothing after observing scores.

5. Select actual validation passages deterministically from the first four
   eligible documents. Use prefixes of both 32 and 64 characters, followed by
   their next 32 reference characters (eight cases). The shorter condition keeps
   prediction within the 64-character training horizon; the longer one measures
   behavior beyond it. Compare teacher-forced predictions with greedy self-fed history.
   Record reference probability/rank, first reference mismatch, and outputs.
   Compare the manually stepped greedy path with ordinary `training::generate`.
   A reference mismatch is not automatically a language error: multiple valid
   continuations exist. Reference surprisal after divergent generated history
   is a diagnostic of altered conditioning, not ordinary held-out likelihood.
   A greedy continuation of the same statistical n-gram may additionally provide
   a local-pattern spelling control, using the same fixed count table and
   lexical metric; it does not demonstrate semantic capability.

6. Report descriptive lexical coverage using a training-only lexicon of ASCII
   alphabetic words of at least two letters. State case handling and boundary
   exclusions. Compare actual validation words, greedy output, and the existing
   fixed sampled output suite where applicable. This lexicon contains source
   noise and omits valid unseen words; a match is not a grammar or meaning score.

7. On the first 16 selected validation windows, record actual pre-reset voltage,
   firing/per-neuron rates, triangular-support occupancy, state RMS, and memory
   bound occupancy. Recombine spike, memory, and bias output terms from the same
   actual states to measure their centered logit magnitudes and conditional loss
   contributions. These are output interventions, not retrained independent
   components: memory already contains spike information and vice versa.
   Differentiate the first two windows through the production loss and report
   all 11 parameter-group gradient norms/maxima without updates. Repeated token
   embeddings prevent interpreting these group norms as isolated temporal
   credit. Check parameter hashes before/after all measurements.

## Interpretation rules

Separate a verified implementation fault from weak learned conditional
predictions, exposure to self-generated histories, a local-context bottleneck,
and state/activity pathologies. Report negative evidence as well as positive
findings. Firing/saturation or a nonzero gradient alone does not establish a
cause. Better n-gram scores would show unused learnable local structure, without
identifying whether width, optimization, or another design choice caused it.
Infer no universal need for another tokenizer or larger model from one checkpoint.

Garbled words already appear within the first 64 generated characters, so the
64-character training versus 256-character generation mismatch cannot alone
explain the observed failure. The one-sentence checkpoint's successful exact
generation is existing evidence against a universal shift/serialization error;
new diagnostic measurements must still examine the current language checkpoint.

Data is the retained 70/8-document FineWeb development sample, not an untouched
test corpus. Preserve all source data and earlier experiment artifacts. Record
actual commands, device, budgets, hashes, tests, measurements, limits, and the
resulting repair recommendation after execution. The frozen first-run results
are recorded below; the later activity controls are identified separately.

The prefix-32 control was added during independent protocol review, before any
of these metric comparisons ran, because prefix 64 plus a 32-character rollout
already exceeds the training horizon. The activity module was reviewed against
the production equations. A preliminary ordinary-CLI check on this checkpoint
produced byte-identical 64-character greedy text on Flex CPU and WGPU for `The `;
logs are `artifacts/text-diagnostic-cpu-generation.log` and
`artifacts/text-diagnostic-gpu-generation.log`. This is one forward-output check,
not a new all-parameter hardware-gradient test.

## Frozen first-run execution and data checks

The recorded invocation was
`cargo run --bin text_probe -- --device gpu --output artifacts/text-diagnostic.json`.
It completed with exit 0; the log records the debug binary invocation and an
incremental compilation of 1.71 seconds. The diagnostic's internal wall time was
42.9606881 seconds, including loading, scoring, generation, activity, and gradient
readbacks, but ending before JSON serialization/writing. This is neither a
training time nor a generation-latency benchmark. It ran on WGPU AMD Radeon RX
9060 XT (`spirv_4098_30096`), using f32 tensors and f64 host scoring of their
actual logits. There were zero optimizer updates and no checkpoint writes.

All prepared-data, split-isolation, and vocabulary checks passed. The retained
FineWeb source was `HuggingFaceFW/fineweb`, `sample-10BT`, observed revision
`9bb295ddab0e05d785b879661af7260fed5140fc`. Training contained 70 documents /
185,628 characters and roundtripped exactly with no unknowns. Validation
contained eight documents / 14,348 characters; its four unknown characters were
the only raw-text differences, and token roundtrip plus expected unknown
substitution checks passed. The saved V148 vocabulary exactly matched the
training-only fit. The observed revision identifies the preparation response;
the `/rows` endpoint did not support requesting that historical revision.

Seed 42 selected 64 windows per split with replacement, each with 65 characters
and 64 scored adjacent pairs. Training selections covered 32 distinct documents,
with two overlapping window pairs; validation covered all eight documents, with
27 overlapping pairs. Neither split had an identical sampled window. Exact
document indices, source IDs, starts, and window hashes are in the report.
These are 4,096 target presentations per split, not 4,096 independent examples.

For the frozen first run, the full suite passed: 56 library plus four integration CPU tests
(60 total), formatting, and Clippy with warnings denied. Tests include shifted
targets, full-path NLL against production loss, reset-each-input equivalence,
generation agreement, n-gram normalization/boundaries, lexical boundaries, and
output guards. The earlier CPU/WGPU neuron/gradient parity checks remain prior
validation; this run did not repeat a full H64 cross-backend gradient comparison.

## Matched next-character results

NLL is mean natural-log loss per character; accuracy is next-character argmax.
Each entry uses the same selected 4,096 targets in its split.

| Condition | Train NLL | Train accuracy | Validation NLL | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| Trained Liminal, full history | 2.384729 | 33.4717% (1,371) | 2.527064 | 31.0303% (1,271) |
| Same weights, reset before every input | 2.719624 | 24.8047% (1,016) | 2.705014 | 24.0479% (985) |
| Fresh same configuration, seed 42 | 4.986247 | 1.3428% (55) | 4.986292 | 0.9766% (40) |
| Training-only order-five n-gram | 1.249543 | 60.4492% (2,476) | 2.012149 | 47.5342% (1,947) |
| Same n-gram, current input character only | 2.586232 | 26.1963% (1,073) | 2.639225 | 25.6348% (1,050) |

Liminal uses history: removing it costs about 0.178 validation NLL and 6.98
accuracy points. Its full history also improves on the one-character n-gram.
However, the fixed five-gram improves on Liminal by 0.515 validation NLL and
16.50 accuracy points. The count model fit 185,558 within-document training
targets, storing 51,166 nonempty contexts and 116,313 context-next entries; it
is not parameter- or compute-matched to the 49,172-parameter network. This is
evidence of unused local character structure, not proof of a particular
architectural repair. The large count model's train/validation gap is retained.

Full-history position bins show no monotonic late-window loss explosion:

| Scored window positions | Targets per split | Train NLL / accuracy | Validation NLL / accuracy |
| --- | ---: | ---: | ---: |
| 1–8 | 512 | 2.501026 / 30.8594% | 2.522464 / 31.4453% |
| 9–16 | 512 | 2.333848 / 30.2734% | 2.605532 / 26.9531% |
| 17–32 | 1,024 | 2.313253 / 36.0352% | 2.452778 / 32.6172% |
| 33–64 | 2,048 | 2.404112 / 33.6426% | 2.545741 / 31.1523% |

Bins contain different characters and contexts, so this is descriptive rather
than a temporal-gradient test. Validation predictions concentrate on space
(1,114), `e` (614), `t` (574), `n` (331), and `a` (328): 2,961 / 4,096 outputs.
For example, a current space predicts `t` 363 times while the corresponding
true space-to-`t` pair appears 89 times. Common wrong true-next-to-predicted-next
pairs include `a`→`e` (79), `i`→`e` (72), `e`→space (71), and `o`→`e` (70).
The report retains ranked frequency/confusion tables; omitted entries are not
zero. Common transitions dominate without reliably disambiguating the next
character, consistent with the observed frequent-word loops.

## True prefixes and generated histories

All eight manual greedy rollouts matched ordinary production generation exactly.
Each row scores the same 32 true reference targets under teacher-forced history
and under greedy-generated history; the latter is not held-out likelihood after
the histories diverge. Rank is `1 + count(logit > reference_logit)`, with ties
sharing best rank. All per-position probabilities, ranks, and outputs are saved.

| Validation document index | Prefix characters | Teacher-forced NLL / accuracy | Reference NLL under greedy history | First reference mismatch position |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32 | 2.470498 / 21.875% | 4.667434 | 2 |
| 0 | 64 | 2.440843 / 31.250% | 3.902524 | 1 |
| 1 | 32 | 2.330295 / 31.250% | 3.796162 | 1 |
| 1 | 64 | 2.291215 / 31.250% | 4.698452 | 1 |
| 2 | 32 | 2.193717 / 37.500% | 3.665317 | 1 |
| 2 | 64 | 2.241653 / 37.500% | 4.989085 | 1 |
| 3 | 32 | 2.532597 / 21.875% | 4.025460 | 1 |
| 3 | 64 | 1.888568 / 50.000% | 3.929464 | 2 |

The report also excludes the initial mismatch position when scoring after its
generated character has actually been fed back; reference NLL is higher than
the paired teacher-forced score in all eight cases there too. This supports
exposure drift as a symptom, while weak teacher-forced scores show that it is
not the only problem. A reference mismatch alone does not establish an error in
an open-ended continuation.

The shorter prefixes produce loops within the training horizon. For document 2,
the 32-character true prefix is `Bible-black with a blinding whit`; its reference
continues `e logo raging across the chest. `, while greedy generation gives
` the the the the and the and the` and temperature 0.8 / seed 42 gives
`e en the seinfe al overicenstis.`. The failure therefore cannot be attributed
solely to requesting a continuation beyond 64 characters. Document 0's shorter
prefix produces ` 201012) - Facebook Google Marde`, showing that collapse is
not literally one identical loop for every context. The count control also
repeats phrases such as `the said should be about`; better local likelihood
does not establish good natural-language generation.

Across these eight short continuations, the 6,002-word training-only lexicon
gives reference hits 30/36 (83.33%), greedy hits 50/53 (94.34%), temperature-0.8
hits 16/40 (40.00%), and n-gram greedy hits 46/46 (100%). These are pooled counts
of eligible completed word occurrences, with the predeclared partial-word
exclusions, not distinct-word coverage or a spelling assessment. The especially
high greedy rate coexists with function-word loops; sampling exposes malformed
local combinations. Valid unseen words, names, noisy training entries, and
ASCII-only splitting limit interpretation. Neither lexical rate measures
grammar, relevance, coherence, or semantic understanding.

## Initial activity and gradient observations

The first 16 validation windows provide 1,024 scored positions / 65,536 neuron
observations. Spikes were verified binary. Mean firing was 62.5519%; one of 64
neurons always fired and none never fired on this sample. The saved individual
rates include many near-constant neurons, but no untrained activity comparison
was part of this frozen first run. Actual pre-reset voltages ranged from
-73.954201 to 54.867863. Strict triangular support `|voltage-1| < 1` was occupied
35.6995% of the time; mean surrogate slope, including zeros, was 0.191134.
Membrane RMS was 19.321151, spike-trace RMS 0.689746, and memory RMS 0.481739.
Memory magnitude exceeded 0.95 in 4.9088% of observations; this measures the
leaky memory state, not saturation of its unobserved tanh candidate.

On the same full-model states, centered vocabulary logit RMS was 3.215020 for
the spike projection, 1.280036 for memory, 0.104732 for bias, and 4.115700 for
their sum. Reconstructing the sum differed by at most 1.90735e-6.

| Same-state output recombination | NLL on 1,024 true targets | Argmax accuracy |
| --- | ---: | ---: |
| Full | 2.569001 | 29.5898% |
| Spike projection + bias | 2.666841 | 29.2969% |
| Memory projection + bias | 3.807114 | 14.0625% |
| Bias only | 4.773356 | 16.8945% |

Memory improves the combined output, and the spike projection has greater
centered magnitude. These are conditional output recombinations, not isolated
recurrent paths: both states already contain information from both feedback
directions. Their score differences do not justify removing either path.

The production masked-mean loss on the first two windows (128 targets) was
2.618139; its global parameter-gradient norm was 1.270195. All 11 groups had
finite nonzero norms: embedding 0.117952, recurrent 0.791339, hidden bias
0.130058, readout 0.592958, output bias 0.097515, semantic input 0.107590,
semantic recurrent 0.147407, spike-to-semantic 0.214826, semantic-to-spike
0.599537, semantic bias 0.037560, and semantic readout 0.393616. These
aggregated gradients exclude a completely dead graph on this batch; they do
not prove useful long-range credit, well-conditioned learning, or that every
neuron/parameter receives a gradient. Before/after parameter hashes matched.
Activity alone does not determine whether learning caused the operating point
or whether it causes garbled text; the separately declared follow-up addresses
part of that uncertainty.

## First-run interpretation and retained evidence

The data/roundtrip, exact checkpoint identity, shifted-target tests, production
greedy agreement, and one CPU/WGPU generation comparison found no tested
tokenization, serialization, generation-feeding, or backend discrepancy that
explains the garbling. This is bounded negative evidence, not a proof that all
implementation defects are absent. The current checkpoint has weak local
conditional prediction relative to a training-only five-gram, strongly favors
common transitions, and deteriorates when conditioned on its own continuations.
The observed state and surrogate operating points warrant controlled follow-up;
they do not yet identify the cause.

Prioritize improving and measuring within-word/local-context learning, with
fixed matched targets and real held-out-prefix rollouts. Establish the fresh
and generated-history activity controls before choosing a neuron-dynamics
change. A future repair needs its own predeclared training budget and generation
criteria; this diagnostic changes no weights or architecture and has not fixed
language generation. Width, tokenizer choice, larger data, and surrogate changes
remain untested causal explanations. (Width was later tested in
[the width-scaling run](2026-10-19-width-scaling.md): a modest gain, not a fix.) The small development split, reused for
earlier experiments, does not support an untouched-test generalization claim.

The immutable initial [JSON report](../../artifacts/text-diagnostic.json) is
423,223 bytes, with its [captured log](../../artifacts/text-diagnostic.log).
Paths below are relative to the project root; all digests are SHA256.

| Evidence | SHA256 |
| --- | --- |
| `artifacts/text-diagnostic.json` | `2617127c10d21820809b9fca73da28caeca3e44727d32e0467a2fda10fb61c2d` |
| `artifacts/text-diagnostic.log` | `965591bfce9f11c95ab37199a7afb06952850c4fa0cb3024fc0c22f7d8281e6d` |
| `artifacts/fineweb-language.bin` | `41aca0870018d462344f2a128d4aadd0d4feda7de156d48f220094adc798572c` |
| Loaded parameters, row-major little-endian f32, before = after | `4d91d7a9b79a33012a69b1ab7f26f7b14b6f5560430b966025fe73615ee2f154` |
| Fresh seed-42 parameters, same encoding | `e1037c3cc094d46d310911e0141d00f09e0669af3300203d8c8fd57537892774` |
| `data/fineweb/manifest.json` | `4b0011a6961e86e6301e0ad849c4ef096bf74dfaeff23023bffce9172c182722` |
| `data/fineweb/train.jsonl` | `ad0b069c93c70dc6fc52000edefbd272a574ae77dd7706ab763eff702ee2633a` |
| `data/fineweb/validation.jsonl` | `3ca429b3909a50a591286cb27f9b650527e2bd2846ffb22e5c460b20ad352866` |
| Training-fitted vocabulary UTF-8 | `47206a6e3e0505d360e3d967d2c7a60e485d65eea175e3fa7c8116cffce192c4` |
| Training selection manifest | `0569ce5217e77961c8a876f443fbeccf8cecf181a4a56eee01acca19147f7ed0` |
| Validation selection manifest | `c3c9ee909c33040c08ce233873fb71cf59954616a19cde3669b8000ac1c85860` |
| CPU and GPU ordinary 64-character greedy logs, each | `f7a54acd3c624dfe5ee40cf925ae1fcf1ae23e588b9927e5dcea55a579a08ce1` |

Selection hashes encode ordered document index, tab, character start, tab,
encoded 65-character window UTF-8 hash, and newline per sample. Re-execution
must choose a fresh JSON filename: the diagnostic refuses to overwrite an
existing report or checkpoint. The later activity-control source additions do
not retroactively change this initial report or its measured controls.

## Follow-up control declared after the first frozen run

The initial report completed at `artifacts/text-diagnostic.json`. Its matched
4,096-target validation sample scored 31.03% next-character argmax accuracy for
Liminal versus 47.53% for the fixed n-gram control. The first 16 neural activity
windows had voltage range about -74 to 55 and many near-constant neuron rates.
These results motivate an additional control before attributing that activity
to learning: run the same activity function on the same 16 windows with the
fresh, same-configuration seed-42 initialization. Add explicit per-neuron counts
above 95% and below 5% firing (strict inequalities).

Also compare activity on matched streams from the same four validation
documents: 32-character true prefix followed by 32 true, greedy, or temperature-.8
sampled characters. Each stream has 63 scored adjacent pairs. These summaries
include their common prefix; generated-stream losses score their own next
characters and must not be substituted for reference held-out likelihoods.
Record all controls in `artifacts/text-activity-control.json`, using the same
maintained probe implementation and unchanged parameters. The initial report
remains intact. No order, smoothing, weights, thresholds, or training setting is
changed. This follow-up was specified before running its new activity controls.

## Follow-up activity-control results

The separate invocation
`cargo run --bin text_probe -- --device gpu --output artifacts/text-activity-control.json`
completed with exit 0 on the same WGPU device and precision. It repeated the
original sampled metrics/prefix cases, which reproduced their reported scores,
and added the declared activity controls. Its internal diagnostic wall time was
60.340659 seconds, with the same loading/scoring/readback scope as the first
run; the separate debug compilation took 11.16 seconds. Neither is training
time. The initial report and candidate checkpoint hashes remained unchanged.

The five activity summaries cover 2,804 scored adjacent pairs in total: 1,024
trained and 1,024 fresh on the identical 16 windows, plus 252 for each of the
three matched four-document streams. Their gradient diagnostics use only the
first two windows of each summary: 128 + 128 + 3×126 = 634 target presentations.
These are repeated diagnostic observations, not new examples or training
updates. Every call had finite nonzero gradient norms in all 11 groups and
unchanged before/after parameter hashes. The fresh model's global gradient
norm was 0.699456 versus 1.270195 for the trained model on their identical
128-target batch. Nine filtered CPU tests passed after the diagnostic metadata
and follow-up additions; final formatting and Clippy checks also passed. The
earlier full 60-test run was before these final diagnostic-only additions.

| History / weights | Scored pairs | Mean firing | Neurons >95% / <5% firing | Strict surrogate support | Pre-reset voltage range | Membrane RMS |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Same 16 true windows, trained | 1,024 | 62.5519% | 24 / 10 | 35.6995% | [-73.954201, 54.867863] | 19.321151 |
| Same 16 true windows, fresh seed 42 | 1,024 | 18.5028% | 0 / 4 | 90.7684% | [-1.790848, 1.760414] | 0.533638 |
| Four prefix-32 + true continuations, trained | 252 | 61.8924% | 24 / 10 | 35.5779% | [-72.184479, 52.165684] | 18.237220 |
| Same prefixes + greedy continuations, trained | 252 | 61.5265% | 24 / 10 | 35.2121% | [-73.850548, 55.331802] | 17.937796 |
| Same prefixes + temperature-0.8 continuations, trained | 252 | 61.6939% | 24 / 10 | 35.8197% | [-73.573212, 54.022453] | 18.250853 |

This comparison establishes a training-associated change in the operating
point for this checkpoint, seed, and sample: much larger voltage/membrane
magnitudes, more nearly constant firing neurons, and less direct triangular
surrogate support than the fresh initialization. It does not prove that the
activity change caused weak language generation. Initial and trained weights
also differ in all their learned predictive functions; the fresh model's much
worse character loss rules out treating its activity as a demonstrated quality
target.

The same 24 nearly always-on and 10 nearly always-off neurons occur on true,
greedy, and sampled streams, with similar voltage/support summaries. The
operating-point shift is therefore already present with truthful histories and
is not unique to free-running generation. These stream summaries include the
common 32-character prefix (31 shared adjacent pairs per document); they do
not isolate continuation-only activity or establish that generated histories
have no further effect. Their own-stream output losses score different targets
and are not a quality ranking or reference held-out likelihood comparison.

The [control report](../../artifacts/text-activity-control.json) is 473,741 bytes;
its [log](../../artifacts/text-activity-control.log) retains the exact invocation.
SHA256 is `8dc28355307eba6b16dc5e99844d838cb5a7819d93b87b12d22ad30d4d685533`
for the JSON and
`8450e6dd723b0c4e4aa41cff6291c2b7c02592e81b206b1d4302bbbde7809def`
for the log. Model, corpus, vocabulary, and window selections match the first
report's recorded hashes. Both JSON reports remain separate immutable evidence.

## Resulting diagnosis and proposed next test

1. The strongest measured quality deficit is weak conditional local prediction:
   the trained network trails the fixed five-gram on the same validation
   targets and overuses common transitions. History helps, so an entirely
   unused recurrent state is not the explanation.
2. Training-associated voltage drift and nearly constant neuron firing are a
   concrete candidate mechanism. Many neuron observations lie outside the
   triangular surrogate's direct support, although the full loss still has
   nonzero gradients in every parameter group. Group norms alone do not measure
   useful temporal credit or prove a globally dead gradient.
3. Self-generated conditioning worsens reference prediction and produces loops
   or malformed sampled words, including within the training horizon. It
   exacerbates an already weak learned predictor; the neuron operating point
   is also shifted on true histories.
4. The bounded data/roundtrip, checkpoint, feeding, and CPU/GPU checks found no
   discrepancy that explains this failure. They reduce those specific concerns
   without excluding every implementation defect. Neither a larger tokenizer,
   greater width, nor a longer generation horizon is an established fix.

The next discriminating experiment should compare the unchanged training
objective with a training-only penalty on excessive pre-reset voltage range,
using matched fresh initial weights, data draws, update/target budgets, and
generation prompts. Keep hard forward spikes, the triangular surrogate, the
attached reset gradient, and the inference equations unchanged. Specify the
voltage band, penalty coefficient, seeds, and acceptance thresholds before
training; a possible penalty is the mean squared distance of pre-reset voltage
outside that fixed band, added to next-character cross-entropy. Measure both
activity and held-out/local-generation outcomes. Better voltage/support
statistics alone must not count as a language-quality success.

That A/B repair has not been implemented or tested in this diagnostic. The
current checkpoint has not demonstrated coherent natural-language generation
in the retained checks, and
these results establish neither symbolic/emotional reasoning nor a real-time
semantic feedback capability.
