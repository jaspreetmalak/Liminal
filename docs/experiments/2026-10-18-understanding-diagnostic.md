# Frozen language-understanding diagnostic

## Question and protocol before scoring

The user asks why Liminal lacks natural-language understanding. Operationalize
one narrow component as choosing a continuation consistent with supplied facts
when assignments, negation, temporal order, or semantic roles change. This is
distinct from fluent free generation. A small synthetic fixture is a focused
test, not training data or a general comprehension benchmark.

Keep the sole Rust/Burn model, tokenizer, weights, neuron equations, surrogate,
reset, optimizer, and training configuration unchanged. Perform **zero updates**.
Inspect the actual FineWeb corpus and supervision, then run frozen tests of:

1. Whether supplied relationships change candidate preferences correctly.
2. How much real-text prediction benefits from additional ordered context.
3. Whether observed failures are consistent with earlier implementation,
   learning, corpus, and generation evidence.

Use three fixed subjects: H64/V148 CE control
`artifacts/fineweb-voltage-control.bin` (SHA256
`fc7b21d3565b242393c0cb330cd862daf75908ffb361b7848338821bad685ed0`),
H64/V148 voltage-regularized checkpoint `artifacts/fineweb-voltage.bin` (SHA256
`53cdccd0cf8198fcc54a4cfb16e6085b5b21052109f22ab6560454558fc40f88`),
and a fresh seed-42 model with the same configuration and vocabulary. Load no
external model. Run the same scoring implementation on WGPU; CPU tests verify
its masks, branching, likelihoods, and bounds. No decoding-temperature choice,
fine-tuning, coefficient search, or checkpoint selection is part of this work.

## Counterfactual continuation test

Fix four three-letter name pairs: Ann/Bob, Eve/Dan, Amy/Tom, Sam/Lee. Cross
two story entity orders and two queried roles in each of four families, giving
**64 pairs / 128 contexts / 256 complete candidate scores per subject**:

- Binding: `Ann wears red. Bob wears tan.` versus exchanged colors;
  complete either `Ann wears` or `Bob wears` with ` red.` / ` tan.`.
- Negation: `Ann is wet. Bob is not wet.` versus exchanged negations;
  answer `Who is wet?` or `Who is not wet?` with the two names.
- Order: `Ann came before Bob.` versus `Ann came after Bob.`;
  complete `First came` or `Last came` with the two names.
- Semantic roles: `Ann thanked Bob.` versus `Ann was thanked by Bob.`;
  answer `Who gave thanks?` or `Who received thanks?` with the two names.

Within every pair the query suffix and two candidates are identical; gold
answers are opposite. Include leading space and terminal period in each
five-character candidate. All complete prompt/candidate spans must fit 64
characters and contain no unknown token. Counterbalance first/last-mention and
constant-answer heuristics to exactly 50% per family. Assert labels with a
separate text parser and test uniqueness, balance, lengths, and shared suffixes.

For candidate a or b, score its **total teacher-forced log likelihood**, from
zero state for each independently batched branch. Predict using the larger
score; ties are not correct. Define D(c)=logP(a|c)-logP(b|c). If the first
context's correct answer is a, the signed pair interaction is D(c1)-D(c2);
reverse its sign when the first context's correct answer is b.

Report individual case accuracy, strict both-members-correct pair accuracy,
each raw margin, interaction sizes, and query-only preferences. Query-only
scores are common within a pair; subtracting them cancels from its interaction
and is not independent additional evidence. Tiny positive interaction alone
does not establish usable meaning. A descriptive focused pass requires at
least 75% cases and 50% strict pairs **in every family**. Do not tune prompts or
change criteria after inspecting results. These are chosen templates, not
independent samples for significance claims.

Repeat the identical first case and reverse candidate evaluation order to
detect state contamination. Bound scores and require unchanged parameter and
checkpoint hashes. Failure on unfamiliar question formats may reflect format
alignment; the binding family uses a declarative completion to reduce that
confound. No result establishes emotion modeling or general reasoning.

## Genuine FineWeb context diagnostic

Keep original source files, source IDs, and whole-document splits. Verify train,
validation, and manifest hashes against the existing preparation record. Fit
no vocabulary or counts on validation. Select **32 document-contained spans per
split**, each with 128 prefix characters and 16 true target characters, uniformly
over eligible starts with replacement using seed 42 independently per split.
Record all IDs, character offsets, target strings, unknown counts, and ordered
selection hashes. This gives **512 scored targets per split per condition**.

Score the identical 16 true targets using the final 1, 4, 16, 32, 48, 64, or
128 prefix characters, resetting all four states before each row. Additionally
shuffle the first 16 of the 32-character prefix, or the first 112 of the
128-character prefix, preserving the last 16 characters exactly. Use seeded
Fisher-Yates permutations fixed before model scoring. Preserve the character
multiset, target strings, and scoring positions. The short shuffle tests order
within the trained horizon; long shuffles are separately out-of-horizon.

Only suffix targets contribute to likelihood. Compute NLL as -sum(logP)/512.
All targets remain teacher-forced across variants; this test measures
conditional prediction, not free generation or a semantic benchmark. The
maximum consumed-input count is prefix length + 15. Thus prefixes through 48
are fully within the 64-input training horizon; 64/128-prefix conditions extend
beyond it and must be labeled accordingly. Shuffled text is a perturbation,
not an ordinary held-out data distribution.

Report absolute NLLs, paired changes from ordered 32-character context, and
per-window scores. Distinguish inability to retain context, failure to exploit
it for a relationship, and insufficient learned lexical prediction. No fixed
passive decay constant alone proves an effective memory limit.

## Reproduction and evidence

Keep one shared frozen continuation scorer using production `Model::step`.
No diagnostic implements a separate neuron recurrence or optimizer. New files
are organized by scoring, controlled cases, and experiment orchestration; the
binary is thin. Preserve all earlier artifacts and create the new report with
exclusive creation under `artifacts/`.

```powershell
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo run --bin understanding_probe -- --device gpu --output artifacts/understanding-diagnostic.json
```

Record environment, source/data/checkpoint/report hashes, actual examples and
measurements, elapsed time with its scope, alternative explanations, and
unverified areas. Mathematical correctness and falling character CE do not by
themselves demonstrate natural-language understanding.

## Results

### Pre-scoring audit: data and learning contract

The retained source is a bounded sample, not the full FineWeb corpus: 78
documents accepted from the first 128 source rows under the 200,000-character
budget. All retained documents come from CC-MAIN-2013-20. Train/validation
contain 70/8 documents and 185,628/14,348 Unicode scalars. Maximal ASCII letter
runs `[A-Za-z]+` number **29,546/2,409**; these are word-like units, while model
tokens remain individual Unicode scalars. There are 6,002 unique lowercase
training ASCII runs of at least two letters. Validation has 274 such types /
430 occurrences absent from that lexicon (18.7691% of eligible validation word
occurrences), despite having only four unknown *characters*.

Training document 33 (JSONL line 34, original source row 38), ID
`07f0d7eb-c5f6-4de5-b87d-ffedbc8204de`, source
`http://www.wordstream.com/larry-kim`, contains 33,891 characters: **18.2575%**
of training characters and **18.6737%** of the 181,148 eligible 64-input starts.
Of its 397 nonempty lines, 377 contain a parenthesized year followed by a dash
headline pattern; those lines contain 31,234 characters. It supplies all 35
literal `May 16, 2012` occurrences, all 12 `Google AdWord` occurrences,
131/141 `Facebook` occurrences, and 57/61 `Marketing` occurrences in training.
This establishes availability and frequent exposure of the generated motifs;
it does not establish exactly which sequences were memorized.

The document's UTF-8 text SHA256 is
`2e118f3f01edee78d674d379313dbd7e454e5610c5dbf611573d993deb999717`.
Its first `May 16, 2012` starts at Unicode-scalar offset 28,421; a snippet at
28,376 reads `Is Facebook really a good business?` followed by a TheDrum
date/headline entry. Source-derived facts also exist: its [48,85) span is
`Larry Kim founded WordStream in 2007.`. Other documents contain coherent
prose, technical discussion, product listings, forums, FAQs, and news. The
problem is neither an entirely meaningless corpus nor widespread exact
duplicate documents: only two normalized long lines repeat under the audit's
40-character cutoff, both inside that same WordStream document.

The five largest training documents contain 63,374 characters (34.1403%) and
receive 34.8080% of eligible-window probability. The top ten contain 87,224
(46.9886%) and receive 47.7974%. Validation is also concentrated: document 3,
source row 18, SeekingAlpha UUID `c6c03e75-ef3c-4e66-80ce-d937222af5e1`,
contains 7,313 characters, **50.9688%** of validation. Report context-sample
source IDs and avoid treating these few documents as a representative benchmark.

The 49,172-parameter H64 model was trained from scratch with 64-**character**
windows. Every sampled window resets u, s, r, c. The 64 equally weighted losses
have 1 through 64 available input characters, averaging **32.5**. Its 3,072,000
targets represent about 16.5555 times the corpus's 185,558 adjacent positions as
an aggregate exposure ratio; they are repeated overlapping samples, not that
many distinct characters or facts. No gradient crosses a window boundary.

The named semantic state is a 64-dimensional leaky tanh recurrence jointly
trained from scratch. Its only language supervision is next-character CE;
optional voltage regularization constrains activity. There is no pretrained
language component, fact store, semantic-role objective, affect label, or
curated conversation training phase in these checkpoint runs. Literal `User:`
and `Assistant:` occur zero times in their training corpus. This helps explain
why an assistant-formatted prompt is an unvalidated use, but does not imply
that next-token learning cannot acquire meaning given suitable data/learning.

Code review found no forced bag-of-characters operation or state reset between
characters: recurrence permits order dependence. Attached reset and all four
temporal state paths remain intact. Passive decay .9 has a 9.49-character time
constant, but learned recurrence can sustain information; it is not a proven
maximum memory span. Earlier delayed-bit recall succeeds, while a different
query-conditioned correction task underfit at 50% even on training. Those
focused tasks establish neither general comprehension nor a universal SNN
incapacity.

These audit facts identify constraints and hypotheses rather than a uniquely
established causal repair. The fixed frozen measurements follow.

### Validation, environment, and actual compute

`cargo test` passed 73 library tests and four integration tests. The hardware
gradient test is opt-in and was not counted as a CPU test; its earlier result
remains applicable evidence about unchanged model equations, not a new result
from this diagnostic. New tests independently parse the printed semantic facts,
verify all gold labels and 50% shortcuts, check causal target-only likelihood
against production loss and a scalar log-sum-exp of production logits, and test
padding, candidate branch order, duplicates, frozen weights, lengths, sampling,
and shuffle multisets. `cargo fmt --check` passed. Clippy initially requested
`args.as_chunks::<2>()` instead of `chunks_exact(2)`; that CLI-only replacement,
the two affected orchestrator tests, and final all-target Clippy all passed.
The scorer and model equations were not changed to improve an observed result.

The actual command was the preregistered debug-profile command, with console
output captured in `artifacts/understanding-diagnostic.log`. Its whole Cargo
command timer was 72.5211121 seconds, including the 11.62-second incremental
build and device/load/scoring/report I/O. The report's internal timer was
59.7296572 seconds through loading, scoring, and readbacks, before writing the
report file. Neither is a training or performance benchmark. Hardware was the
AMD Radeon RX 9060 XT, WGPU fingerprint `spirv_4098_30096`, Burn 0.22.0,
Rust/Cargo 1.98.1 on Windows. Device tensors and per-row likelihood accumulation
are f32; cross-row aggregation and reporting use f64. Exactly **zero updates**
were performed. CPU/GPU subject scoring uses the same maintained model step.

### Frozen fact-consistency results

Each cell below gives correct individual cases out of 32, followed by strict
both-members-correct pairs out of 16. Each family's constant and first/last
mention heuristics are exactly 50% on cases. A constant answer cannot solve
both members of an opposite-answer pair.

| Subject | Binding | Negation | Order | Semantic roles | Overall cases / strict pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE control | 15/32; 0/16 | 17/32; 2/16 | 17/32; 2/16 | 16/32; 0/16 | 65/128 (50.7813%); 4/64 |
| Voltage penalty | 13/32; 0/16 | 16/32; 2/16 | 16/32; 2/16 | 17/32; 1/16 | 62/128 (48.4375%); 5/64 |
| Fresh seed 42 | 14/32; 3/16 | 19/32; 3/16 | 15/32; 3/16 | 15/32; 2/16 | 63/128 (49.2188%); 11/64 |

Every family fails the predeclared 75% case / 50% strict-pair criterion for
every subject. The learned models do not demonstrate reliable short fact
consistency on these fixtures, even when the answers are supplied. This failure
does not depend on stochastic sampling or on a premise longer than the training
horizon. Binding uses declarative completions, reducing the question-format
confound; it fails too. Fresh weights occasionally solve pairs, demonstrating
why isolated successes or tiny signed interactions are not evidence of learning.
These few dependent templates support no statistical claim of equivalence,
significant degradation, or general NLU accuracy.

A concrete binding pair is:

```text
Ann wears red. Bob wears tan. Ann wears   -> correct: " red."
Ann wears tan. Bob wears red. Ann wears   -> correct: " tan."
```

Both candidate strings have five characters including space/period. For CE,
logP(red)-logP(tan) is -3.144730 / +.704949: it selects tan then red, both
wrong for Ann. The penalty model's margins are -1.677310 / -.147821: it selects
tan twice, only one correct. This is one illustrative failure, not proof that
every prediction follows a last-mentioned-entity rule.

### Context changes scores without reliably controlling answers

| Measurement | CE control | Voltage penalty | Fresh seed 42 |
| --- | ---: | ---: | ---: |
| Full-context choice agrees with query-only choice | 117/128 (91.4063%) | 102/128 (79.6875%) | 79/128 (61.7188%) |
| Counterfactual pairs with a changed preferred answer | 7/64 | 12/64 | 23/64 |
| Strict pairs solved | 4/64 | 5/64 | 11/64 |
| Repeat/reverse-branch maximum score error | 0 | 0 | 0 |

Every pair has a nonzero log-margin interaction, including binding pairs with
the identical character multiset. Thus the current model is not forcibly
order-invariant and does not completely ignore history. But changes often
leave the preferred answer unchanged or move it in the wrong direction. The
regularized checkpoint has more preference flips than CE without higher
overall fact accuracy. Sensitivity is not successful semantic interpretation.
Query-only agreement measures persistence of a completion preference; it does
not uniquely identify which internal pathway causes it. No hidden-state
representation decoder was trained or tested here.

### Genuine-text context results

The 32 training spans cover 19 documents and 512 unique target positions.
Validation spans cover six documents and 511 unique positions among 512 scored
targets; document 3 contributes 16/32 spans. There are no unknown characters in
these selected prefixes/targets. The ordered selection metadata hashes are
`33374a3aa31c42f7c87c47ce35d4a9bd95e6a8025419f70653118b7c78be8bdb`
(training) and
`8183a4b92fd2241f6e4cb0c955f9c2ffd3b40632dd66cddc217ded942f4a7867`
(validation). Records include exact Unicode-scalar offsets, raw prefixes and
targets, and shuffled token IDs. These are shared across all subjects.

Validation NLL on the identical 512 targets:

| Initial prefix / intervention | Maximum consumed inputs | CE control | Voltage penalty | Fresh |
| --- | ---: | ---: | ---: | ---: |
| Last 1 character | 16 | 2.356818 | 2.287570 | 5.010286 |
| Last 4 | 19 | 2.335827 | 2.254855 | 5.004188 |
| Last 16 | 31 | 2.333786 | 2.246777 | 4.994552 |
| Last 32 | 47 | 2.330152 | 2.226824 | 4.974252 |
| Last 48 | 63 | 2.323147 | 2.248874 | 4.981367 |
| Last 64, beyond training horizon | 79 | 2.334385 | 2.259122 | 4.987582 |
| Full 128, beyond training horizon | 143 | 2.337910 | 2.253640 | 4.985771 |
| Shuffle first 16 of 32, retain final 16 | 47 | 2.327620 | 2.273979 | 4.989340 |
| Shuffle first 112 of 128, retain final 16 | 143 | 2.362381 | 2.258236 | 4.986362 |

The learned checkpoints predict these true characters far better than fresh
weights. On the regularized checkpoint, shortening initial prefix 32 to 1
increases NLL by .060747; shuffling the older half of the 32-character context
increases it by .047155 despite retaining the last 16 characters and all prefix
character counts. This establishes useful ordered-context influence on this
sample, even though it did not establish fact binding. CE's corresponding
short shuffle changes NLL by -.002533, providing no ordered-context advantage
on that perturbation. More initial context is not monotonically better.

Crucially, all conditions teacher-force the same 16 true targets. Even the
"last 1" condition subsequently receives previous true targets and grows to
16 consumed inputs; it is not a model reset before every one of 512 predictions.
Do not infer a one- or four-character memory limit from this table. Prefixes
64/128 go beyond the trained input horizon during scoring. None of these
observations is a temporal Jacobian, a controlled capacity intervention, or
proof that the passive decay constant is the cause of a memory deficit.

Complete training-split scores and every per-window likelihood are preserved
in the JSON report. For example, training NLL at prefix32 is 2.302416 / 2.101058 /
4.977009 for CE/penalty/fresh. These sampled continuation measurements use
different targets/context policies from the earlier full-corpus CE; do not
substitute them for those complete-corpus results.

### Diagnosis, what was ruled out, and next discriminating test

The demonstrated problem is **weak fact-conditioned choices despite learned
character prediction and measurable context sensitivity**. A lower character
loss and improved firing statistics did not establish relational understanding.
This is broader than free-decoding corruption: supplied candidate answers do
not make these fixed relationships reliable. It is also narrower than a claim
that the architecture cannot process meaning or that no latent semantic signal
exists anywhere in its states.

The strongest observed constraints are the tiny, uneven training sample;
repeated exposure to headline templates; very short supervised character
contexts with all states reset; and an unvalidated 64-dimensional scratch-trained
memory. There is no separately pretrained or independently validated semantic
reasoning module. These constraints are plausible explanations for the observed
learning outcome, not individually isolated causal effects. Character prediction
can in principle learn semantic regularities; a special semantic loss is not
proved mandatory by this experiment.

No input/target shift, unexpected tokenizer substitution, answer-branch state
leak, forced order invariance, frozen-parameter mutation, or broken local
surrogate/reset derivative was found by this audit and its existing numerical
tests. That evidence does not rule out all optimization problems, weak temporal
credit, inadequate capacity, insufficient data, or an unsuitable training budget.
The fixtures also involve particular names and question formats; a failed
zero-shot completion does not uniquely distinguish those causes.

The next discriminating experiment should train the **same model** on a small,
shortcut-controlled relational fixture as a focused learnability test: first
check that it can fit training bindings, then test disjoint names/facts or
paraphrases. Failure to fit would prioritize optimization/capacity investigation;
successful fitting without transfer would prioritize representation/data
generalization. Such synthetic training must remain a focused test, not the
primary language corpus. Separately, any language-training change should expand
and inspect genuine FineWeb coverage, control document concentration, and test
longer contexts under matched budgets. No such repair or additional training
was performed or claimed here.

### Integrity and retained artifacts

- `artifacts/understanding-diagnostic.json`: full frozen report, SHA256
  `7cf71266320339dc41ac603c98685387f322f4ba7781f4bf68e048bb94d09471`.
- `artifacts/understanding-diagnostic.log`: command output/timer, SHA256
  `125b6138ef711e342e23fc93ab3dde6fffd196667362958e99f5c3eddf078ada`.
- `artifacts/understanding-final-source-hashes.txt`: final source/data/subject
  hashes immediately before scoring. The earlier `understanding-source-hashes.txt`
  precedes the Clippy-only CLI iterator correction.
- Debug executable SHA256:
  `d45303d7614aaa1312ac547fba1e82ec5c52e4bde9e35fd43239cb742cb73207`.
- Fixture ordered compact-JSON SHA256:
  `82f09d33e7cb726a347088f0609f6b2c1697d1ac9081a10815ae03e1bbd17c79`.

Both recorded checkpoint hashes match before/after. Parameter f32 little-endian
hashes also match before/after: CE
`f76f53a43492bc20b9672c0114b8dfb2727ae987243695661e44d188cee49606`,
penalty `04bd3a7f4de98c8dc535474ec44809c06b00682108976d8345ac0c69f0275c18`,
fresh `e1037c3cc094d46d310911e0141d00f09e0669af3300203d8c8fd57537892774`.
The production model and trainer hashes still match the completed voltage
experiment: model `83554c3a1b8ec4e03d0da755d0305c44b71702949956b9026f403ad69dac53c2`
and training `eef123d3caeb607adc74c9ddad953ce0aaa825e810b017ccf0e5816c6062b71e`.
