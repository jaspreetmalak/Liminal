# Current subword model: language-generation failure audit

## Fixed subject and question before scoring

The user reports substantial improvements by Claude and asks what the updated
language generation fails at. Assess the current model, without carrying over
the old H64 character-model conclusions. Perform no training, checkpoint
selection, parameter edit, vocabulary fitting, or decoding tuning in this audit.

Subject: `artifacts/subword-xl-h256.bin`, SHA256
`0f6db11e7e7ab5a3546d936690a58ef80446af6e7a1ee0dbd8439352c02c9b60`.
It completed training with exit 0 on 2026-10-20. H256, V9778, 6,000 learned
merges over 3,778 base characters, 10,351,410 parameters, both feedback paths.
The single spiking layer has learned per-neuron membrane/trace/memory decays
and a spike-trace readout. A proposed second layer has not been implemented.

Training used 90,000 FineWeb documents / 277,187,624 characters / 78,086,966
tokens, batch 64, independent 64-token windows, 16,000 updates, 65,536,000
sampled token targets, seed 42, cosine .003 to .0003, global clip 1. This is
about .84 corpus-token passes with replacement, not three million character
targets on 70 documents. Reported complete XL validation loss is 1.284331
nats/character versus 1.343949 for the earlier subword checkpoint on that same
split. Training CE 1.272059 is a 4,500-document sample, not the full training
split. A small train/validation gap alone does not isolate model capacity as
the bottleneck; the loss curve still declined near the training budget's end.

The prior `subword-xl-h256-decoding.log` already shows familiar words and
frequently plausible local syntax, unlike the older character-level garbling.
The remaining question is whether the apparent fluency survives controlled
facts, sustained topic, different decoding, and different random draws.

## Fresh generation protocol

Use the existing production `training::generate`, one loaded model/tokenizer,
zero state per call, and no alternate decoder. Record all outputs verbatim,
excluding echoed prompts, plus token IDs, input/output token counts, character
counts, seeds, settings, unknown counts, and consumed input horizon. Preserve
weights and checkpoint bytes before/after. Output creation must be exclusive
under the project's `artifacts/` directory.

Fix these 12 prompts before reading any fresh output:

1. `A public library gives people access to books, information, and a quiet place to study. For a small town,`
2. `To grow basil in a small pot, choose a sunny windowsill and keep the soil moist. The next step is`
3. `Mira put the blue key in a drawer. Owen put the pink key in a box. Later, Mira returned to collect her key. The color of Mira's key was:`
4. `Mira put the pink key in a drawer. Owen put the blue key in a box. Later, Mira returned to collect her key. The color of Mira's key was:`
5. `The box is not blue. It is pink. The color of the box is:`
6. `The box is not pink. It is blue. The color of the box is:`
7. `The dog chased the cat. The cat ran away. The animal that did the chasing was:`
8. `The cat chased the dog. The dog ran away. The animal that did the chasing was:`
9. `Write exactly the word blue and nothing else. Output:`
10. `User: I spent the afternoon planting basil and now I feel tired. Assistant:`
11. `A shop had 7 apples and received 5 more. The total number of apples is:`
12. `This guide explains how to plant and care for basil in a small pot. First,`

Generate 72 new **subword tokens** for prompts 1-11 and 256 for prompt 12.
Use five fixed settings: plain greedy (temperature 0, top-p 1, penalty 1,
seed 42); greedy with penalty 1.3; plain sampling (temperature .8, top-p 1,
penalty 1, seed 42); nucleus sampling (temperature .8, top-p .9, penalty 1.3,
seed 42); and the same nucleus setting with seed 43. There are **60 outputs**
and **5,240 generated tokens**, with no seed search or output-based revision.

Judge each setting using concrete text evidence:

- Is the first completed statement grammatical and meaningful in context?
- Do the next two completed sentences retain the topic and avoid contradiction?
- For 3-8, does the first relevant assertion retain the supplied fact, contradict
  it, or fail to answer? Do not count any later mention of a color as success.
- Does the continuation abruptly switch to advertising, privacy, citation, or
  other unrelated web templates?
- Is a sequence of at least four words repeated at least three times? Record
  other conspicuous repetitive structures separately.
- For the long basil continuation, distinguish early and later topic failure.

Do not score an incomplete last sentence/word caused by the fixed token cutoff
as an independent grammar failure. A new fictional name is not automatically a
malformed word or false fact. Do not combine the heterogeneous cases into a
general "understanding accuracy". Instruction, dialogue, and arithmetic are
separate applicability checks on a FineWeb completion model, not evidence that
it was instruction/chat trained. Exact one-word termination is also constrained
by the absence of an EOS token and the fixed-budget generation API.

The repetition penalty includes prompt tokens and may discourage necessary
repetition of a supplied name/color. Compare unpenalized settings before
attributing a penalized recall failure to the model alone. Training resets
after 64 subword input steps; inference processes all prompt/generated tokens
without that hard truncation. Report when a failure occurs within the trained
horizon versus only after it, without treating long inference as a new trained
capability.

## Controlled candidate likelihoods

Reuse the existing 64 paired binding/negation/order/semantic-role fixtures as
a separate frozen conditional diagnostic of this current checkpoint. These
are focused synthetic tests only. Report all four family case counts and strict
both-member pair counts. The prior H64 fixture results are not current-model
measurements and must not be reused as its scores.

For subwords, encode **the complete prefix plus candidate**, verify that its
IDs start exactly with `encode(prefix)`, and score the remaining candidate IDs
through the shared continuation scorer. Reject unstable token boundaries and
unknown characters. Record token counts: equal character lengths need not
imply equal token lengths. Use raw summed log probabilities, without repetition
penalty or temperature. Opposite-answer counterfactual pairs cancel a fixed
candidate preference; positive score sensitivity alone does not establish
consistent interpretation. Chosen templates are not a representative benchmark.

## Reproduction

```powershell
cargo test -j 1
cargo fmt --check
cargo clippy -j 1 --all-targets -- -D warnings
cargo run -j 1 --bin generation_probe -- --device gpu --output artifacts/generation-diagnostic-2026-10-20.json
```

Always select the current XL subject explicitly. The ordinary CLI still defaults
to the old `artifacts/liminal.bin`. The existing text/understanding CLIs retain
their original H64-corpus restrictions; this new audit reuses their generic
helpers without silently changing their historical subjects. Model, training,
and decoding implementation stay unchanged.

## Environment and execution

Windows x86_64, Ryzen 7 7800X3D, 32 GB RAM; Rust/cargo 1.98.1, Burn 0.22,
f32, WGPU selected AMD Radeon RX 9060 XT, device fingerprint
`spirv_4098_30096`, driver `32.0.31041.1004`. The diagnostic binary was built
with the normal unoptimized debug profile; this is not a throughput benchmark.

`cargo fmt --check` and all-target Clippy with warnings denied passed. The first
unrestricted parallel `cargo test` failed during compilation when Windows could
not allocate memory/map dependencies (paging-file error 1455). Rerunning with
`cargo test -j 1` passed all **87 library + 4 integration tests**; the explicit
GPU-gradient comparison remains ignored in ordinary tests. The model code was
unchanged and that hardware comparison was not repeated for this audit.

The fixed GPU diagnostic completed with exit 0: **60/60 continuations**, exactly
**5,240/5,240 requested new tokens**, and all 64 semantic pairs. No prompt had
unknown characters/tokens and no generated token was unknown. Run time was
114.098 seconds including loading, generation, scoring, and readbacks; the
whole Cargo command took 116.355 seconds. Semantic scoring took 6.045 seconds.
Zero optimizer updates; checkpoint and in-memory parameter hashes match before
and after. Model/training/tokenizer/checkpoint source hashes also remain unchanged.

Full unedited continuations, prompts, token IDs/counts, settings, and candidate
scores: [JSON report](../../artifacts/generation-diagnostic-2026-10-20.json).
Execution output: [run log](../../artifacts/generation-diagnostic-2026-10-20.log).
All outputs were inspected; the quotations below illustrate the full fixed run,
not selected successful seeds. Manual judgments were independently reviewed.

## Measured fact consistency

Raw summed candidate log probability, without sampling or repetition penalty:

| Family | Correct contexts | Both opposite-fact contexts correct | Constant-candidate baseline |
| --- | ---: | ---: | ---: |
| Entity/color binding | 16/32 (50%) | 0/16 | 50% contexts, 0% pairs |
| Negation | 16/32 (50%) | 0/16 | 50% contexts, 0% pairs |
| Before/after order | 16/32 (50%) | 0/16 | 50% contexts, 0% pairs |
| Active/passive semantic roles | 16/32 (50%) | 2/16 | 50% contexts, 0% pairs |
| Total | **64/128 (50%)** | **2/64 (3.125%)** | 50% contexts, 0% pairs |

Every family fails the existing focused criterion of >=75% context accuracy
and >=50% strict pair accuracy. **60/64 pairs retain the same candidate choice
despite opposite facts**. The choice matches its question-only preference in
100/128 contexts. This is weak conditioning on these facts, not a claim of
complete semantic absence: two role pairs do switch correctly. For example,
the model distinguishes `Ann thanked Bob` from `Ann was thanked by Bob` for
`Who gave thanks?`, but this does not generalize across the role fixtures.

The longest full prefix plus candidate is **22 subword tokens**; this failure
does not require exceeding the 64-step training horizon. Repeated/reordered
candidate branches agree exactly (maximum log-probability error 0, tolerance
2e-5), ruling out that scorer-state confound on the checked branches.

Candidate priors matter: ` red.` is two tokens, ` tan.` is three, and both
binding contexts usually prefer red. These are likelihoods of the exact
canonical token sequences, not calibrated probabilities of truth. Equal
character lengths do not eliminate tokenization/frequency preferences. The
paired test asks whether changing the supplied fact overcomes that preference;
per-token score normalization would define a different measure. The cases are
chosen, correlated synthetic fixtures, not an estimate of general comprehension.

## Free generation: manual results

| Test | Fixed outputs inspected | Result |
| --- | ---: | --- |
| Library, short basil, long basil (01/02/12) | 15 | 0/15 retain the stated subject through two subsequent completed sentences |
| Color binding, negation, chasing roles (03-08) | 30 | 0/30 give the supplied answer; all 30 are unrelated nonanswers |
| Exactly `blue` (09) | 5 | 0/5 begin with the requested word; stopping is separately unsupported |
| Basil/tiredness dialogue (10) | 5 | 0/5 give a relevant reply |
| Seven apples plus five (11) | 5 | 0/5 give 12/twelve; unrelated text rather than a demonstrated arithmetic calculation |

These descriptive counts cover this suite and settings only. In particular,
the dialogue/instruction/arithmetic cases are applicability checks on a plain
web-completion model, not held-out tests of instruction fine-tuning.

### Fact answers and prompt sensitivity

All five settings fail each of cases 03-08. Binding prompts contain 47 tokens,
negation 18, and roles 21. The initial nonanswer begins within the trained
horizon in all cases, including plain greedy and plain sampling without any
repetition penalty. Penalizing the prompt therefore cannot explain the entire
fact-answer failure.

- Swapping blue/pink in cases 03/04 gives **identical full greedy-penalty
  continuations**: `"I think it is very important and that I have been able to
  do this for me."` followed by unrelated text. It never answers the key color.
- Opposite negation prompts 05/06 give **identical full greedy continuations**:
  `The color of the color is the color of the color of the color.` followed by
  repeated `color`, rather than pink or blue.
- The dog/cat prompts lead to `I was a very good thing` or unrelated first-person
  anecdotes. They do not state which animal chased the other.

### Topic, usable meaning, and syntax

Claude's subword/data improvements produce many recognizable words and readable
local constructions. They do not yet establish useful multi-sentence completion.

- **01/library, greedy penalty:** `the city is located in the south of France,
  where they are not only one of those who have been married for children.`
  The next sentence says `The church has lived at the age of 18 years oldest
  daughter`. This mixes location, family, and building predicates immediately.
- **02/basil, greedy:** `a great way to get a good idea to get a good idea.`
  completes the prompt's `The next step is`. The syntax resembles prose but
  gives no planting step.
- **02/basil, greedy penalty:** starts `to be sure that you can’t even get it
  on your own`, then switches to popular brands and workouts. It fails despite
  mostly recognizable words and some grammatical clauses.
- **12/long basil:** all five settings are already off-topic in the first
  completed statement. Greedy switches to `the second half of the first half
  of the season`; penalty gives generic remarks about people/home/jobs; sampling
  splices unrelated engineering, money, sport, or medical prose. Later drift is
  present, but this suite cannot isolate a late-context failure from that early
  failure. The prompt is only 19 tokens.
- Word formation remains imperfect under sampling: case 02/plain has `aushm`
  modifying `cake`; case 12/plain has `scanghuling fit`; case 12/nucleus43
  has `absurdrated`. Fictional names and cutoff-final partial words were not
  counted as malformed words.

First-statement grammar is not uniformly absent. For example, the short basil
greedy/penalty sentences have plausible syntax but empty or unusable meaning.
This is why a word-validity or grammar-only score would miss the main problem.

### Repetition and decoding

Manual threshold: a sequence of >=4 whitespace-separated words appears >=3
times within a continuation, ignoring case/punctuation. **10/12 plain-greedy
outputs meet this threshold**: 01,02,03,05,06,07,08,10,11,12. Cases 04 and 09
have conspicuous shorter/less-frequent repetition but do not meet that exact
threshold. Repeated single-word runs in 01/05/06 also produce repeated
four-word sequences.

Examples include `to get the best possible` (02), `I was able to` (07),
`market share of the` (11), and `the second half of the season` (12).
The penalty improves variety, but the nonanswers, semantic mismatches, and
topic failure persist. Decoder changes alone have not demonstrated a repair
of fact-conditioned generation on this run.

### Dialogue, instructions, arithmetic, and stopping

The tired gardener receives `I'm not sure` loops, Creative Commons attribution,
news-like quotes, or dated fragments across the five settings; none acknowledges
the user's tiredness or basil. The 14-token `Output:` instruction gets unrelated
content from the start, so it has a content failure in addition to the separate
absence of a stop mechanism. The apples case has a 19-token prompt and produces
market/citation/table fragments. One sample starts `$399,000 to produce higher
prices`; this is not evidence that the model performed 7+5 and computed 399,000.

No EOS/turn-end token exists and the API always emits the requested count.
Consequently it cannot naturally emit one word and stop with this interface.
Fixed-budget final fragments were excluded from grammar judgments.

## Interpretation and next decision

The main remaining generation defect is weak control by supplied meaning:
relations, referents, and subject continuity often give way to common web-text
patterns. The current run demonstrates this even in short contexts. Merely
extending inference past 64 tokens would not address those initial failures.

Training's zero-state 64-token windows, a FineWeb next-token objective without
instruction/dialogue fine-tuning, and limited recurrent capacity are plausible
contributors. Of 10,351,410 parameters, 10,012,672 (96.73%) are input and output
vocabulary tables; the four recurrent state vectors each have width 256, and
recurrent/interface matrices total 327,680 parameters. Parameter count alone
does not establish the size or quality of relational processing. The model has
also seen only .84 corpus-token-equivalent sampled targets with replacement.
None of these associations isolates a unique cause or validates a particular fix.

Before judging another model change by nicer samples alone, keep this frozen
subject as a baseline and require better opposite-fact pair consistency plus
on-topic continuations. A subsequent causal training/architecture experiment
should compare matched data, exposure, seeds, and decoding. Chat fine-tuning and
EOS support address distinct behaviors and should not substitute for testing
ordinary completion. No retraining, architecture change, or claimed repair was
performed in this audit.

## Integrity record

SHA256, exact file bytes except the explicitly named f32-LE parameter digest:

| Artifact | SHA256 |
| --- | --- |
| JSON report | `252a7a6ea2f3fcdbfc46984cbe5d20ab8c42aa42c7cfaf8c6ded5af7d4ef71d7` |
| Run log | `b9ca3e5c842267295dc0199c7edf4e23146cd75213ac36f01cdf884cce677a64` |
| Checkpoint, before/after | `0f6db11e7e7ab5a3546d936690a58ef80446af6e7a1ee0dbd8439352c02c9b60` |
| Parameter values f32 LE, before/after | `7550fd663dc0afbdb84e15c5e91ba3ef6a6a68959e92dfef4f18b0c3e264ffed` |
| FineWeb XL manifest | `ccc802a22a002eab37d59b8c27d2039443c731d3d53efa6e23ace8399fa8d834` |
| Cargo.lock | `9f850010ca56a1b7d5eec143f635c0e444252233e226dcb096b7bff6ab883549` |
| src/model.rs, before/after | `77df0cf9ccc97fe220eb094a21c391709a6e7076c0d0b8f19207149082685600` |
| src/training.rs, before/after | `aaa7c9dd17b8aceef31e869fa3e5c34ddec4444ef72197302398f58b16a6cec4` |
| src/data.rs, before/after | `b94570f582402b6c388d39cc945385f7154ac1476cfec07a98ca4a6c9c6d3e17` |
| src/checkpoint.rs, before/after | `3955f62e26468227d8b5face243917620de7914fe33a212df455f094e20aaf6b` |
| src/generation_probe.rs | `54752cffc208c63b96670587770c0218835f780b575674b6fc675d2e338d13f9` |
| src/bin/generation_probe.rs | `0e0bcbb4eee75e6affc8cd4da43dfffc55290d2d9c92f73b07f32773f559af7b` |
| src/continuation_score.rs | `2c47aa585997520004a2c27125e2c7c17591b6ac9e419456fe04bbb45c10b402` |
| src/understanding_probe.rs | `2fea5e49e49124db13a1f36837f8125e482249d0e4df9ca6388b617378f3b445` |
| Debug diagnostic executable | `26c0de0c9e163c390daf1c2ba30b10c133b04ca1ada07d768f294e08df20631d` |
