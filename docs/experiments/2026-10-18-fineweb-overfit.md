# Genuine FineWeb passage fitting diagnostic

## Question and fixed protocol before training

Can the current full spiking/memory model fit a short genuine language sequence
when every next-character position supplies a loss, and reproduce its continuation?
This is an engineering memorization check alongside the sparse one-bit memory
diagnostic. It is not evidence of fluency, dialogue, emotional reasoning, or
generalization. It uses the same current model and shared Burn optimizer step.

Use the first prepared FineWeb training document. Skip pipe-prefixed forum header
lines and select the first complete sentence of at most 128 characters, ending
in `.`, `!`, or `?` followed by whitespace/end. Preserve the exact substring.
The inspected source yields:

> Don't care about Chloe/Taniel/Jen-Jen.

This is 38 Unicode scalar values (37 adjacent targets), starting at character
offset 95 in source document
`<urn:uuid:39147604-bfbe-4ed5-b19c-54105f8ae8a7>`, manifest source row 0.
Document text SHA256:
`f0e0dcc2aa1919b4b26c62f7f9349c770deac4c6c95c3cc44a2bf2b4704626d9`.
Exact passage SHA256:
`c12578c6e0015c9a703687d54a3432edea2b073dffd3c8a414611ec76b0f83ce`.
The source URL/dump metadata and corpus hashes must accompany actual results.

Fit vocabulary on all 70 prepared training documents only (expected 148 entries),
but update parameters using only this one passage. The vocabulary-only use of
other training documents must be reported; no validation content fits vocabulary.
Keep the existing eight whole validation documents disjoint from all training
documents and updates. Report their unknown rate (previously 4/14,348 characters).

Fixed settings: fresh model, full path, both feedback directions; H64; seed 42;
f32 WGPU on RX 9060 XT; batch 1; full passage window of 37 targets; zero all states
at each update. Adam betas .9/.999, epsilon 1e-8, learning rate .003, global clip 1.
At most 800 updates (29,600 adjacent-target presentations).

Check training every 25 updates and at the cap. End early only when teacher-forced
training accuracy is 100% and mean training NLL <=.05 nats/character. Validation
does not select a checkpoint or stopping time. Report final metrics even on failure.

Acceptance requires the training condition plus exact greedy continuation of all
30 remaining characters after the predeclared eight-character prefix `Don't ca`.
Also test greedy continuation from the first character `D`, over 37 new characters.
The latter is additional evidence of deterministic full-sequence reproduction.

Measure initial/final held-out NLL on all eight validation documents, resetting
at document and 37-target window boundaries and weighting actual targets, with
batch 16. Compare an additive-one unigram fitted only to the passage's targets
using the same vocabulary. Report unknown rate and generation from a genuine
held-out sentence prefix. Held-out improvement is not required for the fitting
milestone, and memorization must never be presented as general language quality.

## Reproduction and artifacts

```powershell
cargo run --release --bin language_probe -- --device gpu --steps 800 --output artifacts/fineweb-overfit.json
```

The report will include the exact source, selected passage, configuration, actual
update/target count, learning curve, losses/accuracy, generated text, and hashes.
Save this diagnostic's full-path checkpoint as `artifacts/fineweb-overfit.bin`.
The existing `artifacts/liminal.bin` remains unchanged. FineWeb remains the primary
language corpus; no synthetic text is used in this diagnostic.

## Measured result and acceptance

The predefined **passage-memorization criterion passed**. The first passing
training check was at **325 updates**, with **37/37 correct next characters**
and mean training NLL **0.0415610075 nats/character**, below .05. Greedy decoding
from the fixed eight-character prefix reproduced **all 30 remaining characters
exactly**. The additional first-character test also reproduced all 37 remaining
characters exactly. This is an end-to-end fitting result on one selected
training sentence, not a language-quality result.

| Measurement | Initial fresh model | Final model |
| --- | ---: | ---: |
| Training next-character accuracy | 1/37 = 2.702703% | 37/37 = 100% |
| Training NLL, nats/character | 4.979500771 | 0.0415610075 |
| Training perplexity | 145.401775 | 1.042437 |
| Held-out document NLL, nats/character | 4.995702714 | 8.318885498 |

The passage-target unigram scores **4.308366296 nats/character** on the same
held-out documents. Final model validation NLL is **3.323182784 nats worse than
initial** and **4.010519202 nats worse than that unigram**. Validation therefore
does not show improved language generalization. Its deterioration is consistent
with specialization to the repeated sentence; it does not invalidate the
separate memorization acceptance criterion.

### Actual learning curve and budget

All checks use post-update teacher-forced training metrics, with the complete
37-target window and zero initial state. Validation did not select stopping.

| Update | Correct next characters / 37 | Training NLL |
| ---: | ---: | ---: |
| 0 | 1 | 4.979500771 |
| 25 | 10 | 2.386868000 |
| 50 | 25 | 1.592934489 |
| 75 | 35 | 1.014185071 |
| 100 | 36 | 0.666646242 |
| 125 | 36 | 0.394628525 |
| 150 | 37 | 0.232932627 |
| 175 | 37 | 0.161070123 |
| 200 | 37 | 0.117739476 |
| 225 | 37 | 0.088000581 |
| 250 | 37 | 0.072003365 |
| 275 | 37 | 0.052886393 |
| 300 | 37 | 0.050536357 |
| 325 | 37 | 0.0415610075 |

Accuracy first reached 100% at a recorded check at update 150, but that alone
did not satisfy the NLL gate. At update 300 the post-update NLL was still above
.05; the first observed joint pass was update 325. Checks every 25 updates do
not establish the exact earliest possible passing update.

The actual budget was **325 full-passage updates and 12,025 next-character
target presentations** (325 x 37), using batch 1 and seed 42. There are only
37 distinct adjacent target positions; repetition is not additional corpus
diversity. The model had **49,172 parameters**, H64, and V148. All 70 training
documents supplied vocabulary characters only; only this 38-character sentence
entered weight updates. The report confirms disjoint source IDs and whole
document text across training/validation.

Evaluation covered all eight independent validation documents: **14,348
characters and 14,340 within-document adjacent targets**. The unknown-character
rate was **4/14,348 = 0.02787845%**. Windows reset after 37 targets and at document
boundaries; partial windows were masked and weighted by actual targets, in
batches of 16. The unigram used additive-one smoothing over V148 and counted
only the sentence's 37 targets, excluding its first character.

Training took **20.8124502 seconds**, including training checks and the final
parameter readback. This excludes initial/final full-validation evaluation,
file preparation, and compilation. Execution used Burn 0.22, f32 WGPU on
**AMD Radeon RX 9060 XT**, fingerprint `spirv_4098_30096`. These measurements
do not establish individual response latency, a speedup, or real-time behavior.

### Exact greedy outputs

Temperature was zero. With prompt `Don't ca`, the expected and produced
30-character completion was:

```text
re about Chloe/Taniel/Jen-Jen.
```

With prompt `D`, the expected and produced 37-character completion was:

```text
on't care about Chloe/Taniel/Jen-Jen.
```

Both yield the complete original sentence, with 100% positional agreement and
full-string equality. Generation lengths were fixed to the known remaining
sentence lengths; this does not test learned EOS or stopping behavior.

The unchanged held-out sentence was `If you have set yourself on fire, do not
run.` From its eight-character prompt `If you h`, the model produced:

```text
If you hent aboe/Taniel/Jen-Jen.en..e........
```

Its expected 37-character completion was `ave set yourself on fire, do not run.`
Only **5/37 positions matched (13.513514%)**, and full-string equality failed.
The insertion of memorized names into this unrelated prompt illustrates why
the fitting result must not be described as fluent text generation.

## Provenance, artifacts, and checks

The prepared corpus was original **HuggingFaceFW/fineweb**, configuration
`sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`. No new download
or dataset rewriting occurred. The selected text remains an exact substring of
the training document identified in the preregistration, at byte and character
offset 95. Its source is [the recorded forum post](http://daytimeroyaltyonline.com/single/?p=8906650&t=8780053),
dump `CC-MAIN-2013-20`, crawl date `2013-05-18T05:48:59Z`. Its original JSONL row
SHA256, excluding the line ending, is
`74792501dad89ebea652ec9d0912daf502d16cf916f3eef58f487debb8f5fa4d`.

The held-out example is an unchanged 45-character substring at offset 107 in
validation source row 1, document
`<urn:uuid:ba819eb7-e6e6-415a-87f4-0347b6a4f017>`, from
[the recorded blog comment](http://endogenousretrovirus.blogspot.com/2007/11/if-you-have-set-yourself-on-fire-do-not.html?showComment=1196270520000).
Its source text SHA256 is
`1b244f26c854cb012a0b81274f5b85e3393d5394b56ffc0ca8d7437dc4c7f95e`;
excerpt SHA256 is
`14dca27ad1c42b704699b15911c61df226a395900ac9c7968349276b30564fcc`.
No held-out characters or labels fitted the vocabulary or model.

SHA256 integrity values:

- Prepared training JSONL: `ad0b069c93c70dc6fc52000edefbd272a574ae77dd7706ab763eff702ee2633a`.
- Prepared validation JSONL: `3ca429b3909a50a591286cb27f9b650527e2bd2846ffb22e5c460b20ad352866`.
- Prepared manifest: `4b0011a6961e86e6301e0ad849c4ef096bf74dfaeff23023bffce9172c182722`.
- Initial row-major f32 little-endian parameters: `e1037c3cc094d46d310911e0141d00f09e0669af3300203d8c8fd57537892774`.
- [Full experiment report](../../artifacts/fineweb-overfit.json): `64723b132b5ce702e670f02d4d564bfd8c2b89c6015d7e71e1792132d16cec21`.
- [Training/output log](../../artifacts/fineweb-overfit.log): `264d6e4d3e848cc04e2e33a103063e6763d162559d75ba5f425964b916652478`.
- [Diagnostic checkpoint](../../artifacts/fineweb-overfit.bin): `6b115e1cceb2934c9a6581c0ae5bacfe30e7f210617fc6eade5e5e1568c1f630`.

The combined diagnostic release build completed successfully in **3m 53s**:

```powershell
cargo build --release --bin memory_probe --bin language_probe
```

That is compilation time, separate from the 20.8124502-second training
measurement. The actual experiment command completed with exit code 0:

```powershell
./target/release/language_probe.exe --device gpu --steps 800 --output artifacts/fineweb-overfit.json 2>&1 | Tee-Object -FilePath artifacts/fineweb-overfit.log
```

Ordinary inference independently reloaded the saved diagnostic checkpoint:

```powershell
./target/debug/liminal.exe generate --device gpu --checkpoint artifacts/fineweb-overfit.bin --prompt D --tokens 37 --temperature 0
```

That command produced the same complete sentence exactly. Its
[captured output](../../artifacts/fineweb-overfit-checkpoint-generation.log) has
SHA256 `7fb8694b49192f4a35f5bf6ef4faeed9cc799aac1690608797fb92d3e822cb32`.
This confirms the saved weights work through the existing inference path; the
checkpoint contains model/vocabulary/configuration, not Adam or sampler state.

Implementation validation passed: **52 CPU tests**, `cargo fmt --check`, and
`cargo clippy --all-targets -- -D warnings`. The four language-probe tests cover
unchanged sentence selection/Unicode offsets, provenance and split isolation,
training-only vocabulary, adjacent targets/reset behavior, and bounded protected
outputs. The separately executed RX 9060 XT parity test passed for full,
memory-only, and spikes-only paths on the shared 45-parameter numerical fixture;
its details are in the [memory diagnostic](2026-10-18-memory-diagnostic.md).

The existing `artifacts/liminal.bin` was not replaced. Its SHA256 remains
`4eb56bc9fff8709931313d12e49c1e59d784eccdc1487ff1a21e52a6023f6fca`, verified
again after this run.

## Decision and limits

The current Rust/Burn full model and shared learning implementation can fit this
genuine next-character sequence and reproduce it from a short prefix. That
removes a blanket inability to optimize any natural-text sequence as the
explanation for earlier failures. It does not establish why the earlier sparse
query-conditioned correction task failed or prove that feedback improves
learning: this test has one sentence, one seed, dense supervision, a large
parameter-to-example ratio, and no matched feedback ablations.

The next language-development work needs broader training data and held-out
improvement, preserving the exact SNN math and measured baselines. This
specialized diagnostic checkpoint is evidence of memorization, while its
validation loss and unrelated continuation demonstrate the remaining
generalization gap. No fluency, conversational, emotional, or general semantic
capability has been established.
