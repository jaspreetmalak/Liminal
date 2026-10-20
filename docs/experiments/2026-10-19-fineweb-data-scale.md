# FineWeb data scale: does more text and compute fix generation?

## Question and protocol before training

The [width-scaling run](2026-10-19-width-scaling.md) showed that H256 trained on
185,628 characters reaches 2.341205 validation nats/character. Its train/validation
gap widened, and it reproduced training fragments, so data is a binding limit.
Question: with the model unchanged, does a ~250× larger FineWeb corpus plus a
matching compute budget substantially lower held-out loss and produce word-like
generation?

Corpus: `prepare-fineweb --output data/fineweb-large --rows 16000 --max-chars
60000000`, `sample-10BT` from offset 0, split seed 42. The row budget, not the
character budget, is meant to bind, so document selection is not biased toward
short documents. The existing `data/fineweb` is unchanged.

Contamination: the old 8 validation documents are rows 0-127, which this sample
also contains, so some may now sit in the new training split. All comparisons
therefore use the new validation split. Earlier checkpoints map characters
outside their 148-entry vocabulary to U+FFFD; the unknown rate is reported.

Model: unchanged H256 Liminal, feedback both, sequence length 64, learning rate
.003 (constant), clip 1, seed 42, f32 WGPU on the RX 9060 XT. Batch 64 instead of
16: a 50-step benchmark measured 0.236 s/step at batch 16 versus 0.279 s/step at
batch 64 (4,335 versus 14,705 targets/s), because per-step overhead dominates.
Budget: 20,000 updates = 81,920,000 targets, about 1.8 passes over the new
training split, estimated at 1.5-2 hours. Data and compute both change relative
to `width-h256.bin`; this run tests the combined lever.

Control (run after the main run if the GPU is free): the identical batch-64,
20,000-update budget on the original 185,628-character corpus. It separates the
effect of data from the effect of compute.

Training's own whole-corpus evaluations now use a fixed 256-window batch.
Windows are independent and target-weighted, so the batch size changes speed,
not results: H256 train CE was 2.165226 at both batch 16 and 256.

Acceptance, fixed before training:

1. Primary: validation CE on the new split is at least 0.30 nats/character
   below `width-h256.bin` on the same split.
2. Generation: same six prompts as `artifacts/fineweb-language-generation.log`,
   greedy and temperature 0.8, seed 42, 256 characters. Report them verbatim.
   Word-like English is expected near 1.0 nats/character; above that, judge it
   unproven. Lower loss alone does not establish fluent generation.

## Commands

```text
cargo run --release -- prepare-fineweb --output data/fineweb-large --rows 16000 --max-chars 60000000
cargo run --release -- train --device gpu --data data/fineweb-large/train.jsonl --validation-data data/fineweb-large/validation.jsonl --hidden 256 --sequence-length 64 --batch-size 64 --steps 20000 --learning-rate .003 --gradient-clip 1 --seed 42 --feedback both --checkpoint artifacts/fineweb-large-h256.bin
cargo run --release -- evaluate --checkpoint artifacts/width-h256.bin --data data/fineweb-large/validation.jsonl --sequence-length 64 --batch-size 256
```

## Results

### Corpus

The download first stopped at row 6,400 on HTTP 429 (rate limit) after three
quick retries, and nothing was published. Retries now honor `Retry-After` and
back off exponentially up to 10 attempts. The repeated command then completed:
16,000 rows inspected, 1 duplicate skipped, 0 over budget. That gives 14,399
training documents (44,399,637 characters) and 1,600 validation documents
(4,722,810 characters). Source revision `9bb295ddab0e05d785b879661af7260fed5140fc`.

Overlap with the original corpus, by `text_sha256`: all 8 old validation
documents are in the new training split, as expected. 3 of the 70 old training
documents are in the new validation split, a small advantage for checkpoints
trained on the old corpus.

The training-fitted vocabulary grew from 148 to 1,868 characters, so H256
has 1,764,684 parameters instead of 442,004. The extra parameters are almost
entirely embedding/readout rows for rare characters. Initial CE 7.545 matches
ln(1868) = 7.533.

### Held-out loss on the new validation split

Reset length 64, evaluation batch 256, 4,722,810 targets. Old checkpoints map
1,558 characters (0.033%) to U+FFFD.

| Checkpoint | Training | Validation CE (nats/char) |
| --- | --- | ---: |
| `fineweb-language.bin` | H64, small corpus, 3M targets | 2.494515 |
| `width-h256.bin` | H256, small corpus, 3M targets | 2.325994 |
| `fineweb-large-h256.bin` | H256, new corpus, 81.9M targets | **1.935937** |

The new model's final train CE is 1.943249, with essentially no
train/validation gap. **The primary criterion passes**: 0.390 nats below
`width-h256.bin`, against a required 0.30. Mean batch CE kept falling
throughout, without the small-corpus plateau: 2.874 over steps 1-1,000,
2.276 over 4,001-5,000, and 1.98-2.06 after 12,500. The run passes one epoch at
about step 10,800, so later batch CE includes previously seen text. The
training command took 84.8 minutes of wall time from log creation to last
write, including four whole-corpus evaluations.
Checkpoint SHA256 `cd9c7c648d33860fd95c5b2cd71f98ee544216386fdc46e55fc32491fc7a0aa2`.

### Generation

Full output: `artifacts/fineweb-large-h256-generation.log`. Greedy decoding now
emits real English words, but loops on frequent phrases, e.g.
`The become and the content of the company to the process and the process ...`.
Temperature 0.8 produces mostly real or near-real words and some local syntax,
e.g. `through a have wates and the ground to wher hove on`. It has no sentence-
level coherence and no relevant dialogue reply. Compared with the small-corpus
checkpoints' `the the the` loops and mostly invented words, this is clearly
more word-like, but **generation criterion 2 is not met**: at about 2.8
bits/character the model is still well above the ~1.0 nats/character level
expected for readable text.

## Outcome and decision

More data plus more compute gives the largest improvement measured so far
(0.39 nats) and moves output from invented words to real words. It does not
yield coherent language. With train and validation CE now equal, the
current H256 model is capacity-limited on this corpus rather than data-limited.
The next lever is the
[architecture proposal](2026-10-19-architecture-proposal.md), tested on this
corpus. Keep `data/fineweb-large` as the development corpus. The small-corpus
control below separates data from compute.

### Small-corpus control

First attempt: started with the same old-model binary, then stopped at update
4,350 of 20,000 to free the GPU for the architecture arms. Its partial log is
`artifacts/small-corpus-b64-control-stopped.log`. Batch CE was already 1.90 on
the 185,628-character corpus, which it had repeated about 96 times, so that
number is memorization, not held-out loss.

Completed run (`artifacts/small-corpus-b64-control.bin`, SHA256
`582ccf7c74866d7558e34344da2e6601542db04493400d3e2c2d4afcbe304552`): the same
pre-change H256 model, batch 64, and 20,000 updates (81.92M targets, about 441
passes over the small corpus). Train CE is 1.497014. Validation CE is 2.386565
on its own 8-document split and 2.377130 on the new 1,600-document split, with
1,558 characters (0.033%) unknown. Three of its training documents sit in that
split, which slightly favors it.

| Same model and budget | Train CE | New-split validation CE |
| --- | ---: | ---: |
| Small corpus (185,628 characters) | 1.497014 | 2.377130 |
| Large corpus (44,399,637 characters) | 1.943249 | 1.935937 |

**Data, not compute, explains the gain.** At identical compute the large
corpus is 0.441 nats/character better. On the small corpus, the extra updates
mostly memorize: train CE falls to 1.50 while validation (2.377) is worse than
the 3,000-update `width-h256.bin` (2.326).
