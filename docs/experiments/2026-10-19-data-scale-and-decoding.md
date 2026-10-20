# Larger FineWeb sample for subword training, and decoding controls

## Question and protocol before training

The [subword model](2026-10-19-subword-tokens.md) reached 1.335449 validation
nats/character, but its train/validation gap opened to 0.046 after 1.85 passes
over 12.5M training tokens. Its greedy output loops. This experiment asks two
questions:

1. Does a corpus about 6× larger, with a matching compute increase, lower
   held-out loss for the unchanged subword model?
2. Do nucleus (top-p) sampling and a repetition penalty reduce the loops and
   malformed drift in generation?

Corpus: `prepare-fineweb --output data/fineweb-xl --offset 16000 --rows 100000
--max-chars 400000000`, `sample-10BT`, split seed 42. Rows 16,000-115,999 are
disjoint from rows 0-15,999, which `data/fineweb-large` and the original sample
used. The previous subword model therefore never trained on any new validation
document, and both models can be compared on the new validation split. The
character budget is set so that the row budget binds.

Training-split evaluation: whole-corpus scoring of a training split this size
would take over an hour per pass. `train` now scores the training split exactly
up to 4,000,000 tokens. Above that it scores an evenly spaced sample of every
k-th document, with k = ceil(tokens / 4,000,000), and reports the sample size.
Validation is always scored in full. Earlier runs had fewer than 4M training
tokens or reported full-split scores and are unaffected.

Run: stage 1 model, H256, feedback both, seed 42, clip 1, batch 64, window 64
tokens, learning rate .003 with cosine decay. 6,000 merges are learned on the
new training split only. Budget: 16,000 updates = 65,536,000 token targets,
about 0.8 passes if the split yields about 80M tokens; the actual token count
is recorded. Compute is 2.8× the previous subword run, so this tests data and
compute together. Checkpoint `artifacts/subword-xl-h256.bin`.

```text
cargo run --release -- train --device gpu --data data/fineweb-xl/train.jsonl --validation-data data/fineweb-xl/validation.jsonl --merges 6000 --hidden 256 --sequence-length 64 --batch-size 64 --steps 16000 --learning-rate .003 --lr-schedule cosine --gradient-clip 1 --seed 42 --feedback both --checkpoint artifacts/subword-xl-h256.bin
cargo run --release -- evaluate --checkpoint artifacts/subword-6000-h256.bin --data data/fineweb-xl/validation.jsonl --sequence-length 64 --batch-size 256
```

Acceptance (loss): the new model's validation nats/character on the new split
must be at least 0.05 below `subword-6000-h256.bin` scored on the same split,
with the same 64-token resets. Unknown-character rates are reported.

Decoding (qualitative): the new checkpoint generates 72 tokens for the same six
prompts with seed 42 under four settings:

| Setting | Temperature | Top-p | Repetition penalty |
| --- | ---: | ---: | ---: |
| greedy | 0 | 1 | 1 |
| greedy + penalty | 0 | 1 | 1.3 |
| sampled | 0.8 | 1 | 1 |
| nucleus + penalty | 0.8 | 0.9 | 1.3 |

The penalty applies to tokens among the last 64 context tokens. The same four
settings are applied first to the existing `subword-6000-h256.bin`, while the
corpus downloads, and then to the new checkpoint. Outputs are recorded verbatim
and compared qualitatively for loops, word validity, and local coherence. No
automatic quality score is claimed.

## Results

### Decoding on the existing subword model

Full output: `artifacts/subword-6000-h256-decoding.log` (6 prompts × 4 settings,
72 tokens, seed 42). Settings with temperature 0.8, top-p 1, and penalty 1
reproduce the earlier sampled outputs exactly. Qualitative comparison:

- **Greedy** loops on every prompt (`I am not sure I am not sure`, a list of
  `- Search Engine`, `the world ... the world`).
- **Greedy + penalty 1.3** removes the loops on all six prompts and produces the
  most fluent text, with long, mostly grammatical, plausible-sounding clauses,
  e.g. `It's just how much we had to do with some of our friends," said Brian,
  who has been in the past year and he'd be able to see what they're doing`.
  Content still drifts and does not answer the dialogue prompts. One prompt
  turns into citation boilerplate (`Copyright © 2013.`, `https://doi`).
- **Sampled** (0.8) avoids loops but has invented words (`Chocolaterong`,
  `Bulcood`) and abrupt topic changes.
- **Nucleus 0.9 + penalty 1.3** is more varied than greedy + penalty, with fewer
  malformed words than plain sampling. It still invents names (`Klevenance`,
  `Nalash`) and changes topic within a sentence or two.

For this model, greedy decoding with a 1.3 repetition penalty gives the most
readable output. None of the settings produces coherent multi-sentence content
or relevant dialogue replies. This is a qualitative judgment on 24 samples, not
a measured quality score.

### Larger corpus

Preparation completed with exit 0 in about 43 minutes, with retries handling
rate limits. Of 100,000 rows inspected, none were duplicates or rejected,
leaving 90,000 training documents (277,187,624 characters) and 10,000 validation
documents (31,490,045 characters). The source revision is
`9bb295ddab0e05d785b879661af7260fed5140fc`, the same as `data/fineweb-large`.
Tokenization learned 6,000 merges over 3,778 characters (V = 9,778): 78,086,966
training tokens and 8,881,157 validation tokens, 3.55 characters/token. The
model has 10,351,410 parameters. Training-split scores used an evenly spaced
4,500 of 90,000 documents. The 16,000 updates (65,536,000 token targets, 0.84
passes) ran at about 1.0 s/update with the GPU otherwise idle: 291.9 minutes
including evaluations. Checkpoint SHA256
`0f6db11e7e7ab5a3546d936690a58ef80446af6e7a1ee0dbd8439352c02c9b60`; log
`artifacts/subword-xl-h256-train.log`; queue log `artifacts/gpu-queue-xl.log`.

| Model, scored on the new 10,000-document validation split | Validation nats/character | Unknown characters |
| --- | ---: | ---: |
| `subword-6000-h256.bin` (44M-character corpus, 5,650 updates) | 1.343949 | 953 |
| **`subword-xl-h256.bin` (277M-character corpus, 16,000 updates)** | **1.284331** | 403 |

**The loss criterion passes**: 0.059618 nats/character lower, against a
required 0.05. Train CE on the sample is 1.272059, so the train/validation gap
fell from 0.046 to 0.012: the larger corpus removed most of the overfitting.
Mean batch CE per 2,000 updates, in nats/token, kept falling to the end:
5.457, 4.834, 4.709, 4.649, 4.616, 4.553, 4.519, 4.486. More updates would
probably still help. Data and compute changed together (6.2× the training
characters and 2.8× the updates), so the gain is not attributed to either one
alone. Relative to the cost, the gain is modest. With almost no train/validation
gap, the H256 model's capacity is now the more likely limit.

Decoding on the new checkpoint (`artifacts/subword-xl-h256-decoding.log`)
repeats the pattern above. Greedy still loops (`The Civil War` repeated,
`What is your favorite?`). Greedy + penalty 1.3 is again the most readable,
producing fluent generic web prose (`We are looking forward to our team members
who have experienced professionals with their skills`). One continuation turns
into a privacy-policy template ending in `Please contact us at
email@example.com`. Sampling still invents words and drifts. Qualitatively the
text is similar to the previous model's; the loss gain does not produce
visibly more coherent multi-sentence text.

## Decision

Keep `data/fineweb-xl` and `subword-xl-h256.bin` as the best current corpus and
checkpoint. Recommend greedy decoding with `--repetition-penalty 1.3` for
demonstrations; the default stays 1, which reproduces earlier outputs. Next
levers, in order: model capacity (wider H or the stage 2 second layer), now that
the gap is small; then longer training, since the curve has not flattened.
Both need substantially more compute than a single RX 9060 XT gives quickly.
