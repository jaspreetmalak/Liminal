# Subword tokens: merges over characters

## Question and protocol before training

Character models make about five sequential predictions per English word and
see only 64-128 characters of trained context. Question: with the stage 1 model
unchanged, do subword tokens (characters plus learned merges) lower held-out
loss per character and make generation more word- and phrase-like?

Tokenizer (implemented in `src/data.rs`, Rust): base characters fitted on
training documents only, with U+FFFD as token 0. Pre-tokenization splits text
into runs of one character class: letters, digits, single spaces, other
whitespace, and other characters. The last space before a letter or digit run
joins it (" word"). Merge learning repeatedly merges the most frequent adjacent
token pair inside chunks, breaking ties toward the smallest token pair; merges
never cross chunks. Encoding applies merges in learned rank order. Decoding is
lossless except for unknown characters, which become U+FFFD as before.
Checkpoints (`LIMINAL\x03`) store base characters and the ranked merges.

Measured on `data/fineweb-large` with `--merges 6000`: 7,868 tokens (1,868
characters + 6,000 merges). Training uses 44,399,637 characters = 12,534,196
tokens (3.542 characters/token); validation uses 4,722,810 characters =
1,327,967 tokens (3.556). Merge learning and corpus encoding took about 7 s.

Run: stage 1 model, H256, feedback both, seed 42, clip 1, batch 64, window 64
tokens (about 227 characters), learning rate .003. It uses the schedule that
wins the [schedule experiment](2026-10-19-schedule-and-context.md). Budget:
matched training characters with the 20,000-update character runs
(81,920,000 targets): 81.92M / (4,096 x 3.542) = 5,650 updates, or 23,142,400
token targets. With V=7,868 the model has 4VH + 5H^2 + 5H + V = 8,393,660
parameters, mostly embedding and readout rows.

```text
cargo run --release -- train --device gpu --data data/fineweb-large/train.jsonl --validation-data data/fineweb-large/validation.jsonl --merges 6000 --hidden 256 --sequence-length 64 --batch-size 64 --steps 5650 --learning-rate .003 --lr-schedule <winner> --gradient-clip 1 --seed 42 --feedback both --checkpoint artifacts/subword-6000-h256.bin
```

Metric: validation cross-entropy in nats per character, total NLL divided by
the characters of all scored tokens. Each document's first token is context
only, as with characters. Each model is scored at its own trained context
policy, with resets every 64 tokens here and every 64 or 128 characters for
character models. Longer context per reset is part of what tokens provide.

Acceptance, fixed before training: adopt subword tokens for language training
if validation nats/character are at least 0.10 below the best character model
in the schedule experiment, taking each character model's best reset (64 or
128). Generation: the same six prompts, greedy and temperature 0.8, seed 42,
judged qualitatively, with 72 tokens (about 256 characters) per prompt.

## Results

The schedule experiment selected cosine (lower validation CE), so the run used
`--lr-schedule cosine`. It completed with exit 0 (log
`artifacts/subword-6000-h256-train.log`, about 66 minutes including
evaluations). It shared the GPU with other evaluations and the control for most
of that time, so the wall time is not a speed comparison. The model has
8,393,660 parameters, as predicted. Training covered 23,142,400 token targets,
about 1.85 passes over the 12,534,196 training tokens. Checkpoint SHA256
`cd52a6c66f280d8aa0782c2d5e3065e61aeb65020acfc12072ac5d069fb9c336`.

| Model | Validation nats/character | Note |
| --- | ---: | --- |
| Best character model (S, cosine, reset 128) | 1.619398 | 81.92M character targets |
| **Subword, 6,000 merges** | **1.335449** | 4.752273 nats/token, 3.5586 characters/token |

Train CE is 1.289584 nats/character, a train/validation gap of 0.046. That gap
is larger than the character models' (~0), so this larger-vocabulary model is
starting to fit training text specifically. **The acceptance criterion passes**:
0.283949 nats/character below the best character model, against a required 0.10.

Generation (`artifacts/subword-6000-h256-generation.log`) changes in kind:
whole sentences are mostly grammatical, for example
`The "country" is a good way to get a lot of money.`,
`We are looking for a new business and a new business.`, and
`I love my daughter in my hands and my mother.` Dialogue prompts get first-
person replies. Greedy decoding still loops (`I am not sure I am not sure`,
`- Search Engine`). Sampled text drifts between topics, invents names, and is
not coherent beyond a sentence or two.

Limitations: one seed. This comparison changes the token unit, the context
characters per reset (about 227 versus 128), and the parameter count (8.4M,
mostly rows for 7,868 tokens, versus 2.2M). The gain is the combined effect of
adopting tokens, not tokenization alone.

Decision: adopt subword tokens for language training. The recommended command
is in the README. Next levers: more data, since the train/validation gap has
opened; then larger or deeper models, given that the stage 1 width test showed
little gain at short budgets.
