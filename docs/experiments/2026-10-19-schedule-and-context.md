# Learning-rate decay and longer training windows

## Question and protocol before training

The stage 1 model (learned decays plus spike-trace readout, see the
[architecture experiment](2026-10-19-architecture-proposal.md)) trains with a
constant Adam rate and 64-character truncated windows. This experiment asks two
questions:

1. Does cosine decay of the learning rate, from .003 at update 1 to .0003 at
   the last update, lower held-out loss at a fixed budget?
2. At matched compute (the same number of trained targets), do 128-character
   windows lower held-out loss? Slow learned decays (tau up to 32 steps and
   beyond) can only learn dependencies inside a window, and a 64-character
   window is a few words.

Data: `data/fineweb-large` (44,399,637 training / 4,722,810 validation
characters, V1868). Model: stage 1, H256, feedback both, seed 42, clip 1,
batch 64, f32 WGPU on the RX 9060 XT.

| Run | Window | Updates | Targets | Schedule | Checkpoint |
| --- | ---: | ---: | ---: | --- | --- |
| Baseline | 64 | 20,000 | 81,920,000 | constant | `stage1-large-h256.bin` |
| S | 64 | 20,000 | 81,920,000 | cosine | `stage1-cosine-h256.bin` |
| W | 128 | 10,000 | 81,920,000 | winner of S vs baseline | `stage1-w128-h256.bin` |

The baseline was already running when this protocol was written. It was started
as the 20,000-update follow-up to arm C1 and changes no setting here.

Metrics: validation CE in nats/character from `evaluate --batch-size 256` at
reset lengths 64 and 128 for every checkpoint, so each run is scored
with the same context policy. The train command's own final evaluation uses the
training window and is reported, but it is not the comparison.

Acceptance, fixed before training:

- S: adopt cosine as the default schedule if its reset-64 validation CE is at
  least 0.03 below the baseline's.
- W: adopt 128-character windows if W's reset-128 CE is at least 0.03 below the
  matched 64-window run's reset-128 CE, and W is no more than 0.02 worse at
  reset 64.

Generation: the same six prompts, greedy and temperature 0.8, seed 42, 256
characters, for the best checkpoint.

## Results

All three runs completed with exit 0. The queue log is `artifacts/gpu-queue.log`.
Wall times from log creation to last write: baseline 76.8 minutes, S 71.0,
and W 71.7, including
evaluations; their wall times overlapped other jobs only briefly. Validation
CE in nats/character on the 4,722,810-character split:

| Run | Train CE | Validation, reset 64 | Validation, reset 128 | SHA256 |
| --- | ---: | ---: | ---: | --- |
| Baseline (constant, window 64) | 1.670938 | 1.668781 | 1.651477 | `0f5e5681f6df8f6fc81a12688a24847d9565a1652b469fb73460bded00130db6` |
| S (cosine, window 64) | 1.639588 | **1.639160** | **1.619398** | `3a35758e1047dfa7ca98fc610c1848b66eb77ef69a9135eb73dd6f4940986d5e` |
| W (cosine, window 128, 10,000 updates) | 1.719231 | 1.732808 | 1.714331 | `0f04b6adc2de0c768d6bd25757e47247097adc78cf86005862e84516eac8c81f` |

- **S fails the predeclared criterion narrowly.** Cosine is 0.029621 below
  the baseline at reset 64 against the required 0.03 (0.032079 at reset 128,
  which was not the criterion). Decay helps slightly; the default stays
  `constant` and `--lr-schedule cosine` remains an option. Because cosine had
  the lower loss, W and the subword run used it, as the protocol specified.
- **W fails.** At matched targets, 128-character windows are 0.094933 worse at
  reset 128 and 0.093648 worse at reset 64. Doubling the window halves the
  number of updates at this budget, and the extra context does not make up for
  that. All character models score better with 128-character resets than with
  64 (by 0.017-0.020), so evaluation context helps even though longer training
  windows did not.

The stage 1 baseline also extends the
[architecture experiment](2026-10-19-architecture-proposal.md): with the same
20,000 updates, the previous model reached 1.935937 and stage 1 reaches
1.668781.

Its greedy generations (`artifacts/stage1-large-h256-generation.log`) loop on
`the state ... the company`. Samples at temperature 0.8 mix real words and some
phrases (`The part the management was a risk.`) without coherence.

Decision: keep 64-character windows and the constant default. Cosine is
recorded as a small, sub-threshold gain. The larger lever was tokenization;
see the [subword experiment](2026-10-19-subword-tokens.md).
