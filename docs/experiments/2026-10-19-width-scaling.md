# Width scaling: H64 versus H256

## Question

The [text diagnostic](2026-10-18-text-diagnostic.md) left width untested as a
cause of weak local prediction. The H64 baseline levels off near 2.4-2.5 batch
CE after about 1,000 updates, and its final train CE (2.411103) is
close to its validation CE (2.511597). That pattern points to underfitting.
Does 4× width (9× parameters) at an identical budget substantially lower
held-out loss and repair generation?

## Configuration

Same corpus (70 training / 8 validation FineWeb documents, V148), Burn WGPU f32
on the AMD Radeon RX 9060 XT, seed 42, feedback both, sequence length 64,
batch 16, 3,000 updates / 3,072,000 targets, learning rate .003, clip 1,
no voltage penalty. Only `--hidden` differs from `fineweb-language.bin`. One
seed only; initialization draws differ with width.

```text
cargo run --release -- train --device gpu --hidden 256 --sequence-length 64 --batch-size 16 --steps 3000 --learning-rate .003 --gradient-clip 1 --seed 42 --feedback both --checkpoint artifacts/width-h256.bin
cargo run --release -- generate --checkpoint artifacts/width-h256.bin --prompt "The " --tokens 256 --temperature 0.8 --seed 42
cargo run --release -- generate --checkpoint artifacts/width-h256.bin --prompt "The " --tokens 160 --temperature 0 --seed 42
```

## Results

Final losses come from the train command's own evaluation. Both runs used the
same settings.

| Width | Parameters | Train CE | Validation CE | Train/validation gap |
| --- | ---: | ---: | ---: | ---: |
| H64 (`fineweb-language.bin`) | 49,172 | 2.411103 | 2.511597 | 0.100 |
| H256 (`width-h256.bin`) | 442,004 | 2.165226 | 2.341205 | 0.176 |

Training was about 4× slower than H64 on this GPU.
Checkpoint SHA256: `42c2643d2e201139a244eb9b97f29a9487c5a25e2e2b852804f835e7316746ac`.

Greedy output for `The `: `The Cance Marke (May 16, 2012) - Facebook Google Ad
Markes Mating Marke Marketing ...`, ending in `and and and`. Temperature 0.8
drifted into an all-caps nonsense loop. The log is `artifacts/width-h256-generation.log`.

## Outcome and decision

Width helps, but only modestly: 9× parameters lowered validation CE by 0.170
nats/character. That is still far from the 2.012149 validation NLL of the
training-only five-gram, and far from the roughly 1.0 nats/character at which
character models produce word-like English. The train/validation gap widened, and
greedy output reproduced a training fragment ("Facebook Google"). The 185,628-
character corpus is now a binding limit alongside capacity. Generation remains
incoherent; width alone is not a fix. Next levers are more FineWeb data and
architecture changes to the fixed-time-constant, single-layer, binary-readout
bottleneck. Each requires its own predeclared experiment.
