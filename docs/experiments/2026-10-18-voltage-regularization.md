# Training-only voltage regularization trial

## Hypothesis and fixed protocol before training

The user authorized trying the repair proposed by the frozen text diagnostic.
The preceding H64 checkpoint has weak conditional character prediction and many
nearly constant firing neurons: 24/64 fire above 95% and 10/64 below 5% on the
recorded 16 validation windows. Its pre-reset voltages range roughly -74 to 55;
strict triangular-surrogate support is 35.7%, versus 90.8% at its fresh seeded
initialization. These associations do not establish a causal quality fix.

Test whether discouraging excessive pre-reset voltages during training improves
both prediction and activity, while preserving the SNN's inference equations,
hard binary spikes, normalized triangular surrogate, and attached subtractive
reset. No temperature tuning, tokenizer change, external model, new data, or
architecture/width change is part of this experiment.

## Objective and exact mathematical definition

For actual pre-reset voltage v[j,t,i], threshold theta, surrogate width w, valid
target mask M[j,t], N=sum(M)>0, and H hidden neurons, define:

```text
a = (v - theta) / w
q = max(0, abs(a) - k)
P = sum_(j,t,i) M[j,t] * q[j,t,i]^2 / (N * H)
L_train = L_next_character_CE + lambda * P
```

Fix **k=3** and **lambda=.001** before either training run. At theta=w=1,
the unpenalized voltage band is **[-2,4]**. This band is deliberately wider than
the surrogate's direct support (0,2); the penalty does not force all neurons
to fire or all voltages into that support. P is dimensionless. Its direct local
voltage derivative outside the band is
`2*M*q*sign(v-theta)/(N*H*w)`, and zero within/on the band. Multiply by lambda
for its contribution to total loss. All temporal dependencies remain attached.

The same production step supplies CE and actual voltage in one shared loss loop.
One backward pass differentiates the total, one global clip covers all 11
parameter groups, and one Burn Adam update follows. Exclude padding from both
terms; divide P by valid targets times H. Disabled lambda 0 takes the original CE
path without computing an auxiliary graph. Its logged raw penalty is 0 because
measurement is disabled, not because voltages were measured within the band.
Initial/final corpus evaluation, perplexity, and all inference use **plain CE**,
never the regularized total. Checkpoints retain the unchanged inference format.

Numerical scope: the auxiliary square is computed before multiplying by the
mask. As with other finite-f32 calculations, extreme normalized voltages can
overflow; a padded-only overflow would yield `0*Inf`, not a safely masked zero.
This trial has no padded rows, and checks every combined gradient norm for
finiteness. Analytic masking tests establish finite-range values/gradients;
they do not establish overflow-safe masking for arbitrary external Batch values.

## Matched data, configuration, and budget

Use the existing FineWeb sample-10BT revision
`9bb295ddab0e05d785b879661af7260fed5140fc`. Keep the 70 training documents /
185,628 characters and 8 whole-document validation records /14,348 characters
unchanged. Fit V148 only on training documents. Validation has 4 unknown
characters and 14,340 adjacent targets; training has 185,558 adjacent targets.

- Training JSONL SHA256: `ad0b069c93c70dc6fc52000edefbd272a574ae77dd7706ab763eff702ee2633a`.
- Validation JSONL SHA256: `3ca429b3909a50a591286cb27f9b650527e2bd2846ffb22e5c460b20ad352866`.
- Manifest SHA256: `4b0011a6961e86e6301e0ad849c4ef096bf74dfaeff23023bffce9172c182722`.

Run two **fresh** conditions on f32 Burn 0.22/WGPU, RX 9060 XT:

| Setting | CE control | Voltage trial |
| --- | ---: | ---: |
| Initialization and sampling seed | 42 | 42 |
| Hidden width / vocabulary | 64 /148 | 64 /148 |
| Feedback / path | both /Full | both /Full |
| Updates / batch / sequence length | 3000 /16 /64 | 3000 /16 /64 |
| Learning rate / global clip | .003 /1 | .003 /1 |
| Adam beta1 /beta2 /epsilon | .9 /.999 /1e-8 | .9 /.999 /1e-8 |
| Voltage lambda /band widths k | 0 /3 | .001 /3 |

Keep membrane decay=.9, threshold=1, surrogate width=1, trace decay=.8, memory
decay=.9, no weight decay. Every condition has 49,172 parameters. Match initial
parameter bytes, vocabulary, sampler sequence, and all hyperparameters except
lambda. The sampler draws document-contained windows with replacement; zero all
four states for each window. Minimum training document length 321 makes every
row full: **3,072,000 targets per condition**, **6,144,000 total**. No validation
updates, early stopping, checkpoint selection, learning-rate search, coefficient
search, or extra updates after seeing outcomes. Execute GPU jobs sequentially.
This is one exploratory paired seed, not a multi-seed confirmation or a runtime/
energy comparison. Separate inference artifacts do not create another code path.

Preserve the earlier CE artifact `artifacts/fineweb-language.bin` (SHA256
`41aca0870018d462344f2a128d4aadd0d4feda7de156d48f220094adc798572c`), old
`liminal.bin`, and sentence diagnostic. Save fresh control to
`artifacts/fineweb-voltage-control.bin` and trial to `artifacts/fineweb-voltage.bin`.
Compare fresh lambda 0 with the earlier CE artifact for exact reproducibility;
if values differ, report it and use the new matched pair for decisions.

## Predeclared measurements and acceptance

Evaluate both final checkpoints through ordinary save/load on the same complete
train/validation splits, batch 16, reset lengths 32 and 64. Record plain CE,
character perplexity, and validation-minus-training gap. Use the same frozen
`text_probe` selections, counts/backoff baseline, eight real-prefix cases,
lexical definitions, activity summaries, and seed42 as the preceding diagnostic;
explicitly supply each checkpoint path and its measured SHA256. Keep all prior
reports immutable. Comparisons must match target sets and reset lengths.

Record the same six fixed prompts as the diverse-document run, each 256 new
characters at temperatures 0 and .8, seed 42. Keep all 12 outputs per condition.
Assess connected coherence, repetition, and relevance excluding echoed prompts.
Separately record the descriptive complete-word lexical hit fraction on the
eight real-prefix temperature-.8 samples. Lexical hits alone are not coherence.

The following thresholds are fixed before either run:

1. **Operating-point improvement:** at least 25% fewer neurons firing above 95%
   on the same 16 true validation windows, and at least 10 percentage points more
   strict surrogate-support occupancy than the fresh CE control.
2. **Prediction improvement:** full validation CE at least .05 nats/character
   below control at **both** reset lengths, plus at least 3 percentage points
   higher argmax accuracy on the matched 4096-target validation sample.
3. **Lexical improvement:** at least 10 percentage points higher pooled training-
   lexicon hit fraction on complete eligible words in the eight real-prefix
   stochastic continuations, with all numerator/denominator counts reported.
   These few correlated snippets do not establish a spelling benchmark.
4. **Coherence:** report whether any of the 12 fixed full continuations forms
   coherent connected text, and whether any of 4 dialogue outputs addresses its
   prompt. An unchanged zero count must be reported even if other gates pass.

Support for the targeted mechanism requires gates 1 and 2; count gate 3 separately
as a local-generation check. Improved firing/voltage without prediction gains is
not a quality success. Do not claim garbling fixed or conversation achieved
without corresponding output evidence. If any prediction outcome worsens,
record it without changing lambda or running additional unplanned candidates.
Capacity versus other optimization effects may remain unresolved even on a win.

## Implementation and reproduction

Add the optional objective in the single model/training implementation. Retain
CE-only public losses for evaluation and existing probes. Extend the same train
CLI with `--voltage-penalty` and `--voltage-band-widths`, defaults 0 and 3. Log CE,
raw P, lambda*P, total, gradient norm, and cumulative targets at the established
step 1/every 50/final schedule. Check zero-lambda values/updates, auxiliary analytic
and numerical gradients, mask normalization, and existing neuron/gradient tests.
Verify the new objective on actual WGPU as well as CPU.
Progress uses six decimal places; a small positive weighted penalty can round to
displayed zero even when its coefficient and gradient are active.

```powershell
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --release --bin liminal --bin text_probe
./target/release/liminal.exe train --device gpu --data data/fineweb/train.jsonl --validation-data data/fineweb/validation.jsonl --checkpoint artifacts/fineweb-voltage-control.bin --hidden 64 --sequence-length 64 --batch-size 16 --steps 3000 --learning-rate .003 --gradient-clip 1 --seed 42 --feedback both --voltage-penalty 0 --voltage-band-widths 3
./target/release/liminal.exe train --device gpu --data data/fineweb/train.jsonl --validation-data data/fineweb/validation.jsonl --checkpoint artifacts/fineweb-voltage.bin --hidden 64 --sequence-length 64 --batch-size 16 --steps 3000 --learning-rate .003 --gradient-clip 1 --seed 42 --feedback both --voltage-penalty .001 --voltage-band-widths 3
```

Capture separate control/trial train, evaluation, text-probe, and generation logs
under `artifacts/`. Report source, data, checkpoint, report/log hashes; exact
commands; actual target/update budgets; and wall times with their measured scope.
Pure update throughput is not inferred from a whole-command timer that includes
GPU initialization, full evaluation, readbacks, and checkpoint I/O.

After training, reload each checkpoint and evaluate the four corpus/reset
combinations below. All evaluation targets are scored once per combination.
Use the same release executable and run GPU commands sequentially:

```powershell
foreach ($condition in @('voltage-control','voltage')) {
    foreach ($reset in @(32,64)) {
        foreach ($split in @('train','validation')) {
            ./target/release/liminal.exe evaluate --device gpu --checkpoint "artifacts/fineweb-$condition.bin" --data "data/fineweb/$split.jsonl" --sequence-length $reset --batch-size 16
        }
    }
    $checkpoint = "artifacts/fineweb-$condition.bin"
    $hash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    ./target/release/text_probe.exe --device gpu --checkpoint $checkpoint --expected-checkpoint-sha256 $hash --output "artifacts/text-diagnostic-$condition.json"
}
```

The fixed generation suite uses the same six literal prompts for both conditions
and both temperatures; each command starts with zero recurrent state and its own
sampling seed 42:

```powershell
$prompts = @('The ', 'Language allows us to ', 'When I feel uncertain, ', 'Once upon a time, ', 'User: Hi Liminal! How are you today? Assistant: ', 'User: I had a long day. How was yours? Assistant: ')
foreach ($condition in @('voltage-control','voltage')) {
    foreach ($temperature in @(0,.8)) {
        foreach ($prompt in $prompts) {
            ./target/release/liminal.exe generate --device gpu --checkpoint "artifacts/fineweb-$condition.bin" --prompt $prompt --tokens 256 --temperature $temperature --seed 42
        }
    }
}
```

## Results

Implementation checks passed before either training condition:

- `cargo test`: 64 library and 4 integration tests passed. The hardware test was
  separately invoked rather than counted as an ordinary CPU test.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo test --test gpu -- --ignored --nocapture`: passed on the RX 9060 XT.
  For the active auxiliary fixture (lambda=.17, k=.25), CPU/GPU
  `[CE, P, total]` were `[1.1219785, .020043984, 1.125386]` /
  `[1.1219784, .020043982, 1.1253859]`. Maximum error across all 45 parameter
  gradients was `1.3969839e-9`, below the predeclared `3e-5` tolerance.
  The original three path-specific CE/state/gradient checks and exact hard spike
  and surrogate endpoint checks also passed.
- Independent frozen-tangent finite differences verify all 45 fixture parameters
  for both CE and augmented loss, with nonzero initial states and masked unequal
  lanes. Analytic tests cover band boundaries, nonunit width, direct gradient
  outside surrogate support, valid-target/hidden normalization, and padding.
  Zero coefficient preserves exact CE, gradients, updates, and sampler draws.

Both implementation and fixed-budget training are evaluated separately from
language quality; no repair outcome is claimed from implementation checks.

### Environment and reproducibility observations

Release build completed in 3m 27s. Toolchain: Rust 1.98.1
`48a229cea 2026-09-01`, Cargo 1.98.1 `797e8a9bc 2026-08-05`, Burn 0.22.0,
Windows, WGPU discrete AMD Radeon RX 9060 XT, fingerprint
`spirv_4098_30096`. Source/data hashes before training are retained in
`artifacts/fineweb-voltage-source-hashes.txt`; exact release executable hashes
are in `artifacts/fineweb-voltage-binary-hashes.txt`.

The fresh CE control agrees with the earlier run's displayed step-1 CE/norm,
but its later GPU trajectory is not exactly identical: at update 50 it has
CE 3.419888 rather than 3.418523. CPU zero-coefficient tests establish exact
within-binary loss/gradient/update equivalence. They do not establish bitwise
GPU training reproducibility across builds or processes. The cause of this
small initial divergence is not isolated; hard threshold recurrence can amplify
floating-point differences. As preregistered, quality decisions use the new
matched-budget pair, not the older checkpoint as the control.

### Fresh CE control training

The zero-penalty run completed exactly 3,000 updates and 3,072,000 targets.
Whole-command time was 330.2095684 seconds, including initialization, initial
and final corpus evaluation, updates, progress readbacks, and checkpoint I/O.
Initial full training/validation CE at reset 64 was 4.990149 / 4.984031; final
training/validation CE was 2.408780 / 2.508364 (validation perplexity 12.284821).
All combined gradient norms were finite. Progress was logged at 61 selected
updates; logs do not establish the clipping frequency over all 3,000 updates.

Checkpoint SHA256:
`fc7b21d3565b242393c0cb330cd862daf75908ffb361b7848338821bad685ed0`.
It differs from the older CE checkpoint; its final reset-64 validation CE is
.003233 nats lower. This small difference is not a regularization effect.

### Voltage-penalty training

The fixed lambda .001 / k=3 run completed exactly 3,000 updates and 3,072,000
targets. Whole-command time was 404.0693891 seconds with the same scope as the
control timer. This was not a pure optimizer-throughput benchmark; both runs
used the same target budget rather than a matched time budget.

Initial full training/validation CE at reset 64 was again 4.990149 / 4.984031.
Final training/validation **plain CE** was 2.255314 / 2.408742 (validation
perplexity 11.119958). All combined gradient norms were finite. The final sampled
batch had CE 2.264918 and raw P .001232; its weighted penalty rounds to .000001
at displayed precision. No early stopping, retuning, or extra updates occurred.

Penalty checkpoint SHA256:
`53cdccd0cf8198fcc54a4cfb16e6085b5b21052109f22ab6560454558fc40f88`.

### Reloaded full-corpus prediction

All four corpus/reset combinations used the ordinary evaluation CLI, batch 16,
with each file loaded independently. There are 185,558 actual training targets
and 14,340 validation targets in each respective complete-corpus measurement.
The reset-64 reload results agree with training's final plain-CE measurements.

| Condition | Reset | Training CE | Validation CE | Validation perplexity | Validation minus training CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh CE control | 32 | 2.419050 | 2.509806 | 12.302541 | .090756 |
| Voltage penalty | 32 | 2.274922 | 2.408202 | 11.113955 | .133280 |
| Fresh CE control | 64 | 2.408780 | 2.508364 | 12.284821 | .099584 |
| Voltage penalty | 64 | 2.255314 | 2.408742 | 11.119958 | .153428 |

Validation CE improves by .101604 and .099622 nats at resets 32 and 64,
respectively. Reset-64 perplexity falls by about 9.48%. Training improves more,
so the training/validation gap widens; the held-out gain does not imply that
the model closed that gap. The new loss values pass the loss portion of gate 2;
the accuracy requirement and other gates are judged from the frozen measurements
below.

Whole four-command evaluation times were 16.497617 seconds for control and
15.8249955 seconds for penalty, including repeated load/device/scoring/IO.
These small timings are descriptive and not a speed comparison.

### Frozen paired diagnostics and integrity

The unchanged `text_probe` metric implementation analyzed both selected
checkpoints with explicit expected hashes. The two reports have identical
training/validation data, manifest, and vocabulary hashes. Their ordered
65-character selection manifest hashes are respectively
`0569ce5217e77961c8a876f443fbeccf8cecf181a4a56eee01acca19147f7ed0` and
`c3c9ee909c33040c08ce233873fb71cf59954616a19cde3669b8000ac1c85860`.
Each split contains 64 windows and 4,096 scored targets. Windows can overlap;
these targets are not independent observations. The fresh seed-42 control
parameter hash is identical in both reports:
`e1037c3cc094d46d310911e0141d00f09e0669af3300203d8c8fd57537892774`.
Initialization uses explicit seeded Rust parameter values, and the zero-penalty
update test verifies identical initialization and sampling sequences. No extra
random draws are introduced by the regularizer.

Both frozen reports verify unchanged parameter hashes before/after inspection.
All eight manual greedy rollouts per condition match ordinary production
generation. The data roundtrip checks still report no unknown training
characters, four expected validation unknowns, and no unexpected token changes.

| Matched 4,096-target measurement | CE control | Voltage penalty |
| --- | ---: | ---: |
| Training NLL | 2.386599116 | 2.243032722 |
| Training argmax correct | 1,384 / 4,096 (33.7891%) | 1,515 / 4,096 (36.9873%) |
| Validation NLL | 2.534009208 | 2.423540584 |
| Validation argmax correct | 1,274 / 4,096 (31.1035%) | 1,393 / 4,096 (34.0088%) |
| Validation NLL, reset before each input | 2.732817544 | 2.739211676 |
| Training-only order-five validation NLL | 2.012148650 | 2.012148650 |
| Training-only order-five validation accuracy | 47.5342% | 47.5342% |

The full models benefit from recurrent context relative to their same-weight
one-input reset controls. The regularized reset control itself does not improve.
The count model remains stronger on these targets, with much larger context
tables and no matched capacity/compute claim.

Activity below uses the same first 16 true validation windows, 1,024 target
steps, with zero initial state for each window. It measures actual pre-reset
voltage and hard spikes, not an approximation from readout activations.

| Activity measurement | CE control | Voltage penalty |
| --- | ---: | ---: |
| Neurons firing strictly above 95% | 20 / 64 | 0 / 64 |
| Neurons firing strictly below 5% | 14 / 64 | 0 / 64 |
| Aggregate firing fraction | 57.2113% | 46.5088% |
| Strict triangular-support occupancy | 35.4263% | 73.7808% |
| Mean surrogate slope, including zeros | .18865469 | .39938286 |
| Pre-reset voltage minimum / maximum | -41.7084 / 66.9210 | -3.73235 / 4.54961 |
| Post-reset membrane RMS | 18.51075 | .734809 |

Every observed spike remains exactly binary. All 11 parameter groups have
finite, nonzero CE gradients on the bounded gradient diagnostic. These group
norms neither isolate long-term credit nor measure the auxiliary gradient; they
do not prove that every neuron learned a meaningful language feature.

### Generation, lexical check, and exposure diagnostic

The fixed six-prompt suite produced all 12 continuations per condition, exactly
256 new characters each, with temperature 0 or .8 and sampling seed 42.
Manual review excluded echoed prompts and did not count an output-limit-truncated
final word as a failure by itself. Independent review agreed with the counts:

| Whole-continuation judgment | CE control | Voltage penalty |
| --- | ---: | ---: |
| Connected coherent continuations | 0 / 12 | 0 / 12 |
| Relevant dialogue replies | 0 / 4 | 0 / 4 |

Both greedy suites repeat function words and malformed fragments. For example,
the control starts `The the and the plers...`; the regularized model starts
`The Sourch Engine March and and...`. The regularized greeting generates repeated
date/advertising fragments such as `(May 16, 2012) - Google AdWord...`, without
answering the prompt. Stochastic outputs still contain malformed words such as
`propmentost` and `rempperacis`. Recognizable individual phrases and dates do not
establish connected meaning, dialogue, symbolic reasoning, or emotion modeling.
The repeated motifs alone do not establish which source examples were memorized.

For the eight actual validation prefixes, the training-only ASCII lexical
measurement pooled **17 / 34 = 50%** hits for control and **16 / 37 = 43.2432%**
for penalty: **-6.7568 percentage points**, rather than the required +10 points.
Use pooled counts, not the mean of eight rates. The eligible words and denominator
change with each generated sample. This is a small descriptive lexicon check,
not a significance test or a comprehensive spelling/meaning metric.

The same eight cases also retain substantial self-fed history drift:

| Document / prefix length | Control teacher NLL | Penalty teacher NLL | Control free-history reference NLL | Penalty free-history reference NLL |
| --- | ---: | ---: | ---: | ---: |
| 0 / 32 | 2.28928 | 2.38957 | 3.95152 | 4.36490 |
| 0 / 64 | 2.37356 | 2.26341 | 3.75818 | 4.22577 |
| 1 / 32 | 2.43944 | 2.38428 | 3.81108 | 3.94936 |
| 1 / 64 | 2.13564 | 2.14971 | 4.81176 | 4.84427 |
| 2 / 32 | 2.13658 | 2.15994 | 3.64454 | 4.67883 |
| 2 / 64 | 2.18552 | 2.24249 | 4.37850 | 5.49452 |
| 3 / 32 | 2.51096 | 2.28467 | 4.11558 | 4.30797 |
| 3 / 64 | 2.04424 | 1.92820 | 4.03496 | 4.12280 |

Each case scores the same 32 true targets. Mean teacher NLL is 2.264402 /
2.225284 for control/penalty; mean free-history reference NLL is 4.063263 /
4.498552. The latter worsens in all eight cases. After a different character is
fed, the original reference may no longer be the only sensible continuation;
this comparison is an exposure diagnostic, not ordinary sequence likelihood or
a verdict that every reference mismatch is a language error. Combined with the
actual outputs and lexical counts, it provides no evidence of repaired generation.

### Predeclared gates and decision

1. **Activity: passes.** High-firing neurons fall 20 to 0 (100% fewer), and
   strict support rises 38.3545 percentage points, exceeding both thresholds.
2. **Prediction: fails narrowly overall.** Both corpus CE reductions exceed .05,
   but matched validation accuracy gains only **2.9052734375 points**, below the
   required 3 points. There are 119 additional correct predictions; at least
   123 were required, four more than observed. Rounded 31.1% to 34.0% must not
   be used to call this gate a pass.
3. **Lexical: fails.** The pooled hit fraction decreases 6.7568 points.
4. **Coherence: unchanged failure on this suite.** Both conditions have zero
   coherent full continuations and zero relevant dialogue replies.

Thus the predeclared combined activity-plus-prediction criterion is **not met**.
This intervention demonstrably controls observed voltage excursions and improves
held-out CE in this paired run. It does **not** fix garbled generation or meet
the lexical criterion. Do not dismiss the continuous CE/accuracy improvements
because of the narrow threshold miss, or promote a failed gate by rounding.
One paired seed and eight held-out documents do not establish reproducible causal
language-quality gains, and the earlier GPU trajectory divergence remains a limit.

Keep the tested objective as an optional, documented setting in the single
maintained training implementation; its default remains zero. Do not promote
it to a proven language repair or retune this completed experiment. The next
language study should target the remaining conditional orthography and self-fed
generation failure, retaining a CE-only comparison and these fixed probes.
More data, capacity, context, or a revised learning objective remain hypotheses,
not outcomes established by this trial.

### Artifacts and measured scope

- `artifacts/fineweb-voltage-control-train.log` and
  `artifacts/fineweb-voltage-train.log`: 61 progress records per condition and
  initial/final CE, configuration, target budgets, and whole-command times.
- `artifacts/fineweb-voltage-control-evaluate.log` and
  `artifacts/fineweb-voltage-evaluate.log`: all reloaded corpus checks.
- `artifacts/text-diagnostic-voltage-control.json` and
  `artifacts/text-diagnostic-voltage.json`: all paired selections, scores, true
  prefixes, free histories, lexical counts, states, gradients, and integrity data.
- `artifacts/text-diagnostic-voltage-control.log` and
  `artifacts/text-diagnostic-voltage.log`: diagnostic console output.
- `artifacts/fineweb-voltage-control-generation.log` and
  `artifacts/fineweb-voltage-generation.log`: all 24 full outputs.
- `artifacts/fineweb-voltage-source-hashes.txt`,
  `artifacts/fineweb-voltage-binary-hashes.txt`, and
  `artifacts/fineweb-voltage-result-hashes.txt`: exact file digests.

Report SHA256 values are
`fbf64f2b5af529509f110b2b8b5e00f160ca8f8160707d8849bd79e07e967fb4`
(control) and
`f3b4c0b78a7bc2426624a84d925765a13992bcd7321cb4ef1c3932e434facd60`
(penalty). Frozen diagnostic times including load/scoring/activity were
19.3255568 / 17.9762779 seconds. Twelve-command generation times including
load/device/generation/IO were 11.2953642 / 11.2224015 seconds. Neither pair is
a controlled runtime comparison. All GPU commands ran sequentially; all model,
training, tokenization, evaluation, and diagnostic code remains Rust/Burn.
After the experiment, every preregistered source/data hash still matched. The
three earlier checkpoint hashes also matched their recorded values, and both
new checkpoints matched their frozen-report hashes. A final formatting check
passed; no executable source changed after the pre-training test/build checks.
