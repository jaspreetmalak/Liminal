# Small-talk evaluation of the current checkpoint

## Question and protocol

Can the trained Liminal checkpoint produce coherent, relevant replies to simple
small talk? This is an exploratory qualitative check, not a calibrated benchmark.
A usable reply must form a coherent sentence and address the supplied prompt;
isolated readable words do not suffice. The follow-up also checks continuity with
a supplied conversation. Only generated continuation text is assessed: the CLI
echoes the prompt, which is not evidence of model ability.

Five distinct prompts cover a greeting, wellbeing, hobbies, a follow-up about
books, and an unformatted greeting. Each generates 120 additional characters at
temperature 0.8, seed 42. The first greeting is also run with temperature 0
(greedy), giving six continuations and 720 generated characters in total.
Each invocation starts with zero neuron state and consumes its full prompt.
Earlier turns in the follow-up are supplied text, not successful model replies.

## Checkpoint, data, and compute

- Date: 2026-10-18. Working directory:
  `C:\Users\jassm\Desktop\Projects\Liminal`.
- Existing `artifacts/liminal.bin`, SHA-256 verified as
  `4eb56bc9fff8709931313d12e49c1e59d784eccdc1487ff1a21e52a6023f6fca`.
- Same Rust release executable and Burn 0.22 WGPU implementation as the training
  experiment; default discrete GPU is the AMD Radeon RX 9060 XT. Float32 tensors.
- 16,708 parameters, 48 recurrent spiking neurons, 148 character vocabulary entries.
- Previously trained for 500 updates, batch 16, 32-character windows, seed 42:
  256,000 sampled prediction targets from 70 FineWeb training documents containing
  185,628 characters. Validation: 8 disjoint documents, 14,348 characters.
- No new training, vocabulary fitting, or model changes in this evaluation.
  Prompts produced no unknown-character warnings. No inference speed measurement.
- Prior evidence: [GPU training record](2026-10-18-burn-gpu-training.md) reports
  validation character cross-entropy 2.747231 nats, perplexity 15.599385, and
  fragmented generated text. Those are existing measurements, not rerun here.

## Exact outputs

The following blocks contain only the newly generated characters, without the
echoed prompt. All six invocations completed successfully.

1. Prompt: `User: Hi Liminal! How are you today? Assistant: `

```text
od aat the fhera part he arltisiorg Afran asak ante Ge ropr. wh apvetosa tor the ol. Forthhe fl iires ror Seerapat efrey
```

2. Prompt: `User: I had a long day. How was yours? Assistant: `

```text
od aat the mhel 21 ew mialrlsisios fEAs gallin Inte GaMrint aur apwetsre tmoat o seer tishee ns ofoar tor Seeran ugees y
```

3. Prompt: `User: What do you like doing for fun? Assistant: `

```text
od SItes sange MEl, wand asitisiorin as haerge Enul Be sopr Iur ary tos' tor the sellaserore or ofe. Cori5 Tas fet alrey
```

4. Prompt: `User: I like reading books. Assistant: What kind of books? User: Science fiction. How about you? Assistant: `

```text
od aatas ofaonig icoubme asrtisn the as oecore Enul Be. llo csially tos Moro tea te satllis Ner oflcieris Secocilt datey
```

5. Prompt: `Hello, how are you today? `

```text
od bat the henid acouand Wlouros tik Woan core Inte Ma tint ath aow totacsor the teis tisere fo ofoas too an raogud cony
```

6. Greedy control, same prompt as item 1, temperature 0:

```text
an an the the an the the the at the the ant ant ant at ore at the ant the athe the the the the ate the the an the athe t
```

## Assessment and decision

| Check | Observed result |
| --- | --- |
| Coherent, relevant conversational reply | 0 of 6 continuations |
| Coherent sentences | 0 of 6 continuations |
| Some readable individual words | Present in all 6, amid fragments or repetition |
| Follow-up continuity | No recognizable response about books or science fiction |
| Greedy control | Repeats short fragments, especially `the` and `an` |

The current checkpoint cannot sustain small talk in these probes. Sampling
temperature and role-label formatting do not account for all observed failures:
both the greedy control and the plain greeting also fail the usability criterion.
The measured character-loss improvement remains evidence of learning character
statistics; it has not yet translated into coherent conversation.

This small, single-seed check does not establish general capability scores or
isolate the cause. The model is small, was trained on a bounded web-text slice
with short reset windows, and has no dialogue-specific training or learned stop
token. More training alone is not demonstrated to solve these failures. Retain
this checkpoint as the current development baseline and require stronger
held-out language-generation evidence before describing it as conversational.

## Reproduction and artifacts

Run from the project root using the existing release binary:

```powershell
./target/release/liminal.exe generate --device gpu --prompt "User: Hi Liminal! How are you today? Assistant: " --tokens 120 --temperature 0.8 --seed 42
./target/release/liminal.exe generate --device gpu --prompt "User: I had a long day. How was yours? Assistant: " --tokens 120 --temperature 0.8 --seed 42
./target/release/liminal.exe generate --device gpu --prompt "User: What do you like doing for fun? Assistant: " --tokens 120 --temperature 0.8 --seed 42
./target/release/liminal.exe generate --device gpu --prompt "User: I like reading books. Assistant: What kind of books? User: Science fiction. How about you? Assistant: " --tokens 120 --temperature 0.8 --seed 42
./target/release/liminal.exe generate --device gpu --prompt "Hello, how are you today? " --tokens 120 --temperature 0.8 --seed 42
./target/release/liminal.exe generate --device gpu --prompt "User: Hi Liminal! How are you today? Assistant: " --tokens 120 --temperature 0 --seed 42
```

The first raw output is preserved above. Raw stdout for items 2-6 is also saved
under `artifacts/` as `smalltalk-mood.txt`, `smalltalk-hobbies.txt`,
`smalltalk-followup.txt`, `smalltalk-plain-greeting.txt`, and
`smalltalk-greedy-greeting.txt`. These include the echoed prompt.
