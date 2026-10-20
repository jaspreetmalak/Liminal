# FineWeb data preparation

Liminal's default corpus comes from the original
[HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
dataset, using its `sample-10BT` configuration and upstream `train` split. This
is English web text collected from CommonCrawl. Liminal prepares its own local
training and validation documents; it does not use FineWeb-Edu.

Data preparation, document loading, tokenization, training, and evaluation are
implemented in Rust. The importer uses `reqwest` for HTTPS, `serde_json` for
JSON, and `sha2` for SHA-256. Model execution and automatic differentiation use
Burn tensors; optimization uses Burn Adam after a global tensor-gradient clip.

## Prepare a bounded sample

From the project root:

```text
cargo run --release -- prepare-fineweb
```

The defaults are equivalent to:

```text
cargo run --release -- prepare-fineweb --subset sample-10BT --offset 0 --rows 128 --max-chars 200000 --seed 42 --output data/fineweb
```

| Option | Meaning |
| --- | --- |
| `--subset` | Upstream FineWeb configuration, default `sample-10BT` |
| `--offset` | First source row, default `0` |
| `--rows` | Maximum source rows inspected, default `128`; allowed range `2..=100000` |
| `--max-chars` | Maximum retained Unicode scalar values across both splits, default `200000` |
| `--seed` | Seed for document split membership, default `42` |
| `--output` | New or empty corpus directory, default `data/fineweb` |

The public [Dataset Viewer rows API](https://huggingface.co/docs/dataset-viewer/en/rows)
returns at most 100 rows per request. Liminal requests contiguous pages from:

```text
https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb&config=sample-10BT&split=train&offset=0&length=100
```

The character budget bounds retained text, not HTTP transfer bytes. The importer
can inspect more documents after rejecting one that does not fit the remaining
budget. It caps each response at 16 MiB and applies connection/request timeouts.
Rows that are malformed, blank, shorter than two characters, or marked as
truncated are rejected. A document too large for the remaining character budget
is skipped whole: its prefix is never substituted for the original text.
Preparation requires at least two accepted documents.

The source order and offset determine the candidate sample. Offset zero yields
a biased prefix, and the retained size budget further favors documents that fit.
The split seed randomizes local split membership only; it does not make this a
random sample of all FineWeb. This pathway supports bounded engineering
experiments. Respect service rate limits and retry preparation later if requests
fail. The public preview API is not a scalable full-corpus training feed. A Rust
reader for direct, revision-pinned Parquet access is a future capability and is
not implemented here.

## Whole-document splits

Deduplication happens before splitting. A previously accepted FineWeb `id` or
SHA-256 of the exact UTF-8 `text` rejects a duplicate. Text is not normalized.
This prevents accepted identical texts from appearing in both local splits; it
does not detect near duplicates, shared passages, or equivalent text with
different whitespace. Domains and URLs are not isolated between splits.

For `N` accepted documents, a seeded Fisher-Yates shuffle selects
`ceil(N / 10)` validation members, clamped to at least one and at most `N - 1`.
The remaining documents are training members. Each output preserves source order
within its own membership. The ratio is by document count, so character counts
can differ substantially from 90/10. Changing the selected documents can change
membership even with the same seed.

The output contains:

| File | Contents |
| --- | --- |
| `train.jsonl` | Complete original FineWeb row objects selected for training |
| `validation.jsonl` | Complete original FineWeb row objects selected for validation |
| `manifest.json` | Source, selection, split policy, counts, provenance, and hashes |

Each JSONL line represents one document; newlines inside text are JSON escaped.
The original text and row metadata are preserved, including available `id`,
`url`, `dump`, `date`, `file_path`, `language`, `language_score`, and `token_count`
fields. FineWeb's `token_count` uses its GPT-2 tokenizer and is not a Liminal
character count.

The training loader keeps JSONL documents separate. It fits the character
vocabulary on training documents only and maps unseen validation characters to
U+FFFD, reporting their count. Model state resets for every sampled training
window and independent evaluation window. Windows and next-character targets
stay within a document; no target predicts the beginning of an unrelated page.
Evaluation sums loss over valid target pairs and divides by their count. Record
its sequence/reset length alongside cross-entropy in nats/character and
per-character perplexity.

## Revision and local reproduction

Every page must have the same 40-digit hexadecimal `X-Revision` response header. A revision
change or partial page fails preparation before publication. The manifest records
the observed revision, page offsets/counts and response hashes, original row
indices, document identifiers/provenance, exact text hashes, split membership,
and byte hashes of both output files. Corpus files are published together and
an existing nonempty output directory is never overwritten.

The `/rows` endpoint follows the server's currently indexed dataset revision.
It does not support requesting a historical commit: adding `revision=` does not
pin retrieval. This follows the official
[rows route implementation](https://github.com/huggingface/dataset-viewer/blob/main/services/rows/src/rows/routes/rows.py)
and [response header implementation](https://github.com/huggingface/dataset-viewer/blob/main/libs/libapi/src/libapi/utils.py).
The manifest describes what was observed, rather than promising that a later
network import can reconstruct it.

Preserve the prepared files and manifest for stable local replay. Keep their
recorded SHA-256 values with each experiment and verify the corpus bytes before
comparing runs; the training command does not automatically enforce manifest
hashes. Running the same commands and seed against unchanged local files avoids
dependence on subsequent upstream changes. Separate preparation with a different
offset/output directory is a different data selection and needs its own record.

## Attribution and source terms

Source: **FineWeb**, Guilherme Penedo and collaborators, Hugging Face FineData.
See the [official dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
and [The FineWeb Datasets paper](https://arxiv.org/abs/2406.17557).

The dataset card specifies the
[Open Data Commons Attribution License (ODC-By) 1.0](https://opendatacommons.org/licenses/by/1-0/)
and [CommonCrawl Terms of Use](https://commoncrawl.org/terms-of-use).
Keep source attribution and these links with prepared data and experiment
reports. These dataset terms do not imply that every original webpage has the
same content license; retain its URL/provenance. Raw web content can still
include personal information, biased material, or unwanted text despite upstream
filtering. This importer adds no content classifier or privacy filtering.

## Train and record evidence

```text
cargo run --release -- train
cargo run --release -- evaluate
```

Defaults load the prepared training/validation files. Preparation and training
are separate commands so repeated training can reuse identical local data.
The default model has 48 hidden neurons, 32-character training/evaluation
windows, batches of 16, 500 updates, and seed 42. WGPU is the default device;
`--device cpu` executes the same Burn implementation on Flex. `train` initializes a fresh model and replaces
`artifacts/liminal.bin`; optimizer state is not saved for resume.

For a separately prepared directory, supply both files:

```text
cargo run --release -- train --data data/my-corpus/train.jsonl --validation-data data/my-corpus/validation.jsonl
cargo run --release -- evaluate --data data/my-corpus/validation.jsonl
```

Custom JSONL needs a string `text` field on every line and a separate validation
file. Custom plain UTF-8 is treated as one document per file. Passing plain text
without `--validation-data` uses a 90/10 character split for focused smoke checks.

Record the manifest, preparation/training commands, seeds, model configuration,
training budget, environment, precision, unknown-character counts, losses,
unigram control, and limitations under `docs/experiments/`. The previous tiny
corpus results are not FineWeb results and should not be compared as though
their evaluation populations were the same.
