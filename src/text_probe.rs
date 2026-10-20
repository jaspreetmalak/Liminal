//! Bounded frozen-model text diagnostics. No optimizer or training updates.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::data::{Tokenizer, UNKNOWN_CHARACTER};
use crate::model::{Model, State};
use crate::rng::Rng;
use crate::{checkpoint, runtime, training};

const CHECKPOINT_SHA256: &str = "41aca0870018d462344f2a128d4aadd0d4feda7de156d48f220094adc798572c";
const DEFAULT_CHECKPOINT: &str = "artifacts/fineweb-language.bin";
const SEED: u64 = 42;
const WINDOWS: usize = 64;
const TARGETS: usize = 64;
const BATCH: usize = 16;
const MAX_ORDER: usize = 5;
const BACKOFF_ALPHA: f64 = 10.0;

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parameter_hash(model: &Model) -> Result<String, String> {
    Ok(sha256(
        &model
            .parameter_values()?
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    ))
}

struct Corpus {
    documents: Vec<String>,
    ids: Vec<String>,
    file_hash: String,
}

fn load_corpus(path: &Path) -> Result<Corpus, String> {
    let bytes = fs::read(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    let text = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
    let mut documents = Vec::new();
    let mut ids = Vec::new();
    for (line, row) in text.lines().enumerate() {
        let row: Value = serde_json::from_str(row)
            .map_err(|error| format!("JSONL line {}: {error}", line + 1))?;
        let document = row["text"].as_str().ok_or("FineWeb row needs text")?;
        let id = row["id"].as_str().ok_or("FineWeb row needs source id")?;
        if document.chars().count() < 2 || id.is_empty() {
            return Err("FineWeb row is empty or too short".into());
        }
        documents.push(document.to_owned());
        ids.push(id.to_owned());
    }
    if documents.is_empty() || ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err("FineWeb corpus is empty or repeats a source id".into());
    }
    Ok(Corpus {
        documents,
        ids,
        file_hash: sha256(&bytes),
    })
}

fn verify_splits(train: &Corpus, validation: &Corpus) -> Result<(), String> {
    let ids: BTreeSet<_> = train.ids.iter().collect();
    let texts: BTreeSet<_> = train.documents.iter().collect();
    if validation.ids.iter().any(|id| ids.contains(id))
        || validation.documents.iter().any(|text| texts.contains(text))
    {
        return Err("FineWeb source ids and complete documents must be disjoint".into());
    }
    Ok(())
}

fn roundtrip(corpus: &Corpus, tokenizer: &Tokenizer) -> Result<Value, String> {
    let mut raw_mismatches = 0;
    let mut unknown = 0;
    let mut characters = 0;
    for document in &corpus.documents {
        let encoded = tokenizer.encode(document);
        let decoded = tokenizer.decode(&encoded)?;
        if tokenizer.encode(&decoded) != encoded {
            return Err("token encode/decode roundtrip changed token ids".into());
        }
        let expected: String = document
            .chars()
            .map(|character| {
                if tokenizer.encode(&character.to_string())[0] == 0 {
                    UNKNOWN_CHARACTER
                } else {
                    character
                }
            })
            .collect();
        if decoded != expected {
            return Err("decoded text differs beyond documented UNKNOWN substitution".into());
        }
        raw_mismatches += document
            .chars()
            .zip(decoded.chars())
            .filter(|(a, b)| a != b)
            .count();
        unknown += encoded.iter().filter(|&&token| token == 0).count();
        characters += encoded.len();
    }
    Ok(
        json!({"characters": characters, "unknown_tokens": unknown, "raw_text_mismatching_characters": raw_mismatches, "raw_text_roundtrip_exact": raw_mismatches == 0, "token_roundtrip_exact": true, "matches_expected_unknown_substitution": true}),
    )
}

#[derive(Clone)]
struct Window {
    document: usize,
    start: usize,
    tokens: Vec<usize>,
}

fn select_windows(documents: &[Vec<usize>]) -> Result<Vec<Window>, String> {
    let mut cumulative = Vec::with_capacity(documents.len());
    let mut total = 0_usize;
    for document in documents {
        total = total
            .checked_add(document.len().saturating_sub(TARGETS))
            .ok_or("window count overflow")?;
        cumulative.push(total);
    }
    if total == 0 {
        return Err("diagnostic needs a document with at least65 characters".into());
    }
    let mut random = Rng::new(SEED);
    let mut windows = Vec::with_capacity(WINDOWS);
    for _ in 0..WINDOWS {
        let draw = random.index(total);
        let document = cumulative.partition_point(|&end| end <= draw);
        let previous = document.checked_sub(1).map_or(0, |index| cumulative[index]);
        let start = draw - previous;
        windows.push(Window {
            document,
            start,
            tokens: documents[document][start..start + TARGETS + 1].to_vec(),
        });
    }
    Ok(windows)
}

fn window_report(
    windows: &[Window],
    corpus: &Corpus,
    tokenizer: &Tokenizer,
) -> Result<Value, String> {
    let mut manifest = String::new();
    let mut rows = Vec::new();
    for window in windows {
        let text = tokenizer.decode(&window.tokens)?;
        let hash = sha256(text.as_bytes());
        manifest.push_str(&format!("{}\t{}\t{hash}\n", window.document, window.start));
        rows.push(json!({"document_index": window.document, "source_id": corpus.ids[window.document], "start_character": window.start, "targets": TARGETS, "encoded_window_text": text, "encoded_window_utf8_sha256": hash}));
    }
    let mut overlapping = 0;
    let mut identical = 0;
    for (index, first) in windows.iter().enumerate() {
        for second in &windows[index + 1..] {
            if first.document == second.document {
                identical += usize::from(first.start == second.start);
                overlapping += usize::from(
                    first.start < second.start + TARGETS + 1
                        && second.start < first.start + TARGETS + 1,
                );
            }
        }
    }
    Ok(
        json!({"selection_seed_per_split": SEED, "sampling": "uniform eligible65-character starts with replacement; reset seed independently per split", "window_count": windows.len(), "targets": windows.len() * TARGETS, "distinct_documents": windows.iter().map(|window| window.document).collect::<BTreeSet<_>>().len(), "overlapping_window_pairs": overlapping, "identical_window_pairs": identical, "independence": "windows can overlap and share documents; targets are not independent observations", "manifest_encoding": "ordered document index + TAB + character start + TAB + encoded65-character UTF8 SHA256 + LF", "manifest_sha256": sha256(manifest.as_bytes()), "windows": rows}),
    )
}

struct Scored {
    prediction: usize,
    nll: f64,
    true_probability: f64,
    true_rank: usize,
}

fn score_logits(logits: &[f32], target: usize) -> Result<Scored, String> {
    if logits.is_empty() || target >= logits.len() || logits.iter().any(|value| !value.is_finite())
    {
        return Err("invalid or non-finite diagnostic logits".into());
    }
    let mut prediction = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[prediction] {
            prediction = index;
        }
    }
    let maximum = f64::from(logits[prediction]);
    let log_z = maximum
        + logits
            .iter()
            .map(|&value| (f64::from(value) - maximum).exp())
            .sum::<f64>()
            .ln();
    let nll = (log_z - f64::from(logits[target])).max(0.0);
    Ok(Scored {
        prediction,
        nll,
        true_probability: (-nll).exp(),
        true_rank: 1 + logits
            .iter()
            .filter(|&&value| value > logits[target])
            .count(),
    })
}

#[derive(Default)]
struct Aggregate {
    targets: usize,
    correct: usize,
    nll: f64,
}

impl Aggregate {
    fn add(&mut self, target: usize, score: &Scored) {
        self.targets += 1;
        self.correct += usize::from(target == score.prediction);
        self.nll += score.nll;
    }
    fn json(&self) -> Value {
        json!({"targets": self.targets, "correct": self.correct, "argmax_accuracy": self.correct as f64 / self.targets as f64, "nll_nats_per_character": self.nll / self.targets as f64})
    }
}

#[derive(Default)]
struct Metrics {
    total: Aggregate,
    bins: [Aggregate; 4],
    predictions: BTreeMap<usize, usize>,
    confusions: BTreeMap<(usize, usize), usize>,
    current_predictions: BTreeMap<(usize, usize), usize>,
    current_targets: BTreeMap<(usize, usize), usize>,
}

impl Metrics {
    fn observe(&mut self, position: usize, input: usize, target: usize, score: Scored) {
        self.total.add(target, &score);
        let bin = match position {
            0..=7 => 0,
            8..=15 => 1,
            16..=31 => 2,
            _ => 3,
        };
        self.bins[bin].add(target, &score);
        *self.predictions.entry(score.prediction).or_default() += 1;
        if target != score.prediction {
            *self
                .confusions
                .entry((target, score.prediction))
                .or_default() += 1;
        }
        *self
            .current_predictions
            .entry((input, score.prediction))
            .or_default() += 1;
        *self.current_targets.entry((input, target)).or_default() += 1;
    }
    fn json(&self, tokenizer: &Tokenizer) -> Result<Value, String> {
        let mut common: Vec<_> = self.predictions.iter().collect();
        common.sort_by(|(ka, va), (kb, vb)| vb.cmp(va).then_with(|| ka.cmp(kb)));
        let common = common
            .into_iter()
            .take(20)
            .map(|(&token, &count)| {
                Ok(json!({"token": tokenizer.decode(&[token])?, "token_id": token, "count": count}))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let pairs = |counts: &BTreeMap<(usize, usize), usize>| -> Result<Vec<Value>, String> {
            let mut rows: Vec<_> = counts.iter().collect();
            rows.sort_by(|(ka, va), (kb, vb)| vb.cmp(va).then_with(|| ka.cmp(kb)));
            rows.into_iter().take(40).map(|(&(from, to), &count)| Ok(json!({"from": tokenizer.decode(&[from])?, "to": tokenizer.decode(&[to])?, "count": count}))).collect()
        };
        Ok(
            json!({"overall": self.total.json(), "position_bins": self.bins.iter().zip(["1-8", "9-16", "17-32", "33-64"]).map(|(metrics, label)| json!({"positions_1_based": label, "metrics": metrics.json()})).collect::<Vec<_>>(), "most_common_predictions_top20": common, "wrong_true_next_to_predicted_next_top40": pairs(&self.confusions)?, "true_current_to_predicted_next_top40": pairs(&self.current_predictions)?, "true_current_to_true_next_top40": pairs(&self.current_targets)?, "confusion_tables": "frequency-ranked summaries; ties sorted by token IDs; omitted low-count entries are not zero"}),
        )
    }
}

fn neural_metrics(
    model: &Model,
    windows: &[Window],
    tokenizer: &Tokenizer,
    reset_each: bool,
) -> Result<Value, String> {
    let model = model.clone().valid();
    let device = model.device();
    let mut metrics = Metrics::default();
    for rows in windows.chunks(BATCH) {
        let mut state = State::new(rows.len(), model.config.hidden_size, &device);
        for time in 0..TARGETS {
            if reset_each {
                state = State::new(rows.len(), model.config.hidden_size, &device);
            }
            let input: Vec<_> = rows
                .iter()
                .map(|window| window.tokens[time] as i32)
                .collect();
            let logits = model
                .step(
                    Tensor::<1, Int>::from_data(TensorData::new(input, [rows.len()]), &device),
                    &mut state,
                )?
                .try_into_vec_as::<f32>()
                .map_err(|error| error.to_string())?;
            for (row, values) in rows.iter().zip(logits.chunks(model.config.vocab_size)) {
                metrics.observe(
                    time,
                    row.tokens[time],
                    row.tokens[time + 1],
                    score_logits(values, row.tokens[time + 1])?,
                );
            }
        }
    }
    metrics.json(tokenizer)
}

#[derive(Default)]
struct ContextCounts {
    total: u64,
    next: BTreeMap<usize, u64>,
}

struct Ngram {
    root: Vec<u64>,
    total: u64,
    contexts: BTreeMap<Vec<usize>, ContextCounts>,
}

impl Ngram {
    fn fit(documents: &[Vec<usize>], vocabulary: usize) -> Self {
        let mut model = Self {
            root: vec![0; vocabulary],
            total: 0,
            contexts: BTreeMap::new(),
        };
        for document in documents {
            for position in 1..document.len() {
                let target = document[position];
                model.root[target] += 1;
                model.total += 1;
                for length in 1..=position.min(MAX_ORDER - 1) {
                    let counts = model
                        .contexts
                        .entry(document[position - length..position].to_vec())
                        .or_default();
                    counts.total += 1;
                    *counts.next.entry(target).or_default() += 1;
                }
            }
        }
        model
    }
    fn probabilities(&self, prefix: &[usize]) -> Vec<f64> {
        let mut probabilities: Vec<_> = self
            .root
            .iter()
            .map(|&count| (count + 1) as f64 / (self.total + self.root.len() as u64) as f64)
            .collect();
        for length in 1..=prefix.len().min(MAX_ORDER - 1) {
            if let Some(counts) = self.contexts.get(&prefix[prefix.len() - length..]) {
                for (token, probability) in probabilities.iter_mut().enumerate() {
                    *probability = (counts.next.get(&token).copied().unwrap_or(0) as f64
                        + BACKOFF_ALPHA * *probability)
                        / (counts.total as f64 + BACKOFF_ALPHA);
                }
            }
        }
        probabilities
    }

    fn greedy(&self, prefix: &[usize], count: usize) -> Vec<usize> {
        let mut history = prefix.to_vec();
        for _ in 0..count {
            let probabilities = self.probabilities(&history);
            let mut prediction = 0;
            for index in 1..probabilities.len() {
                if probabilities[index] > probabilities[prediction] {
                    prediction = index;
                }
            }
            history.push(prediction);
        }
        history[prefix.len()..].to_vec()
    }
    fn score(
        &self,
        windows: &[Window],
        tokenizer: &Tokenizer,
        current_only: bool,
    ) -> Result<Value, String> {
        let mut metrics = Metrics::default();
        for window in windows {
            for time in 0..TARGETS {
                let prefix = if current_only {
                    &window.tokens[time..time + 1]
                } else {
                    &window.tokens[..time + 1]
                };
                let probabilities = self.probabilities(prefix);
                let mut prediction = 0;
                for index in 1..probabilities.len() {
                    if probabilities[index] > probabilities[prediction] {
                        prediction = index;
                    }
                }
                let target = window.tokens[time + 1];
                metrics.observe(
                    time,
                    window.tokens[time],
                    target,
                    Scored {
                        prediction,
                        nll: -probabilities[target].ln(),
                        true_probability: probabilities[target],
                        true_rank: 1 + probabilities
                            .iter()
                            .filter(|&&value| value > probabilities[target])
                            .count(),
                    },
                );
            }
        }
        metrics.json(tokenizer)
    }
}

fn ascii_words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|character: char| !character.is_ascii_alphabetic())
        .filter(|word| word.len() >= 2)
        .map(str::to_ascii_lowercase)
}

fn lexical_summary(previous: Option<char>, text: &str, lexicon: &BTreeSet<String>) -> Value {
    let left_partial = previous.is_some_and(|character| character.is_ascii_alphabetic())
        && text.starts_with(|character: char| character.is_ascii_alphabetic());
    let mut eligible = Vec::new();
    let mut start = None;
    for (offset, character) in text.char_indices() {
        if character.is_ascii_alphabetic() {
            start.get_or_insert(offset);
        } else if let Some(begin) = start.take() {
            let word = &text[begin..offset];
            if word.len() >= 2 && !(begin == 0 && left_partial) {
                eligible.push(word.to_ascii_lowercase());
            }
        }
    }
    // Deliberately drop a final letter run: its word may be cut by the budget.
    let hits = eligible
        .iter()
        .filter(|word| lexicon.contains(*word))
        .count();
    json!({"eligible_complete_ascii_words": eligible.len(), "training_lexicon_hits": hits, "hit_rate": if eligible.is_empty() { None } else { Some(hits as f64 / eligible.len() as f64) }, "words": eligible, "excluded_left_partial_word": left_partial, "excluded_right_partial_word": start.is_some(), "definition": "lowercase ASCII alphabetic runs length>=2; omit a prefix-continuing first word and unterminated final word; punctuation/apostrophes/hyphens split words; lexicon hit is not spelling/grammar/meaning validation"})
}

fn step_single(model: &Model, state: &mut State, token: usize) -> Result<Vec<f32>, String> {
    model
        .step(
            Tensor::<1, Int>::from_data([token as i32], &model.device()),
            state,
        )?
        .try_into_vec_as::<f32>()
        .map_err(|error| error.to_string())
}

fn compare_prefix(
    model: &Model,
    tokenizer: &Tokenizer,
    prefix: &[usize],
    reference: &[usize],
    lexicon: &BTreeSet<String>,
) -> Result<Value, String> {
    if prefix.is_empty() || reference.is_empty() {
        return Err("prefix/reference must be nonempty".into());
    }
    let model = model.clone().valid();
    let mut initial_state = State::new(1, model.config.hidden_size, &model.device());
    let mut initial_logits = Vec::new();
    for &token in prefix {
        initial_logits = step_single(&model, &mut initial_state, token)?;
    }
    let mut teacher_state = initial_state.clone();
    let mut free_state = initial_state;
    let mut teacher_logits = initial_logits.clone();
    let mut free_logits = initial_logits;
    let mut teacher = Vec::new();
    let mut free = Vec::new();
    let mut records = Vec::new();
    let mut teacher_nll = Vec::new();
    let mut free_nll = Vec::new();
    for (index, &target) in reference.iter().enumerate() {
        let teacher_score = score_logits(&teacher_logits, target)?;
        let free_score = score_logits(&free_logits, target)?;
        teacher.push(teacher_score.prediction);
        free.push(free_score.prediction);
        teacher_nll.push(teacher_score.nll);
        free_nll.push(free_score.nll);
        records.push(json!({"position_1_based": index + 1, "common_true_target": tokenizer.decode(&[target])?, "teacher_forced_argmax": tokenizer.decode(&[teacher_score.prediction])?, "free_history_argmax": tokenizer.decode(&[free_score.prediction])?, "teacher_forced_true_probability": teacher_score.true_probability, "free_history_true_probability": free_score.true_probability, "teacher_forced_true_rank": teacher_score.true_rank, "free_history_true_rank": free_score.true_rank, "rank_ties": "best rank:1+count(logit strictly greater than true-target logit); ties share rank", "teacher_forced_nll": teacher_score.nll, "free_history_nll": free_score.nll}));
        if index + 1 < reference.len() {
            teacher_logits = step_single(&model, &mut teacher_state, target)?;
            free_logits = step_single(&model, &mut free_state, free_score.prediction)?;
        }
    }
    let ordinary = training::generate(
        &model,
        prefix,
        reference.len(),
        training::Decoding::sampled(0.0),
        SEED,
    )?;
    let sampled = training::generate(
        &model,
        prefix,
        reference.len(),
        training::Decoding::sampled(0.8),
        SEED,
    )?;
    let first_mismatch = free.iter().zip(reference).position(|(a, b)| a != b);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let after = |values: &[f64]| {
        first_mismatch
            .and_then(|index| (index + 1 < values.len()).then(|| mean(&values[index + 1..])))
    };
    let decoded_prefix = tokenizer.decode(prefix)?;
    let decoded_reference = tokenizer.decode(reference)?;
    let teacher_text = tokenizer.decode(&teacher)?;
    let free_text = tokenizer.decode(&free)?;
    let sampled_text = tokenizer.decode(&sampled)?;
    Ok(
        json!({"prefix": decoded_prefix, "reference_next_characters": decoded_reference, "teacher_forced_argmax_text": teacher_text, "greedy_free_run_text": free_text, "ordinary_generate_text": tokenizer.decode(&ordinary)?, "ordinary_generate_matches_manual_step_rollout": ordinary == free, "sampled_text": sampled_text, "sampled_temperature": 0.8, "sampled_seed": SEED, "lexical_sampled": lexical_summary(decoded_prefix.chars().last(), &sampled_text, lexicon), "first_greedy_reference_mismatch_position_1_based": first_mismatch.map(|index| index + 1), "teacher_forced_next_character_accuracy": teacher.iter().zip(reference).filter(|(a, b)| a == b).count() as f64 / reference.len() as f64, "greedy_reference_position_match_fraction": free.iter().zip(reference).filter(|(a, b)| a == b).count() as f64 / reference.len() as f64, "teacher_forced_mean_nll": mean(&teacher_nll), "free_history_mean_nll_on_same_true_targets": mean(&free_nll), "teacher_forced_nll_after_first_reference_mismatch_is_fed": after(&teacher_nll), "free_history_nll_after_first_reference_mismatch_is_fed": after(&free_nll), "position_scores": records, "lexical_reference": lexical_summary(decoded_prefix.chars().last(), &decoded_reference, lexicon), "lexical_teacher_argmax": lexical_summary(decoded_prefix.chars().last(), &teacher_text, lexicon), "lexical_free_run": lexical_summary(decoded_prefix.chars().last(), &free_text, lexicon), "interpretation": "both scored paths use identical true targets; histories diverge only when a mismatching generated character is fed at the next step; a reference mismatch is not automatically a language error; teacher argmax characters are not an autoregressive sample; sampled text has descriptive lexical counts only"}),
    )
}

fn checked_output(path: &Path) -> Result<PathBuf, String> {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    if path
        .components()
        .any(|component| component == Component::ParentDir)
        || path.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err("output must be a new artifacts/*.json path without '..'".into());
    }
    let output = if path.is_absolute() {
        path.to_path_buf()
    } else {
        project.join(path)
    };
    if !output.starts_with(project.join("artifacts")) || fs::symlink_metadata(&output).is_ok() {
        return Err("output must be a new file inside project/artifacts".into());
    }
    let mut ancestor = output.parent().ok_or("output needs a parent")?;
    while fs::symlink_metadata(ancestor).is_err() {
        ancestor = ancestor.parent().ok_or("cannot resolve output")?;
    }
    if !ancestor
        .canonicalize()
        .map_err(|error| error.to_string())?
        .starts_with(project.canonicalize().map_err(|error| error.to_string())?)
    {
        return Err("output resolves outside the project".into());
    }
    Ok(output)
}

fn checked_sha256(value: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(
            "expected checkpoint SHA256 must contain exactly 64 hexadecimal characters".into(),
        );
    }
    Ok(value.to_ascii_lowercase())
}

struct CheckedCheckpoint {
    path: PathBuf,
    actual_sha256: String,
    expected_sha256: String,
}

/// Validate the subject before initializing a device or loading model tensors.
fn checked_checkpoint(path: &Path, expected_sha256: &str) -> Result<CheckedCheckpoint, String> {
    let expected_sha256 = checked_sha256(expected_sha256)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let artifacts = root.join("artifacts");
    if path
        .components()
        .any(|component| component == Component::ParentDir)
        || path.extension().and_then(|value| value.to_str()) != Some("bin")
    {
        return Err(
            "checkpoint must be an existing project artifacts/*.bin path without '..'".into(),
        );
    }
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    if !path.starts_with(&artifacts) {
        return Err("checkpoint must remain inside project/artifacts".into());
    }
    let canonical_artifacts = artifacts
        .canonicalize()
        .map_err(|error| format!("resolving artifacts directory: {error}"))?;
    let canonical_root = root.canonicalize().map_err(|error| error.to_string())?;
    let canonical_path = path
        .canonicalize()
        .map_err(|error| format!("resolving checkpoint: {error}"))?;
    if !canonical_artifacts.starts_with(canonical_root)
        || !canonical_path.starts_with(&canonical_artifacts)
        || !canonical_path.is_file()
    {
        return Err("checkpoint must resolve to an existing file inside project/artifacts".into());
    }
    let bytes = fs::read(&path).map_err(|error| format!("reading checkpoint: {error}"))?;
    let actual_sha256 = sha256(&bytes);
    if actual_sha256 != expected_sha256 {
        return Err(
            "candidate checkpoint SHA256 differs from the expected diagnostic subject".into(),
        );
    }
    Ok(CheckedCheckpoint {
        path,
        actual_sha256,
        expected_sha256,
    })
}

/// Preserve the original fixed subject and protocol for existing callers.
pub fn run(device_name: &str, output: &Path) -> Result<PathBuf, String> {
    run_configured(
        device_name,
        output,
        Path::new(DEFAULT_CHECKPOINT),
        CHECKPOINT_SHA256,
    )
}

/// All comparisons are frozen inference or gradient observations; no optimizer.
/// Selecting a different checkpoint does not change sampling or metrics.
pub fn run_configured(
    device_name: &str,
    output: &Path,
    checkpoint_path: &Path,
    expected_checkpoint_sha256: &str,
) -> Result<PathBuf, String> {
    if !matches!(device_name, "gpu" | "cpu") {
        return Err("--device must be gpu or cpu".into());
    }
    let output = checked_output(output)?;
    let started = Instant::now();
    let subject = checked_checkpoint(checkpoint_path, expected_checkpoint_sha256)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest_bytes =
        fs::read(root.join("data/fineweb/manifest.json")).map_err(|error| error.to_string())?;
    let manifest: Value =
        serde_json::from_slice(&manifest_bytes).map_err(|error| error.to_string())?;
    let train = load_corpus(&root.join("data/fineweb/train.jsonl"))?;
    let validation = load_corpus(&root.join("data/fineweb/validation.jsonl"))?;
    for (name, corpus) in [("train.jsonl", &train), ("validation.jsonl", &validation)] {
        if manifest["files"][name]["sha256"].as_str() != Some(corpus.file_hash.as_str()) {
            return Err(format!("{name} hash differs from prepared manifest"));
        }
    }
    if manifest["source"]["dataset"].as_str() != Some("HuggingFaceFW/fineweb") {
        return Err("expected original FineWeb source".into());
    }
    verify_splits(&train, &validation)?;
    let model_path = subject.path;
    let device: Device = runtime::device(device_name)?;
    eprintln!(
        "Text diagnostic device: {device:?}; identity: {:?}",
        device.identity()
    );
    let (model, tokenizer) = checkpoint::load(&model_path, &device)?;
    if model.config.hidden_size != 64 || tokenizer != Tokenizer::from_documents(&train.documents) {
        return Err(
            "checkpoint configuration/vocabulary disagrees with fixed H64 training corpus".into(),
        );
    }
    let before_hash = parameter_hash(&model)?;
    let training_roundtrip = roundtrip(&train, &tokenizer)?;
    let validation_roundtrip = roundtrip(&validation, &tokenizer)?;
    if training_roundtrip["raw_text_roundtrip_exact"] != true {
        return Err("training text did not roundtrip exactly".into());
    }
    let training_tokens: Vec<_> = train
        .documents
        .iter()
        .map(|text| tokenizer.encode(text))
        .collect();
    let validation_tokens: Vec<_> = validation
        .documents
        .iter()
        .map(|text| tokenizer.encode(text))
        .collect();
    let train_windows = select_windows(&training_tokens)?;
    let validation_windows = select_windows(&validation_tokens)?;
    let mut untrained = Model::new(model.config.clone(), SEED, &device)?;
    untrained.feedback = model.feedback;
    let ngram = Ngram::fit(&training_tokens, tokenizer.vocab_size());
    let lexicon: BTreeSet<_> = train
        .documents
        .iter()
        .flat_map(|text| ascii_words(text))
        .collect();
    let mut split_reports = Vec::new();
    for (label, corpus, windows) in [
        ("training", &train, &train_windows),
        ("validation", &validation, &validation_windows),
    ] {
        let full = neural_metrics(&model, windows, &tokenizer, false)?;
        let reset = neural_metrics(&model, windows, &tokenizer, true)?;
        let random = neural_metrics(&untrained, windows, &tokenizer, false)?;
        println!(
            "{label}: trained full NLL {}, reset-each-token NLL {}, untrained NLL {}",
            full["overall"]["nll_nats_per_character"],
            reset["overall"]["nll_nats_per_character"],
            random["overall"]["nll_nats_per_character"]
        );
        split_reports.push(json!({"split": label, "selection": window_report(windows, corpus, &tokenizer)?, "trained_full": full, "trained_reset_before_each_input": reset, "untrained_same_configuration_seed42": random, "training_only_order5_ngram": ngram.score(windows, &tokenizer, false)?, "training_only_ngram_current_input_context": ngram.score(windows, &tokenizer, true)?}));
    }
    let mut prefix_reports = Vec::new();
    for (document_index, tokens) in validation_tokens
        .iter()
        .enumerate()
        .filter(|(_, tokens)| tokens.len() >= 96)
        .take(4)
    {
        for prefix_length in [32, 64] {
            let end = prefix_length + 32;
            let ngram_tokens = ngram.greedy(&tokens[..prefix_length], 32);
            let ngram_text = tokenizer.decode(&ngram_tokens)?;
            let decoded_prefix = tokenizer.decode(&tokens[..prefix_length])?;
            prefix_reports.push(json!({"document_index": document_index, "source_id": validation.ids[document_index], "raw_prefix_and_reference_span": validation.documents[document_index].chars().take(end).collect::<String>(), "total_reference_span_characters": end, "model_input_steps_through_last_prediction": end - 1, "within64_input_training_horizon": end - 1 <= 64, "prefix_characters": prefix_length, "scored_true_targets": 32, "unknown_tokens_in_prefix_and_targets": tokens[..end].iter().filter(|&&token| token == 0).count(), "comparison": compare_prefix(&model, &tokenizer, &tokens[..prefix_length], &tokens[prefix_length..end], &lexicon)?, "ngram_greedy_next32": ngram_text, "ngram_lexical_free_run": lexical_summary(decoded_prefix.chars().last(), &ngram_text, &lexicon), "ngram_reference_position_match_fraction": ngram_tokens.iter().zip(&tokens[prefix_length..end]).filter(|(a, b)| a == b).count() as f64 / 32.0}));
        }
    }
    let activity_windows: Vec<_> = validation_windows
        .iter()
        .take(16)
        .map(|window| window.tokens.clone())
        .collect();
    let activity = crate::text_activity::run(&model, &tokenizer, &activity_windows)?;
    let untrained_activity = crate::text_activity::run(&untrained, &tokenizer, &activity_windows)?;
    let rollout_activity = |field: &str| -> Result<Value, String> {
        let windows: Result<Vec<_>, String> = prefix_reports
            .iter()
            .filter(|case| case["prefix_characters"].as_u64() == Some(32))
            .map(|case| {
                let comparison = &case["comparison"];
                let prefix = comparison["prefix"]
                    .as_str()
                    .ok_or("missing rollout prefix")?;
                let continuation = comparison[field]
                    .as_str()
                    .ok_or("missing rollout continuation")?;
                Ok(tokenizer.encode(&format!("{prefix}{continuation}")))
            })
            .collect();
        crate::text_activity::run(&model, &tokenizer, &windows?)
    };
    let matched_rollout_activity = json!({
        "reference": rollout_activity("reference_next_characters")?,
        "greedy": rollout_activity("greedy_free_run_text")?,
        "sampled_temperature_0_8": rollout_activity("sampled_text")?,
        "scope": "Same first four validation documents:32-character prefix plus32-character continuation,63 actual input/prediction steps each. Summaries include the common prefix, not only the differing continuation. Self-generated next-character targets are part of each generated stream, so their output-loss scores are not held-out reference likelihoods.",
    });
    let after_hash = parameter_hash(&model)?;
    if before_hash != after_hash {
        return Err("frozen text diagnostic changed model parameters".into());
    }
    let report = json!({
        "experiment": "frozen_fineweb_text_diagnostic", "scope": "bounded character prediction/context/exposure diagnostics, not training or general semantic evaluation", "optimizer_updates": 0,
        "device": format!("{device:?}"), "device_identity": format!("{:?}", device.identity()), "precision": "f32 tensors; f64 stable logsumexp scoring from actual f32 logits",
        "checkpoint": {"path": model_path, "sha256": subject.actual_sha256, "expected_sha256": subject.expected_sha256, "parameters": model.parameter_count(), "feedback": model.feedback.as_str(), "parameter_f32_le_sha256_before": before_hash, "parameter_f32_le_sha256_after": after_hash, "unchanged": true},
        "data": {"source": manifest["source"], "manifest_sha256": sha256(&manifest_bytes), "training_file_sha256": train.file_hash, "validation_file_sha256": validation.file_hash, "training_documents": train.documents.len(), "validation_documents": validation.documents.len(), "vocabulary_size": tokenizer.vocab_size(), "vocabulary_matches_training_only_fit": true, "training_roundtrip": training_roundtrip, "validation_roundtrip": validation_roundtrip, "vocabulary_utf8_sha256": sha256(tokenizer.vocab().iter().collect::<String>().as_bytes())},
        "controls": {"sample_targets_per_split": WINDOWS * TARGETS, "bins_1_based": ["1-8", "9-16", "17-32", "33-64"], "trained_reset": "all four states zero BEFORE each input; one current-character input context", "untrained": "same current Model::new, same configuration and feedback, seed42; no optimizer", "untrained_initial_parameter_f32_le_sha256": parameter_hash(&untrained)?, "history_shuffling": "not performed; reset control only"},
        "ngram": {"maximum_order": MAX_ORDER, "maximum_previous_characters": MAX_ORDER - 1, "alpha": BACKOFF_ALPHA, "root": "additive-one frequencies over training targets only, excluding each document's first character", "recurrence": "p(c|h)=(count(h,c)+alpha*p(c|suffix(h)))/(N(h)+alpha), shortest to longest suffix; unseen context backs off unchanged", "fit_documents": train.documents.len(), "fit_target_count": ngram.total, "distinct_nonempty_contexts": ngram.contexts.len(), "distinct_context_next_entries": ngram.contexts.values().map(|counts| counts.next.len()).sum::<usize>(), "boundary": "counts never cross documents; scoring context is only the observed window prefix, maximum4 characters; reset control uses current input only", "meaning": "count baseline, not a second neural model or neural training implementation"},
        "sampled_split_metrics": split_reports, "real_validation_prefix_comparisons": prefix_reports, "training_ascii_lexicon_unique_words": lexicon.len(), "activity_first16_validation_windows": activity,
        "activity_untrained_same16_validation_windows": untrained_activity,
        "activity_matched_32prefix_rollouts": matched_rollout_activity,
        "whole_diagnostic_seconds_including_loading_scoring_and_activity": started.elapsed().as_secs_f64(),
        "limitations": ["Sampled windows can overlap and share documents; no independent-target confidence interval", "Known development validation split; no architecture tuning or checkpoint selection", "Frequency confusions and lexical hits are descriptive, not a semantic or grammar metric", "Teacher-forced and greedy-history loss use identical true targets but distinct histories after generated errors", "No parameters updated and no trained checkpoint saved"],
    });
    checked_output(&output)?;
    fs::create_dir_all(output.parent().ok_or("output needs parent")?)
        .map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|error| error.to_string())?;
    file.write_all(&serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?)
        .and_then(|()| file.sync_all())
        .map_err(|error| error.to_string())?;
    eprintln!("Saved frozen text diagnostic {}", output.display());
    Ok(output)
}

struct ProbeArguments {
    device: String,
    output: PathBuf,
    checkpoint: PathBuf,
    expected_checkpoint_sha256: String,
}

fn parse_arguments(args: &[String]) -> Result<ProbeArguments, String> {
    let mut parsed = ProbeArguments {
        device: "gpu".into(),
        output: "artifacts/text-diagnostic.json".into(),
        checkpoint: DEFAULT_CHECKPOINT.into(),
        expected_checkpoint_sha256: CHECKPOINT_SHA256.into(),
    };
    let mut explicit_checkpoint = false;
    let mut explicit_hash = false;
    let mut seen = BTreeSet::new();
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        if !seen.insert(flag) {
            return Err(format!("duplicate {flag}"));
        }
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--device" => parsed.device = value.clone(),
            "--output" => parsed.output = value.into(),
            "--checkpoint" => {
                parsed.checkpoint = value.into();
                explicit_checkpoint = true;
            }
            "--expected-checkpoint-sha256" => {
                parsed.expected_checkpoint_sha256 = checked_sha256(value)?;
                explicit_hash = true;
            }
            _ => return Err(format!("unknown {flag}")),
        }
    }
    if explicit_checkpoint != explicit_hash {
        return Err(
            "--checkpoint and --expected-checkpoint-sha256 must be provided together".into(),
        );
    }
    if !matches!(parsed.device.as_str(), "gpu" | "cpu") {
        return Err("--device must be gpu or cpu".into());
    }
    Ok(parsed)
}

pub fn cli(args: Vec<String>) -> Result<(), String> {
    if args == ["--help"] || args == ["-h"] {
        println!(
            "Usage: text_probe [--device gpu|cpu] [--output artifacts/text-diagnostic.json] [--checkpoint artifacts/model.bin --expected-checkpoint-sha256 HASH]"
        );
        return Ok(());
    }
    let parsed = parse_arguments(&args)?;
    run_configured(
        &parsed.device,
        &parsed.output,
        &parsed.checkpoint,
        &parsed.expected_checkpoint_sha256,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    #[test]
    fn windows_are_bounded_deterministic_and_document_contained() {
        let documents = vec![vec![1; 65], (0..200).map(|index| index % 3).collect()];
        let first = select_windows(&documents).unwrap();
        let second = select_windows(&documents).unwrap();
        assert_eq!(first.len() * TARGETS, 4096);
        for (a, b) in first.iter().zip(second) {
            assert_eq!((a.document, a.start), (b.document, b.start));
            assert_eq!(a.tokens, documents[a.document][a.start..a.start + 65]);
        }
        let mut metrics = Metrics::default();
        for time in 0..TARGETS {
            metrics.observe(
                time,
                1,
                1,
                Scored {
                    prediction: 1,
                    nll: 0.0,
                    true_probability: 1.0,
                    true_rank: 1,
                },
            );
        }
        assert_eq!(
            metrics
                .bins
                .iter()
                .map(|bin| bin.targets)
                .collect::<Vec<_>>(),
            [8, 8, 16, 32]
        );
        assert!(select_windows(&[vec![1; 64]]).is_err());
    }

    #[test]
    fn ngram_has_normalized_backoff_no_cross_document_counts_or_first_target_leak() {
        let model = Ngram::fit(&[vec![1, 2], vec![3, 1]], 4);
        assert_eq!(model.total, 2);
        assert_eq!(model.root, [0, 1, 1, 0]);
        assert!(!model.contexts.contains_key(&vec![2, 3]));
        for prefix in [vec![], vec![0], vec![1], vec![2, 3, 1], vec![0; 20]] {
            let probabilities = model.probabilities(&prefix);
            assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(probabilities.iter().all(|&probability| probability > 0.0));
        }
        assert_eq!(model.probabilities(&[0]), model.probabilities(&[]));
    }

    #[test]
    fn lexical_counts_exclude_truncated_words_and_do_not_claim_semantics() {
        let lexicon = ascii_words("Cat dog. CAT!").collect();
        let statistics = lexical_summary(Some('h'), "ing cat dog", &lexicon);
        assert_eq!(statistics["eligible_complete_ascii_words"], 1);
        assert_eq!(statistics["training_lexicon_hits"], 1);
        let statistics = lexical_summary(Some(' '), "Cat zorb.", &lexicon);
        assert_eq!(statistics["hit_rate"], 0.5);
        assert!(lexical_summary(None, "x", &lexicon)["hit_rate"].is_null());
    }

    #[test]
    fn scored_windows_match_production_loss_and_one_character_reset_equivalence() {
        let tokenizer = Tokenizer::from_text("abc");
        let device = Device::flex();
        let model = Model::new(
            ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                hidden_size: 2,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            42,
            &device,
        )
        .unwrap();
        let windows: Vec<_> = (0..2)
            .map(|offset| Window {
                document: offset,
                start: 0,
                tokens: (0..65).map(|index| 1 + (index + offset) % 3).collect(),
            })
            .collect();
        let samples: Vec<_> = windows
            .iter()
            .map(|window| window.tokens.as_slice())
            .collect();
        let batch = training::build_batch(&samples, TARGETS, &device).unwrap();
        let expected = model.loss(&batch).unwrap().into_scalar::<f32>() as f64;
        let measured = neural_metrics(&model, &windows, &tokenizer, false).unwrap();
        assert!(
            (measured["overall"]["nll_nats_per_character"]
                .as_f64()
                .unwrap()
                - expected)
                .abs()
                < 2e-6
        );
        let pairs: Vec<_> = windows
            .iter()
            .flat_map(|window| window.tokens.windows(2))
            .collect();
        let batch = training::build_batch(&pairs, 1, &device).unwrap();
        let expected = model.loss(&batch).unwrap().into_scalar::<f32>() as f64;
        let measured = neural_metrics(&model, &windows, &tokenizer, true).unwrap();
        assert!(
            (measured["overall"]["nll_nats_per_character"]
                .as_f64()
                .unwrap()
                - expected)
                .abs()
                < 2e-6
        );
    }

    #[test]
    fn same_prefix_scoring_and_greedy_match_production_without_updates() {
        let tokenizer = Tokenizer::from_text("abc?");
        let model = Model::new(
            ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                hidden_size: 2,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            42,
            &Device::flex(),
        )
        .unwrap();
        let before = parameter_hash(&model).unwrap();
        let result = compare_prefix(
            &model,
            &tokenizer,
            &tokenizer.encode("abc?"),
            &tokenizer.encode("abca"),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(
            result["ordinary_generate_matches_manual_step_rollout"],
            true
        );
        assert_eq!(
            result["position_scores"][0]["teacher_forced_nll"],
            result["position_scores"][0]["free_history_nll"]
        );
        assert_eq!(parameter_hash(&model).unwrap(), before);
        let certain = score_logits(&[-1000.0, 1000.0], 1).unwrap();
        assert_eq!(certain.nll, 0.0);
        assert_eq!(certain.true_rank, 1);
        assert_eq!(score_logits(&[0.0, 0.0], 1).unwrap().true_rank, 1);
        assert_eq!(score_logits(&[-1000.0, 1000.0], 0).unwrap().nll, 2000.0);
    }

    #[test]
    fn output_rejects_outside_and_checkpoint_paths() {
        for path in [
            "../outside.json",
            "src/result.json",
            "artifacts/liminal.bin",
            "artifacts/fineweb-language.bin",
        ] {
            assert!(checked_output(Path::new(path)).is_err());
        }
        let path = PathBuf::from(format!(
            "artifacts/text-probe-output-test-{}.json",
            std::process::id()
        ));
        let output = checked_output(&path).unwrap();
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .unwrap();
        drop(file);
        assert!(checked_output(&path).is_err());
        fs::remove_file(output).unwrap();
    }

    #[test]
    fn checkpoint_arguments_are_paired_and_preserve_default_subject() {
        let default = parse_arguments(&[]).unwrap();
        assert_eq!(default.device, "gpu");
        assert_eq!(default.output, Path::new("artifacts/text-diagnostic.json"));
        assert_eq!(default.checkpoint, Path::new(DEFAULT_CHECKPOINT));
        assert_eq!(default.expected_checkpoint_sha256, CHECKPOINT_SHA256);
        for args in [
            vec!["--checkpoint".into(), DEFAULT_CHECKPOINT.into()],
            vec![
                "--expected-checkpoint-sha256".into(),
                CHECKPOINT_SHA256.into(),
            ],
            vec![
                "--checkpoint".into(),
                DEFAULT_CHECKPOINT.into(),
                "--expected-checkpoint-sha256".into(),
                "xyz".into(),
            ],
        ] {
            assert!(parse_arguments(&args).is_err());
        }
        let selected = parse_arguments(&[
            "--checkpoint".into(),
            "artifacts/fineweb-voltage.bin".into(),
            "--expected-checkpoint-sha256".into(),
            CHECKPOINT_SHA256.to_ascii_uppercase(),
        ])
        .unwrap();
        assert_eq!(
            selected.checkpoint,
            Path::new("artifacts/fineweb-voltage.bin")
        );
        assert_eq!(selected.expected_checkpoint_sha256, CHECKPOINT_SHA256);
        assert!(checked_sha256(&"g".repeat(64)).is_err());
        assert!(checked_sha256(&"0".repeat(63)).is_err());
        assert!(checked_sha256(&"0".repeat(65)).is_err());
    }

    #[test]
    fn checkpoint_subject_path_and_hash_are_checked_without_device_initialization() {
        let relative = PathBuf::from(format!(
            "artifacts/text-probe-subject-test-{}.bin",
            std::process::id()
        ));
        let absolute = Path::new(env!("CARGO_MANIFEST_DIR")).join(&relative);
        fs::create_dir_all(absolute.parent().unwrap()).unwrap();
        let bytes = b"frozen text diagnostic subject hash fixture";
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&absolute)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        drop(file);
        let expected = sha256(bytes);
        let checked = checked_checkpoint(&relative, &expected.to_ascii_uppercase()).unwrap();
        assert_eq!(checked.path, absolute);
        assert_eq!(checked.actual_sha256, expected);
        assert_eq!(checked.expected_sha256, expected);
        assert!(checked_checkpoint(&relative, &"0".repeat(64)).is_err());
        assert!(checked_checkpoint(&absolute, &expected).is_ok());
        for invalid in [
            Path::new("../outside.bin"),
            Path::new("src/checkpoint.bin"),
            Path::new("artifacts/checkpoint.json"),
            Path::new("artifacts/nonexistent-text-probe-subject.bin"),
        ] {
            assert!(checked_checkpoint(invalid, &expected).is_err());
        }
        fs::remove_file(absolute).unwrap();
    }
}
