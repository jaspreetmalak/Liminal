//! Genuine FineWeb passage memorization with the production model and trainer.
//! This bounded diagnostic does not establish fluency or select on validation.

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use burn::module::Module;
use burn::tensor::{Int, Tensor};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::data::Tokenizer;
use crate::model::{FeedbackMode, Model, ModelConfig, State};
use crate::{checkpoint, runtime, training};

const MAX_STEPS: usize = 800;
const HIDDEN_SIZE: usize = 64;
const SEED: u64 = 42;
const LEARNING_RATE: f64 = 0.003;
const GRADIENT_CLIP: f64 = 1.0;
const MAX_PASSAGE_CHARACTERS: usize = 128;

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug, Clone)]
struct Passage {
    text: String,
    byte_offset: usize,
    character_offset: usize,
}

/// Select an unchanged complete sentence within a source line. Leading pipe
/// table/header lines are skipped. A boundary is .!? followed by whitespace or
/// line end; sentences over the cap are skipped, never truncated or rewritten.
fn select_passage(text: &str) -> Result<Passage, String> {
    let mut line_offset = 0;
    for original_line in text.split_inclusive('\n') {
        let line = original_line.trim_end_matches(['\r', '\n']);
        let line_start = line.len() - line.trim_start().len();
        if !line[line_start..].starts_with('|') {
            let mut sentence_start = line_start;
            for (offset, character) in line.char_indices() {
                if offset < sentence_start || !matches!(character, '.' | '!' | '?') {
                    continue;
                }
                let end = offset + character.len_utf8();
                if line[end..]
                    .chars()
                    .next()
                    .is_some_and(|next| !next.is_whitespace())
                {
                    continue;
                }
                let sentence = &line[sentence_start..end];
                if (2..=MAX_PASSAGE_CHARACTERS).contains(&sentence.chars().count()) {
                    let byte_offset = line_offset + sentence_start;
                    return Ok(Passage {
                        text: sentence.to_owned(),
                        byte_offset,
                        character_offset: text[..byte_offset].chars().count(),
                    });
                }
                sentence_start = end + line[end..].len() - line[end..].trim_start().len();
            }
        }
        line_offset += original_line.len();
    }
    Err("first source document has no complete 2..=128-character sentence".into())
}

struct Corpus {
    path: PathBuf,
    documents: Vec<String>,
    ids: BTreeSet<String>,
    file_sha256: String,
    first_row: Value,
    first_row_sha256: String,
}

impl Corpus {
    fn from_bytes(path: PathBuf, bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes).map_err(|error| format!("corpus UTF-8: {error}"))?;
        let mut documents = Vec::new();
        let mut ids = BTreeSet::new();
        let mut first_row = Value::Null;
        let mut first_row_sha256 = String::new();
        for (index, line) in text.lines().enumerate() {
            let row: Value = serde_json::from_str(line)
                .map_err(|error| format!("invalid corpus JSONL line {}: {error}", index + 1))?;
            let document = row["text"].as_str().ok_or("FineWeb row needs text")?;
            let id = row["id"]
                .as_str()
                .ok_or("FineWeb row needs its source id")?;
            if document.chars().count() < 2 || id.is_empty() || !ids.insert(id.to_owned()) {
                return Err("FineWeb rows need unique ids and at least two text characters".into());
            }
            documents.push(document.to_owned());
            if index == 0 {
                first_row_sha256 = sha256(line.as_bytes());
                first_row = row;
            }
        }
        if documents.is_empty() {
            return Err("FineWeb corpus is empty".into());
        }
        Ok(Self {
            path,
            documents,
            ids,
            file_sha256: sha256(bytes),
            first_row,
            first_row_sha256,
        })
    }

    fn load(path: PathBuf) -> Result<Self, String> {
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        Self::from_bytes(path, &bytes)
    }

    fn source(&self, manifest: &Value, split: &str, passage: &Passage) -> Result<Value, String> {
        let id = self.first_row["id"].as_str().ok_or("missing source id")?;
        let entry = manifest["documents"]
            .as_array()
            .and_then(|documents| {
                documents
                    .iter()
                    .find(|entry| entry["id"].as_str() == Some(id))
            })
            .ok_or("first source document is absent from the FineWeb manifest")?;
        let document_hash = sha256(self.documents[0].as_bytes());
        if entry["split"].as_str() != Some(split)
            || entry["text_sha256"].as_str() != Some(document_hash.as_str())
        {
            return Err("first source document disagrees with its FineWeb manifest".into());
        }
        Ok(json!({
            "file": self.path,
            "jsonl_line_index": 0,
            "source_row_index": entry["row_idx"],
            "id": id,
            "url": self.first_row["url"],
            "date": self.first_row["date"],
            "dump": self.first_row["dump"],
            "file_path": self.first_row["file_path"],
            "language": self.first_row["language"],
            "document_text_utf8_sha256": document_hash,
            "jsonl_row_utf8_sha256_excluding_line_ending": self.first_row_sha256,
            "selection": "first complete <=128-character sentence, skipping pipe-header lines; unchanged substring",
            "passage": passage.text,
            "passage_characters": passage.text.chars().count(),
            "passage_utf8_sha256": sha256(passage.text.as_bytes()),
            "document_byte_offset": passage.byte_offset,
            "document_character_offset": passage.character_offset,
        }))
    }
}

fn validate_splits(train: &Corpus, validation: &Corpus) -> Result<(), String> {
    let training_texts: BTreeSet<&str> = train.documents.iter().map(String::as_str).collect();
    if !train.ids.is_disjoint(&validation.ids)
        || validation
            .documents
            .iter()
            .any(|document| training_texts.contains(document.as_str()))
    {
        return Err(
            "FineWeb training/validation must have disjoint whole documents and ids".into(),
        );
    }
    Ok(())
}

fn teacher_forced(model: &Model, tokens: &[usize], tokenizer: &Tokenizer) -> Result<Value, String> {
    let started = Instant::now();
    let nll = training::evaluate(model, &[tokens.to_vec()], tokens.len() - 1, 1)?;
    let model = model.clone().valid();
    let device = model.device();
    let mut state = State::new(1, model.config.hidden_size, &device);
    let mut predictions = Vec::with_capacity(tokens.len() - 1);
    for &token in &tokens[..tokens.len() - 1] {
        let logits = model
            .step(
                Tensor::<1, Int>::from_data([token as i32], &device),
                &mut state,
            )?
            .try_into_vec_as::<f32>()
            .map_err(|error| format!("reading teacher-forced logits: {error}"))?;
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("teacher-forced logits are non-finite".into());
        }
        let mut best = 0;
        for index in 1..logits.len() {
            if logits[index] > logits[best] {
                best = index;
            }
        }
        predictions.push(best);
    }
    let correct = predictions
        .iter()
        .zip(&tokens[1..])
        .filter(|(a, b)| a == b)
        .count();
    Ok(json!({
        "answer_nll_nats": nll,
        "perplexity": nll.exp(),
        "next_character_accuracy": correct as f64 / predictions.len() as f64,
        "correct_targets": correct,
        "targets": predictions.len(),
        "predicted_next_characters": tokenizer.decode(&predictions)?,
        "wall_seconds_including_host_readbacks": started.elapsed().as_secs_f64(),
    }))
}

fn fit_met(metrics: &Value) -> bool {
    metrics["next_character_accuracy"].as_f64() == Some(1.0)
        && metrics["answer_nll_nats"]
            .as_f64()
            .is_some_and(|nll| nll <= 0.05)
}

fn greedy_completion(
    model: &Model,
    tokenizer: &Tokenizer,
    text: &str,
    prefix_length: usize,
) -> Result<Value, String> {
    let characters: Vec<_> = text.chars().collect();
    let split = prefix_length.min(characters.len() - 1);
    let prefix: String = characters[..split].iter().collect();
    let expected: String = characters[split..].iter().collect();
    let predicted_tokens = training::generate(
        model,
        &tokenizer.encode(&prefix),
        characters.len() - split,
        training::Decoding::sampled(0.0),
        SEED,
    )?;
    let completion = tokenizer.decode(&predicted_tokens)?;
    let correct = completion
        .chars()
        .zip(expected.chars())
        .filter(|(a, b)| a == b)
        .count();
    Ok(json!({
        "prompt": prefix,
        "prompt_characters": split,
        "expected_completion": expected,
        "completion": completion,
        "generated_text": format!("{prefix}{completion}"),
        "completion_characters": characters.len() - split,
        "position_match_fraction": correct as f64 / (characters.len() - split) as f64,
        "full_match": completion == expected,
        "temperature": 0,
        "generation_length": "fixed to the known remaining passage length; no EOS or stopping behavior is learned",
    }))
}

#[derive(Debug)]
pub struct LanguageProbeConfig {
    pub device: String,
    pub steps: usize,
    pub output: PathBuf,
}

impl Default for LanguageProbeConfig {
    fn default() -> Self {
        Self {
            device: "gpu".into(),
            steps: MAX_STEPS,
            output: "artifacts/fineweb-overfit.json".into(),
        }
    }
}

impl LanguageProbeConfig {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut config = Self::default();
        let mut seen = BTreeSet::new();
        let mut options = args.iter();
        while let Some(flag) = options.next() {
            if !seen.insert(flag.as_str()) {
                return Err(format!("duplicate argument {flag}"));
            }
            let value = options
                .next()
                .ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--device" => config.device = value.clone(),
                "--steps" => {
                    config.steps = value.parse().map_err(|_| "--steps needs an integer")?
                }
                "--output" => config.output = value.into(),
                _ => return Err(format!("unknown argument {flag}")),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if !matches!(self.device.as_str(), "gpu" | "cpu") || !(1..=MAX_STEPS).contains(&self.steps)
        {
            return Err("--device must be gpu|cpu and --steps must be in 1..=800".into());
        }
        checked_outputs(&self.output)?;
        Ok(())
    }
}

fn checked_outputs(path: &Path) -> Result<(PathBuf, PathBuf), String> {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = project.canonicalize().map_err(|error| error.to_string())?;
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| component == Component::ParentDir)
        || path.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err(
            "--output must be a new project-local artifacts/*.json path without '..'".into(),
        );
    }
    let output = if path.is_absolute() {
        path.to_path_buf()
    } else {
        project.join(path)
    };
    if !output.starts_with(project.join("artifacts")) {
        return Err("language diagnostic output must stay inside project/artifacts".into());
    }
    let model = output.with_extension("bin");
    if model
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("liminal.bin"))
    {
        return Err("language diagnostic must not replace liminal.bin".into());
    }
    for target in [&output, &model] {
        if fs::symlink_metadata(target).is_ok() {
            return Err(format!(
                "diagnostic output already exists: {}",
                target.display()
            ));
        }
    }
    let mut ancestor = output.parent().ok_or("output needs a parent directory")?;
    while fs::symlink_metadata(ancestor).is_err() {
        ancestor = ancestor.parent().ok_or("cannot resolve output parent")?;
    }
    if !ancestor
        .canonicalize()
        .map_err(|error| format!("resolving output ancestor: {error}"))?
        .starts_with(root)
    {
        return Err("--output resolves outside the project root".into());
    }
    Ok((output, model))
}

/// Fit one genuine source passage. Validation is scored only before/after the
/// run and never participates in updates, stopping, or vocabulary fitting.
pub fn run(config: &LanguageProbeConfig) -> Result<PathBuf, String> {
    config.validate()?;
    let (output, model_output) = checked_outputs(&config.output)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let corpus_root = root.join("data/fineweb");
    let manifest_bytes = fs::read(corpus_root.join("manifest.json"))
        .map_err(|error| format!("reading prepared FineWeb manifest: {error}"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("FineWeb manifest: {error}"))?;
    if manifest["source"]["dataset"].as_str() != Some("HuggingFaceFW/fineweb") {
        return Err("language diagnostic requires the prepared original FineWeb corpus".into());
    }
    let train = Corpus::load(corpus_root.join("train.jsonl"))?;
    let validation = Corpus::load(corpus_root.join("validation.jsonl"))?;
    for (name, corpus) in [("train.jsonl", &train), ("validation.jsonl", &validation)] {
        if manifest["files"][name]["sha256"].as_str() != Some(corpus.file_sha256.as_str()) {
            return Err(format!("{name} does not match its prepared FineWeb hash"));
        }
    }
    validate_splits(&train, &validation)?;
    let passage = select_passage(&train.documents[0])?;
    let validation_passage = select_passage(&validation.documents[0])?;
    let training_source = train.source(&manifest, "train", &passage)?;
    let validation_source = validation.source(&manifest, "validation", &validation_passage)?;
    let tokenizer = Tokenizer::from_documents(&train.documents);
    let train_tokens = tokenizer.encode(&passage.text);
    let validation_tokens: Vec<_> = validation
        .documents
        .iter()
        .map(|document| tokenizer.encode(document))
        .collect();
    let validation_characters: usize = validation_tokens.iter().map(Vec::len).sum();
    let validation_unknown: usize = validation_tokens
        .iter()
        .flatten()
        .filter(|&&token| token == 0)
        .count();
    let length = train_tokens.len() - 1;
    let unigram = training::unigram_loss(
        std::slice::from_ref(&train_tokens),
        &validation_tokens,
        tokenizer.vocab_size(),
    )?;
    let device = runtime::device(&config.device)?;
    eprintln!(
        "FineWeb passage device: {device:?}; identity: {:?}",
        device.identity()
    );
    println!(
        "Selected {}-character genuine training sentence: {}",
        train_tokens.len(),
        passage.text
    );
    let mut model = Model::new(
        ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_size: HIDDEN_SIZE,
            threshold: 1.0,
            surrogate_width: 1.0,
        },
        SEED,
        &device,
    )?;
    model.feedback = FeedbackMode::Both;
    let initial_parameters: Vec<_> = model
        .parameter_values()?
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let initial_train = teacher_forced(&model, &train_tokens, &tokenizer)?;
    let initial_validation = training::evaluate(&model, &validation_tokens, length, 16)?;
    model = model.train();
    let batch = training::build_batch(&[train_tokens.as_slice()], length, &model.device())?;
    let mut optimizer = training::adam();
    let mut checks = Vec::new();
    let mut updates = 0;
    let started = Instant::now();
    for step in 1..=config.steps {
        let check = step % 25 == 0 || step == config.steps;
        let (loss, norm) = training::optimize_batch(
            &mut model,
            &batch,
            &mut optimizer,
            LEARNING_RATE,
            GRADIENT_CLIP,
            check,
        )?;
        updates = step;
        if check {
            let metrics = teacher_forced(&model, &train_tokens, &tokenizer)?;
            eprintln!(
                "passage step {step}/{}: train NLL {:.6}, accuracy {:.3}%, gradient norm {norm:.6}",
                config.steps,
                metrics["answer_nll_nats"].as_f64().unwrap(),
                100.0 * metrics["next_character_accuracy"].as_f64().unwrap()
            );
            let passed = fit_met(&metrics);
            checks.push(json!({"step": step, "pre_update_batch_nll": loss, "preclip_global_gradient_norm": norm, "post_update_training": metrics, "training_fit_met": passed}));
            if passed {
                break;
            }
        }
    }
    model.parameter_values()?;
    let training_seconds = started.elapsed().as_secs_f64();
    let final_train = teacher_forced(&model, &train_tokens, &tokenizer)?;
    let final_validation = training::evaluate(&model, &validation_tokens, length, 16)?;
    let from_eight = greedy_completion(&model, &tokenizer, &passage.text, 8)?;
    let from_one = greedy_completion(&model, &tokenizer, &passage.text, 1)?;
    let validation_greedy = greedy_completion(&model, &tokenizer, &validation_passage.text, 8)?;
    println!(
        "Greedy training continuation: {}",
        from_eight["generated_text"].as_str().unwrap()
    );
    println!(
        "Greedy validation continuation: {}",
        validation_greedy["generated_text"].as_str().unwrap()
    );
    println!(
        "Validation NLL: initial {initial_validation:.6}, final {final_validation:.6}, passage-fitted unigram {unigram:.6}"
    );
    fs::create_dir_all(output.parent().ok_or("output needs parent")?)
        .map_err(|error| error.to_string())?;
    checkpoint::save(&model_output, &model, &tokenizer)?;
    let model_bytes = fs::read(&model_output).map_err(|error| error.to_string())?;
    let report = json!({
        "experiment": "genuine_fineweb_training_passage_overfit",
        "scope": "Exact training-passage memorization diagnostic; validation is descriptive, not a stopping criterion or fluency test",
        "created_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_secs(),
        "device": format!("{device:?}"), "device_identity": format!("{:?}", device.identity()), "precision": "f32",
        "data": {"source": manifest["source"], "manifest_sha256": sha256(&manifest_bytes), "train_file_sha256": train.file_sha256, "validation_file_sha256": validation.file_sha256, "training_documents": train.documents.len(), "validation_documents": validation.documents.len(), "vocabulary_source": "all training documents only; parameter updates use selected passage only", "vocabulary": tokenizer.vocab().iter().collect::<String>(), "vocab_size": tokenizer.vocab_size(), "training_passage": training_source, "validation_excerpt": validation_source, "validation_characters": validation_characters, "validation_unknown_characters": validation_unknown, "validation_unknown_fraction": validation_unknown as f64 / validation_characters as f64, "whole_document_splits_disjoint": true},
        "configuration": {"model_seed": SEED, "hidden_size": HIDDEN_SIZE, "batch_size": 1, "sequence_length": length, "maximum_steps": config.steps, "learning_rate": LEARNING_RATE, "global_gradient_clip": GRADIENT_CLIP, "feedback_mode": model.feedback.as_str(), "loss": "mean full-vocabulary cross entropy over every within-passage adjacent character pair", "threshold": model.config.threshold, "surrogate_width": model.config.surrogate_width, "decays": "learned per neuron, 0.995*sigmoid(logit); initial tau log-uniform in [2, 32] steps", "reset": "all states zero for every full passage update and evaluation window", "optimizer": {"name": "Adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0}, "training_check_interval": 25, "training_stop": "100% teacher-forced next-character accuracy and NLL<=0.05; train-only checks every25 and cap", "validation_window_length": length, "validation_batch_size": 16},
        "parameters": model.parameter_count(), "initial_parameters_f32_le_sha256": sha256(&initial_parameters),
        "initial_training": initial_train, "final_training": final_train, "training_fit_met": fit_met(&final_train), "training_checks": checks,
        "updates": updates, "training_prediction_targets": updates * length, "training_wall_seconds_including_training_checks_and_final_parameter_readback": training_seconds,
        "initial_validation_nll_nats": initial_validation, "final_validation_nll_nats": final_validation, "passage_target_unigram_validation_nll_nats": unigram, "unigram_fit": "additive-one smoothing over V entries; counts only selected training passage targets, excluding its first character",
        "greedy_training_prefix_eight": from_eight, "greedy_training_first_character": from_one, "greedy_validation_prefix_eight": validation_greedy,
        "checkpoint": {"path": model_output, "sha256": sha256(&model_bytes), "contains": "one model, vocabulary, configuration and feedback mode; optimizer/sampler state not saved"},
        "existing_language_checkpoint_updated": false,
    });
    let mut report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|error| format!("creating report: {error}"))?;
    report_file
        .write_all(&serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?)
        .and_then(|()| report_file.sync_all())
        .map_err(|error| format!("writing report: {error}"))?;
    eprintln!(
        "Saved FineWeb overfit report {} and model {}",
        output.display(),
        model_output.display()
    );
    Ok(output)
}

pub fn cli(args: Vec<String>) -> Result<(), String> {
    if args == ["--help"] || args == ["-h"] {
        println!(
            "Usage: language_probe [--device gpu|cpu] [--steps 1..800] [--output artifacts/fineweb-overfit.json]"
        );
        return Ok(());
    }
    run(&LanguageProbeConfig::parse(&args)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn sentence_selection_preserves_original_substring_and_unicode_offsets() {
        let source = "|Header. Skip this.|\n|Other header!|\n  Don't care about Chloe/Taniel/Jen-Jen. Another sentence.";
        let selected = select_passage(source).unwrap();
        assert_eq!(selected.text, "Don't care about Chloe/Taniel/Jen-Jen.");
        assert_eq!(selected.text.chars().count(), 38);
        assert_eq!(
            sha256(selected.text.as_bytes()),
            "c12578c6e0015c9a703687d54a3432edea2b073dffd3c8a414611ec76b0f83ce"
        );
        assert_eq!(
            &source[selected.byte_offset..selected.byte_offset + selected.text.len()],
            selected.text
        );
        assert_eq!(
            source[..selected.byte_offset].chars().count(),
            selected.character_offset
        );
        let unicode = select_passage("|é. header|\n  Élan works!\n").unwrap();
        assert_eq!(unicode.text, "Élan works!");
        assert_ne!(unicode.byte_offset, unicode.character_offset);
        assert!(select_passage(&format!("{}.", "x".repeat(128))).is_err());
        assert!(select_passage("|Only header.|\nNo complete sentence").is_err());
    }

    fn fixture(id: &str, text: &str) -> Corpus {
        let bytes = serde_json::to_vec(&json!({"id": id, "text": text})).unwrap();
        Corpus::from_bytes("fixture.jsonl".into(), &bytes).unwrap()
    }

    #[test]
    fn provenance_splits_and_training_only_vocabulary_are_enforced() {
        let train = fixture("train", "Alpha text.");
        let validation = fixture("validation", "Ω held out.");
        validate_splits(&train, &validation).unwrap();
        assert!(validate_splits(&train, &fixture("other", "Alpha text.")).is_err());
        assert!(validate_splits(&train, &fixture("train", "Other text.")).is_err());
        let vocabulary = Tokenizer::from_documents(&train.documents);
        assert_eq!(vocabulary.encode("Ω"), vec![0]);
        let passage = select_passage(&train.documents[0]).unwrap();
        let mut manifest = json!({"documents": [{"id": "train", "split": "train", "row_idx": 7, "text_sha256": sha256(train.documents[0].as_bytes())}]});
        let source = train.source(&manifest, "train", &passage).unwrap();
        assert_eq!(source["source_row_index"], 7);
        assert_eq!(
            source["passage_utf8_sha256"],
            sha256(passage.text.as_bytes())
        );
        manifest["documents"][0]["split"] = json!("validation");
        assert!(train.source(&manifest, "train", &passage).is_err());
    }

    #[test]
    fn passage_batch_uses_adjacent_targets_and_loss_resets_state() {
        let device = Device::flex();
        let tokenizer = Tokenizer::from_text("A real sentence.");
        let tokens = tokenizer.encode("A real sentence.");
        let batch = training::build_batch(&[tokens.as_slice()], tokens.len() - 1, &device).unwrap();
        assert_eq!(
            batch.inputs.clone().try_into_vec_as::<i32>().unwrap(),
            tokens[..tokens.len() - 1]
                .iter()
                .map(|&token| token as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            batch.targets.clone().try_into_vec_as::<i32>().unwrap(),
            tokens[1..]
                .iter()
                .map(|&token| token as i32)
                .collect::<Vec<_>>()
        );
        assert!(
            batch
                .mask
                .try_into_vec_as::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 1.0)
        );
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
        let first = teacher_forced(&model, &tokens, &tokenizer).unwrap();
        let second = teacher_forced(&model, &tokens, &tokenizer).unwrap();
        assert_eq!(first["answer_nll_nats"], second["answer_nll_nats"]);
        assert_eq!(
            first["predicted_next_characters"],
            second["predicted_next_characters"]
        );
    }

    #[test]
    fn cli_rejects_outside_existing_and_protected_model_outputs() {
        assert!(LanguageProbeConfig::parse(&["--steps".into(), "0".into()]).is_err());
        assert!(LanguageProbeConfig::parse(&["--steps".into(), "801".into()]).is_err());
        assert!(LanguageProbeConfig::parse(&["--device".into(), "cuda".into()]).is_err());
        for path in [
            "../outside.json",
            "src/result.json",
            "artifacts/liminal.json",
            "artifacts/liminal.bin",
        ] {
            assert!(checked_outputs(Path::new(path)).is_err());
        }
        let fresh = format!(
            "artifacts/language-probe-test-output-{}.json",
            std::process::id()
        );
        let (_, checkpoint) = checked_outputs(Path::new(&fresh)).unwrap();
        assert_eq!(checkpoint.extension().unwrap(), "bin");
        fs::create_dir_all(checkpoint.parent().unwrap()).unwrap();
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&checkpoint)
            .unwrap();
        drop(file);
        assert!(checked_outputs(Path::new(&fresh)).is_err());
        fs::remove_file(checkpoint).unwrap();
    }
}
