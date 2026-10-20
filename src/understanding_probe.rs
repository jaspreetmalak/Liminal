//! Frozen, bounded fact-consistency and real-text context diagnostics.
//! All likelihoods use production Model::step, with no optimizer updates.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use burn::tensor::Device;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::continuation_score::{Continuation, score};
use crate::data::Tokenizer;
use crate::model::{FeedbackMode, Model};
use crate::rng::Rng;
use crate::understanding_cases::{ContrastPair, dataset};
use crate::{checkpoint, runtime};

const SUBJECTS: [(&str, &str, &str); 2] = [
    (
        "ce_control",
        "fineweb-voltage-control.bin",
        "fc7b21d3565b242393c0cb330cd862daf75908ffb361b7848338821bad685ed0",
    ),
    (
        "voltage_penalty",
        "fineweb-voltage.bin",
        "53cdccd0cf8198fcc54a4cfb16e6085b5b21052109f22ab6560454558fc40f88",
    ),
];
const PREFIX: usize = 128;
const TARGETS: usize = 16;
const WINDOWS: usize = 32;
const SEED: u64 = 42;
const BATCH: usize = 16;

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
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    ))
}

struct Corpus {
    texts: Vec<String>,
    ids: Vec<String>,
    hash: String,
}

fn load_corpus(path: &Path, expected: &str) -> Result<Corpus, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let hash = sha256(&bytes);
    if hash != expected {
        return Err(format!("{} differs from fixed source hash", path.display()));
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let mut texts = Vec::new();
    let mut ids = Vec::new();
    for line in text.lines() {
        let row: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        texts.push(
            row["text"]
                .as_str()
                .ok_or("missing document text")?
                .to_owned(),
        );
        ids.push(row["id"].as_str().ok_or("missing document ID")?.to_owned());
    }
    if texts.is_empty()
        || texts.iter().any(|text| text.chars().count() < 2)
        || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
    {
        return Err("corpus is empty, short, or repeats source IDs".into());
    }
    Ok(Corpus { texts, ids, hash })
}

fn corpus_summary(corpus: &Corpus, manifest: &Value) -> Value {
    let lengths: Vec<_> = corpus.texts.iter().map(|s| s.chars().count()).collect();
    let total: usize = lengths.iter().sum();
    let eligible: usize = lengths.iter().map(|&n| n.saturating_sub(64)).sum();
    let mut indices: Vec<_> = (0..lengths.len()).collect();
    indices.sort_by_key(|&i| std::cmp::Reverse(lengths[i]));
    let words: usize = corpus
        .texts
        .iter()
        .map(|text| {
            text.split(|c: char| !c.is_ascii_alphabetic())
                .filter(|word| !word.is_empty())
                .count()
        })
        .sum();
    let top: Vec<_> = indices.iter().take(10).map(|&i| {
        let source = manifest["documents"].as_array().and_then(|docs| docs.iter().find(|doc| doc["id"].as_str() == Some(corpus.ids[i].as_str()))).cloned();
        json!({"document_index": i, "source_id": corpus.ids[i], "characters": lengths[i], "character_fraction": lengths[i] as f64 / total as f64, "training_length64_eligible_start_fraction": lengths[i].saturating_sub(64) as f64 / eligible as f64, "manifest_source": source})
    }).collect();
    let concentration = |n: usize| json!({"documents": n.min(indices.len()), "character_fraction": indices.iter().take(n).map(|&i| lengths[i]).sum::<usize>() as f64 / total as f64, "length64_eligible_start_fraction": indices.iter().take(n).map(|&i| lengths[i].saturating_sub(64)).sum::<usize>() as f64 / eligible as f64});
    json!({"documents": lengths.len(), "characters_unicode_scalars": total, "ascii_letter_runs": words, "word_count_definition": "maximal nonempty ASCII alphabetic runs, not model tokens", "file_sha256": corpus.hash, "length64_eligible_starts": eligible, "top_documents": top, "top5": concentration(5), "top10": concentration(10)})
}

struct Window {
    document: usize,
    start: usize,
    raw_prefix: String,
    raw_targets: String,
    prefix: Vec<usize>,
    targets: Vec<usize>,
    short_shuffled: Vec<usize>,
    long_shuffled: Vec<usize>,
}

fn shuffle_prefix(tokens: &[usize], random: &mut Rng) -> Vec<usize> {
    let mut output = tokens.to_vec();
    let shuffled = output.len() - 16;
    for i in (1..shuffled).rev() {
        let j = random.index(i + 1);
        output.swap(i, j);
    }
    output
}

fn select_windows(corpus: &Corpus, tokenizer: &Tokenizer) -> Result<Vec<Window>, String> {
    let chars: Vec<Vec<_>> = corpus
        .texts
        .iter()
        .map(|text| text.chars().collect())
        .collect();
    let mut cumulative = Vec::new();
    let mut total = 0_usize;
    for text in &chars {
        total += text.len().saturating_sub(PREFIX + TARGETS - 1);
        cumulative.push(total);
    }
    if total == 0 {
        return Err("no complete diagnostic span".into());
    }
    let mut random = Rng::new(SEED);
    let mut windows = Vec::new();
    for _ in 0..WINDOWS {
        let draw = random.index(total);
        let document = cumulative.partition_point(|&end| end <= draw);
        let before = document.checked_sub(1).map_or(0, |i| cumulative[i]);
        let start = draw - before;
        let raw_prefix: String = chars[document][start..start + PREFIX].iter().collect();
        let raw_targets: String = chars[document][start + PREFIX..start + PREFIX + TARGETS]
            .iter()
            .collect();
        let prefix = tokenizer.encode(&raw_prefix);
        let targets = tokenizer.encode(&raw_targets);
        windows.push(Window {
            document,
            start,
            raw_prefix,
            raw_targets,
            short_shuffled: Vec::new(),
            long_shuffled: Vec::new(),
            prefix,
            targets,
        });
    }
    // Selection RNG is independent of shuffle draws; all subjects reuse these rows.
    let mut shuffle_random = Rng::new(SEED ^ 0x7368_7566_666c_6500);
    for window in &mut windows {
        window.short_shuffled = shuffle_prefix(&window.prefix[PREFIX - 32..], &mut shuffle_random);
        window.long_shuffled = shuffle_prefix(&window.prefix, &mut shuffle_random);
    }
    Ok(windows)
}

fn selection_report(corpus: &Corpus, windows: &[Window]) -> Value {
    let rows: Vec<_> = windows.iter().map(|window| json!({"document_index": window.document, "source_id": corpus.ids[window.document], "prefix_start_unicode_scalar": window.start, "target_start_unicode_scalar": window.start + PREFIX, "raw_prefix": window.raw_prefix, "raw_targets": window.raw_targets, "unknown_prefix_tokens": window.prefix.iter().filter(|&&v| v == 0).count(), "unknown_target_tokens": window.targets.iter().filter(|&&v| v == 0).count(), "short_shuffled_ids": window.short_shuffled, "long_shuffled_ids": window.long_shuffled})).collect();
    let unique: BTreeSet<_> = windows
        .iter()
        .flat_map(|window| (0..TARGETS).map(|t| (window.document, window.start + PREFIX + t)))
        .collect();
    json!({"seed": SEED, "sampling": "uniform eligible 144-character starts, with replacement; same seed independently per split; shuffle RNG separate", "windows": rows, "window_count": windows.len(), "targets": windows.len() * TARGETS, "unique_target_positions": unique.len(), "manifest_encoding": "serde_json compact ordered window-record array UTF8", "manifest_sha256": sha256(&serde_json::to_vec(&rows).expect("serializable window metadata")), "independence": "windows may overlap and share documents; no significance test"})
}

fn context_metrics(model: &Model, windows: &[Window]) -> Result<Value, String> {
    let mut reports = Vec::new();
    for (name, length, shuffled) in [
        ("suffix1", 1, false),
        ("suffix4", 4, false),
        ("suffix16", 16, false),
        ("suffix32", 32, false),
        ("suffix48", 48, false),
        ("suffix64", 64, false),
        ("prefix128", 128, false),
        ("shuffle_first16_keep_last16", 32, true),
        ("shuffle_first112_keep_last16", 128, true),
    ] {
        let rows: Vec<_> = windows
            .iter()
            .map(|window| Continuation {
                prefix: if shuffled {
                    if length == 32 {
                        window.short_shuffled.clone()
                    } else {
                        window.long_shuffled.clone()
                    }
                } else {
                    window.prefix[PREFIX - length..].to_vec()
                },
                targets: window.targets.clone(),
            })
            .collect();
        let scores = score(model, &rows, BATCH)?;
        reports.push(json!({"condition": name, "prefix_characters": length, "max_input_steps_through_last_prediction": length + TARGETS - 1, "within64_input_training_horizon": length + TARGETS - 1 <= 64, "character_order_perturbed": shuffled, "targets": windows.len() * TARGETS, "nll_nats_per_character": -scores.iter().sum::<f64>() / (windows.len() * TARGETS) as f64, "per_window_summed_target_log_probability": scores}));
    }
    Ok(json!(reports))
}

#[derive(Default)]
struct FamilyMetrics {
    cases: usize,
    correct: usize,
    pairs: usize,
    both: usize,
    signed_interaction: f64,
    first: usize,
    last: usize,
}

fn choice(margin: f64) -> Option<usize> {
    if margin > 0.0 {
        Some(0)
    } else if margin < 0.0 {
        Some(1)
    } else {
        None
    }
}

pub(crate) fn semantic_metrics(
    model: &Model,
    tokenizer: &Tokenizer,
    pairs: &[ContrastPair],
) -> Result<Value, String> {
    let make = |prefix: &str, target: &str| Continuation::from_text(tokenizer, prefix, target);
    let rows: Vec<_> = pairs
        .iter()
        .flat_map(|pair| {
            pair.contexts.iter().flat_map(|context| {
                pair.candidates
                    .iter()
                    .map(|candidate| make(context, candidate))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if rows.iter().any(|row| {
        row.prefix.iter().chain(&row.targets).any(|&v| v == 0)
            || row.prefix.len() + row.targets.len() > 64
    }) {
        return Err("controlled fixture has unknown token or exceeds short horizon".into());
    }
    let scores = score(model, &rows, BATCH)?;
    let query_rows: Vec<_> = pairs
        .iter()
        .flat_map(|pair| {
            pair.candidates
                .iter()
                .map(|candidate| make(&pair.query, candidate))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let query_scores = score(model, &query_rows, BATCH)?;
    let mut families: BTreeMap<&str, FamilyMetrics> = BTreeMap::new();
    let mut details = Vec::new();
    for (i, pair) in pairs.iter().enumerate() {
        let margins = [
            scores[i * 4] - scores[i * 4 + 1],
            scores[i * 4 + 2] - scores[i * 4 + 3],
        ];
        let correct = [
            choice(margins[0]) == Some(pair.gold[0]),
            choice(margins[1]) == Some(pair.gold[1]),
        ];
        let interaction = (if pair.gold[0] == 0 { 1.0 } else { -1.0 }) * (margins[0] - margins[1]);
        let family = families.entry(pair.family).or_default();
        family.cases += 2;
        family.correct += correct.iter().filter(|&&v| v).count();
        family.pairs += 1;
        family.both += usize::from(correct[0] && correct[1]);
        family.signed_interaction += interaction;
        family.first += (0..2)
            .filter(|&j| pair.first_mention[j] == pair.gold[j])
            .count();
        family.last += (0..2)
            .filter(|&j| pair.last_mention[j] == pair.gold[j])
            .count();
        details.push(json!({"id": pair.id, "family": pair.family, "contexts": pair.contexts, "candidates": pair.candidates, "gold_candidate_indices": pair.gold, "query": pair.query, "full_prompt_plus_candidate_characters": pair.contexts.iter().map(|s| s.chars().count() + 5).collect::<Vec<_>>(), "prefix_token_counts": [rows[i * 4].prefix.len(), rows[i * 4 + 2].prefix.len()], "candidate_token_counts_context0_then_context1": rows[i * 4..i * 4 + 4].iter().map(|row| row.targets.len()).collect::<Vec<_>>(), "full_prompt_plus_candidate_token_counts_context0_then_context1": rows[i * 4..i * 4 + 4].iter().map(|row| row.prefix.len() + row.targets.len()).collect::<Vec<_>>(), "query_only_prefix_token_count": query_rows[i * 2].prefix.len(), "query_only_candidate_token_counts": query_rows[i * 2..i * 2 + 2].iter().map(|row| row.targets.len()).collect::<Vec<_>>(), "candidate_summed_log_probabilities_context0_then_context1": &scores[i * 4..i * 4 + 4], "candidate0_minus_candidate1_log_probability": margins, "correct": correct, "strict_both_correct": correct[0] && correct[1], "signed_pair_interaction_nats": interaction, "query_only_candidate_log_probabilities": &query_scores[i * 2..i * 2 + 2], "first_mention_candidate_indices": pair.first_mention, "last_mention_candidate_indices": pair.last_mention}));
    }
    let summaries: Vec<_> = families.iter().map(|(name, family)| json!({"family": name, "cases": family.cases, "correct": family.correct, "case_accuracy": family.correct as f64 / family.cases as f64, "pairs": family.pairs, "strict_both_correct": family.both, "strict_pair_accuracy": family.both as f64 / family.pairs as f64, "mean_signed_pair_interaction_nats": family.signed_interaction / family.pairs as f64, "first_mention_baseline_accuracy": family.first as f64 / family.cases as f64, "last_mention_baseline_accuracy": family.last as f64 / family.cases as f64, "constant_candidate_baseline_accuracy": 0.5, "focused_criterion_met": family.correct as f64 / family.cases as f64 >= 0.75 && family.both as f64 / family.pairs as f64 >= 0.5})).collect();
    // Independent branches must not depend on candidate evaluation order.
    let controls = [
        make(&pairs[0].contexts[0], &pairs[0].candidates[0])?,
        make(&pairs[0].contexts[0], &pairs[0].candidates[1])?,
        make(&pairs[0].contexts[0], &pairs[0].candidates[0])?,
    ];
    let repeat = score(model, &controls, 1)?;
    let reversed = score(
        model,
        &[
            make(&pairs[0].contexts[0], &pairs[0].candidates[1])?,
            make(&pairs[0].contexts[0], &pairs[0].candidates[0])?,
        ],
        1,
    )?;
    let error = (repeat[0] - repeat[2])
        .abs()
        .max((repeat[0] - reversed[1]).abs())
        .max((repeat[1] - reversed[0]).abs());
    if error > 2e-5 {
        return Err(format!("candidate branch/repeat control failed: {error}"));
    }
    Ok(
        json!({"families": summaries, "cases": details, "candidate_branch_control_max_log_probability_error": error, "candidate_branch_control_tolerance": 2e-5, "interpretation": "chosen short fact-consistency fixtures; not a general comprehension or dialogue benchmark; positive interaction alone can be too small or arbitrary to yield correct choices"}),
    )
}

pub(crate) fn checked_output(path: &Path) -> Result<PathBuf, String> {
    let parts: Vec<_> = path.components().collect();
    if !matches!(parts.as_slice(), [Component::Normal(dir), Component::Normal(_)] if *dir == "artifacts")
        || path.extension().is_none_or(|ext| ext != "json")
    {
        return Err("output must be a direct artifacts/*.json relative path".into());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let artifacts = root
        .join("artifacts")
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if artifacts.parent() != Some(root.canonicalize().map_err(|e| e.to_string())?.as_path()) {
        return Err("artifacts directory must remain inside the project".into());
    }
    let output = artifacts.join(path.file_name().ok_or("missing output filename")?);
    if output.exists() {
        return Err("refusing to replace existing diagnostic".into());
    }
    Ok(output)
}

pub fn cli(args: Vec<String>) -> Result<(), String> {
    if args == ["--help"] || args == ["-h"] {
        println!(
            "Usage: understanding_probe [--device gpu|cpu] [--output artifacts/understanding-diagnostic.json]"
        );
        return Ok(());
    }
    if !args.len().is_multiple_of(2) {
        return Err("options need values".into());
    }
    let mut device_name = "gpu";
    let mut output_arg = "artifacts/understanding-diagnostic.json";
    let mut seen = BTreeSet::new();
    for pair in args.as_chunks::<2>().0 {
        if !seen.insert(pair[0].as_str()) {
            return Err("duplicate option".into());
        }
        match pair[0].as_str() {
            "--device" => device_name = &pair[1],
            "--output" => output_arg = &pair[1],
            _ => return Err(format!("unknown option {}", pair[0])),
        }
    }
    if !matches!(device_name, "gpu" | "cpu") {
        return Err("device must be gpu or cpu".into());
    }
    let output = checked_output(Path::new(output_arg))?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let start = Instant::now();
    for (_, filename, expected) in SUBJECTS {
        if sha256(&fs::read(root.join("artifacts").join(filename)).map_err(|e| e.to_string())?)
            != expected
        {
            return Err(format!("{filename} checkpoint hash changed"));
        }
    }
    let manifest_bytes =
        fs::read(root.join("data/fineweb/manifest.json")).map_err(|e| e.to_string())?;
    if sha256(&manifest_bytes) != "4b0011a6961e86e6301e0ad849c4ef096bf74dfaeff23023bffce9172c182722"
    {
        return Err("FineWeb manifest changed".into());
    }
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let train = load_corpus(
        &root.join("data/fineweb/train.jsonl"),
        "ad0b069c93c70dc6fc52000edefbd272a574ae77dd7706ab763eff702ee2633a",
    )?;
    let validation = load_corpus(
        &root.join("data/fineweb/validation.jsonl"),
        "3ca429b3909a50a591286cb27f9b650527e2bd2846ffb22e5c460b20ad352866",
    )?;
    if validation.ids.iter().any(|id| train.ids.contains(id))
        || validation
            .texts
            .iter()
            .any(|text| train.texts.contains(text))
    {
        return Err("document split leakage".into());
    }
    let device: Device = runtime::device(device_name)?;
    println!(
        "Understanding diagnostic: {device:?}; identity: {:?}",
        device.identity()
    );
    let (control, tokenizer) =
        checkpoint::load(&root.join("artifacts").join(SUBJECTS[0].1), &device)?;
    let (penalty, penalty_tokenizer) =
        checkpoint::load(&root.join("artifacts").join(SUBJECTS[1].1), &device)?;
    if tokenizer != penalty_tokenizer
        || tokenizer != Tokenizer::from_documents(&train.texts)
        || control.config.hidden_size != 64
        || penalty.config.hidden_size != 64
        || control.config.vocab_size != 148
        || penalty.config.vocab_size != 148
        || control.feedback != FeedbackMode::Both
        || penalty.feedback != FeedbackMode::Both
    {
        return Err("unexpected fixed subject vocabulary or architecture".into());
    }
    let mut fresh = Model::new(control.config.clone(), SEED, &device)?;
    fresh.feedback = control.feedback;
    let pairs = dataset();
    let pair_manifest: Vec<_> = pairs.iter().map(|p| json!({"id": p.id, "family": p.family, "contexts": p.contexts, "candidates": p.candidates, "gold": p.gold, "query": p.query})).collect();
    let train_windows = select_windows(&train, &tokenizer)?;
    let validation_windows = select_windows(&validation, &tokenizer)?;
    let mut reports = Vec::new();
    for (label, model) in [
        ("ce_control", control),
        ("voltage_penalty", penalty),
        ("untrained_seed42", fresh),
    ] {
        let before = parameter_hash(&model)?;
        let semantic = semantic_metrics(&model, &tokenizer, &pairs)?;
        println!("{label} fact consistency: {}", semantic["families"]);
        let training_context = context_metrics(&model, &train_windows)?;
        let validation_context = context_metrics(&model, &validation_windows)?;
        for condition in validation_context
            .as_array()
            .ok_or("missing context scores")?
        {
            println!(
                "{label} validation {} NLL {}",
                condition["condition"], condition["nll_nats_per_character"]
            );
        }
        let after = parameter_hash(&model)?;
        if before != after {
            return Err("frozen diagnostic changed parameters".into());
        }
        reports.push(json!({"subject": label, "parameter_count": model.parameter_count(), "feedback": model.feedback.as_str(), "parameter_f32_le_sha256_before": before, "parameter_f32_le_sha256_after": after, "optimizer_updates": 0, "semantic_contrasts": semantic, "training_context": training_context, "validation_context": validation_context}));
    }
    for (_, filename, expected) in SUBJECTS {
        if sha256(&fs::read(root.join("artifacts").join(filename)).map_err(|e| e.to_string())?)
            != expected
        {
            return Err("checkpoint changed during analysis".into());
        }
    }
    let report = json!({"experiment": "frozen fact consistency and natural-context diagnostic", "device": format!("{device:?}"), "device_identity": format!("{:?}", device.identity()), "precision": "f32 Burn0.22 forward and per-row logP accumulation; cross-row aggregation/reporting f64", "optimizer_updates": 0, "fixed_subjects": SUBJECTS.iter().map(|(name, file, hash)| json!({"subject": name, "path": format!("artifacts/{file}"), "sha256": hash})).collect::<Vec<_>>(), "data": {"manifest_sha256": sha256(&manifest_bytes), "training": corpus_summary(&train, &manifest), "validation": corpus_summary(&validation, &manifest)}, "semantic_fixture": {"pairs": pair_manifest.len(), "contexts": pair_manifest.len() * 2, "candidates_per_context": 2, "encoding": "compact serde_json ordered pair record array UTF8", "sha256": sha256(&serde_json::to_vec(&pair_manifest).map_err(|e| e.to_string())?), "records": pair_manifest, "focused_per_family_criteria": "case_accuracy>=.75 AND strict_pair_accuracy>=.5 in every family; descriptive, no p-value"}, "training_selection": selection_report(&train, &train_windows), "validation_selection": selection_report(&validation, &validation_windows), "subjects": reports, "whole_diagnostic_seconds_including_loading_scoring_and_readbacks": start.elapsed().as_secs_f64(), "limitations": ["Synthetic focused templates, not representative semantic benchmark or training data.", "Question formats may be unfamiliar; declarative binding is reported separately.", "Context corruption is a perturbation; prefixes64/128 plus16targets extend beyond64input training horizon.", "No training-budget/capacity/data intervention: associations do not isolate a unique causal repair.", "State exists and correct local derivatives do not prove useful semantic representations."]});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut file, &report).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    println!("Saved frozen understanding diagnostic {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_fixtures_keep_the_original_prefix_and_candidate_tokens() {
        let pairs = dataset();
        let texts: Vec<_> = pairs
            .iter()
            .flat_map(|pair| {
                pair.contexts.iter().flat_map(|context| {
                    pair.candidates
                        .iter()
                        .map(move |candidate| format!("{context}{candidate}"))
                })
            })
            .collect();
        let tokenizer = Tokenizer::from_documents(&texts);
        for pair in &pairs {
            for prefix in pair.contexts.iter().chain([&pair.query]) {
                for candidate in &pair.candidates {
                    let row = Continuation::from_text(&tokenizer, prefix, candidate).unwrap();
                    assert_eq!(row.prefix, tokenizer.encode(prefix));
                    assert_eq!(row.targets, tokenizer.encode(candidate));
                    assert_eq!(row.targets.len(), 5);
                    assert!(row.prefix.len() + row.targets.len() <= 64);
                }
            }
        }
    }

    #[test]
    fn sampled_spans_and_shuffles_preserve_targets_suffix_and_multiset() {
        let texts = vec![
            "abcdefghijklmnopqrstuvwxyz ".repeat(10),
            "a fine day. ".repeat(30),
        ];
        let tokenizer = Tokenizer::from_documents(&texts);
        let corpus = Corpus {
            texts,
            ids: vec!["a".into(), "b".into()],
            hash: "fixture".into(),
        };
        let first = select_windows(&corpus, &tokenizer).unwrap();
        let second = select_windows(&corpus, &tokenizer).unwrap();
        assert_eq!(first.len(), 32);
        assert_eq!(
            selection_report(&corpus, &first),
            selection_report(&corpus, &second)
        );
        for window in &first {
            assert_eq!(window.prefix.len(), 128);
            assert_eq!(window.targets.len(), 16);
            assert_eq!(window.short_shuffled[16..], window.prefix[112..]);
            assert_eq!(window.long_shuffled[112..], window.prefix[112..]);
            for (original, shuffled) in [
                (&window.prefix[96..], &window.short_shuffled),
                (&window.prefix[..], &window.long_shuffled),
            ] {
                let mut a = original.to_vec();
                let mut b = shuffled.clone();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b);
            }
            let source: String = corpus.texts[window.document]
                .chars()
                .skip(window.start + 128)
                .take(16)
                .collect();
            assert_eq!(window.raw_targets, source);
        }
    }

    #[test]
    fn ties_do_not_count_and_output_paths_are_bounded() {
        assert_eq!(choice(0.0), None);
        assert_eq!(choice(-0.1), Some(1));
        assert_eq!(choice(0.1), Some(0));
        for path in [
            "../outside.json",
            "artifacts/../bad.json",
            "src/bad.json",
            "artifacts/model.bin",
            "artifacts/sub/report.json",
        ] {
            assert!(checked_output(Path::new(path)).is_err());
        }
        assert!(cli(vec!["--device".into(), "invalid".into()]).is_err());
        assert!(cli(vec!["--unrecognized".into(), "value".into()]).is_err());
    }
}
