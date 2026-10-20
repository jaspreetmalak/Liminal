//! Frozen generation and fact-consistency audit of the current XL checkpoint.
//! Every continuation uses production generation; this module never trains.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::data::Tokenizer;
use crate::model::Model;
use crate::training::Decoding;
use crate::{checkpoint, runtime, training, understanding_cases, understanding_probe};

const CHECKPOINT: &str = "artifacts/subword-xl-h256.bin";
const CHECKPOINT_SHA256: &str = "0f6db11e7e7ab5a3546d936690a58ef80446af6e7a1ee0dbd8439352c02c9b60";
const DEFAULT_OUTPUT: &str = "artifacts/generation-diagnostic-2026-10-20.json";

struct Case {
    id: &'static str,
    category: &'static str,
    prompt: &'static str,
    expected_claim: Option<&'static str>,
    new_tokens: usize,
}

fn cases() -> Vec<Case> {
    [
        ("01_library", "topic_continuation", "A public library gives people access to books, information, and a quiet place to study. For a small town,", None),
        ("02_basil", "procedural_continuation", "To grow basil in a small pot, choose a sunny windowsill and keep the soil moist. The next step is", None),
        ("03_binding_blue", "binding", "Mira put the blue key in a drawer. Owen put the pink key in a box. Later, Mira returned to collect her key. The color of Mira's key was:", Some("blue")),
        ("04_binding_pink", "binding", "Mira put the pink key in a drawer. Owen put the blue key in a box. Later, Mira returned to collect her key. The color of Mira's key was:", Some("pink")),
        ("05_negation_pink", "negation", "The box is not blue. It is pink. The color of the box is:", Some("pink")),
        ("06_negation_blue", "negation", "The box is not pink. It is blue. The color of the box is:", Some("blue")),
        ("07_roles_dog", "semantic_roles", "The dog chased the cat. The cat ran away. The animal that did the chasing was:", Some("dog")),
        ("08_roles_cat", "semantic_roles", "The cat chased the dog. The dog ran away. The animal that did the chasing was:", Some("cat")),
        ("09_exact_output", "instruction_following", "Write exactly the word blue and nothing else. Output:", Some("exactly blue and nothing else")),
        ("10_dialogue", "dialogue", "User: I spent the afternoon planting basil and now I feel tired. Assistant:", None),
        ("11_arithmetic", "arithmetic", "A shop had 7 apples and received 5 more. The total number of apples is:", Some("12 or twelve")),
        ("12_long_basil", "long_continuation", "This guide explains how to plant and care for basil in a small pot. First,", None),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (id, category, prompt, expected_claim))| Case {
        id,
        category,
        prompt,
        expected_claim,
        new_tokens: if index == 11 { 256 } else { 72 },
    })
    .collect()
}

#[derive(Clone, Copy)]
struct Setting {
    id: &'static str,
    decoding: Decoding,
    seed: u64,
}

fn settings() -> [Setting; 5] {
    [
        Setting {
            id: "greedy",
            decoding: Decoding::sampled(0.0),
            seed: 42,
        },
        Setting {
            id: "greedy_penalty",
            decoding: Decoding {
                temperature: 0.0,
                top_p: 1.0,
                repetition_penalty: 1.3,
            },
            seed: 42,
        },
        Setting {
            id: "sampled",
            decoding: Decoding::sampled(0.8),
            seed: 42,
        },
        Setting {
            id: "nucleus_penalty_seed42",
            decoding: Decoding {
                temperature: 0.8,
                top_p: 0.9,
                repetition_penalty: 1.3,
            },
            seed: 42,
        },
        Setting {
            id: "nucleus_penalty_seed43",
            decoding: Decoding {
                temperature: 0.8,
                top_p: 0.9,
                repetition_penalty: 1.3,
            },
            seed: 43,
        },
    ]
}

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

fn prompt_information(tokenizer: &Tokenizer, prompt: &str, ids: &[usize]) -> Value {
    json!({
        "characters_unicode_scalars": prompt.chars().count(),
        "tokens": ids.len(),
        "token_ids": ids,
        "unknown_characters": prompt.chars().filter(|character| !tokenizer.vocab().contains(character)).count(),
        "unknown_token_ids": ids.iter().filter(|&&id| id == 0).count(),
        "decoded_prompt": tokenizer.decode(ids).expect("encoded IDs belong to tokenizer"),
    })
}

struct Arguments {
    device: String,
    output: PathBuf,
}

fn parse(arguments: &[String]) -> Result<Arguments, String> {
    let mut device = None;
    let mut output = None;
    let mut arguments = arguments.iter();
    while let Some(option) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {option}"))?;
        match option.as_str() {
            "--device" if device.is_none() => {
                if !matches!(value.as_str(), "gpu" | "cpu") {
                    return Err("device must be gpu or cpu".into());
                }
                device = Some(value.clone());
            }
            "--output" if output.is_none() => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown or duplicate option {option}")),
        }
    }
    Ok(Arguments {
        device: device.unwrap_or_else(|| "gpu".into()),
        output: output.unwrap_or_else(|| DEFAULT_OUTPUT.into()),
    })
}

pub fn cli(arguments: Vec<String>) -> Result<(), String> {
    let arguments = parse(&arguments)?;
    run(&arguments.device, &arguments.output)
}

pub fn run(device_name: &str, output: &Path) -> Result<(), String> {
    let started = Instant::now();
    let output = understanding_probe::checked_output(output)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let checkpoint_path = root.join(CHECKPOINT);
    let checkpoint_bytes = fs::read(&checkpoint_path).map_err(|error| error.to_string())?;
    let checkpoint_hash = sha256(&checkpoint_bytes);
    if checkpoint_hash != CHECKPOINT_SHA256 {
        return Err("current XL checkpoint differs from the fixed expected SHA256".into());
    }
    drop(checkpoint_bytes);
    let device = runtime::device(device_name)?;
    let (model, tokenizer) = checkpoint::load(&checkpoint_path, &device)?;
    if model.config.hidden_size != 256 || model.config.vocab_size != 9778 {
        return Err("fixed XL checkpoint must have H256 and V9778".into());
    }
    let parameters_before = parameter_hash(&model)?;
    let mut generations = Vec::with_capacity(60);
    for case in cases() {
        let prompt_ids = tokenizer.encode(case.prompt);
        for setting in settings() {
            let generation_started = Instant::now();
            let generated = training::generate(
                &model,
                &prompt_ids,
                case.new_tokens,
                setting.decoding,
                setting.seed,
            )?;
            let text = tokenizer.decode(&generated)?;
            generations.push(json!({
                "case_id": case.id,
                "category": case.category,
                "prompt": case.prompt,
                "expected_claim": case.expected_claim,
                "setting": setting.id,
                "decoding": {
                    "temperature": setting.decoding.temperature,
                    "top_p": setting.decoding.top_p,
                    "repetition_penalty": setting.decoding.repetition_penalty,
                    "seed": setting.seed,
                    "repetition_window_tokens": 64,
                },
                "prompt_encoding": prompt_information(&tokenizer, case.prompt, &prompt_ids),
                "requested_new_tokens": case.new_tokens,
                "consumed_new_tokens": generated.len(),
                "total_context_and_generated_tokens": prompt_ids.len() + generated.len(),
                "model_input_steps_through_last_prediction": prompt_ids.len() + generated.len().saturating_sub(1),
                "trained_window_tokens": 64,
                "generated_token_ids": generated,
                "continuation": text,
                "continuation_characters_unicode_scalars": text.chars().count(),
                "continuation_unknown_token_ids": generated.iter().filter(|&&id| id == 0).count(),
                "seconds_including_readbacks": generation_started.elapsed().as_secs_f64(),
            }));
            println!("completed {} / {}", case.id, setting.id);
        }
    }
    let semantic_started = Instant::now();
    let pairs = understanding_cases::dataset();
    let semantic_metrics = understanding_probe::semantic_metrics(&model, &tokenizer, &pairs)?;
    let semantic_seconds = semantic_started.elapsed().as_secs_f64();
    let parameters_after = parameter_hash(&model)?;
    let checkpoint_after = sha256(&fs::read(&checkpoint_path).map_err(|error| error.to_string())?);
    if parameters_before != parameters_after || checkpoint_hash != checkpoint_after {
        return Err("checkpoint or model parameters changed during frozen diagnostic".into());
    }
    let report = json!({
        "experiment": "frozen current XL generation and fact-consistency diagnostic",
        "device": format!("{device:?}"),
        "device_identity": format!("{:?}", device.identity()),
        "precision": "f32 Burn 0.22 production model; Rust production decoding",
        "optimizer_updates": 0,
        "checkpoint": {
            "path": CHECKPOINT,
            "expected_sha256": CHECKPOINT_SHA256,
            "sha256_before": checkpoint_hash,
            "sha256_after": checkpoint_after,
            "parameter_f32_le_sha256_before": parameters_before,
            "parameter_f32_le_sha256_after": parameters_after,
            "unchanged": true,
        },
        "model": {
            "parameters": model.parameter_count(),
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "base_characters": tokenizer.vocab().len(),
            "merges": tokenizer.merges().len(),
            "feedback": model.feedback.as_str(),
            "threshold": model.config.threshold,
            "surrogate_width": model.config.surrogate_width,
            "decays": "learned per-neuron membrane, trace, and memory; 0.995*sigmoid(logit)",
        },
        "protocol": {
            "cases": 12,
            "settings_per_case": 5,
            "continuations": generations.len(),
            "generated_tokens_requested": 5240,
            "generation_implementation": "training::generate, each call begins with zero state and consumes every prompt token exactly once",
            "expected_claims": "descriptive labels for manual review; no automatic coherence or correctness judgment",
        },
        "generations": generations,
        "semantic_fixture": {"pairs": pairs.len(), "contexts": pairs.len() * 2, "families": 4, "used_for_training": false},
        "semantic_metrics": semantic_metrics,
        "semantic_seconds_including_readbacks": semantic_seconds,
        "whole_diagnostic_seconds_including_loading_and_readbacks": started.elapsed().as_secs_f64(),
        "limitations": [
            "One frozen checkpoint, fixed prompts, and two sampling seeds; this is a descriptive audit, not an independent general language benchmark.",
            "The model is trained for plain next-token web completion, not an instruction-tuned chat assistant.",
            "The tokenizer has no EOS token or learned stop rule; production generation always consumes its requested budget and may end mid-word or mid-sentence.",
            "Training windows are 64 subword tokens with zero initial state; longer inference keeps state continuously without a hard 64-token input truncation.",
            "Subword tokens contain variable numbers of characters and may split words. Candidate semantic scores use the shared BPE-safe continuation scorer.",
            "Repetition penalty includes prompt tokens among the last 64 context tokens and can discourage legitimate entity or topic repetition.",
            "Fact-consistency contrasts test four narrow families; candidate preference is not a general understanding score.",
        ],
    });
    let mut bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|error| format!("cannot create fresh report {}: {error}", output.display()))?;
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    println!("saved frozen generation diagnostic to {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_suite_has_opposing_facts_and_prescribed_budgets() {
        let cases = cases();
        assert_eq!(cases.len(), 12);
        assert!(cases[..11].iter().all(|case| case.new_tokens == 72));
        assert_eq!(cases[11].new_tokens, 256);
        assert_eq!(
            cases.iter().map(|case| case.new_tokens).sum::<usize>(),
            1048
        );
        for index in [2, 4, 6] {
            assert_ne!(cases[index].prompt, cases[index + 1].prompt);
            assert_ne!(cases[index].expected_claim, cases[index + 1].expected_claim);
        }
        assert_eq!(settings().len(), 5);
        assert_eq!(settings()[3].decoding, settings()[4].decoding);
        assert_ne!(settings()[3].seed, settings()[4].seed);
    }

    #[test]
    fn encoding_metadata_counts_known_and_unknown_characters() {
        let documents: Vec<_> = cases().iter().map(|case| case.prompt.to_owned()).collect();
        let tokenizer = Tokenizer::from_documents(&documents)
            .with_merges(&documents, 40)
            .unwrap();
        for case in cases() {
            let ids = tokenizer.encode(case.prompt);
            let information = prompt_information(&tokenizer, case.prompt, &ids);
            assert_eq!(information["unknown_characters"], 0);
            assert_eq!(information["unknown_token_ids"], 0);
            assert_eq!(information["decoded_prompt"], case.prompt);
            assert!(ids.iter().all(|&id| id < tokenizer.vocab_size()));
        }
        let prompt = "blue 🧬";
        let information = prompt_information(&tokenizer, prompt, &tokenizer.encode(prompt));
        assert_eq!(information["unknown_characters"], 1);
        assert_eq!(information["unknown_token_ids"], 1);
    }

    #[test]
    fn cli_rejects_missing_unknown_duplicate_and_invalid_options() {
        let parse_words = |words: &[&str]| {
            parse(
                &words
                    .iter()
                    .map(|word| (*word).to_owned())
                    .collect::<Vec<_>>(),
            )
        };
        for words in [
            vec!["--device"],
            vec!["--device", "cuda"],
            vec!["--checkpoint", "other.bin"],
            vec!["--device", "cpu", "--device", "gpu"],
            vec![
                "--output",
                "artifacts/a.json",
                "--output",
                "artifacts/b.json",
            ],
        ] {
            assert!(parse_words(&words).is_err());
        }
        let defaults = parse_words(&[]).unwrap();
        assert_eq!(defaults.device, "gpu");
        assert_eq!(defaults.output, PathBuf::from(DEFAULT_OUTPUT));
        let cpu = parse_words(&["--device", "cpu", "--output", "artifacts/new.json"]).unwrap();
        assert_eq!(cpu.device, "cpu");
    }
}
