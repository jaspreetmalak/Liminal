//! A controlled synthetic test of the two directed feedback connections.
//! This measures delayed binding and correction, not conversational reasoning.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::data::Tokenizer;
use crate::model::{Batch, FeedbackMode, Model, ModelConfig, State};
use crate::rng::Rng;
use crate::{runtime, training};

const DATA_SEED: u64 = 42;
const SCHEDULE_SEED: u64 = 0x7363_6865_6475_0042;
const TRAIN_CASES: usize = 1024;
const TEST_CASES: usize = 256;
const PROMPT_LENGTH: usize = 32;
const BATCH_SIZE: usize = 32;
const HIDDEN_SIZE: usize = 32;
const LEARNING_RATE: f64 = 0.003;
const GRADIENT_CLIP: f64 = 1.0;
const MODES: [FeedbackMode; 4] = [
    FeedbackMode::Both,
    FeedbackMode::TopDown,
    FeedbackMode::BottomUp,
    FeedbackMode::Neither,
];

#[derive(Debug, Clone)]
struct Case {
    prompt: String,
    answer: char,
    query: char,
    corrected: char,
}

impl Case {
    fn json(&self) -> Value {
        json!({
            "prompt": self.prompt,
            "answer": self.answer.to_string(),
            "query": self.query.to_string(),
            "corrected_entity": self.corrected.to_string(),
            "prompt_sha256": sha256(self.prompt.as_bytes()),
        })
    }
}

struct Dataset {
    train: Vec<Case>,
    test: Vec<Case>,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn cases_hash(cases: &[Case]) -> String {
    let mut text = String::new();
    for case in cases {
        text.push_str(&case.prompt);
        text.push('\t');
        text.push(case.answer);
        text.push('\n');
    }
    sha256(text.as_bytes())
}

/// Each unique history produces both entity queries with opposite answers.
/// Prefix groups, rather than just full prompts, are disjoint across splits.
/// Corrected entity x final-a value is balanced exactly within each split.
fn dataset() -> Dataset {
    let mut random = Rng::new(DATA_SEED);
    let mut seen = BTreeSet::new();
    let mut make = |count: usize| {
        let mut cases = Vec::with_capacity(count);
        for index in 0..count / 2 {
            let corrected = index % 2;
            let final_a = (index / 2) % 2;
            let final_values = [final_a, 1 - final_a];
            loop {
                let mut initial = final_values;
                initial[corrected] = 1 - final_values[corrected];
                let order = if random.index(2) == 0 { [0, 1] } else { [1, 0] };
                let mut prompt = String::with_capacity(PROMPT_LENGTH);
                for entity in order {
                    prompt.push(['a', 'b'][entity]);
                    prompt.push(['0', '1'][initial[entity]]);
                    prompt.push(';');
                }
                let before = 2 + random.index(11);
                let after = 20 - before;
                for _ in 0..before {
                    prompt.push(['x', 'y', 'z'][random.index(3)]);
                }
                prompt.push(['a', 'b'][corrected]);
                prompt.push(['0', '1'][1 - initial[corrected]]);
                prompt.push(';');
                for _ in 0..after {
                    prompt.push(['x', 'y', 'z'][random.index(3)]);
                }
                if seen.insert(prompt.clone()) {
                    for (query, &final_value) in final_values.iter().enumerate() {
                        cases.push(Case {
                            prompt: format!("{prompt}?{}=", ['a', 'b'][query]),
                            answer: ['0', '1'][final_value],
                            query: ['a', 'b'][query],
                            corrected: ['a', 'b'][corrected],
                        });
                    }
                    break;
                }
            }
        }
        // Shuffle independently of model seeds, removing the strata's order.
        for i in (1..cases.len()).rev() {
            cases.swap(i, random.index(i + 1));
        }
        cases
    };
    Dataset {
        train: make(TRAIN_CASES),
        test: make(TEST_CASES),
    }
}

struct ParsedPrompt {
    query: char,
    corrected: char,
    latest_bit: char,
    first_bit: char,
    queried_bit: char,
}

/// Independent deterministic interpretation, using only the prompt text.
/// This is a diagnostic oracle, not a second learned model implementation.
fn parse_prompt(prompt: &str) -> Result<ParsedPrompt, String> {
    let bytes = prompt.as_bytes();
    if bytes.len() < 3 {
        return Err("probe prompt is missing its query".into());
    }
    let end = bytes.len() - 3;
    if bytes[end] != b'?' || !matches!(bytes[end + 1], b'a' | b'b') || bytes[end + 2] != b'=' {
        return Err("probe prompt must end in ?a= or ?b=".into());
    }
    let query = bytes[end + 1] as char;
    let mut values = [None, None];
    let mut latest = None;
    let mut first_bit = None;
    let mut position = 0;
    while position < end {
        match bytes[position] {
            b'x' | b'y' | b'z' => position += 1,
            entity @ (b'a' | b'b') => {
                if position + 2 >= end
                    || !matches!(bytes[position + 1], b'0' | b'1')
                    || bytes[position + 2] != b';'
                {
                    return Err(
                        "probe assignment must have entity, binary bit, and semicolon".into(),
                    );
                }
                let bit = bytes[position + 1] as char;
                first_bit.get_or_insert(bit);
                values[(entity - b'a') as usize] = Some(bit);
                latest = Some((entity as char, bit));
                position += 3;
            }
            _ => return Err("probe history contains an unsupported character".into()),
        }
    }
    if values.iter().any(Option::is_none) {
        return Err("probe history must assign both entities".into());
    }
    let (corrected, latest_bit) = latest.ok_or("probe history has no assignments")?;
    Ok(ParsedPrompt {
        query,
        corrected,
        latest_bit,
        first_bit: first_bit.unwrap(),
        queried_bit: values[(query as u8 - b'a') as usize].unwrap(),
    })
}

fn baseline_metrics(cases: &[Case]) -> Result<Value, String> {
    if cases.is_empty() {
        return Err("baseline split must be nonempty".into());
    }
    let mut correct = [0_usize; 6];
    for case in cases {
        let parsed = parse_prompt(&case.prompt)?;
        let correction_only = if parsed.query == parsed.corrected {
            parsed.latest_bit
        } else if parsed.latest_bit == '0' {
            '1'
        } else {
            '0'
        };
        for (count, prediction) in correct.iter_mut().zip([
            '0',
            '1',
            parsed.latest_bit,
            parsed.first_bit,
            parsed.queried_bit,
            correction_only,
        ]) {
            *count += usize::from(prediction == case.answer);
        }
    }
    Ok(json!({
        "cases": cases.len(),
        "predictors": ([
            "constant_zero", "constant_one", "latest_assigned_bit_ignoring_query",
            "first_assigned_bit_ignoring_query", "latest_queried_entity_oracle",
            "correction_only_plus_query_opposite_final_bits_oracle",
        ].iter().zip(correct).map(|(name, count)| json!({
            "name": name, "correct": count, "total": cases.len(),
            "accuracy": count as f64 / cases.len() as f64,
        })).collect::<Vec<_>>()),
    }))
}

fn baseline_diagnostics(data: &Dataset) -> Result<Value, String> {
    Ok(json!({
        "kind": "deterministic dataset diagnostics; no learned model, GPU execution, or optimizer updates",
        "scope_limit": "The correction-only-plus-query rule assumes opposite final entity bits and can solve the paired task without retaining initial assignments. It is not a valid general binding oracle.",
        "train": baseline_metrics(&data.train)?,
        "test": baseline_metrics(&data.test)?,
    }))
}

fn read_dataset_report(path: &Path) -> Result<(Dataset, Value), String> {
    let path = checked_output(path)?;
    let metadata =
        fs::metadata(&path).map_err(|error| format!("reading dataset report metadata: {error}"))?;
    if !metadata.is_file() || metadata.len() > 16 * 1024 * 1024 {
        return Err(
            "dataset report must be a project-local JSON file no larger than 16 MiB".into(),
        );
    }
    let bytes = fs::read(&path).map_err(|error| format!("reading dataset report: {error}"))?;
    let report: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parsing dataset report: {error}"))?;
    let read_cases = |name: &str, expected: usize| -> Result<Vec<Case>, String> {
        let values = report["dataset"][name]
            .as_array()
            .ok_or_else(|| format!("dataset report needs dataset.{name}"))?;
        if values.len() != expected {
            return Err(format!(
                "dataset report {name} must contain {expected} cases"
            ));
        }
        values
            .iter()
            .map(|value| {
                let prompt = value["prompt"]
                    .as_str()
                    .ok_or("dataset case needs a prompt string")?;
                let answer = match value["answer"].as_str() {
                    Some("0") => '0',
                    Some("1") => '1',
                    _ => return Err("dataset case answer must be 0 or 1".into()),
                };
                if prompt.len() != PROMPT_LENGTH {
                    return Err("saved dataset prompts must have 32 ASCII characters".into());
                }
                let parsed = parse_prompt(prompt)?;
                if parsed.queried_bit != answer {
                    return Err(
                        "saved dataset answer disagrees with the prompt's latest queried value"
                            .into(),
                    );
                }
                Ok(Case {
                    prompt: prompt.into(),
                    answer,
                    query: parsed.query,
                    corrected: parsed.corrected,
                })
            })
            .collect()
    };
    let data = Dataset {
        train: read_cases("train_cases", TRAIN_CASES)?,
        test: read_cases("test_cases", TEST_CASES)?,
    };
    for (name, cases) in [("train_sha256", &data.train), ("test_sha256", &data.test)] {
        if report["dataset"][name].as_str() != Some(cases_hash(cases).as_str()) {
            return Err(format!("saved dataset {name} does not match its cases"));
        }
    }
    Ok((
        data,
        json!({
            "kind": "saved report cases; no historical generator retained",
            "path": path.to_string_lossy(),
            "report_sha256": sha256(&bytes),
            "experiment": report["experiment"],
            "posthoc_diagnostic": true,
        }),
    ))
}

struct EncodedCase {
    inputs: Vec<usize>,
    answer: usize,
}

fn encode(cases: &[Case], tokenizer: &Tokenizer) -> Result<Vec<EncodedCase>, String> {
    cases
        .iter()
        .map(|case| {
            let inputs = tokenizer.encode(&case.prompt);
            let answer = tokenizer.encode(&case.answer.to_string())[0];
            if inputs.len() != PROMPT_LENGTH || inputs.contains(&0) || answer == 0 {
                return Err("probe case has an invalid length or unknown token".into());
            }
            Ok(EncodedCase { inputs, answer })
        })
        .collect()
}

/// The answer is a target only. It is never appended to the input sequence.
fn batch(cases: &[&EncodedCase], device: &Device) -> Result<Batch, String> {
    if cases.is_empty() || cases.len() > BATCH_SIZE {
        return Err("probe batch must contain 1..=32 cases".into());
    }
    let mut inputs = Vec::with_capacity(cases.len() * PROMPT_LENGTH);
    let mut targets = vec![0_i32; cases.len() * PROMPT_LENGTH];
    let mut mask = vec![0.0_f32; cases.len() * PROMPT_LENGTH];
    for (row, case) in cases.iter().enumerate() {
        if case.inputs.len() != PROMPT_LENGTH {
            return Err("probe inputs must have exactly 32 tokens".into());
        }
        for &token in &case.inputs {
            inputs.push(i32::try_from(token).map_err(|_| "probe token exceeds i32")?);
        }
        let position = (row + 1) * PROMPT_LENGTH - 1;
        targets[position] = i32::try_from(case.answer).map_err(|_| "answer token exceeds i32")?;
        mask[position] = 1.0;
    }
    let shape = [cases.len(), PROMPT_LENGTH];
    Ok(Batch {
        inputs: Tensor::<2, Int>::from_data(TensorData::new(inputs, shape), device),
        targets: Tensor::<2, Int>::from_data(TensorData::new(targets, shape), device),
        mask: Tensor::from_data(TensorData::new(mask, shape), device),
        valid_targets: cases.len(),
    })
}

#[derive(Debug, Clone)]
struct Evaluation {
    nll: f64,
    accuracy: f64,
    binary_accuracy: f64,
    seconds: f64,
    predictions: Vec<usize>,
}

impl Evaluation {
    fn json(&self, tokenizer: &Tokenizer, include_predictions: bool) -> Result<Value, String> {
        let mut value = json!({
            "answer_nll_nats": self.nll,
            "accuracy": self.accuracy,
            "binary_forced_choice_accuracy": self.binary_accuracy,
            "forward_wall_seconds_including_host_transfers": self.seconds,
            "cases": self.predictions.len(),
        });
        if include_predictions {
            value["predictions"] = json!(
                self.predictions
                    .iter()
                    .map(|&token| tokenizer.decode(&[token]))
                    .collect::<Result<Vec<_>, _>>()?
            );
        }
        Ok(value)
    }
}

fn evaluate(
    model: &Model,
    cases: &[EncodedCase],
    tokenizer: &Tokenizer,
) -> Result<Evaluation, String> {
    if cases.is_empty() {
        return Err("probe evaluation split must be nonempty".into());
    }
    let model = model.clone().valid();
    // Finish queued parameter updates before starting the forward-only timer.
    model.parameter_values()?;
    let device = model.device();
    let mut predictions = Vec::with_capacity(cases.len());
    let mut total_nll = 0.0;
    let mut correct = 0;
    let mut binary_correct = 0;
    let binary = tokenizer.encode("01");
    let start = Instant::now();
    for rows in cases.chunks(BATCH_SIZE) {
        let mut state = State::new(rows.len(), model.config.hidden_size, &device);
        let mut final_logits = None;
        for time in 0..PROMPT_LENGTH {
            let tokens: Vec<i32> = rows.iter().map(|case| case.inputs[time] as i32).collect();
            final_logits = Some(model.step(
                Tensor::from_data(TensorData::new(tokens, [rows.len()]), &device),
                &mut state,
            )?);
        }
        // Reading all final logits synchronizes completion of the forward work.
        let logits = final_logits
            .expect("prompt length is positive")
            .try_into_vec_as::<f32>()
            .map_err(|error| format!("reading probe logits: {error}"))?;
        for (case, row) in rows.iter().zip(logits.chunks(model.config.vocab_size)) {
            if row.iter().any(|value| !value.is_finite()) {
                return Err("probe produced non-finite logits".into());
            }
            let mut prediction = 0;
            for index in 1..row.len() {
                if row[index] > row[prediction] {
                    prediction = index;
                }
            }
            predictions.push(prediction);
            correct += usize::from(prediction == case.answer);
            let binary_prediction = if row[binary[1]] > row[binary[0]] {
                binary[1]
            } else {
                binary[0]
            };
            binary_correct += usize::from(binary_prediction == case.answer);
            let maximum = row[prediction] as f64;
            let partition = row.iter().map(|&v| (v as f64 - maximum).exp()).sum::<f64>();
            total_nll += maximum + partition.ln() - row[case.answer] as f64;
        }
    }
    Ok(Evaluation {
        nll: total_nll / cases.len() as f64,
        accuracy: correct as f64 / cases.len() as f64,
        binary_accuracy: binary_correct as f64 / cases.len() as f64,
        seconds: start.elapsed().as_secs_f64(),
        predictions,
    })
}

#[derive(Debug)]
struct RunResult {
    seed: u64,
    mode: FeedbackMode,
    final_test: Evaluation,
    report: Value,
}

/// CLI configuration is deliberately bounded because this is a small diagnostic.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub device: String,
    pub steps: usize,
    pub seeds: Vec<u64>,
    pub output: PathBuf,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            device: "gpu".into(),
            steps: 300,
            seeds: vec![42, 43, 44],
            output: PathBuf::from("artifacts/feedback-counterfactual.json"),
        }
    }
}

impl ProbeConfig {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut config = Self::default();
        let mut used = BTreeSet::new();
        let mut args = args.iter();
        while let Some(flag) = args.next() {
            if !used.insert(flag.clone()) {
                return Err(format!("duplicate argument {flag}"));
            }
            let value = args
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))?;
            match flag.as_str() {
                "--device" => config.device = value.clone(),
                "--steps" => {
                    config.steps = value.parse().map_err(|_| "--steps must be an integer")?
                }
                "--seeds" => {
                    config.seeds = value
                        .split(',')
                        .map(|seed| {
                            seed.parse()
                                .map_err(|_| "--seeds must be comma-separated u64 integers")
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                "--output" => config.output = PathBuf::from(value),
                _ => return Err(format!("unknown argument {flag}")),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if !matches!(self.device.as_str(), "gpu" | "cpu") {
            return Err("--device must be gpu or cpu".into());
        }
        if !(1..=10_000).contains(&self.steps) {
            return Err("--steps must be in 1..=10000".into());
        }
        if self.seeds.is_empty()
            || self.seeds.len() > 8
            || self.seeds.iter().copied().collect::<BTreeSet<_>>().len() != self.seeds.len()
        {
            return Err("--seeds requires 1..=8 distinct u64 values".into());
        }
        checked_output(&self.output)?;
        Ok(())
    }
}

fn checked_output(path: &Path) -> Result<PathBuf, String> {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = project
        .canonicalize()
        .map_err(|error| format!("resolving project root: {error}"))?;
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        || path.extension().and_then(|ext| ext.to_str()) != Some("json")
    {
        return Err("--output must be a project-local .json path without '..'".into());
    }
    let output = if path.is_absolute() {
        path.to_path_buf()
    } else {
        project.join(path)
    };
    if !output.starts_with(project) {
        return Err("--output must remain inside the Liminal project root".into());
    }
    let mut ancestor = output.as_path();
    while !ancestor.exists() {
        ancestor = ancestor.parent().ok_or("cannot resolve output parent")?;
    }
    if !ancestor
        .canonicalize()
        .map_err(|error| format!("resolving output ancestor: {error}"))?
        .starts_with(&root)
    {
        return Err("--output resolves outside the Liminal project root".into());
    }
    Ok(output)
}

fn summary(runs: &[RunResult], config: &ProbeConfig) -> Value {
    let mean = |mode: FeedbackMode, metric: fn(&Evaluation) -> f64| {
        let values: Vec<_> = runs.iter().filter(|run| run.mode == mode).collect();
        values
            .iter()
            .map(|run| metric(&run.final_test))
            .sum::<f64>()
            / values.len() as f64
    };
    let mut best_other = FeedbackMode::TopDown;
    for mode in [FeedbackMode::BottomUp, FeedbackMode::Neither] {
        let accuracy = mean(mode, |evaluation| evaluation.accuracy);
        let best_accuracy = mean(best_other, |evaluation| evaluation.accuracy);
        if accuracy > best_accuracy
            || (accuracy == best_accuracy
                && mean(mode, |evaluation| evaluation.nll)
                    < mean(best_other, |evaluation| evaluation.nll))
        {
            best_other = mode;
        }
    }
    let accuracy_advantage =
        mean(FeedbackMode::Both, |e| e.accuracy) - mean(best_other, |e| e.accuracy);
    let nll_difference = mean(FeedbackMode::Both, |e| e.nll) - mean(best_other, |e| e.nll);
    let mut positive_seeds = 0;
    let mut paired = Vec::new();
    for &seed in &config.seeds {
        let both = &runs
            .iter()
            .find(|run| run.seed == seed && run.mode == FeedbackMode::Both)
            .unwrap()
            .final_test;
        let other = &runs
            .iter()
            .find(|run| run.seed == seed && run.mode == best_other)
            .unwrap()
            .final_test;
        positive_seeds += usize::from(both.accuracy > other.accuracy);
        paired.push(json!({
            "seed": seed,
            "accuracy_advantage": both.accuracy - other.accuracy,
            "nll_difference_nats": both.nll - other.nll,
        }));
    }
    let preregistered = config.steps == 300 && config.seeds == [42, 43, 44];
    let passed = accuracy_advantage >= 0.05 && positive_seeds >= 2 && nll_difference < 0.0;
    json!({
        "means_by_mode": MODES.iter().map(|&mode| json!({
            "mode": mode.as_str(),
            "answer_accuracy": mean(mode, |e| e.accuracy),
            "answer_nll_nats": mean(mode, |e| e.nll),
        })).collect::<Vec<_>>(),
        "best_other_mode_by_mean_accuracy_then_nll": best_other.as_str(),
        "both_accuracy_advantage": accuracy_advantage,
        "both_nll_difference_nats": nll_difference,
        "positive_paired_seeds": positive_seeds,
        "paired": paired,
        "uses_preregistered_steps_and_seeds": preregistered,
        "meets_numerical_criterion": passed,
        "decision": if !preregistered {
            "exploratory_run_not_preregistered_confirmation"
        } else if passed {
            "supported_on_this_synthetic_task"
        } else {
            "not_supported_by_preregistered_test"
        },
    })
}

/// Run all matched modes without touching the natural-language checkpoint.
pub fn run(config: &ProbeConfig) -> Result<PathBuf, String> {
    config.validate()?;
    let output = checked_output(&config.output)?;
    protect_historical_report(&output)?;
    let device = runtime::device(&config.device)?;
    eprintln!(
        "feedback probe device: {device:?}; identity: {:?}",
        device.identity()
    );
    let data = dataset();
    let tokenizer = Tokenizer::from_documents(
        &data
            .train
            .iter()
            .map(|case| case.prompt.clone())
            .collect::<Vec<_>>(),
    );
    let train = encode(&data.train, &tokenizer)?;
    let test = encode(&data.test, &tokenizer)?;
    let mut random = Rng::new(SCHEDULE_SEED);
    let schedule: Vec<Vec<usize>> = (0..config.steps)
        .map(|_| (0..BATCH_SIZE).map(|_| random.index(train.len())).collect())
        .collect();
    let schedule_bytes: Vec<u8> = schedule
        .iter()
        .flatten()
        .flat_map(|&index| (index as u64).to_le_bytes())
        .collect();
    let mut runs = Vec::new();
    for &seed in &config.seeds {
        let original = Model::new(
            ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                hidden_size: HIDDEN_SIZE,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            seed,
            &device,
        )?;
        let parameter_bytes: Vec<u8> = original
            .parameter_values()?
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let initial_parameter_hash = sha256(&parameter_bytes);
        for mode in MODES {
            let mut model = original.clone();
            model.feedback = mode;
            let initial = evaluate(&model, &test, &tokenizer)?;
            model = model.train();
            let training_device = model.device();
            let mut optimizer = training::adam();
            let mut logged_updates = Vec::new();
            let start = Instant::now();
            for (step, indices) in schedule.iter().enumerate() {
                let rows: Vec<_> = indices.iter().map(|&index| &train[index]).collect();
                let batch = batch(&rows, &training_device)?;
                let report = step == 0 || (step + 1) % 50 == 0 || step + 1 == config.steps;
                let (loss, norm) = training::optimize_batch(
                    &mut model,
                    &batch,
                    &mut optimizer,
                    LEARNING_RATE,
                    GRADIENT_CLIP,
                    report,
                )?;
                if let Some(loss) = loss {
                    eprintln!(
                        "seed {seed}, {}: step {}/{}, loss {loss:.6}, gradient norm {norm:.6}",
                        mode.as_str(),
                        step + 1,
                        config.steps
                    );
                    logged_updates.push(json!({"step": step + 1, "answer_nll": loss, "preclip_global_gradient_norm": norm}));
                }
            }
            model.parameter_values()?;
            let training_seconds = start.elapsed().as_secs_f64();
            let final_train = evaluate(&model, &train, &tokenizer)?;
            let final_test = evaluate(&model, &test, &tokenizer)?;
            eprintln!(
                "seed {seed}, {}: test accuracy {:.3}%, answer NLL {:.6}",
                mode.as_str(),
                100.0 * final_test.accuracy,
                final_test.nll
            );
            let mut interventions = Vec::new();
            if mode == FeedbackMode::Both {
                for intervention in [
                    FeedbackMode::TopDown,
                    FeedbackMode::BottomUp,
                    FeedbackMode::Neither,
                ] {
                    let mut changed = model.clone().valid();
                    changed.feedback = intervention;
                    interventions.push(json!({
                        "retained_directions": intervention.as_str(),
                        "retrained": false,
                        "test": evaluate(&changed, &test, &tokenizer)?.json(&tokenizer, true)?,
                    }));
                }
            }
            let report = json!({
                "seed": seed,
                "feedback_mode": mode.as_str(),
                "parameters": model.parameter_count(),
                "initial_parameters_f32_le_sha256": initial_parameter_hash,
                "initial_test": initial.json(&tokenizer, false)?,
                "final_train": final_train.json(&tokenizer, false)?,
                "final_test": final_test.json(&tokenizer, true)?,
                "training_wall_seconds_with_final_parameter_readback": training_seconds,
                "training_answer_targets": config.steps * BATCH_SIZE,
                "training_context_tokens": config.steps * BATCH_SIZE * PROMPT_LENGTH,
                "logged_updates": logged_updates,
                "posttraining_interventions": interventions,
            });
            runs.push(RunResult {
                seed,
                mode,
                final_test,
                report,
            });
        }
    }
    let result_summary = summary(&runs, config);
    let train_hash = cases_hash(&data.train);
    let test_hash = cases_hash(&data.test);
    let report = json!({
        "experiment": "delayed_binding_counterfactual_feedback",
        "scope": "Focused synthetic mechanism test; not natural-language, emotional, or symbolic reasoning validation",
        "created_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_secs(),
        "device_request": config.device,
        "device": format!("{device:?}"),
        "device_identity": format!("{:?}", device.identity()),
        "precision": "f32",
        "configuration": {
            "steps": config.steps, "model_seeds": config.seeds, "hidden_size": HIDDEN_SIZE,
            "batch_size": BATCH_SIZE, "prompt_tokens": PROMPT_LENGTH,
            "learning_rate": LEARNING_RATE, "global_gradient_clip": GRADIENT_CLIP,
            "threshold": 1.0, "surrogate_width": 1.0,
            "decays": "learned per neuron, 0.995*sigmoid(logit); initial tau log-uniform in [2, 32] steps",
            "optimizer": {"name": "Adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0},
            "loss": "mean full-vocabulary cross entropy at final answer position only",
            "reset": "all states zero independently for every prompt",
        },
        "protocol": {
            "criterion": "Both mean accuracy >= best other retrained mode + 0.05; positive paired accuracy difference for at least 2/3 default seeds; lower paired mean answer NLL",
            "accuracy": "argmax over the entire character vocabulary; binary forced-choice accuracy is secondary",
            "matched_controls": "all modes clone identical initial parameters per seed; same batches, tensor operations, parameter count, update count, optimizer, and token budget; only interface gates differ",
            "shuffled_spike_feedback_control": "not implemented; no claim about resistance to shuffled feedback",
            "timing": "forward evaluation includes host input transfers and synchronized final-logit readbacks; excludes queued parameter updates; no real-time claim",
        },
        "dataset": {
            "source": "Rust deterministic synthetic diagnostic; no FineWeb or language-checkpoint updates",
            "seed": DATA_SEED,
            "grammar": "two assignments; 2..12 xyz distractors; one opposite-value correction; remaining xyz distractors; each history paired with ?a= and ?b=; opposite final entity values",
            "split": "512 unique training-prefix groups and 128 unique test-prefix groups; two opposite-answer queries per group produce 1024/256 cases; reject duplicate histories globally before assigning split",
            "balance": "exact corrected-entity x final-a-bit strata within each split; paired queries have opposite answers; initial assignment order and distractor gaps random",
            "scope_limit": "opposite final bits imply equal initial bits; a correction-plus-query rule can solve this task without retaining both initial assignments",
            "hash_encoding": "SHA256 over each ordered ASCII prompt + TAB + answer + LF; combined hash over train hash + LF + test hash + LF",
            "train_sha256": train_hash,
            "test_sha256": test_hash,
            "combined_sha256": sha256(format!("{train_hash}\n{test_hash}\n").as_bytes()),
            "vocabulary_fitted_on": "training prompts only",
            "vocabulary": tokenizer.vocab().iter().map(char::to_string).collect::<Vec<_>>(),
            "train_cases": data.train.iter().map(Case::json).collect::<Vec<_>>(),
            "test_cases": data.test.iter().map(Case::json).collect::<Vec<_>>(),
        },
        "minibatch_schedule": {
            "seed": SCHEDULE_SEED,
            "sampling": "uniform training-case indices with replacement; reused identically across all modes and model seeds",
            "hash_encoding": "SHA256 over flattened indices as little-endian u64",
            "sha256": sha256(&schedule_bytes),
        },
        "runs": runs.into_iter().map(|run| run.report).collect::<Vec<_>>(),
        "summary": result_summary,
        "baseline_diagnostics": baseline_diagnostics(&data)?,
    });
    write_report(&output, &report)
}

fn protect_historical_report(output: &Path) -> Result<(), String> {
    if output
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("feedback-probe.json"))
    {
        return Err(
            "feedback-probe.json is the historical first-run report and must not be overwritten"
                .into(),
        );
    }
    let historical = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/feedback-probe.json");
    if output.exists()
        && historical.exists()
        && output.canonicalize().map_err(|error| error.to_string())?
            == historical
                .canonicalize()
                .map_err(|error| error.to_string())?
    {
        return Err("output resolves to the protected historical first-run report".into());
    }
    Ok(())
}

fn write_report(output: &Path, report: &Value) -> Result<PathBuf, String> {
    let parent = output.parent().ok_or("output needs a parent directory")?;
    fs::create_dir_all(parent).map_err(|error| format!("creating report directory: {error}"))?;
    checked_output(output)?;
    fs::write(
        output,
        serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("writing probe report: {error}"))?;
    eprintln!("saved feedback probe report to {}", output.display());
    Ok(output.to_path_buf())
}

/// Audit either the current paired dataset or saved cases from an earlier run.
/// This path never initializes a device or constructs/trains a learned model.
fn run_baselines(output: &Path, dataset_report: Option<&Path>) -> Result<PathBuf, String> {
    let output = checked_output(output)?;
    protect_historical_report(&output)?;
    if output
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("feedback-counterfactual.json"))
    {
        return Err(
            "baseline diagnostics must not overwrite the full counterfactual model report".into(),
        );
    }
    let (data, source) = match dataset_report {
        Some(path) => {
            let input = checked_output(path)?;
            if output.exists()
                && input.canonicalize().map_err(|error| error.to_string())?
                    == output.canonicalize().map_err(|error| error.to_string())?
            {
                return Err("baseline output must differ from the audited dataset report".into());
            }
            read_dataset_report(&input)?
        }
        None => (
            dataset(),
            json!({
                "kind": "current Rust query-counterfactual generator", "seed": DATA_SEED,
                "posthoc_diagnostic": false,
            }),
        ),
    };
    let train_hash = cases_hash(&data.train);
    let test_hash = cases_hash(&data.test);
    let report = json!({
        "experiment": "feedback_shortcut_baseline_diagnostic",
        "scope": "Dataset diagnostic only; no model inference, GPU initialization, training, or checkpoint writes",
        "criterion": "The preregistered model-comparison threshold is unchanged; this artifact adds deterministic task controls",
        "created_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_secs(),
        "source": source,
        "dataset": {
            "train_cases": data.train.len(), "test_cases": data.test.len(),
            "hash_encoding": "SHA256 over each ordered ASCII prompt + TAB + answer + LF; combined hash over train hash + LF + test hash + LF",
            "train_sha256": train_hash, "test_sha256": test_hash,
            "combined_sha256": sha256(format!("{train_hash}\n{test_hash}\n").as_bytes()),
        },
        "baseline_diagnostics": baseline_diagnostics(&data)?,
    });
    write_report(&output, &report)
}

pub fn cli(args: Vec<String>) -> Result<(), String> {
    if args == ["--help"] || args == ["-h"] {
        println!(
            "Usage: feedback_probe [--device gpu|cpu] [--steps 1..10000] [--seeds 42,43,44] [--output artifacts/feedback-counterfactual.json]\n       feedback_probe --baselines-only [--dataset-report artifacts/feedback-probe.json] [--output artifacts/feedback-shortcut-baselines.json]"
        );
        return Ok(());
    }
    let mut baselines_only = false;
    let mut dataset_report = None;
    let mut paired_args = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--baselines-only" if !baselines_only => baselines_only = true,
            "--baselines-only" => return Err("duplicate --baselines-only".into()),
            "--dataset-report" if dataset_report.is_none() => {
                dataset_report = Some(PathBuf::from(
                    args.next().ok_or("--dataset-report requires a value")?,
                ));
            }
            "--dataset-report" => return Err("duplicate --dataset-report".into()),
            _ => paired_args.push(arg),
        }
    }
    if dataset_report.is_some() && !baselines_only {
        return Err("--dataset-report is supported only with --baselines-only".into());
    }
    if baselines_only && !paired_args.iter().any(|arg| arg == "--output") {
        paired_args.extend([
            "--output".into(),
            "artifacts/feedback-shortcut-baselines.json".into(),
        ]);
    }
    let config = ProbeConfig::parse(&paired_args)?;
    if baselines_only {
        run_baselines(&config.output, dataset_report.as_deref())?;
    } else {
        run(&config)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn latest_answer(prompt: &str) -> char {
        let bytes = prompt.as_bytes();
        let query = bytes[bytes.len() - 2];
        let mut value = None;
        for pair in bytes[..bytes.len() - 3].windows(2) {
            if pair[0] == query && matches!(pair[1], b'0' | b'1') {
                value = Some(pair[1] as char);
            }
        }
        value.unwrap()
    }

    #[test]
    fn splits_are_disjoint_balanced_and_labels_follow_latest_queried_value() {
        let data = dataset();
        let mut seen = BTreeSet::new();
        for cases in [&data.train, &data.test] {
            let mut strata = [0; 8];
            for case in cases {
                assert_eq!(case.prompt.len(), PROMPT_LENGTH);
                assert!(seen.insert(case.prompt.clone()));
                assert_eq!(latest_answer(&case.prompt), case.answer);
                let index = usize::from(case.query == 'b')
                    + 2 * usize::from(case.answer == '1')
                    + 4 * usize::from(case.corrected == 'b');
                strata[index] += 1;
            }
            assert!(strata.iter().all(|&count| count == cases.len() / 8));
        }
        let again = dataset();
        assert_eq!(cases_hash(&data.train), cases_hash(&again.train));
        assert_eq!(cases_hash(&data.test), cases_hash(&again.test));
        assert_ne!(cases_hash(&data.train), cases_hash(&data.test));
    }

    #[test]
    fn counterfactual_prefix_pairs_are_disjoint_and_query_blind_baselines_are_chance() {
        let data = dataset();
        let mut all_prefixes = BTreeSet::new();
        for cases in [&data.train, &data.test] {
            let mut groups: BTreeMap<&str, Vec<&Case>> = BTreeMap::new();
            for case in cases {
                groups
                    .entry(&case.prompt[..PROMPT_LENGTH - 3])
                    .or_default()
                    .push(case);
            }
            assert_eq!(groups.len(), cases.len() / 2);
            let mut strata = [0; 4];
            for (prefix, pair) in groups {
                assert!(all_prefixes.insert(prefix.to_string()));
                assert_eq!(pair.len(), 2);
                assert_ne!(pair[0].answer, pair[1].answer);
                assert_ne!(pair[0].query, pair[1].query);
                assert_eq!(pair[0].corrected, pair[1].corrected);
                let final_a = pair.iter().find(|case| case.query == 'a').unwrap().answer;
                strata[usize::from(pair[0].corrected == 'b') + 2 * usize::from(final_a == '1')] +=
                    1;
            }
            assert!(strata.iter().all(|&count| count == cases.len() / 8));
            let diagnostics = baseline_metrics(cases).unwrap();
            for predictor in diagnostics["predictors"].as_array().unwrap() {
                let name = predictor["name"].as_str().unwrap();
                let expected = if name.contains("oracle") {
                    cases.len()
                } else {
                    cases.len() / 2
                };
                assert_eq!(
                    predictor["correct"].as_u64().unwrap(),
                    expected as u64,
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn prompt_parser_reads_latest_queried_value_independently_of_last_assignment() {
        let parsed = parse_prompt("a0;b0;xxb1;yy?a=").unwrap();
        assert_eq!(parsed.query, 'a');
        assert_eq!(parsed.corrected, 'b');
        assert_eq!(parsed.latest_bit, '1');
        assert_eq!(parsed.queried_bit, '0');
        assert_eq!(parsed.first_bit, '0');
        let parsed = parse_prompt("b1;a0;xxa1;yy?b=").unwrap();
        assert_eq!(parsed.query, 'b');
        assert_eq!(parsed.corrected, 'a');
        assert_eq!(parsed.latest_bit, '1');
        assert_eq!(parsed.queried_bit, '1');
        assert_eq!(parsed.first_bit, '1');
        for invalid in [
            "",
            "a0;?b=",
            "a2;b1;?a=",
            "a0;b1;?c=",
            "a0;b1;!a=",
            "a0;b1;wa1;?a=",
        ] {
            assert!(parse_prompt(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn final_only_mask_and_inputs_do_not_include_future_answer() {
        let data = dataset();
        let tokenizer = Tokenizer::from_documents(
            &data
                .train
                .iter()
                .map(|case| case.prompt.clone())
                .collect::<Vec<_>>(),
        );
        let encoded = encode(&data.train[..2], &tokenizer).unwrap();
        let rows: Vec<_> = encoded.iter().collect();
        let batch = batch(&rows, &Device::flex()).unwrap();
        let inputs = batch.inputs.try_into_vec_as::<i32>().unwrap();
        let targets = batch.targets.try_into_vec_as::<i32>().unwrap();
        let mask = batch.mask.try_into_vec_as::<f32>().unwrap();
        assert_eq!(batch.valid_targets, 2);
        for (row, case) in encoded.iter().enumerate() {
            assert_eq!(
                inputs[row * PROMPT_LENGTH..(row + 1) * PROMPT_LENGTH],
                case.inputs
                    .iter()
                    .map(|&token| token as i32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(case.inputs.last(), tokenizer.encode("=").first());
            for time in 0..PROMPT_LENGTH {
                let position = row * PROMPT_LENGTH + time;
                assert_eq!(mask[position], f32::from(time + 1 == PROMPT_LENGTH));
                assert_eq!(
                    targets[position],
                    if time + 1 == PROMPT_LENGTH {
                        case.answer as i32
                    } else {
                        0
                    }
                );
            }
        }
    }

    #[test]
    fn arguments_and_output_are_bounded() {
        assert!(ProbeConfig::parse(&["--steps".into(), "0".into()]).is_err());
        assert!(ProbeConfig::parse(&["--seeds".into(), "42,42".into()]).is_err());
        assert!(ProbeConfig::parse(&["--device".into(), "cuda".into()]).is_err());
        assert!(checked_output(Path::new("../outside.json")).is_err());
        assert!(checked_output(Path::new("artifacts/file.bin")).is_err());
        assert!(checked_output(Path::new("artifacts/probe-test.json")).is_ok());
        assert!(protect_historical_report(Path::new("artifacts/feedback-probe.json")).is_err());
        assert_eq!(
            ProbeConfig::default().output,
            Path::new("artifacts/feedback-counterfactual.json")
        );
    }
}
