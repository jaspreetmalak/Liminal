//! Paired delayed-bit recall curriculum and frozen state/credit diagnostics.
//! Uses the production model, hard spikes, surrogate, reset, and optimizer.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::data::Tokenizer;
use crate::model::{Batch, FeedbackMode, Model, ModelConfig, ModelPath, PARAMETER_NAMES, State};
use crate::rng::Rng;
use crate::{runtime, training};

const DELAYS: [usize; 5] = [1, 2, 4, 8, 16];
const PATHS: [ModelPath; 3] = [
    ModelPath::MemoryOnly,
    ModelPath::SpikesOnly,
    ModelPath::Full,
];
const DATA_SEED: u64 = 42;
const HIDDEN_SIZE: usize = 32;
const BATCH_SIZE: usize = 16;
const TRAIN_HISTORIES: usize = 8;
const MAX_TEST_HISTORIES: usize = 32;
const CHECK_INTERVAL: usize = 25;
const LEARNING_RATE: f64 = 0.003;
const GRADIENT_CLIP: f64 = 1.0;

#[derive(Debug, Clone)]
struct Case {
    history: String,
    prompt: String,
    answer: char,
}

impl Case {
    fn new(history: &str, answer: char) -> Self {
        Self {
            history: history.into(),
            prompt: format!("{}{}{}?", &history[..2], answer, &history[2..]),
            answer,
        }
    }

    fn json(&self) -> Value {
        json!({"history": self.history, "prompt": self.prompt, "answer": self.answer.to_string()})
    }
}

struct Stage {
    delay: usize,
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
    let text: String = cases
        .iter()
        .map(|case| format!("{}\t{}\n", case.prompt, case.answer))
        .collect();
    sha256(text.as_bytes())
}

fn datasets() -> Vec<Stage> {
    let mut random = Rng::new(DATA_SEED);
    DELAYS
        .iter()
        .map(|&delay| {
            let space = 3_usize.pow((delay + 2) as u32);
            let test_histories = MAX_TEST_HISTORIES.min(space - TRAIN_HISTORIES);
            let mut seen = BTreeSet::new();
            let mut histories = Vec::new();
            while histories.len() < TRAIN_HISTORIES + test_histories {
                let history: String = (0..delay + 2)
                    .map(|_| ['x', 'y', 'z'][random.index(3)])
                    .collect();
                if seen.insert(history.clone()) {
                    histories.push(history);
                }
            }
            let paired = |histories: &[String]| {
                histories
                    .iter()
                    .flat_map(|history| ['0', '1'].map(|bit| Case::new(history, bit)))
                    .collect::<Vec<_>>()
            };
            Stage {
                delay,
                train: paired(&histories[..TRAIN_HISTORIES]),
                test: paired(&histories[TRAIN_HISTORIES..]),
            }
        })
        .collect()
}

fn tokenizer(stages: &[Stage]) -> Tokenizer {
    // Only training-stage prompts fit the vocabulary, including stages which
    // a particular seed may not reach. No held-out characters are consulted.
    Tokenizer::from_documents(
        &stages
            .iter()
            .flat_map(|stage| &stage.train)
            .map(|case| case.prompt.clone())
            .collect::<Vec<_>>(),
    )
}

struct Encoded {
    tokens: Vec<usize>,
    answer: usize,
}

fn encode(cases: &[Case], tokenizer: &Tokenizer) -> Result<Vec<Encoded>, String> {
    cases
        .iter()
        .map(|case| {
            let tokens = tokenizer.encode(&case.prompt);
            let answer = tokenizer.encode(&case.answer.to_string())[0];
            if tokens.contains(&0) || answer == 0 {
                return Err("memory diagnostic encountered an unknown token".into());
            }
            Ok(Encoded { tokens, answer })
        })
        .collect()
}

fn batch(cases: &[Encoded], device: &Device) -> Result<Batch, String> {
    let length = cases
        .first()
        .ok_or("memory batch must be nonempty")?
        .tokens
        .len();
    if length < 5 || cases.len() > 64 || cases.iter().any(|case| case.tokens.len() != length) {
        return Err("memory batch requires equal prompt lengths and at most 64 cases".into());
    }
    let inputs: Vec<i32> = cases
        .iter()
        .flat_map(|case| case.tokens.iter().map(|&token| token as i32))
        .collect();
    let mut targets = vec![0_i32; cases.len() * length];
    let mut mask = vec![0.0_f32; cases.len() * length];
    for (row, case) in cases.iter().enumerate() {
        targets[(row + 1) * length - 1] = case.answer as i32;
        mask[(row + 1) * length - 1] = 1.0;
    }
    let shape = [cases.len(), length];
    Ok(Batch {
        inputs: Tensor::<2, Int>::from_data(TensorData::new(inputs, shape), device),
        targets: Tensor::<2, Int>::from_data(TensorData::new(targets, shape), device),
        mask: Tensor::from_data(TensorData::new(mask, shape), device),
        valid_targets: cases.len(),
    })
}

#[derive(Clone)]
struct Score {
    accuracy: f64,
    nll: f64,
    predictions: Vec<usize>,
}

impl Score {
    fn json(&self, tokenizer: &Tokenizer) -> Result<Value, String> {
        Ok(json!({
            "accuracy": self.accuracy, "answer_nll_nats": self.nll,
            "cases": self.predictions.len(),
            "predictions": self.predictions.iter().map(|&token| tokenizer.decode(&[token])).collect::<Result<Vec<_>, _>>()?,
        }))
    }
}

fn score_logits(logits: &[f32], cases: &[Encoded], vocabulary: usize) -> Result<Score, String> {
    if cases.is_empty() || logits.len() != cases.len() * vocabulary {
        return Err("memory score dimensions do not match cases".into());
    }
    let mut correct = 0;
    let mut nll = 0.0;
    let mut predictions = Vec::with_capacity(cases.len());
    for (row, case) in logits.chunks(vocabulary).zip(cases) {
        finite(row)?;
        let mut prediction = 0;
        for index in 1..row.len() {
            if row[index] > row[prediction] {
                prediction = index;
            }
        }
        let maximum = row[prediction] as f64;
        nll += maximum
            + row
                .iter()
                .map(|&value| (value as f64 - maximum).exp())
                .sum::<f64>()
                .ln()
            - row[case.answer] as f64;
        predictions.push(prediction);
        correct += usize::from(prediction == case.answer);
    }
    Ok(Score {
        accuracy: correct as f64 / cases.len() as f64,
        nll: nll / cases.len() as f64,
        predictions,
    })
}

fn evaluate(model: &Model, cases: &[Encoded], path: ModelPath) -> Result<Score, String> {
    let model = model.clone().valid();
    let device = model.device();
    let mut logits = Vec::new();
    for rows in cases.chunks(BATCH_SIZE) {
        let length = rows
            .first()
            .ok_or("memory evaluation must be nonempty")?
            .tokens
            .len();
        let mut state = State::new(rows.len(), model.config.hidden_size, &device);
        let mut final_logits = None;
        for time in 0..length {
            let tokens: Vec<i32> = rows.iter().map(|case| case.tokens[time] as i32).collect();
            final_logits = Some(model.step_for_path(
                Tensor::from_data(TensorData::new(tokens, [rows.len()]), &device),
                &mut state,
                path,
            )?);
        }
        logits.extend(
            final_logits
                .ok_or("empty memory prompt")?
                .try_into_vec_as::<f32>()
                .map_err(|error| error.to_string())?,
        );
    }
    score_logits(&logits, cases, model.config.vocab_size)
}

fn finite(values: &[f32]) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("memory diagnostic encountered a non-finite value".into());
    }
    Ok(())
}

fn values<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>, String> {
    let values = tensor
        .try_into_vec_as::<f32>()
        .map_err(|error| error.to_string())?;
    finite(&values)?;
    Ok(values)
}

fn l2(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn rms(values: &[f32]) -> f64 {
    l2(values) / (values.len() as f64).sqrt()
}

fn pair_distance(values: &[f32]) -> f64 {
    let width = values.len() / 2;
    values[..width]
        .iter()
        .zip(&values[width..])
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn state_metrics(values: &[f32]) -> Value {
    json!({"rms": rms(values), "bit_pair_l2_distance": pair_distance(values)})
}

fn gradient_statistics(values: &[f32]) -> Value {
    json!({
        "l2_norm": l2(values),
        "maximum_absolute": values.iter().map(|value| value.abs() as f64).fold(0.0, f64::max),
        "nonzero_values": values.iter().filter(|&&value| value != 0.0).count(),
        "values": values.len(),
    })
}

fn frozen_pair(delay: usize) -> Vec<Case> {
    let history = format!("xy{}", "x".repeat(delay));
    ['0', '1'].map(|bit| Case::new(&history, bit)).into()
}

fn forward_diagnostic(
    model: &Model,
    cases: &[Encoded],
    path: ModelPath,
    tokenizer: &Tokenizer,
) -> Result<Value, String> {
    let model = model.clone().valid();
    let device = model.device();
    let hidden = model.config.hidden_size;
    let mut state = State::new(2, hidden, &device);
    let mut steps = Vec::new();
    let mut firing_totals = vec![0.0; hidden];
    let length = cases[0].tokens.len();
    let mut final_logits = Vec::new();
    let threshold = model.config.threshold as f32 as f64;
    let width = model.config.surrogate_width as f32 as f64;
    for time in 0..length {
        let ids: Vec<i32> = cases.iter().map(|case| case.tokens[time] as i32).collect();
        let (logits, trace) = model.step_traced_for_path(
            Tensor::from_data(TensorData::new(ids, [2]), &device),
            &mut state,
            path,
        )?;
        let voltage = values(trace.voltage)?;
        let spikes = values(state.spikes.clone())?;
        let membrane = values(state.membrane.clone())?;
        let activity_trace = values(state.trace.clone())?;
        let memory = values(state.semantic.clone())?;
        final_logits = values(logits)?;
        let mut per_neuron = vec![0.0; hidden];
        for row in spikes.chunks(hidden) {
            for ((rate, total), &spike) in per_neuron.iter_mut().zip(&mut firing_totals).zip(row) {
                *rate += spike as f64 / 2.0;
                *total += spike as f64;
            }
        }
        let support = voltage
            .iter()
            .filter(|&&value| (value as f64 - threshold).abs() < width)
            .count();
        let mean_slope = voltage
            .iter()
            .map(|&value| (1.0 - (value as f64 - threshold).abs() / width).max(0.0) / width)
            .sum::<f64>()
            / voltage.len() as f64;
        steps.push(json!({
            "position_zero_based": time,
            "phase": if time < 2 {"irrelevant_prefix"} else if time == 2 {"payload"} else if time + 1 == length {"query"} else {"distractor"},
            "input_pair": cases.iter().map(|case| tokenizer.decode(&[case.tokens[time]])).collect::<Result<Vec<_>, _>>()?,
            "pre_reset_voltage": {
                "minimum": voltage.iter().copied().fold(f32::INFINITY, f32::min),
                "maximum": voltage.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                "bit_pair_l2_distance": pair_distance(&voltage),
            },
            "firing_fraction": spikes.iter().map(|&value| value as f64).sum::<f64>() / spikes.len() as f64,
            "per_neuron_firing_fraction": per_neuron,
            "surrogate_support_fraction": support as f64 / voltage.len() as f64,
            "mean_triangular_surrogate_slope": mean_slope,
            "spikes": state_metrics(&spikes),
            "membrane": state_metrics(&membrane), "trace": state_metrics(&activity_trace),
            "memory": state_metrics(&memory), "logits": state_metrics(&final_logits),
        }));
    }
    Ok(json!({
        "steps": steps,
        "per_neuron_firing_fraction_over_all_steps_and_both_payloads": firing_totals.iter().map(|total| total / (2 * length) as f64).collect::<Vec<_>>(),
        "final_answer": score_logits(&final_logits, cases, model.config.vocab_size)?.json(tokenizer)?,
    }))
}

fn gradient_diagnostic(
    model: &Model,
    cases: &[Encoded],
    path: ModelPath,
    tokenizer: &Tokenizer,
) -> Result<Value, String> {
    let model = model.clone().train();
    let batch = batch(cases, &model.device())?;
    let loss = model.loss_for_path(&batch, path)?;
    let nll = loss.clone().into_scalar::<f32>() as f64;
    if !nll.is_finite() {
        return Err("memory diagnostic loss is non-finite".into());
    }
    let gradients = loss.backward();
    let parameters = model.parameter_tensors();
    if parameters.len() != PARAMETER_NAMES.len() {
        return Err("memory diagnostic parameter names no longer match the model".into());
    }
    let mut groups = Vec::new();
    let mut payload_rows = Vec::new();
    for (index, ((_, parameter), name)) in parameters.into_iter().zip(PARAMETER_NAMES).enumerate() {
        let shape = parameter.dims();
        let gradient = values(
            parameter
                .grad(&gradients)
                .ok_or_else(|| format!("missing diagnostic gradient for {name}"))?,
        )?;
        groups.push(
            json!({"name": name, "shape": shape, "gradient": gradient_statistics(&gradient)}),
        );
        if index == 0 {
            for bit in ['0', '1'] {
                let token = tokenizer.encode(&bit.to_string())[0];
                let start = token * model.config.hidden_size;
                payload_rows.push(json!({
                    "payload": bit.to_string(), "token_id": token,
                    "input_occurrences_per_bit_in_pair": 1,
                    "gradient": gradient_statistics(&gradient[start..start + model.config.hidden_size]),
                }));
            }
        }
    }
    Ok(json!({
        "answer_nll_nats": nll, "parameter_groups": groups, "payload_embedding_rows": payload_rows,
        "interpretation": "Each payload digit occurs only at input position 2, so its embedding-row gradient measures loss credit reaching that input. Full-path credit sums the spike and memory input routes. These are not temporal Jacobians: confidence and operating points differ across delays. No claim about unretained intermediate-state gradients.",
        "optimizer_updates": 0,
    }))
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

fn margin_diagnostic(
    model: &Model,
    cases: &[Encoded],
    path: ModelPath,
    tokenizer: &Tokenizer,
) -> Result<Value, String> {
    // A separate forward graph is required: CE's graph was consumed by backward.
    // Logit margin avoids the softmax/CE confidence multiplier.
    let model = model.clone().train();
    let device = model.device();
    let binary_ids = tokenizer.encode("01");
    let mut state = State::new(cases.len(), model.config.hidden_size, &device);
    let mut final_logits = None;
    for time in 0..cases[0].tokens.len() {
        let ids: Vec<i32> = cases.iter().map(|case| case.tokens[time] as i32).collect();
        final_logits = Some(model.step_for_path(
            Tensor::from_data(TensorData::new(ids, [cases.len()]), &device),
            &mut state,
            path,
        )?);
    }
    let logits = final_logits.ok_or("margin diagnostic needs nonempty prompts")?;
    let correct: Vec<i32> = cases.iter().map(|case| case.answer as i32).collect();
    let other: Vec<i32> = cases
        .iter()
        .map(|case| {
            if case.answer == binary_ids[0] {
                binary_ids[1] as i32
            } else {
                binary_ids[0] as i32
            }
        })
        .collect();
    let shape = [cases.len(), 1];
    let margin = (logits.clone().gather(
        1,
        Tensor::<2, Int>::from_data(TensorData::new(correct, shape), &device),
    ) - logits.gather(
        1,
        Tensor::<2, Int>::from_data(TensorData::new(other, shape), &device),
    ))
    .sum()
        / cases.len() as f64;
    let value = margin.clone().into_scalar::<f32>() as f64;
    if !value.is_finite() {
        return Err("memory diagnostic margin is non-finite".into());
    }
    let gradients = margin.backward();
    let embedding_gradient = values(
        model
            .embedding
            .val()
            .grad(&gradients)
            .ok_or("margin diagnostic has no embedding gradient")?,
    )?;
    let rows: Vec<_> = ['0', '1'].into_iter().zip(binary_ids).map(|(bit, id)| {
        let start = id * model.config.hidden_size;
        json!({
            "payload": bit.to_string(), "token_id": id,
            "gradient": gradient_statistics(&embedding_gradient[start..start + model.config.hidden_size]),
        })
    }).collect();
    Ok(json!({
        "objective": "mean final correct-payload-bit logit minus other-payload-bit logit",
        "mean_logit_margin": value, "payload_embedding_rows": rows, "optimizer_updates": 0,
        "interpretation": "Local output-margin sensitivity without a softmax/CE confidence multiplier. It uses the specified surrogate for spikes and is not a pure temporal Jacobian; operating points differ across delays. Full-path embedding credit combines both input routes.",
    }))
}

fn frozen_sweep(model: &Model, path: ModelPath, tokenizer: &Tokenizer) -> Result<Value, String> {
    let before = parameter_hash(model)?;
    let mut delays = Vec::new();
    for delay in DELAYS {
        let cases = frozen_pair(delay);
        let encoded = encode(&cases, tokenizer)?;
        delays.push(json!({
            "delay": delay, "cases": cases.iter().map(Case::json).collect::<Vec<_>>(),
            "forward": forward_diagnostic(model, &encoded, path, tokenizer)?,
            "backward": gradient_diagnostic(model, &encoded, path, tokenizer)?,
            "margin_backward": margin_diagnostic(model, &encoded, path, tokenizer)?,
        }));
    }
    if parameter_hash(model)? != before {
        return Err("frozen diagnostics unexpectedly changed model parameters".into());
    }
    Ok(json!({
        "parameters_f32_le_sha256": before, "optimizer_updates": 0,
        "not_a_heldout_quality_metric": true,
        "precision": "f32 states and gradients read back; summary arithmetic f64 using rounded f32 threshold/width",
        "delays": delays,
    }))
}

#[derive(Debug, Clone)]
pub struct MemoryProbeConfig {
    pub device: String,
    pub steps: usize,
    pub seeds: Vec<u64>,
    pub output: PathBuf,
}

impl Default for MemoryProbeConfig {
    fn default() -> Self {
        Self {
            device: "gpu".into(),
            steps: 300,
            seeds: vec![42, 43, 44],
            output: "artifacts/memory-diagnostic.json".into(),
        }
    }
}

impl MemoryProbeConfig {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut config = Self::default();
        let mut seen = BTreeSet::new();
        let mut args = args.iter();
        while let Some(flag) = args.next() {
            if !seen.insert(flag) {
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
                        .collect::<Result<Vec<_>, _>>()?
                }
                "--output" => config.output = value.into(),
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
            return Err("--seeds needs 1..=8 distinct u64 values".into());
        }
        checked_output(&self.output)?;
        Ok(())
    }
}

fn checked_output(path: &Path) -> Result<PathBuf, String> {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = project.canonicalize().map_err(|error| error.to_string())?;
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| component == Component::ParentDir)
        || path.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err("--output must be a project-local .json path without '..'".into());
    }
    let output = if path.is_absolute() {
        path.to_path_buf()
    } else {
        project.join(path)
    };
    if !output.starts_with(project) {
        return Err("--output must remain inside the project root".into());
    }
    let mut ancestor = output.as_path();
    while !ancestor.exists() {
        ancestor = ancestor.parent().ok_or("cannot resolve output parent")?;
    }
    if !ancestor
        .canonicalize()
        .map_err(|error| error.to_string())?
        .starts_with(&root)
    {
        return Err("--output resolves outside the project root".into());
    }
    if output
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "feedback-probe.json" | "feedback-counterfactual.json"))
    {
        return Err(
            "memory diagnostic must not overwrite earlier feedback experiment reports".into(),
        );
    }
    Ok(output)
}

/// Run the fixed curriculum with shared production model and optimizer paths.
pub fn run(config: &MemoryProbeConfig) -> Result<PathBuf, String> {
    config.validate()?;
    let output = checked_output(&config.output)?;
    let device = runtime::device(&config.device)?;
    eprintln!(
        "memory probe device: {device:?}; identity: {:?}",
        device.identity()
    );
    let stages = datasets();
    let tokenizer = tokenizer(&stages);
    if tokenizer.vocab_size() != 7 {
        return Err("memory diagnostic vocabulary must have exactly 7 tokens".into());
    }
    let mut reports = Vec::new();
    let mut milestones: BTreeMap<&str, usize> =
        PATHS.iter().map(|path| (path.as_str(), 0)).collect();
    for &seed in &config.seeds {
        let mut original = Model::new(
            ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                hidden_size: HIDDEN_SIZE,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            seed,
            &device,
        )?;
        original.feedback = FeedbackMode::Both;
        let initial_hash = parameter_hash(&original)?;
        for path in PATHS {
            let mut model = original.clone().train();
            if parameter_hash(&model)? != initial_hash {
                return Err("diagnostic paths do not share initial parameters".into());
            }
            let mut optimizer = training::adam();
            let before = frozen_sweep(&model, path, &tokenizer)?;
            let mut stage_reports = Vec::new();
            let mut total_updates = 0;
            let mut total_targets = 0;
            let mut total_tokens = 0;
            let mut all_training_gates = true;
            let mut final_milestone = false;
            let mut reached_final_delay = false;
            for stage in &stages {
                let train = encode(&stage.train, &tokenizer)?;
                let test = encode(&stage.test, &tokenizer)?;
                let batch = batch(&train, &model.device())?;
                let mut checks = Vec::new();
                let mut progress = Vec::new();
                let mut final_train = None;
                let mut updates = 0;
                let mut training_gate = false;
                for step in 1..=config.steps {
                    let check = step % CHECK_INTERVAL == 0 || step == config.steps;
                    let (loss, norm) = training::optimize_batch_for_path(
                        &mut model,
                        &batch,
                        &mut optimizer,
                        LEARNING_RATE,
                        GRADIENT_CLIP,
                        check || step == 1,
                        path,
                    )?;
                    updates = step;
                    if let Some(loss) = loss {
                        progress.push(json!({"stage_update": step, "preupdate_answer_nll": loss, "preclip_global_gradient_norm": norm}));
                    }
                    if check {
                        let score = evaluate(&model, &train, path)?;
                        training_gate = score.accuracy == 1.0 && score.nll <= 0.1;
                        eprintln!(
                            "seed {seed}, {}, delay {}: step {step}/{}, train {:.2}%, NLL {:.6}",
                            path.as_str(),
                            stage.delay,
                            config.steps,
                            100.0 * score.accuracy,
                            score.nll
                        );
                        checks.push(json!({"stage_update": step, "train": score.json(&tokenizer)?, "training_gate_passed": training_gate}));
                        final_train = Some(score);
                        if training_gate {
                            break;
                        }
                    }
                }
                let train_score =
                    final_train.ok_or("curriculum stage has no final training measurement")?;
                // This is the only held-out evaluation for a curriculum stage.
                // It is never consulted by the stop/advance decision above.
                let test_score = evaluate(&model, &test, path)?;
                total_updates += updates;
                total_targets += updates * train.len();
                total_tokens += updates * train.len() * (stage.delay + 4);
                reached_final_delay = stage.delay == 16;
                if reached_final_delay {
                    final_milestone = train_score.accuracy >= 0.99 && test_score.accuracy >= 0.95;
                }
                stage_reports.push(json!({
                    "delay": stage.delay, "prompt_tokens": stage.delay + 4,
                    "updates": updates, "training_answer_targets": updates * train.len(),
                    "training_input_tokens": updates * train.len() * (stage.delay + 4),
                    "training_gate_passed": training_gate,
                    "final_train": train_score.json(&tokenizer)?, "final_validation": test_score.json(&tokenizer)?,
                    "training_only_stop_checks": checks, "progress": progress,
                }));
                if !training_gate {
                    all_training_gates = false;
                    eprintln!(
                        "seed {seed}, {}: curriculum stopped at delay {}; later stages not trained",
                        path.as_str(),
                        stage.delay
                    );
                    break;
                }
            }
            let after = frozen_sweep(&model, path, &tokenizer)?;
            if final_milestone {
                *milestones.get_mut(path.as_str()).unwrap() += 1;
            }
            let not_reached: Vec<_> = DELAYS.iter().copied().skip(stage_reports.len()).collect();
            reports.push(json!({
                "seed": seed, "path": path.as_str(), "feedback_mode": "both",
                "parameters": model.parameter_count(), "initial_parameters_f32_le_sha256": initial_hash,
                "final_parameters_f32_le_sha256": parameter_hash(&model)?,
                "stage_reports": stage_reports, "remaining_delays_not_reached": not_reached,
                "actual_updates": total_updates, "actual_training_answer_targets": total_targets,
                "actual_training_input_tokens": total_tokens,
                "all_stage_training_gates_passed": all_training_gates,
                "reached_final_delay": reached_final_delay, "final_delay_milestone_passed": final_milestone,
                "status": if final_milestone {"final_delay_milestone_passed"} else if reached_final_delay {"final_delay_milestone_failed"} else {"stopped_before_final_delay"},
                "frozen_sweep_before": before, "frozen_sweep_after": after,
            }));
        }
    }
    let dataset_reports: Vec<_> = stages
        .iter()
        .map(|stage| {
            let train_hash = cases_hash(&stage.train);
            let test_hash = cases_hash(&stage.test);
            json!({
                "delay": stage.delay, "prompt_tokens": stage.delay + 4,
                "possible_noise_histories": 3_usize.pow((stage.delay + 2) as u32),
                "training_histories": TRAIN_HISTORIES, "validation_histories": stage.test.len() / 2,
                "train_sha256": train_hash, "validation_sha256": test_hash,
                "train_cases": stage.train.iter().map(Case::json).collect::<Vec<_>>(),
                "validation_cases": stage.test.iter().map(Case::json).collect::<Vec<_>>(),
            })
        })
        .collect();
    let hash_manifest: String = stages
        .iter()
        .map(|stage| {
            format!(
                "{}\t{}\t{}\n",
                stage.delay,
                cases_hash(&stage.train),
                cases_hash(&stage.test)
            )
        })
        .collect();
    let preregistered = config.steps == 300 && config.seeds == [42, 43, 44];
    let report = json!({
        "experiment": "delayed_bit_memory_curriculum_diagnostic",
        "scope": "Focused synthetic memory/credit diagnostic; not language, feedback advantage, or emotional reasoning validation",
        "created_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_secs(),
        "device_request": config.device, "device": format!("{device:?}"), "device_identity": format!("{:?}", device.identity()),
        "configuration": {
            "model_seeds": config.seeds, "stage_update_cap": config.steps,
            "hidden_size": HIDDEN_SIZE, "batch_size": BATCH_SIZE, "delays": DELAYS,
            "learning_rate": LEARNING_RATE, "global_gradient_clip": GRADIENT_CLIP,
            "precision": "f32", "threshold": 1.0, "surrogate_width": 1.0,
            "decays": "learned per neuron, 0.995*sigmoid(logit); initial tau log-uniform in [2, 32] steps",
            "optimizer": {"name": "Adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0},
        },
        "protocol": {
            "paths": PATHS.iter().map(|path| path.as_str()).collect::<Vec<_>>(),
            "isolation": "memory-only/spikes-only gate both couplings off and gate the opposite readout off; full uses Both; same parameters, forward graph, surrogate/reset equations, and shared optimizer implementation",
            "initialization": "fresh identical initial parameters per seed/path, fresh Adam per path; parameters and Adam moments carry across that path's curriculum stages",
            "stop_rule": "training accuracy exactly 100% and answer NLL <= 0.1 at each 25-update check or last update; advance if passed, otherwise stop path after stage cap",
            "validation": "evaluate once per reached stage after training stops; never used for updates, stopping, or checkpoint selection",
            "loss": "full-vocabulary mean cross entropy only after final '?' input; all earlier masks zero; each target equals sole payload digit at position 2",
            "state": "all recurrent states reset independently per prompt/fullbatch update",
            "milestone": "at final delay 16: training accuracy >=99% and held-out accuracy >=95%, for at least 2 of 3 default seeds per path",
            "frozen_diagnostics": "same xy0+x^D+? and xy1+x^D+? pair, all delays, before curriculum and after its last reached stage; no optimizer updates during diagnostics",
            "checkpoint_writes": 0,
        },
        "dataset": {
            "source": "Rust deterministic synthetic diagnostic", "seed": DATA_SEED,
            "grammar": "two xyz prefix chars + one payload bit + D xyz distractors + '?'",
            "split": "sample unique noise histories per delay; first 8 train, next min(32,3^(D+2)-8) validation; each history paired with both payloads; no history shared across splits",
            "vocabulary_fitted_on": "union of training prompts from all stages, never validation",
            "vocabulary": tokenizer.vocab().iter().map(char::to_string).collect::<Vec<_>>(),
            "case_hash_encoding": "SHA256 over ordered ASCII prompt + TAB + answer + LF",
            "combined_hash_encoding": "SHA256 over each delay + TAB + train hash + TAB + validation hash + LF",
            "combined_sha256": sha256(hash_manifest.as_bytes()), "stages": dataset_reports,
            "history_only_or_constant_predictor_accuracy": 0.5, "read_payload_oracle_accuracy": 1.0,
        },
        "schedule": {
            "kind": "every update uses all 16 stage-training cases in their recorded fixed order; same schedule across paths and seeds",
            "indices": (0..BATCH_SIZE).collect::<Vec<_>>(),
            "index_encoding": "little-endian u64 indices, once; identical complete minibatch repeated each update",
            "sha256": sha256(&(0..BATCH_SIZE).flat_map(|index| (index as u64).to_le_bytes()).collect::<Vec<_>>()),
        },
        "runs": reports,
        "summary": {
            "uses_preregistered_steps_and_seeds": preregistered,
            "paths": PATHS.iter().map(|path| json!({
                "path": path.as_str(), "final_delay_milestone_passing_seeds": milestones[path.as_str()],
                "total_seeds": config.seeds.len(), "meets_two_of_three_default_seed_milestone": preregistered && milestones[path.as_str()] >= 2,
            })).collect::<Vec<_>>(),
        },
    });
    fs::create_dir_all(output.parent().ok_or("output has no parent")?)
        .map_err(|error| error.to_string())?;
    checked_output(&output)?;
    fs::write(
        &output,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("writing memory diagnostic: {error}"))?;
    eprintln!("saved memory diagnostic to {}", output.display());
    Ok(output)
}

const FEEDBACK_MODES: [FeedbackMode; 4] = [
    FeedbackMode::Both,
    FeedbackMode::TopDown,
    FeedbackMode::BottomUp,
    FeedbackMode::Neither,
];

struct ComparisonRun {
    seed: u64,
    mode: FeedbackMode,
    validation: Score,
    report: Value,
}

fn comparison_summary(runs: &[ComparisonRun], config: &MemoryProbeConfig) -> Value {
    let mean = |mode: FeedbackMode, metric: fn(&Score) -> f64| {
        let matching: Vec<_> = runs.iter().filter(|run| run.mode == mode).collect();
        matching
            .iter()
            .map(|run| metric(&run.validation))
            .sum::<f64>()
            / matching.len() as f64
    };
    let mut best_other = FeedbackMode::TopDown;
    for mode in [FeedbackMode::BottomUp, FeedbackMode::Neither] {
        let accuracy = mean(mode, |score| score.accuracy);
        let best_accuracy = mean(best_other, |score| score.accuracy);
        if accuracy > best_accuracy
            || (accuracy == best_accuracy
                && mean(mode, |score| score.nll) < mean(best_other, |score| score.nll))
        {
            best_other = mode;
        }
    }
    let accuracy_advantage =
        mean(FeedbackMode::Both, |score| score.accuracy) - mean(best_other, |score| score.accuracy);
    let nll_difference =
        mean(FeedbackMode::Both, |score| score.nll) - mean(best_other, |score| score.nll);
    let mut positive_seeds = 0;
    let mut paired = Vec::new();
    for &seed in &config.seeds {
        let both = &runs
            .iter()
            .find(|run| run.seed == seed && run.mode == FeedbackMode::Both)
            .unwrap()
            .validation;
        let other = &runs
            .iter()
            .find(|run| run.seed == seed && run.mode == best_other)
            .unwrap()
            .validation;
        positive_seeds += usize::from(both.accuracy > other.accuracy);
        paired.push(json!({"seed": seed, "accuracy_advantage": both.accuracy - other.accuracy, "nll_difference_nats": both.nll - other.nll}));
    }
    let preregistered = config.steps == 300 && config.seeds == [42, 43, 44];
    let passed = accuracy_advantage >= 0.05 && positive_seeds >= 2 && nll_difference < 0.0;
    json!({
        "means_by_mode": FEEDBACK_MODES.iter().map(|&mode| json!({
            "mode": mode.as_str(), "validation_accuracy": mean(mode, |score| score.accuracy),
            "validation_answer_nll_nats": mean(mode, |score| score.nll),
        })).collect::<Vec<_>>(),
        "best_other_mode_by_mean_accuracy_then_nll": best_other.as_str(),
        "both_accuracy_advantage": accuracy_advantage, "both_nll_difference_nats": nll_difference,
        "positive_paired_seeds": positive_seeds, "paired": paired,
        "uses_preregistered_steps_and_seeds": preregistered, "meets_numerical_criterion": passed,
        "decision": if !preregistered { "exploratory_run_not_preregistered_confirmation" }
            else if passed { "supported_on_this_cold_start_bit_recall_task" }
            else { "not_supported_by_preregistered_test" },
    })
}

fn comparison_output(path: &Path) -> Result<PathBuf, String> {
    let output = checked_output(path)?;
    if output
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("memory-diagnostic.json"))
    {
        return Err(
            "feedback comparison must not overwrite the curriculum memory-diagnostic.json report"
                .into(),
        );
    }
    let memory_report =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/memory-diagnostic.json");
    if output.exists()
        && memory_report.exists()
        && output.canonicalize().map_err(|error| error.to_string())?
            == memory_report
                .canonicalize()
                .map_err(|error| error.to_string())?
    {
        return Err(
            "feedback comparison output resolves to the protected curriculum report".into(),
        );
    }
    Ok(output)
}

/// A separate matched, cold-start delay-16 experiment. It does not load or
/// continue the curriculum's weights, and never performs early stopping.
fn run_feedback_comparison(config: &MemoryProbeConfig) -> Result<PathBuf, String> {
    config.validate()?;
    let output = comparison_output(&config.output)?;
    let device = runtime::device(&config.device)?;
    eprintln!(
        "memory feedback comparison device: {device:?}; identity: {:?}",
        device.identity()
    );
    let stages = datasets();
    let tokenizer = tokenizer(&stages);
    if tokenizer.vocab_size() != 7 {
        return Err("feedback comparison vocabulary must have 7 tokens".into());
    }
    let stage = stages
        .iter()
        .find(|stage| stage.delay == 16)
        .ok_or("delay-16 data missing")?;
    let train = encode(&stage.train, &tokenizer)?;
    let validation = encode(&stage.test, &tokenizer)?;
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
        let initial_hash = parameter_hash(&original)?;
        for mode in FEEDBACK_MODES {
            let mut model = original.clone();
            model.feedback = mode;
            if parameter_hash(&model)? != initial_hash {
                return Err("feedback conditions do not share initial parameters".into());
            }
            let initial_train = evaluate(&model, &train, ModelPath::Full)?;
            let initial_validation = evaluate(&model, &validation, ModelPath::Full)?;
            model = model.train();
            let batch = batch(&train, &model.device())?;
            let mut optimizer = training::adam();
            let mut progress = Vec::new();
            for step in 1..=config.steps {
                let (loss, norm) = training::optimize_batch_for_path(
                    &mut model,
                    &batch,
                    &mut optimizer,
                    LEARNING_RATE,
                    GRADIENT_CLIP,
                    step == 1 || step % 50 == 0 || step == config.steps,
                    ModelPath::Full,
                )?;
                if let Some(loss) = loss {
                    eprintln!(
                        "cold-start seed {seed}, {}: step {step}/{}, loss {loss:.6}, gradient norm {norm:.6}",
                        mode.as_str(),
                        config.steps
                    );
                    progress.push(json!({"update": step, "preupdate_answer_nll": loss, "preclip_global_gradient_norm": norm}));
                }
            }
            let final_train = evaluate(&model, &train, ModelPath::Full)?;
            let final_validation = evaluate(&model, &validation, ModelPath::Full)?;
            eprintln!(
                "cold-start seed {seed}, {}: train {:.2}%, held-out {:.2}%, held-out NLL {:.6}",
                mode.as_str(),
                100.0 * final_train.accuracy,
                100.0 * final_validation.accuracy,
                final_validation.nll
            );
            let report = json!({
                "seed": seed, "feedback_mode": mode.as_str(), "model_path": "full",
                "parameters": model.parameter_count(), "initial_parameters_f32_le_sha256": initial_hash,
                "final_parameters_f32_le_sha256": parameter_hash(&model)?,
                "initial_train": initial_train.json(&tokenizer)?, "initial_validation": initial_validation.json(&tokenizer)?,
                "final_train": final_train.json(&tokenizer)?, "final_validation": final_validation.json(&tokenizer)?,
                "actual_updates": config.steps, "actual_training_answer_targets": config.steps * BATCH_SIZE,
                "actual_training_input_tokens": config.steps * BATCH_SIZE * (stage.delay + 4), "progress": progress,
            });
            runs.push(ComparisonRun {
                seed,
                mode,
                validation: final_validation,
                report,
            });
        }
    }
    let summary = comparison_summary(&runs, config);
    let hash_manifest: String = stages
        .iter()
        .map(|stage| {
            format!(
                "{}\t{}\t{}\n",
                stage.delay,
                cases_hash(&stage.train),
                cases_hash(&stage.test)
            )
        })
        .collect();
    let report = json!({
        "experiment": "cold_start_delay16_memory_feedback_comparison",
        "scope": "Matched feedback comparison on one-bit long-delay recall; independent cold starts, no curriculum or transferred weights; not emotional reasoning or language generation validation",
        "created_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_secs(),
        "device_request": config.device, "device": format!("{device:?}"), "device_identity": format!("{:?}", device.identity()),
        "configuration": {
            "model_seeds": config.seeds, "updates_per_condition": config.steps, "delay": stage.delay,
            "hidden_size": HIDDEN_SIZE, "batch_size": BATCH_SIZE, "prompt_tokens": stage.delay + 4,
            "learning_rate": LEARNING_RATE, "global_gradient_clip": GRADIENT_CLIP, "precision": "f32",
            "threshold": 1.0, "surrogate_width": 1.0,
            "decays": "learned per neuron, 0.995*sigmoid(logit); initial tau log-uniform in [2, 32] steps",
            "optimizer": {"name": "Adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0},
        },
        "protocol": {
            "conditions": FEEDBACK_MODES.iter().map(|mode| mode.as_str()).collect::<Vec<_>>(),
            "model_path": "full", "cold_start": true, "curriculum": false, "transferred_weights": false,
            "initialization": "identical initial parameters per seed across all modes; fresh Adam for every condition",
            "updates": "exactly the configured count for every condition, with no early stopping or validation-based selection",
            "loss": "full-vocabulary mean cross entropy only at final '?' position; all earlier masks zero",
            "state": "zero all states independently for every prompt/update",
            "criterion": "Both mean held-out accuracy >= best other retrained mode +0.05; positive paired accuracy difference for at least 2 of 3 default seeds; lower paired mean held-out NLL",
            "checkpoint_writes": 0,
        },
        "dataset": {
            "seed": DATA_SEED, "source": "existing Rust memory diagnostic delay-16 stage, unchanged",
            "grammar": "two xyz prefix chars + one payload bit + 16 xyz distractors + '?'",
            "training_histories": TRAIN_HISTORIES, "validation_histories": stage.test.len() / 2,
            "training_case_count": stage.train.len(), "validation_case_count": stage.test.len(),
            "vocabulary_fitted_on": "union of all curriculum training-stage prompts only, to preserve the same 7-token vocabulary; no held-out prompts consulted",
            "vocabulary": tokenizer.vocab().iter().map(char::to_string).collect::<Vec<_>>(),
            "case_hash_encoding": "SHA256 over ordered ASCII prompt + TAB + answer + LF",
            "train_sha256": cases_hash(&stage.train), "validation_sha256": cases_hash(&stage.test),
            "all_curriculum_stages_combined_sha256": sha256(hash_manifest.as_bytes()),
            "train_cases": stage.train.iter().map(Case::json).collect::<Vec<_>>(),
            "validation_cases": stage.test.iter().map(Case::json).collect::<Vec<_>>(),
            "history_only_or_constant_predictor_accuracy": 0.5, "read_payload_oracle_accuracy": 1.0,
        },
        "schedule": {
            "kind": "full batch of the same 16 delay-16 training cases, in the recorded order, every update for every condition/seed",
            "indices": (0..BATCH_SIZE).collect::<Vec<_>>(),
            "index_encoding": "little-endian u64 indices, once; repeat identical batch each update",
            "sha256": sha256(&(0..BATCH_SIZE).flat_map(|index| (index as u64).to_le_bytes()).collect::<Vec<_>>()),
        },
        "runs": runs.into_iter().map(|run| run.report).collect::<Vec<_>>(), "summary": summary,
    });
    fs::create_dir_all(output.parent().ok_or("output has no parent")?)
        .map_err(|error| error.to_string())?;
    comparison_output(&output)?;
    fs::write(
        &output,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("writing memory feedback comparison: {error}"))?;
    eprintln!("saved memory feedback comparison to {}", output.display());
    Ok(output)
}

fn cli_config(args: &[String]) -> Result<(bool, MemoryProbeConfig), String> {
    let mut compare = false;
    let mut paired = Vec::new();
    for arg in args {
        if arg == "--compare-feedback" {
            if compare {
                return Err("duplicate --compare-feedback".into());
            }
            compare = true;
        } else {
            paired.push(arg.clone());
        }
    }
    if compare && !paired.iter().any(|arg| arg == "--output") {
        paired.extend([
            "--output".into(),
            "artifacts/memory-feedback-comparison.json".into(),
        ]);
    }
    let config = MemoryProbeConfig::parse(&paired)?;
    if compare {
        comparison_output(&config.output)?;
    }
    Ok((compare, config))
}

pub fn cli(args: Vec<String>) -> Result<(), String> {
    if args == ["--help"] || args == ["-h"] {
        println!(
            "Usage: memory_probe [--device gpu|cpu] [--steps 1..10000] [--seeds 42,43,44] [--output artifacts/memory-diagnostic.json]\n       memory_probe --compare-feedback [--device gpu|cpu] [--steps 300] [--seeds 42,43,44] [--output artifacts/memory-feedback-comparison.json]"
        );
        return Ok(());
    }
    let (compare, config) = cli_config(&args)?;
    if compare {
        run_feedback_comparison(&config)?;
    } else {
        run(&config)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_splits_are_disjoint_paired_balanced_and_have_expected_budgets() {
        let stages = datasets();
        assert_eq!(
            stages.iter().map(|stage| stage.delay).collect::<Vec<_>>(),
            DELAYS
        );
        for stage in &stages {
            assert_eq!(stage.train.len(), 16);
            assert_eq!(stage.test.len(), if stage.delay == 1 { 38 } else { 64 });
            let mut seen = BTreeSet::new();
            for cases in [&stage.train, &stage.test] {
                let mut groups: BTreeMap<&str, Vec<&Case>> = BTreeMap::new();
                for case in cases {
                    groups.entry(&case.history).or_default().push(case);
                    assert_eq!(case.history.len(), stage.delay + 2);
                    assert!(
                        case.history
                            .chars()
                            .all(|character| matches!(character, 'x' | 'y' | 'z'))
                    );
                    assert_eq!(case.prompt.len(), stage.delay + 4);
                    assert_eq!(case.prompt.as_bytes()[2] as char, case.answer);
                    assert_eq!(
                        case.prompt
                            .chars()
                            .filter(|character| matches!(character, '0' | '1'))
                            .count(),
                        1
                    );
                    assert_eq!(case.prompt.chars().last(), Some('?'));
                }
                for (history, pair) in groups {
                    assert!(seen.insert(history.to_string()));
                    assert_eq!(pair.len(), 2);
                    assert_ne!(pair[0].answer, pair[1].answer);
                }
                assert_eq!(
                    cases.iter().filter(|case| case.answer == '0').count(),
                    cases.len() / 2
                );
            }
        }
        let repeated = datasets();
        for (first, second) in stages.iter().zip(repeated) {
            assert_eq!(cases_hash(&first.train), cases_hash(&second.train));
            assert_eq!(cases_hash(&first.test), cases_hash(&second.test));
        }
    }

    #[test]
    fn training_only_vocabulary_covers_stages_and_final_only_masks_do_not_append_answers() {
        let stages = datasets();
        let tokenizer = tokenizer(&stages);
        assert_eq!(
            tokenizer.vocab(),
            &['\u{fffd}', '0', '1', '?', 'x', 'y', 'z']
        );
        for stage in &stages {
            let encoded = encode(&stage.train, &tokenizer).unwrap();
            encode(&stage.test, &tokenizer).unwrap();
            let batch = batch(&encoded, &Device::flex()).unwrap();
            assert_eq!(batch.valid_targets, BATCH_SIZE);
            let input = batch.inputs.try_into_vec_as::<i32>().unwrap();
            let target = batch.targets.try_into_vec_as::<i32>().unwrap();
            let mask = batch.mask.try_into_vec_as::<f32>().unwrap();
            let length = stage.delay + 4;
            for (row, case) in encoded.iter().enumerate() {
                assert_eq!(
                    input[row * length..(row + 1) * length],
                    case.tokens
                        .iter()
                        .map(|&token| token as i32)
                        .collect::<Vec<_>>()
                );
                assert_eq!(case.tokens.last(), tokenizer.encode("?").first());
                for time in 0..length {
                    let index = row * length + time;
                    assert_eq!(mask[index], f32::from(time + 1 == length));
                    assert_eq!(
                        target[index],
                        if time + 1 == length {
                            case.answer as i32
                        } else {
                            0
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn config_is_bounded_and_model_path_names_round_trip() {
        for path in PATHS {
            assert_eq!(ModelPath::parse(path.as_str()).unwrap(), path);
        }
        assert!(MemoryProbeConfig::parse(&["--steps".into(), "0".into()]).is_err());
        assert!(MemoryProbeConfig::parse(&["--seeds".into(), "42,42".into()]).is_err());
        assert!(MemoryProbeConfig::parse(&["--device".into(), "cuda".into()]).is_err());
        assert!(checked_output(Path::new("../outside.json")).is_err());
        assert!(checked_output(Path::new("artifacts/liminal.bin")).is_err());
        assert!(checked_output(Path::new("artifacts/memory-diagnostic.json")).is_ok());
    }

    #[test]
    fn feedback_comparison_cli_is_separate_and_protects_curriculum_report() {
        let (compare, default) = cli_config(&[]).unwrap();
        assert!(!compare);
        assert_eq!(
            default.output,
            Path::new("artifacts/memory-diagnostic.json")
        );
        let (compare, configured) = cli_config(&["--compare-feedback".into()]).unwrap();
        assert!(compare);
        assert_eq!(
            configured.output,
            Path::new("artifacts/memory-feedback-comparison.json")
        );
        assert_eq!(configured.steps, 300);
        assert_eq!(configured.seeds, [42, 43, 44]);
        assert!(cli_config(&["--compare-feedback".into(), "--compare-feedback".into()]).is_err());
        assert!(
            cli_config(&[
                "--compare-feedback".into(),
                "--output".into(),
                "artifacts/memory-diagnostic.json".into()
            ])
            .is_err()
        );
        assert!(
            cli_config(&[
                "--compare-feedback".into(),
                "--output".into(),
                "artifacts/liminal.bin".into()
            ])
            .is_err()
        );
    }

    #[test]
    fn feedback_control_summary_requires_all_three_predeclared_conditions() {
        let config = MemoryProbeConfig::default();
        let make = |accuracy_advantage: f64, nll_advantage: f64| {
            config
                .seeds
                .iter()
                .flat_map(|&seed| {
                    FEEDBACK_MODES.map(|mode| {
                        let both = mode == FeedbackMode::Both;
                        ComparisonRun {
                            seed,
                            mode,
                            report: Value::Null,
                            validation: Score {
                                accuracy: 0.75 + if both { accuracy_advantage } else { 0.0 },
                                nll: 0.5 - if both { nll_advantage } else { 0.0 },
                                predictions: Vec::new(),
                            },
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            comparison_summary(&make(0.10, 0.1), &config)["meets_numerical_criterion"],
            true
        );
        assert_eq!(
            comparison_summary(&make(0.0, 0.1), &config)["meets_numerical_criterion"],
            false
        );
        assert_eq!(
            comparison_summary(&make(0.10, -0.1), &config)["meets_numerical_criterion"],
            false
        );
        let mut one_positive_seed = make(0.0, 0.1);
        one_positive_seed
            .iter_mut()
            .find(|run| run.seed == 42 && run.mode == FeedbackMode::Both)
            .unwrap()
            .validation
            .accuracy = 1.0;
        let summary = comparison_summary(&one_positive_seed, &config);
        assert!(summary["both_accuracy_advantage"].as_f64().unwrap() >= 0.05);
        assert_eq!(summary["positive_paired_seeds"], 1);
        assert_eq!(summary["meets_numerical_criterion"], false);
    }

    #[test]
    fn answer_scoring_uses_full_vocabulary_and_stable_logsumexp() {
        let cases = [
            Encoded {
                tokens: vec![1; 5],
                answer: 1,
            },
            Encoded {
                tokens: vec![2; 5],
                answer: 2,
            },
        ];
        let confident = score_logits(
            &[-1000.0, 1000.0, -1000.0, -1000.0, -1000.0, 1000.0],
            &cases,
            3,
        )
        .unwrap();
        assert_eq!(confident.accuracy, 1.0);
        assert_eq!(confident.nll, 0.0);
        let uniform = score_logits(&[0.0; 6], &cases, 3).unwrap();
        assert!((uniform.nll - 3.0_f64.ln()).abs() < 1e-12);
        assert_eq!(uniform.accuracy, 0.0);
        assert!(score_logits(&[f32::NAN; 6], &cases, 3).is_err());
    }

    #[test]
    fn frozen_diagnostics_use_encoded_payload_rows_and_do_not_update_parameters() {
        let stages = datasets();
        let tokenizer = tokenizer(&stages);
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
        let sweep = frozen_sweep(&model, ModelPath::MemoryOnly, &tokenizer).unwrap();
        assert_eq!(parameter_hash(&model).unwrap(), before);
        assert_eq!(sweep["optimizer_updates"], 0);
        assert_eq!(sweep["delays"].as_array().unwrap().len(), DELAYS.len());
        for delay in sweep["delays"].as_array().unwrap() {
            assert_eq!(
                delay["backward"]["parameter_groups"]
                    .as_array()
                    .unwrap()
                    .len(),
                PARAMETER_NAMES.len()
            );
            for kind in ["backward", "margin_backward"] {
                let rows = delay[kind]["payload_embedding_rows"].as_array().unwrap();
                assert_eq!(rows.len(), 2);
                for (row, &token) in rows.iter().zip(&tokenizer.encode("01")) {
                    assert_eq!(row["token_id"].as_u64().unwrap(), token as u64);
                    assert_eq!(row["gradient"]["values"], 2);
                }
                assert_eq!(delay[kind]["optimizer_updates"], 0);
            }
            for step in &delay["forward"]["steps"].as_array().unwrap()[..2] {
                for state in ["membrane", "trace", "memory", "logits"] {
                    assert_eq!(step[state]["bit_pair_l2_distance"].as_f64().unwrap(), 0.0);
                }
            }
        }
    }
}
