//! Document-contained batches, Burn autodiff/Adam, evaluation, and generation.

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::model::{Batch, Model, ModelPath, State, VoltagePenalty};
use crate::rng::Rng;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub steps: usize,
    pub sequence_length: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub learning_rate_schedule: LearningRateSchedule,
    pub gradient_clip: f64,
    pub seed: u64,
    pub voltage_penalty: VoltagePenalty,
}

/// Cosine decay ends at this fraction of the base learning rate.
const COSINE_FLOOR: f64 = 0.1;

/// Per-update Adam learning rate. Cosine decays from the base rate at update 1
/// to COSINE_FLOOR times it at the final update; constant keeps the base rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningRateSchedule {
    Constant,
    Cosine,
}

impl LearningRateSchedule {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "constant" => Ok(Self::Constant),
            "cosine" => Ok(Self::Cosine),
            _ => Err("learning-rate schedule must be constant or cosine".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Cosine => "cosine",
        }
    }

    /// The rate for 1-based `step` of `steps` updates.
    pub fn rate(self, base: f64, step: usize, steps: usize) -> f64 {
        match self {
            Self::Constant => base,
            Self::Cosine => {
                let progress = if steps > 1 {
                    (step - 1) as f64 / (steps - 1) as f64
                } else {
                    0.0
                };
                let floor = base * COSINE_FLOOR;
                floor + (base - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        }
    }
}

impl TrainConfig {
    fn validate(&self) -> Result<(), String> {
        if self.steps == 0 || self.sequence_length == 0 || self.batch_size == 0 {
            return Err("training steps, sequence length, and batch size must be positive".into());
        }
        if !self.learning_rate.is_finite()
            || !(self.learning_rate as f32).is_finite()
            || self.learning_rate as f32 <= 0.0
        {
            return Err("learning rate must be finite, positive, and representable in f32".into());
        }
        if !self.gradient_clip.is_finite()
            || !(self.gradient_clip as f32).is_finite()
            || self.gradient_clip as f32 <= 0.0
        {
            return Err("gradient clip must be finite, positive, and representable in f32".into());
        }
        self.batch_size
            .checked_mul(self.sequence_length)
            .ok_or("batch dimensions exceed the addressable range")?;
        self.voltage_penalty.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub initial_train_loss: f64,
    pub final_train_loss: f64,
    pub initial_validation_loss: f64,
    pub final_validation_loss: f64,
    pub steps: usize,
    /// Number of non-padding next-token targets actually used in updates.
    pub trained_targets: usize,
}

/// Windows are independent and target-weighted, so evaluation batch size only
/// changes speed. A wide batch keeps whole-corpus scoring fast on large corpora.
const EVALUATION_BATCH: usize = 256;

pub(crate) fn adam() -> ModuleOptimizer {
    // Burn defaults to epsilon=1e-5; specify Liminal's equation explicitly.
    AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init()
}

/// One global L2 norm before Adam. Burn's built-in Norm clipping is per tensor.
/// Reductions and scaling remain on the selected device; read only one scalar.
fn clip_global_norm(
    gradients: &mut GradientsParams,
    model: &Model,
    maximum: f64,
) -> Result<f64, String> {
    let ids: Vec<_> = model
        .parameter_tensors()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let mut sum: Option<Tensor<1>> = None;
    for &id in &ids {
        let gradient = gradients
            .get::<2>(id)
            .ok_or("backward did not produce a gradient for every model parameter")?;
        let squares = gradient.square().sum();
        sum = Some(match sum {
            Some(previous) => previous + squares,
            None => squares,
        });
    }
    let norm = sum
        .expect("model has parameters")
        .sqrt()
        .into_scalar::<f32>() as f64;
    if !norm.is_finite() {
        return Err("training produced a non-finite global gradient norm".into());
    }
    if norm > maximum {
        for id in ids {
            let gradient = gradients
                .remove::<2>(id)
                .expect("gradient was checked above");
            gradients.register(id, gradient.mul_scalar(maximum / norm));
        }
    }
    Ok(norm)
}

/// The shared Burn update for corpus training and focused behavioral probes.
/// Global clipping includes every current model parameter before Adam updates.
pub(crate) fn optimize_batch(
    model: &mut Model,
    batch: &Batch,
    optimizer: &mut ModuleOptimizer,
    learning_rate: f64,
    gradient_clip: f64,
    report_loss: bool,
) -> Result<(Option<f64>, f64), String> {
    optimize_batch_for_path(
        model,
        batch,
        optimizer,
        learning_rate,
        gradient_clip,
        report_loss,
        ModelPath::Full,
    )
}

/// The same update with a diagnostic path gate. Use a fresh optimizer for each
/// isolated condition so moments from previously active paths cannot carry over.
pub(crate) fn optimize_batch_for_path(
    model: &mut Model,
    batch: &Batch,
    optimizer: &mut ModuleOptimizer,
    learning_rate: f64,
    gradient_clip: f64,
    report_loss: bool,
    path: ModelPath,
) -> Result<(Option<f64>, f64), String> {
    let (progress, norm) = optimize_batch_objective(
        model,
        batch,
        optimizer,
        UpdateSettings {
            learning_rate,
            gradient_clip,
            report_loss,
            path,
            voltage_penalty: VoltagePenalty::default(),
        },
    )?;
    Ok((progress.map(|value| value.cross_entropy), norm))
}

#[derive(Clone, Copy)]
struct UpdateSettings {
    learning_rate: f64,
    gradient_clip: f64,
    report_loss: bool,
    path: ModelPath,
    voltage_penalty: VoltagePenalty,
}

#[derive(Debug, Clone, Copy)]
struct TrainingProgress {
    cross_entropy: f64,
    /// Unweighted activity penalty; exactly zero and not evaluated when disabled.
    voltage_penalty: f64,
    weighted_voltage_penalty: f64,
    total: f64,
}

/// The only loss/backward/global-clip/Adam body. CE-only probes pass coefficient
/// zero, which preserves the original loss graph and does not evaluate activity.
fn optimize_batch_objective(
    model: &mut Model,
    batch: &Batch,
    optimizer: &mut ModuleOptimizer,
    settings: UpdateSettings,
) -> Result<(Option<TrainingProgress>, f64), String> {
    let terms = model.training_loss_for_path(batch, settings.path, settings.voltage_penalty)?;
    let progress = if settings.report_loss {
        let cross_entropy = terms.cross_entropy.clone().into_scalar::<f32>() as f64;
        let (voltage_penalty, weighted_voltage_penalty) =
            if settings.voltage_penalty.coefficient == 0.0 {
                (0.0, 0.0)
            } else {
                let weighted = terms
                    .voltage_penalty
                    .clone()
                    .mul_scalar(settings.voltage_penalty.coefficient)
                    .into_scalar::<f32>() as f64;
                (
                    terms.voltage_penalty.clone().into_scalar::<f32>() as f64,
                    weighted,
                )
            };
        let total = terms.total.clone().into_scalar::<f32>() as f64;
        if [
            cross_entropy,
            voltage_penalty,
            weighted_voltage_penalty,
            total,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err("training produced a non-finite objective term".into());
        }
        Some(TrainingProgress {
            cross_entropy,
            voltage_penalty,
            weighted_voltage_penalty,
            total,
        })
    } else {
        None
    };
    let mut gradients = GradientsParams::from_grads(terms.total.backward(), model);
    let norm = clip_global_norm(&mut gradients, model, settings.gradient_clip)?;
    // Burn consumes graph gradients and restores updated trainable leaf tensors.
    *model = optimizer.step(settings.learning_rate, model.clone(), gradients);
    Ok((progress, norm))
}

/// Each row contains only adjacent pairs from a single document.
#[derive(Clone, Copy)]
struct Window<'a> {
    tokens: &'a [usize],
    start: usize,
    length: usize,
}

fn make_batch(windows: &[Window<'_>], length: usize, device: &Device) -> Result<Batch, String> {
    if windows.is_empty() || length == 0 {
        return Err("a batch requires nonempty rows and a positive sequence length".into());
    }
    let elements = windows
        .len()
        .checked_mul(length)
        .ok_or("batch dimensions exceed the addressable range")?;
    let mut inputs = vec![0_i32; elements];
    let mut targets = vec![0_i32; elements];
    let mut mask = vec![0.0_f32; elements];
    let mut valid_targets = 0_usize;
    for (row, window) in windows.iter().enumerate() {
        if window.length == 0
            || window.length > length
            || window.start >= window.tokens.len()
            || window.length >= window.tokens.len() - window.start
        {
            return Err("batch window is outside its document".into());
        }
        valid_targets = valid_targets
            .checked_add(window.length)
            .ok_or("batch target count exceeds the addressable range")?;
        for time in 0..window.length {
            let index = row * length + time;
            inputs[index] = i32::try_from(window.tokens[window.start + time])
                .map_err(|_| "token index exceeds Burn's integer representation")?;
            targets[index] = i32::try_from(window.tokens[window.start + time + 1])
                .map_err(|_| "token index exceeds Burn's integer representation")?;
            mask[index] = 1.0;
        }
    }
    let shape = [windows.len(), length];
    Ok(Batch {
        inputs: Tensor::<2, Int>::from_data(TensorData::new(inputs, shape), device),
        targets: Tensor::<2, Int>::from_data(TensorData::new(targets, shape), device),
        mask: Tensor::from_data(TensorData::new(mask, shape), device),
        valid_targets,
    })
}

/// Build padded rows from complete token slices (inputs plus their final target).
/// Every slice must remain within one document and have 2..=sequence_length+1 tokens.
pub fn build_batch(
    samples: &[&[usize]],
    sequence_length: usize,
    device: &Device,
) -> Result<Batch, String> {
    if samples
        .iter()
        .any(|tokens| tokens.len() < 2 || tokens.len() - 1 > sequence_length)
    {
        return Err("each batch sample needs between two and sequence_length+1 tokens".into());
    }
    let windows: Vec<_> = samples
        .iter()
        .map(|tokens| Window {
            tokens,
            start: 0,
            length: tokens.len() - 1,
        })
        .collect();
    make_batch(&windows, sequence_length, device)
}

/// Uniform eligible starts: short documents have one start, while longer
/// documents have every start admitting a full row. Sample with replacement.
struct WindowSampler<'a> {
    documents: &'a [Vec<usize>],
    cumulative_starts: Vec<usize>,
    start_count: usize,
    length: usize,
}

impl<'a> WindowSampler<'a> {
    fn new(documents: &'a [Vec<usize>], length: usize) -> Result<Self, String> {
        let mut cumulative_starts = Vec::with_capacity(documents.len());
        let mut start_count = 0_usize;
        for tokens in documents {
            let window = length.min(tokens.len() - 1);
            start_count = start_count
                .checked_add(tokens.len() - window)
                .ok_or("training window count exceeds the addressable range")?;
            cumulative_starts.push(start_count);
        }
        Ok(Self {
            documents,
            cumulative_starts,
            start_count,
            length,
        })
    }
    fn draw(&self, random: &mut Rng) -> Window<'a> {
        let draw = random.index(self.start_count);
        let document = self.cumulative_starts.partition_point(|&end| end <= draw);
        let previous_end = document
            .checked_sub(1)
            .map_or(0, |previous| self.cumulative_starts[previous]);
        let tokens = self.documents[document].as_slice();
        Window {
            tokens,
            start: draw - previous_end,
            length: self.length.min(tokens.len() - 1),
        }
    }
}

/// Train independent zero-state rows in parallel through Burn tensor operations.
/// Full BPTT runs through each row. Loss is mean over all valid batch targets;
/// padded suffixes have no loss or gradient. Validation never enters updates.
pub fn train(
    model: &mut Model,
    documents: &[Vec<usize>],
    validation_documents: &[Vec<usize>],
    config: &TrainConfig,
) -> Result<TrainingReport, String> {
    config.validate()?;
    validate_documents(documents, model.config.vocab_size)?;
    validate_documents(validation_documents, model.config.vocab_size)?;
    let train_sample = training_evaluation_sample(documents);
    let initial_train_loss = evaluate(
        model,
        &train_sample,
        config.sequence_length,
        EVALUATION_BATCH,
    )?;
    let initial_validation_loss = evaluate(
        model,
        validation_documents,
        config.sequence_length,
        EVALUATION_BATCH,
    )?;
    *model = model.clone().train();
    let device = model.device();
    let sampler = WindowSampler::new(documents, config.sequence_length)?;
    let mut random = Rng::new(config.seed);
    let mut optimizer = adam();
    let mut trained_targets = 0_usize;
    if config.voltage_penalty.coefficient == 0.0 {
        eprintln!(
            "Voltage penalty disabled: raw/weighted penalty logs are zero, not measured activity."
        );
    }
    for step in 1..=config.steps {
        let windows: Vec<_> = (0..config.batch_size)
            .map(|_| sampler.draw(&mut random))
            .collect();
        let batch = make_batch(&windows, config.sequence_length, &device)?;
        trained_targets = trained_targets
            .checked_add(batch.valid_targets)
            .ok_or("training target count exceeds the addressable range")?;
        let learning_rate =
            config
                .learning_rate_schedule
                .rate(config.learning_rate, step, config.steps);
        let (progress, norm) = optimize_batch_objective(
            model,
            &batch,
            &mut optimizer,
            UpdateSettings {
                learning_rate,
                gradient_clip: config.gradient_clip,
                report_loss: step == 1 || step % 50 == 0 || step == config.steps,
                path: ModelPath::Full,
                voltage_penalty: config.voltage_penalty,
            },
        )?;
        if let Some(progress) = progress {
            eprintln!(
                "step {step}/{}: batch CE {:.6}, raw voltage penalty {:.6}, weighted voltage penalty {:.6}, total {:.6}, gradient norm {norm:.6}, targets {trained_targets}, learning rate {learning_rate:.6}",
                config.steps,
                progress.cross_entropy,
                progress.voltage_penalty,
                progress.weighted_voltage_penalty,
                progress.total,
            );
        }
    }
    Ok(TrainingReport {
        initial_train_loss,
        final_train_loss: evaluate(
            model,
            &train_sample,
            config.sequence_length,
            EVALUATION_BATCH,
        )?,
        initial_validation_loss,
        final_validation_loss: evaluate(
            model,
            validation_documents,
            config.sequence_length,
            EVALUATION_BATCH,
        )?,
        steps: config.steps,
        trained_targets,
    })
}

fn validate_documents(documents: &[Vec<usize>], vocab_size: usize) -> Result<(), String> {
    if documents.is_empty() {
        return Err("a text split needs at least one document".into());
    }
    for tokens in documents {
        if tokens.len() < 2 {
            return Err("every document needs at least two characters".into());
        }
        if tokens.iter().any(|&token| token >= vocab_size) {
            return Err("token is outside the model vocabulary".into());
        }
    }
    Ok(())
}

/// Mean next-character NLL in nats. Reset each document and maximum-length
/// window. Count each adjacent pair once with equal weight, including short
/// final windows and a partial final batch. Autodiff is disabled during scoring.
pub fn evaluate(
    model: &Model,
    documents: &[Vec<usize>],
    sequence_length: usize,
    batch_size: usize,
) -> Result<f64, String> {
    validate_documents(documents, model.config.vocab_size)?;
    if sequence_length == 0 || batch_size == 0 {
        return Err("evaluation sequence length and batch size must be positive".into());
    }
    let model = model.clone().valid();
    let device = model.device();
    let mut total_loss = 0.0;
    let mut target_count = 0_usize;
    let mut windows = Vec::with_capacity(batch_size);
    let mut score = |windows: &[Window<'_>]| -> Result<(), String> {
        let batch = make_batch(windows, sequence_length, &device)?;
        let loss = model.loss(&batch)?.into_scalar::<f32>() as f64;
        if !loss.is_finite() {
            return Err("evaluation loss is not finite".into());
        }
        total_loss += loss * batch.valid_targets as f64;
        target_count = target_count
            .checked_add(batch.valid_targets)
            .ok_or("evaluation target count exceeds the addressable range")?;
        Ok(())
    };
    for tokens in documents {
        let mut start = 0;
        while start < tokens.len() - 1 {
            let length = sequence_length.min(tokens.len() - 1 - start);
            windows.push(Window {
                tokens,
                start,
                length,
            });
            if windows.len() == batch_size {
                score(&windows)?;
                windows.clear();
            }
            start += length;
        }
    }
    if !windows.is_empty() {
        score(&windows)?;
    }
    let loss = total_loss / target_count as f64;
    if !loss.is_finite() {
        return Err("evaluation loss is not finite".into());
    }
    Ok(loss)
}

/// Context-free control fitted only to training targets with additive-one
/// smoothing. Like model evaluation, skip the first token of every document.
pub fn unigram_loss(
    training_documents: &[Vec<usize>],
    evaluation_documents: &[Vec<usize>],
    vocab_size: usize,
) -> Result<f64, String> {
    if vocab_size < 2 {
        return Err("unigram evaluation requires at least two vocabulary entries".into());
    }
    validate_documents(training_documents, vocab_size)?;
    validate_documents(evaluation_documents, vocab_size)?;
    let mut counts = vec![1.0_f64; vocab_size];
    for tokens in training_documents {
        for &token in &tokens[1..] {
            counts[token] += 1.0;
        }
    }
    let normalizer: f64 = counts.iter().sum();
    let mut total_loss = 0.0;
    let mut target_count = 0_usize;
    for tokens in evaluation_documents {
        target_count = target_count
            .checked_add(tokens.len() - 1)
            .ok_or("unigram target count exceeds the addressable range")?;
        total_loss += tokens[1..]
            .iter()
            .map(|&token| (normalizer / counts[token]).ln())
            .sum::<f64>();
    }
    Ok(total_loss / target_count as f64)
}

/// Generate after processing each prompt token exactly once. Zero temperature
/// chooses argmax; positive temperatures use stable softmax sampling in Rust.
/// Recent context tokens subject to the repetition penalty.
const REPETITION_WINDOW: usize = 64;

/// How next-token logits become a choice during generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decoding {
    /// Softmax temperature; 0 chooses the argmax.
    pub temperature: f64,
    /// Nucleus sampling: sample only the smallest set of most probable tokens
    /// whose probability reaches `top_p`. 1 keeps the full distribution.
    pub top_p: f64,
    /// CTRL-style penalty (Keskar et al., 2019) for tokens among the last
    /// REPETITION_WINDOW context tokens: positive logits are divided by it and
    /// negative logits multiplied, before temperature. 1 disables it.
    pub repetition_penalty: f64,
}

impl Decoding {
    pub fn sampled(temperature: f64) -> Self {
        Self {
            temperature,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be finite and nonnegative".into());
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err("top-p must lie in (0, 1]".into());
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty < 1.0 {
            return Err("repetition penalty must be finite and at least 1".into());
        }
        Ok(())
    }
}

pub fn generate(
    model: &Model,
    prompt: &[usize],
    count: usize,
    decoding: Decoding,
    seed: u64,
) -> Result<Vec<usize>, String> {
    if prompt.is_empty() {
        return Err("generation requires a nonempty prompt".into());
    }
    if prompt.iter().any(|&token| token >= model.config.vocab_size) {
        return Err("generation prompt token is outside the model vocabulary".into());
    }
    decoding.validate()?;
    let model = model.clone().valid();
    let device = model.device();
    let mut state = State::new(1, model.config.hidden_size, &device);
    let mut logits = Vec::new();
    let mut advance = |token: usize| -> Result<Vec<f64>, String> {
        let token = i32::try_from(token)
            .map_err(|_| "token index exceeds Burn's integer representation")?;
        let output = model.step(Tensor::<1, Int>::from_data([token], &device), &mut state)?;
        Ok(output.into_data().iter::<f32>().map(f64::from).collect())
    };
    for &token in prompt {
        logits = advance(token)?;
    }
    let mut random = Rng::new(seed);
    let mut context = prompt.to_vec();
    let mut generated = Vec::with_capacity(count);
    for index in 0..count {
        let recent = &context[context.len().saturating_sub(REPETITION_WINDOW)..];
        let token = sample(&logits, decoding, recent, &mut random)?;
        generated.push(token);
        context.push(token);
        if index + 1 < count {
            logits = advance(token)?;
        }
    }
    Ok(generated)
}

/// One draw per call. With top_p = 1 and no penalty this is plain
/// temperature sampling over token order, reproducing earlier outputs.
fn sample(
    logits: &[f64],
    decoding: Decoding,
    recent: &[usize],
    random: &mut Rng,
) -> Result<usize, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("sampling requires finite, nonempty logits".into());
    }
    let mut logits = logits.to_vec();
    if decoding.repetition_penalty > 1.0 {
        let mut seen = vec![false; logits.len()];
        for &token in recent {
            if token < logits.len() && !std::mem::replace(&mut seen[token], true) {
                let value = logits[token];
                logits[token] = if value > 0.0 {
                    value / decoding.repetition_penalty
                } else {
                    value * decoding.repetition_penalty
                };
            }
        }
    }
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    if decoding.temperature == 0.0 {
        return Ok(best);
    }
    let maximum = logits[best];
    let weights: Vec<f64> = logits
        .iter()
        .map(|value| ((value - maximum) / decoding.temperature).exp())
        .collect();
    // Candidate order: token order for the full distribution, otherwise the
    // most probable tokens until their probability mass reaches top_p.
    let candidates: Vec<usize> = if decoding.top_p < 1.0 {
        let total: f64 = weights.iter().sum();
        let mut order: Vec<usize> = (0..weights.len()).collect();
        order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
        let mut mass = 0.0;
        let mut kept = 0;
        for &index in &order {
            kept += 1;
            mass += weights[index] / total;
            if mass >= decoding.top_p {
                break;
            }
        }
        order.truncate(kept);
        order
    } else {
        (0..weights.len()).collect()
    };
    let draw = random.next_f64() * candidates.iter().map(|&index| weights[index]).sum::<f64>();
    let mut cumulative = 0.0;
    for &index in &candidates {
        cumulative += weights[index];
        if draw < cumulative {
            return Ok(index);
        }
    }
    Ok(best)
}

/// Training-split scoring is exact up to this many tokens. Larger splits are
/// scored on every k-th document, with k fixed by the split size, which bounds
/// evaluation cost. Validation is always scored in full.
pub const TRAIN_EVALUATION_TOKENS: usize = 4_000_000;

pub fn training_evaluation_sample(documents: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let total: usize = documents.iter().map(Vec::len).sum();
    let stride = total.div_ceil(TRAIN_EVALUATION_TOKENS).max(1);
    documents.iter().step_by(stride).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    fn model() -> Model {
        Model::new(
            ModelConfig {
                vocab_size: 3,
                hidden_size: 6,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            42,
            &Device::flex(),
        )
        .unwrap()
    }

    fn training_config() -> TrainConfig {
        TrainConfig {
            steps: 1,
            sequence_length: 3,
            batch_size: 4,
            learning_rate: 0.01,
            learning_rate_schedule: LearningRateSchedule::Constant,
            gradient_clip: 1.0,
            seed: 7,
            voltage_penalty: VoltagePenalty::default(),
        }
    }

    #[test]
    fn repetition_penalty_and_top_p_shape_the_choice() {
        let greedy = Decoding::sampled(0.0);
        let logits = [2.0, 1.0, -1.0];
        assert_eq!(sample(&logits, greedy, &[0], &mut Rng::new(0)).unwrap(), 0);
        // 2/4 = .5 now trails token 1; a negative logit is pushed further down.
        let penalized = Decoding {
            repetition_penalty: 4.0,
            ..greedy
        };
        assert_eq!(
            sample(&logits, penalized, &[0, 0], &mut Rng::new(0)).unwrap(),
            1
        );
        assert_eq!(
            sample(&[-1.0, -2.0], penalized, &[0], &mut Rng::new(0)).unwrap(),
            1
        );
        // Probabilities .5/.3/.2: top-p .6 keeps tokens 0 and 1; .45 keeps token 0.
        let logits = [0.5_f64.ln(), 0.3_f64.ln(), 0.2_f64.ln()];
        let nucleus = |top_p| Decoding {
            temperature: 1.0,
            top_p,
            repetition_penalty: 1.0,
        };
        let mut random = Rng::new(3);
        let draws: Vec<usize> = (0..400)
            .map(|_| sample(&logits, nucleus(0.6), &[], &mut random).unwrap())
            .collect();
        assert!(!draws.contains(&2) && draws.contains(&0) && draws.contains(&1));
        assert!((0..50).all(|_| sample(&logits, nucleus(0.45), &[], &mut random).unwrap() == 0));
        let draws: Vec<usize> = (0..400)
            .map(|_| sample(&logits, nucleus(1.0), &[], &mut random).unwrap())
            .collect();
        assert!(draws.contains(&2));
        for invalid in [
            nucleus(0.0),
            nucleus(1.5),
            Decoding {
                repetition_penalty: 0.5,
                ..greedy
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn cosine_schedule_decays_from_base_to_floor_and_constant_stays_fixed() {
        let cosine = LearningRateSchedule::Cosine;
        assert_eq!(cosine.rate(0.003, 1, 101), 0.003);
        assert!((cosine.rate(0.003, 51, 101) - 0.00165).abs() < 1e-12);
        assert!((cosine.rate(0.003, 101, 101) - 0.0003).abs() < 1e-12);
        assert_eq!(cosine.rate(0.003, 1, 1), 0.003);
        for step in [1, 50, 101] {
            assert_eq!(LearningRateSchedule::Constant.rate(0.003, step, 101), 0.003);
        }
        for schedule in [LearningRateSchedule::Constant, cosine] {
            assert_eq!(
                LearningRateSchedule::parse(schedule.as_str()).unwrap(),
                schedule
            );
        }
        assert!(LearningRateSchedule::parse("linear").is_err());
    }

    fn filled_gradients(model: &Model, value: f32) -> GradientsParams {
        let mut gradients = GradientsParams::new();
        for (id, parameter) in model.parameter_tensors() {
            gradients.register(
                id,
                Tensor::<2>::full(parameter.dims(), value, &Device::flex()),
            );
        }
        gradients
    }

    #[test]
    fn adam_matches_two_bias_corrected_updates_and_keeps_trainable_leaves() {
        let mut model = model().train();
        let initial = model.parameter_values().unwrap();
        let mut optimizer = adam();
        let mut first = 0.0_f64;
        let mut second = 0.0_f64;
        let mut expected_delta = 0.0_f64;
        for (step, gradient) in [(1, 2.0_f64), (2, -0.5)] {
            let gradients = filled_gradients(&model, gradient as f32);
            model = optimizer.step(0.01, model, gradients);
            first = 0.9 * first + 0.1 * gradient;
            second = 0.999 * second + 0.001 * gradient * gradient;
            expected_delta += 0.01 * (first / (1.0 - 0.9_f64.powi(step)))
                / ((second / (1.0 - 0.999_f64.powi(step))).sqrt() + 1e-8);
        }
        for (actual, original) in model.parameter_values().unwrap().iter().zip(initial) {
            assert!((f64::from(*actual - original) + expected_delta).abs() < 2e-6);
        }
        assert!(
            model
                .parameter_tensors()
                .iter()
                .all(|(_, parameter)| parameter.is_require_grad())
        );
        let batch = build_batch(&[&[1, 2, 1]], 2, &model.device()).unwrap();
        let gradients = GradientsParams::from_grads(model.loss(&batch).unwrap().backward(), &model);
        assert_eq!(gradients.len(), 15);
    }

    #[test]
    fn global_clipping_spans_tensors_and_preserves_direction() {
        let model = model();
        let mut gradients = filled_gradients(&model, 0.0);
        let device = Device::flex();
        let mut embedding = vec![0.0_f32; 18];
        embedding[0] = 3.0;
        let mut semantic_readout = vec![0.0_f32; 18];
        semantic_readout[0] = 4.0;
        gradients.register(
            model.embedding.id,
            Tensor::<2>::from_data(TensorData::new(embedding, [3, 6]), &device),
        );
        gradients.register(
            model.semantic_readout.id,
            Tensor::<2>::from_data(TensorData::new(semantic_readout, [3, 6]), &device),
        );
        assert_eq!(clip_global_norm(&mut gradients, &model, 1.0).unwrap(), 5.0);
        let first = gradients.get::<2>(model.embedding.id).unwrap().into_data();
        let second = gradients
            .get::<2>(model.semantic_readout.id)
            .unwrap()
            .into_data();
        assert!((first.as_slice::<f32>().unwrap()[0] - 0.6).abs() < 1e-6);
        assert!((second.as_slice::<f32>().unwrap()[0] - 0.8).abs() < 1e-6);
        assert!((clip_global_norm(&mut gradients, &model, 2.0).unwrap() - 1.0).abs() < 1e-6);
        let mut invalid = filled_gradients(&model, f32::NAN);
        assert!(clip_global_norm(&mut invalid, &model, 1.0).is_err());
    }

    #[test]
    fn isolated_path_updates_only_its_active_parameter_groups() {
        for (path, active_groups) in [
            (ModelPath::MemoryOnly, vec![0, 4, 5, 6, 9, 10, 13]),
            (ModelPath::SpikesOnly, vec![0, 1, 2, 3, 4, 11, 12, 14]),
        ] {
            let mut model = model();
            // Ensure recurrent spikes occur within this short update fixture.
            model.hidden_bias = burn::module::Param::from_tensor(Tensor::full(
                [1, model.config.hidden_size],
                0.8,
                &model.device(),
            ));
            let mut model = model.train();
            let before: Vec<_> = model
                .parameter_tensors()
                .into_iter()
                .map(|(_, tensor)| tensor.try_into_vec_as::<f32>().unwrap())
                .collect();
            let batch =
                build_batch(&[&[1, 2, 1, 2, 1], &[2, 1, 2, 1, 2]], 4, &model.device()).unwrap();
            let mut optimizer = adam();
            let (loss, norm) =
                optimize_batch_for_path(&mut model, &batch, &mut optimizer, 0.01, 1.0, true, path)
                    .unwrap();
            assert!(loss.unwrap().is_finite());
            assert!(norm.is_finite() && norm > 0.0);
            for (group, ((_, parameter), original)) in model
                .parameter_tensors()
                .into_iter()
                .zip(before)
                .enumerate()
            {
                let changed = parameter.try_into_vec_as::<f32>().unwrap() != original;
                assert_eq!(
                    changed,
                    active_groups.contains(&group),
                    "unexpected parameter update for {path:?}, group {group}"
                );
            }
        }
    }

    #[test]
    fn zero_coefficient_preserves_ce_updates_and_training_draws_exactly() {
        let mut reference = model().train();
        let mut wrapper = model().train();
        let mut objective = model().train();
        let mut reference_optimizer = adam();
        let mut wrapper_optimizer = adam();
        let mut objective_optimizer = adam();
        for tokens in [&[1, 2, 1, 1][..], &[2, 1, 2, 2][..]] {
            let batch = build_batch(&[tokens], 3, &reference.device()).unwrap();
            let loss = reference.loss(&batch).unwrap();
            let expected_loss = loss.clone().into_scalar::<f32>() as f64;
            let mut gradients = GradientsParams::from_grads(loss.backward(), &reference);
            let expected_norm = clip_global_norm(&mut gradients, &reference, 0.2).unwrap();
            reference = reference_optimizer.step(0.01, reference, gradients);
            let (wrapper_loss, wrapper_norm) = optimize_batch(
                &mut wrapper,
                &batch,
                &mut wrapper_optimizer,
                0.01,
                0.2,
                true,
            )
            .unwrap();
            let (progress, objective_norm) = optimize_batch_objective(
                &mut objective,
                &batch,
                &mut objective_optimizer,
                UpdateSettings {
                    learning_rate: 0.01,
                    gradient_clip: 0.2,
                    report_loss: true,
                    path: ModelPath::Full,
                    voltage_penalty: VoltagePenalty::default(),
                },
            )
            .unwrap();
            let progress = progress.unwrap();
            assert_eq!(wrapper_loss, Some(expected_loss));
            assert_eq!(progress.cross_entropy, expected_loss);
            assert_eq!(progress.total, expected_loss);
            assert_eq!(progress.voltage_penalty, 0.0);
            assert_eq!(progress.weighted_voltage_penalty, 0.0);
            assert_eq!(wrapper_norm, expected_norm);
            assert_eq!(objective_norm, expected_norm);
            assert_eq!(
                wrapper.parameter_values().unwrap(),
                reference.parameter_values().unwrap()
            );
            assert_eq!(
                objective.parameter_values().unwrap(),
                reference.parameter_values().unwrap()
            );
        }

        // The corpus loop must retain the same draw sequence as the CE-only
        // shared update; evaluations and objective reporting consume no RNG.
        let documents = [vec![1, 2], vec![2, 1, 1, 2, 1, 2, 2, 1]];
        let config = TrainConfig {
            steps: 3,
            ..training_config()
        };
        let mut corpus = model();
        let mut expected = model().train();
        let sampler = WindowSampler::new(&documents, config.sequence_length).unwrap();
        let mut random = Rng::new(config.seed);
        let mut optimizer = adam();
        for _ in 0..config.steps {
            let windows: Vec<_> = (0..config.batch_size)
                .map(|_| sampler.draw(&mut random))
                .collect();
            let batch = make_batch(&windows, config.sequence_length, &expected.device()).unwrap();
            optimize_batch(
                &mut expected,
                &batch,
                &mut optimizer,
                config.learning_rate,
                config.gradient_clip,
                false,
            )
            .unwrap();
        }
        let report = train(&mut corpus, &documents, &documents, &config).unwrap();
        assert_eq!(
            corpus.parameter_values().unwrap(),
            expected.parameter_values().unwrap()
        );
        assert_eq!(
            report.final_train_loss,
            evaluate(
                &corpus,
                &documents,
                config.sequence_length,
                EVALUATION_BATCH
            )
            .unwrap()
        );
    }

    #[test]
    fn combined_objective_updates_voltage_path_and_reports_plain_ce_separately() {
        let mut initial = model();
        initial.hidden_bias = burn::module::Param::from_tensor(Tensor::full(
            [1, initial.config.hidden_size],
            10.0,
            &initial.device(),
        ));
        let mut ce_only = initial.clone().train();
        let mut regularized = initial.train();
        let batch = build_batch(&[&[1, 2]], 1, &regularized.device()).unwrap();
        let expected_ce = regularized.loss(&batch).unwrap().into_scalar::<f32>() as f64;
        let mut ce_optimizer = adam();
        let mut penalty_optimizer = adam();
        let (ce_loss, _) =
            optimize_batch(&mut ce_only, &batch, &mut ce_optimizer, 0.01, 1.0, true).unwrap();
        let (progress, norm) = optimize_batch_objective(
            &mut regularized,
            &batch,
            &mut penalty_optimizer,
            UpdateSettings {
                learning_rate: 0.01,
                gradient_clip: 1.0,
                report_loss: true,
                path: ModelPath::Full,
                voltage_penalty: VoltagePenalty {
                    coefficient: 0.001,
                    band_widths: 3.0,
                },
            },
        )
        .unwrap();
        let progress = progress.unwrap();
        assert_eq!(ce_loss, Some(expected_ce));
        assert_eq!(progress.cross_entropy, expected_ce);
        assert!(progress.voltage_penalty > 0.0 && progress.weighted_voltage_penalty > 0.0);
        assert!(
            (progress.total - progress.cross_entropy - progress.weighted_voltage_penalty).abs()
                < 2e-7
        );
        assert!(norm.is_finite() && norm > 0.0);
        assert_eq!(
            ce_only.hidden_bias.val().try_into_vec_as::<f32>().unwrap(),
            vec![10.0; 6]
        );
        assert!(
            regularized
                .hidden_bias
                .val()
                .try_into_vec_as::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value < 10.0)
        );
        assert_ne!(
            ce_only.parameter_values().unwrap(),
            regularized.parameter_values().unwrap()
        );
    }

    #[test]
    fn batch_rows_are_document_contained_and_padding_is_masked() {
        let documents = [vec![1, 2], vec![0, 2, 2, 1, 0, 1]];
        let sampler = WindowSampler::new(&documents, 3).unwrap();
        assert_eq!(sampler.cumulative_starts, vec![1, 4]);
        let mut seen = [false; 4];
        for seed in 0..64 {
            let draw = Rng::new(seed).index(4);
            let window = sampler.draw(&mut Rng::new(seed));
            seen[draw] = true;
            if draw == 0 {
                assert_eq!(window.tokens, documents[0]);
                assert_eq!((window.start, window.length), (0, 1));
            } else {
                assert_eq!(window.tokens, documents[1]);
                assert_eq!((window.start, window.length), (draw - 1, 3));
            }
        }
        assert!(seen.into_iter().all(|value| value));
        let batch = build_batch(&[&documents[0], &documents[1][1..5]], 3, &Device::flex()).unwrap();
        assert_eq!(batch.valid_targets, 4);
        assert_eq!(
            batch.inputs.into_data().as_slice::<i32>().unwrap(),
            &[1, 0, 0, 2, 2, 1]
        );
        assert_eq!(
            batch.targets.into_data().as_slice::<i32>().unwrap(),
            &[2, 0, 0, 2, 1, 0]
        );
        assert_eq!(
            batch.mask.into_data().as_slice::<f32>().unwrap(),
            &[1., 0., 0., 1., 1., 1.]
        );
        assert!(build_batch(&[], 3, &Device::flex()).is_err());
        assert!(build_batch(&[&[1]], 3, &Device::flex()).is_err());
        assert!(build_batch(&[&[1, 2, 1]], 1, &Device::flex()).is_err());
    }

    #[test]
    fn evaluation_weights_targets_and_resets_each_document_and_window() {
        let model = model().valid();
        let documents = [vec![1, 2], vec![1, 1, 1, 1, 0, 2]];
        let mut expected = 0.0;
        for (tokens, start, length) in [
            (documents[0].as_slice(), 0, 1),
            (documents[1].as_slice(), 0, 2),
            (documents[1].as_slice(), 2, 2),
            (documents[1].as_slice(), 4, 1),
        ] {
            let batch = make_batch(
                &[Window {
                    tokens,
                    start,
                    length,
                }],
                2,
                &model.device(),
            )
            .unwrap();
            expected += model.loss(&batch).unwrap().into_scalar::<f32>() as f64 * length as f64;
        }
        for batch_size in [1, 2, 3, 8] {
            assert!(
                (evaluate(&model, &documents, 2, batch_size).unwrap() - expected / 6.0).abs()
                    < 5e-7
            );
        }
        assert!(!model.embedding.val().is_autodiff());
    }

    #[test]
    fn unigram_uses_only_training_targets_and_skips_document_starts() {
        let actual = unigram_loss(&[vec![0, 1, 1, 2]], &[vec![1, 0, 2]], 3).unwrap();
        let expected = (6.0_f64.ln() + 3.0_f64.ln()) / 2.0;
        assert!((actual - expected).abs() < 1e-12);
        let training = [vec![0, 1, 1], vec![2, 1]];
        let evaluation = [vec![0, 2], vec![2, 0, 1]];
        let expected = (2.0 * 6.0_f64.ln() + 1.5_f64.ln()) / 3.0;
        assert!((unigram_loss(&training, &evaluation, 3).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn corpus_operations_reject_empty_short_invalid_documents_and_batches() {
        let valid = [vec![1, 2]];
        let config = training_config();
        for invalid in [vec![], vec![vec![]], vec![vec![1]], vec![vec![1, 3]]] {
            assert!(evaluate(&model(), &invalid, 2, 4).is_err());
            assert!(unigram_loss(&invalid, &valid, 3).is_err());
            assert!(unigram_loss(&valid, &invalid, 3).is_err());
            assert!(train(&mut model(), &invalid, &valid, &config).is_err());
            assert!(train(&mut model(), &valid, &invalid, &config).is_err());
        }
        let mut invalid = config.clone();
        invalid.batch_size = 0;
        assert!(train(&mut model(), &valid, &valid, &invalid).is_err());
        for voltage_penalty in [
            VoltagePenalty {
                coefficient: -1.0,
                band_widths: 3.0,
            },
            VoltagePenalty {
                coefficient: f64::NAN,
                band_widths: 3.0,
            },
            VoltagePenalty {
                coefficient: 0.001,
                band_widths: 0.0,
            },
            VoltagePenalty {
                coefficient: 0.001,
                band_widths: f64::INFINITY,
            },
        ] {
            let invalid = TrainConfig {
                voltage_penalty,
                ..config.clone()
            };
            assert!(train(&mut model(), &valid, &valid, &invalid).is_err());
        }
        assert!(evaluate(&model(), &valid, 2, 0).is_err());
    }

    #[test]
    fn batched_alternating_language_is_learnable_and_deterministic() {
        let documents = [(0..64).map(|index| 1 + index % 2).collect()];
        let mut first = model();
        let mut second = model();
        let config = TrainConfig {
            steps: 160,
            sequence_length: 8,
            ..training_config()
        };
        let report = train(&mut first, &documents, &documents, &config).unwrap();
        train(&mut second, &documents, &documents, &config).unwrap();
        assert!(report.final_train_loss < report.initial_train_loss * 0.7);
        assert_eq!(report.trained_targets, 160 * 4 * 8);
        assert_eq!(
            first.parameter_values().unwrap(),
            second.parameter_values().unwrap()
        );
        assert_eq!(
            generate(&first, &[1], 12, Decoding::sampled(0.0), 0).unwrap(),
            (0..12).map(|i| 2 - i % 2).collect::<Vec<_>>()
        );
    }

    #[test]
    fn generation_primes_prompt_once_and_sampling_is_reproducible() {
        let model = model().valid();
        let prompt = [1, 2, 1];
        let mut state = State::new(1, 6, &model.device());
        let mut logits = Vec::new();
        for token in prompt {
            logits = model
                .step(
                    Tensor::<1, Int>::from_data([token as i32], &model.device()),
                    &mut state,
                )
                .unwrap()
                .into_data()
                .iter::<f32>()
                .map(f64::from)
                .collect();
        }
        let expected = sample(&logits, Decoding::sampled(0.0), &[], &mut Rng::new(0)).unwrap();
        assert_eq!(
            generate(&model, &prompt, 1, Decoding::sampled(0.0), 0).unwrap(),
            vec![expected]
        );
        assert_eq!(
            generate(&model, &prompt, 20, Decoding::sampled(0.8), 8).unwrap(),
            generate(&model, &prompt, 20, Decoding::sampled(0.8), 8).unwrap()
        );
        assert!(generate(&model, &[], 1, Decoding::sampled(1.0), 0).is_err());
        assert!(generate(&model, &[3], 1, Decoding::sampled(1.0), 0).is_err());
        assert!(generate(&model, &prompt, 1, Decoding::sampled(f64::NAN), 0).is_err());
    }
}
