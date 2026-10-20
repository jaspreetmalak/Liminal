//! A causal LIF language model coupled to a slower continuous context state.
//! Hard spikes use an attached triangular surrogate, including through reset.
//! Context at t-1 modulates spikes at t; current spike traces update context at t.

use crate::rng::Rng;
use burn::module::{Module, Param, ParamId};
use burn::tensor::{
    Device, Int, Tensor, TensorData,
    activation::{log_softmax, relu, sigmoid},
};

/// Learned per-neuron decays are MAX_DECAY * sigmoid(logit), so every membrane,
/// trace, and memory decay stays in (0, MAX_DECAY) and state bounds stay finite.
pub const MAX_DECAY: f64 = 0.995;
/// Fresh models draw each decay's time constant log-uniformly from this range,
/// with decay = exp(-1/tau), so the population starts with fast and slow neurons.
const INITIAL_TIME_CONSTANTS: (f64, f64) = (2.0, 32.0);

/// The logit whose learned decay equals `decay`; requires 0 < decay < MAX_DECAY.
pub fn decay_logit(decay: f64) -> f64 {
    let fraction = decay / MAX_DECAY;
    (fraction / (1.0 - fraction)).ln()
}

/// Only the two directed interfaces are gated. All parameters and computations
/// remain present, including zero-gradient paths, for comparable ablations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackMode {
    Both,
    TopDown,
    BottomUp,
    Neither,
}

impl FeedbackMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "both" => Ok(Self::Both),
            "top-down" => Ok(Self::TopDown),
            "bottom-up" => Ok(Self::BottomUp),
            "neither" => Ok(Self::Neither),
            _ => Err("feedback must be both, top-down, bottom-up, or neither".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::TopDown => "top-down",
            Self::BottomUp => "bottom-up",
            Self::Neither => "neither",
        }
    }

    fn gates(self) -> (f64, f64) {
        (
            f64::from(matches!(self, Self::Both | Self::TopDown)),
            f64::from(matches!(self, Self::Both | Self::BottomUp)),
        )
    }
}

/// Component diagnostics use the same equations and parameters as production.
/// A component-only path disables both feedback interfaces and the other output.
/// The path is an execution choice, never stored in a model or checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPath {
    Full,
    MemoryOnly,
    SpikesOnly,
}

impl ModelPath {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "full" => Ok(Self::Full),
            "memory-only" => Ok(Self::MemoryOnly),
            "spikes-only" => Ok(Self::SpikesOnly),
            _ => Err("model path must be full, memory-only, or spikes-only".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::MemoryOnly => "memory-only",
            Self::SpikesOnly => "spikes-only",
        }
    }
}

/// Attached values from the same forward execution, for diagnostic measurements.
#[derive(Debug, Clone)]
pub struct StepTrace {
    /// Actual pre-reset membrane voltage [batch, hidden].
    pub voltage: Tensor<2>,
}

/// Training-only pressure against large pre-reset voltage excursions.
/// Voltage is measured in surrogate-width units around the spike threshold.
#[derive(Debug, Clone, Copy)]
pub struct VoltagePenalty {
    pub coefficient: f64,
    pub band_widths: f64,
}

impl Default for VoltagePenalty {
    fn default() -> Self {
        Self {
            coefficient: 0.0,
            band_widths: 3.0,
        }
    }
}

impl VoltagePenalty {
    pub fn validate(&self) -> Result<(), String> {
        if !self.coefficient.is_finite()
            || self.coefficient < 0.0
            || !(self.coefficient as f32).is_finite()
            || (self.coefficient > 0.0 && self.coefficient as f32 == 0.0)
        {
            return Err(
                "voltage penalty coefficient must be finite, nonnegative, and preserve its sign in f32"
                    .into(),
            );
        }
        if !self.band_widths.is_finite()
            || !(self.band_widths as f32).is_finite()
            || self.band_widths as f32 <= 0.0
        {
            return Err(
                "voltage penalty band must be finite, positive, and representable in f32".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LossTerms {
    pub cross_entropy: Tensor<1>,
    /// Raw normalized P, before multiplication by its coefficient. When the
    /// coefficient is zero this is a constant zero: activity was not measured.
    pub voltage_penalty: Tensor<1>,
    pub total: Tensor<1>,
}

fn squared_voltage_excess(
    voltage: Tensor<2>,
    threshold: f64,
    width: f64,
    band_widths: f64,
) -> Tensor<2> {
    relu(((voltage - threshold) / width).abs() - band_widths).square()
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub threshold: f64,
    pub surrogate_width: f64,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size < 2 {
            return Err("vocabulary must contain at least two tokens".into());
        }
        if self.hidden_size == 0 {
            return Err("hidden size must be positive".into());
        }
        for (name, value) in [
            ("spike threshold", self.threshold),
            ("surrogate width", self.surrogate_width),
        ] {
            if !value.is_finite() || !(value as f32).is_finite() || value as f32 <= 0.0 {
                return Err(format!(
                    "{name} must be finite, positive, and representable in f32"
                ));
            }
        }
        if !(1.0_f32 / self.surrogate_width as f32).is_finite() {
            return Err("inverse surrogate width must be finite in f32".into());
        }
        self.checked_parameter_count()?;
        Ok(())
    }

    pub(crate) fn checked_parameter_count(&self) -> Result<usize, String> {
        let embeddings = self.vocab_size.checked_mul(self.hidden_size);
        let recurrence = self.hidden_size.checked_mul(self.hidden_size);
        embeddings
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(recurrence?.checked_mul(5)?))
            .and_then(|n| n.checked_add(self.hidden_size.checked_mul(5)?))
            .and_then(|n| n.checked_add(self.vocab_size))
            .ok_or_else(|| "model dimensions overflow the parameter count".into())
    }
}

#[derive(Debug, Clone)]
pub struct State {
    /// Post-reset membrane potential [batch, hidden].
    pub membrane: Tensor<2>,
    /// Previous binary spikes [batch, hidden].
    pub spikes: Tensor<2>,
    /// Filtered spike activity [batch, hidden].
    pub trace: Tensor<2>,
    /// Continuous context state [batch, hidden], not a claim of semantic ability.
    pub semantic: Tensor<2>,
}

impl State {
    pub fn new(batch_size: usize, hidden_size: usize, device: &Device) -> Self {
        Self {
            membrane: Tensor::zeros([batch_size, hidden_size], device),
            spikes: Tensor::zeros([batch_size, hidden_size], device),
            trace: Tensor::zeros([batch_size, hidden_size], device),
            semantic: Tensor::zeros([batch_size, hidden_size], device),
        }
    }
}

/// Independent windows shaped [batch, time], with a binary target-loss mask.
/// Natural-text batches mask right padding; focused probes may mask prompt losses.
/// Producers validate token bounds and count actual supervised targets.
/// Padding IDs must still be in vocabulary.
#[derive(Debug, Clone)]
pub struct Batch {
    pub inputs: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
    pub mask: Tensor<2>,
    pub valid_targets: usize,
}

#[derive(Module, Debug)]
pub struct Model {
    #[module(skip)]
    pub config: ModelConfig,
    #[module(skip)]
    pub feedback: FeedbackMode,
    /// Token-major V x H input currents.
    pub embedding: Param<Tensor<2>>,
    /// H x H recurrent weights: destination neuron first.
    pub recurrent: Param<Tensor<2>>,
    pub hidden_bias: Param<Tensor<2>>,
    /// V x H spike readout weights.
    pub readout: Param<Tensor<2>>,
    pub output_bias: Param<Tensor<2>>,
    /// C: embedding current to context candidate, H x H.
    pub semantic_input: Param<Tensor<2>>,
    /// A: recurrent context candidate weights, H x H.
    pub semantic_recurrent: Param<Tensor<2>>,
    /// B: spike trace to context candidate, H x H.
    pub spike_to_semantic: Param<Tensor<2>>,
    /// T: previous context to spike voltage, H x H.
    pub semantic_to_spike: Param<Tensor<2>>,
    pub semantic_bias: Param<Tensor<2>>,
    /// D: context to next-token logits, V x H.
    pub semantic_readout: Param<Tensor<2>>,
    /// Per-neuron decay logits, each 1 x H: membrane beta, spike trace rho,
    /// and memory gamma, each MAX_DECAY * sigmoid(logit).
    pub membrane_decay: Param<Tensor<2>>,
    pub trace_decay: Param<Tensor<2>>,
    pub semantic_decay: Param<Tensor<2>>,
    /// P: spike trace to next-token logits, V x H.
    pub trace_readout: Param<Tensor<2>>,
}

/// Report names for `Model::parameter_tensors`, in the same wire order.
pub const PARAMETER_NAMES: [&str; 15] = [
    "embedding",
    "recurrent",
    "hidden_bias",
    "readout",
    "output_bias",
    "semantic_input",
    "semantic_recurrent",
    "spike_to_semantic",
    "semantic_to_spike",
    "semantic_bias",
    "semantic_readout",
    "membrane_decay",
    "trace_decay",
    "semantic_decay",
    "trace_readout",
];

/// Exact hard spikes with ds/dv=max(0,1-|v-theta|/w)/w during autodiff.
/// F(z)=1/2+z-z*|z|/2, z=clamp((v-theta)/w,-1,1), integrates the triangle.
/// It gives the correct derivative at the peak and support endpoints regardless
/// of abs/clamp boundary conventions. The equal smooth values are subtracted
/// BEFORE adding the hard spike, preserving exactly binary f32 forward values.
pub fn surrogate_spike(voltage: Tensor<2>, threshold: f64, width: f64) -> Tensor<2> {
    let hard = voltage.clone().greater_equal_elem(threshold).float();
    let z = ((voltage - threshold) / width).clamp(-1.0, 1.0);
    let smooth = z.clone() - z.clone() * z.abs() * 0.5 + 0.5;
    let correction = smooth.clone() - smooth.detach();
    hard + correction
}

impl Model {
    pub fn new(config: ModelConfig, seed: u64, device: &Device) -> Result<Self, String> {
        config.validate()?;
        let mut rng = Rng::new(seed);
        let h = config.hidden_size;
        let v = config.vocab_size;
        let scale = 1.0 / (h as f64).sqrt();
        let mut p = Vec::with_capacity(config.checked_parameter_count()?);
        p.extend((0..v * h).map(|_| rng.uniform(-0.5, 0.5) as f32));
        p.extend((0..h * h).map(|_| rng.uniform(-0.2 * scale, 0.2 * scale) as f32));
        p.extend((0..h).map(|_| rng.uniform(0.2, 0.3) as f32));
        p.extend((0..v * h).map(|_| rng.uniform(-scale, scale) as f32));
        p.extend(std::iter::repeat_n(0.0, v));
        for _ in 0..4 {
            p.extend((0..h * h).map(|_| rng.uniform(-scale, scale) as f32));
        }
        p.extend(std::iter::repeat_n(0.0, h));
        p.extend((0..v * h).map(|_| rng.uniform(-scale, scale) as f32));
        let (shortest, longest) = INITIAL_TIME_CONSTANTS;
        for _ in 0..3 {
            p.extend((0..h).map(|_| {
                let tau = rng.uniform(shortest.ln(), longest.ln()).exp();
                decay_logit((-1.0 / tau).exp()) as f32
            }));
        }
        p.extend((0..v * h).map(|_| rng.uniform(-scale, scale) as f32));
        Self::from_parameters(config, p, device)
    }

    pub fn from_parameters(
        config: ModelConfig,
        parameters: Vec<f32>,
        device: &Device,
    ) -> Result<Self, String> {
        config.validate()?;
        let expected = config.checked_parameter_count()?;
        if parameters.len() != expected {
            return Err(format!(
                "expected {expected} model parameters, received {}",
                parameters.len()
            ));
        }
        if parameters.iter().any(|value| !value.is_finite()) {
            return Err("model parameters must be finite".into());
        }
        let h = config.hidden_size;
        let v = config.vocab_size;
        let mut cursor = parameters.into_iter();
        let mut take = |shape: [usize; 2]| {
            let values: Vec<f32> = cursor.by_ref().take(shape[0] * shape[1]).collect();
            Param::from_data(TensorData::new(values, shape), device)
        };
        Ok(Self {
            embedding: take([v, h]),
            recurrent: take([h, h]),
            hidden_bias: take([1, h]),
            readout: take([v, h]),
            output_bias: take([1, v]),
            semantic_input: take([h, h]),
            semantic_recurrent: take([h, h]),
            spike_to_semantic: take([h, h]),
            semantic_to_spike: take([h, h]),
            semantic_bias: take([1, h]),
            semantic_readout: take([v, h]),
            membrane_decay: take([1, h]),
            trace_decay: take([1, h]),
            semantic_decay: take([1, h]),
            trace_readout: take([v, h]),
            feedback: FeedbackMode::Both,
            config,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.config
            .checked_parameter_count()
            .expect("constructed dimensions are valid")
    }

    /// Shared wire/gradient order: E, R, b, O, output bias, C, A, B, T, a, D,
    /// then the membrane, trace, and memory decay logits and P.
    /// Every parameter is a row-major rank-two float tensor.
    pub fn parameter_tensors(&self) -> Vec<(ParamId, Tensor<2>)> {
        vec![
            (self.embedding.id, self.embedding.val()),
            (self.recurrent.id, self.recurrent.val()),
            (self.hidden_bias.id, self.hidden_bias.val()),
            (self.readout.id, self.readout.val()),
            (self.output_bias.id, self.output_bias.val()),
            (self.semantic_input.id, self.semantic_input.val()),
            (self.semantic_recurrent.id, self.semantic_recurrent.val()),
            (self.spike_to_semantic.id, self.spike_to_semantic.val()),
            (self.semantic_to_spike.id, self.semantic_to_spike.val()),
            (self.semantic_bias.id, self.semantic_bias.val()),
            (self.semantic_readout.id, self.semantic_readout.val()),
            (self.membrane_decay.id, self.membrane_decay.val()),
            (self.trace_decay.id, self.trace_decay.val()),
            (self.semantic_decay.id, self.semantic_decay.val()),
            (self.trace_readout.id, self.trace_readout.val()),
        ]
    }

    /// Checkpoint parameter order is defined by `parameter_tensors`.
    pub fn parameter_values(&self) -> Result<Vec<f32>, String> {
        let mut values = Vec::with_capacity(self.parameter_count());
        for (_, tensor) in self.parameter_tensors() {
            values.extend(
                tensor
                    .try_to_vec_as::<f32>()
                    .map_err(|error| format!("reading model parameters: {error}"))?,
            );
        }
        Ok(values)
    }

    pub fn device(&self) -> Device {
        self.embedding.val().device()
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        for (name, tensor, expected) in [
            ("embedding", self.embedding.val(), [v, h]),
            ("recurrent", self.recurrent.val(), [h, h]),
            ("hidden bias", self.hidden_bias.val(), [1, h]),
            ("readout", self.readout.val(), [v, h]),
            ("output bias", self.output_bias.val(), [1, v]),
            ("semantic input", self.semantic_input.val(), [h, h]),
            ("semantic recurrence", self.semantic_recurrent.val(), [h, h]),
            ("spike to semantic", self.spike_to_semantic.val(), [h, h]),
            ("semantic to spike", self.semantic_to_spike.val(), [h, h]),
            ("semantic bias", self.semantic_bias.val(), [1, h]),
            ("semantic readout", self.semantic_readout.val(), [v, h]),
            ("membrane decay", self.membrane_decay.val(), [1, h]),
            ("trace decay", self.trace_decay.val(), [1, h]),
            ("semantic decay", self.semantic_decay.val(), [1, h]),
            ("trace readout", self.trace_readout.val(), [v, h]),
        ] {
            if tensor.dims() != expected {
                return Err(format!("{name} shape must be {expected:?}"));
            }
        }
        if self
            .parameter_values()?
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("model parameters must be finite".into());
        }
        Ok(())
    }

    /// One causal token per batch row. Token IDs must be within the vocabulary.
    /// No host transfer or graph detach occurs in state/reset/recurrent paths.
    pub fn step(&self, tokens: Tensor<1, Int>, state: &mut State) -> Result<Tensor<2>, String> {
        self.step_for_path(tokens, state, ModelPath::Full)
    }

    pub fn step_for_path(
        &self,
        tokens: Tensor<1, Int>,
        state: &mut State,
        path: ModelPath,
    ) -> Result<Tensor<2>, String> {
        self.step_traced_for_path(tokens, state, path)
            .map(|(logits, _)| logits)
    }

    /// The sole state-update implementation, shared by full and component paths.
    /// Zero gates preserve differentiable paths for every parameter group.
    pub fn step_traced_for_path(
        &self,
        tokens: Tensor<1, Int>,
        state: &mut State,
        path: ModelPath,
    ) -> Result<(Tensor<2>, StepTrace), String> {
        let [batch] = tokens.dims();
        let expected = [batch, self.config.hidden_size];
        if batch == 0
            || state.membrane.dims() != expected
            || state.spikes.dims() != expected
            || state.trace.dims() != expected
            || state.semantic.dims() != expected
        {
            return Err(format!(
                "state shape must be {expected:?} with nonempty batch"
            ));
        }
        let (top_down, bottom_up, spike_output, memory_output) = match path {
            ModelPath::Full => {
                let (top_down, bottom_up) = self.feedback.gates();
                (top_down, bottom_up, 1.0, 1.0)
            }
            ModelPath::MemoryOnly => (0.0, 0.0, 0.0, 1.0),
            ModelPath::SpikesOnly => (0.0, 0.0, 1.0, 0.0),
        };
        let decay = |logit: &Param<Tensor<2>>| sigmoid(logit.val()) * MAX_DECAY;
        let (trace_decay, semantic_decay) = (decay(&self.trace_decay), decay(&self.semantic_decay));
        let embedding = self.embedding.val().select(0, tokens);
        let voltage = state.membrane.clone() * decay(&self.membrane_decay)
            + embedding.clone()
            + state
                .spikes
                .clone()
                .matmul(self.recurrent.val().transpose())
            + self.hidden_bias.val()
            + state
                .semantic
                .clone()
                .matmul(self.semantic_to_spike.val().transpose())
                * top_down;
        let spikes = surrogate_spike(
            voltage.clone(),
            self.config.threshold,
            self.config.surrogate_width,
        );
        state.membrane = voltage.clone() - spikes.clone() * self.config.threshold;
        state.spikes = spikes.clone();
        state.trace =
            state.trace.clone() * trace_decay.clone() + spikes.clone() * (-trace_decay + 1.0);
        let candidate = (state
            .semantic
            .clone()
            .matmul(self.semantic_recurrent.val().transpose())
            + embedding.matmul(self.semantic_input.val().transpose())
            + state
                .trace
                .clone()
                .matmul(self.spike_to_semantic.val().transpose())
                * bottom_up
            + self.semantic_bias.val())
        .tanh();
        state.semantic =
            state.semantic.clone() * semantic_decay.clone() + candidate * (-semantic_decay + 1.0);
        let logits = (spikes.matmul(self.readout.val().transpose())
            + state
                .trace
                .clone()
                .matmul(self.trace_readout.val().transpose()))
            * spike_output
            + state
                .semantic
                .clone()
                .matmul(self.semantic_readout.val().transpose())
                * memory_output
            + self.output_bias.val();
        Ok((logits, StepTrace { voltage }))
    }

    /// Mean next-token cross entropy over actual targets, excluding padding.
    /// Each independent document window begins with zero state.
    pub fn loss(&self, batch: &Batch) -> Result<Tensor<1>, String> {
        self.loss_for_path(batch, ModelPath::Full)
    }

    pub fn loss_for_path(&self, batch: &Batch, path: ModelPath) -> Result<Tensor<1>, String> {
        self.loss_components(batch, path, None)
            .map(|(cross_entropy, _)| cross_entropy)
    }

    /// The forward dynamics and the specified spike/reset derivatives stay the
    /// same. Only this training objective may add a voltage cost to masked CE.
    pub fn training_loss_for_path(
        &self,
        batch: &Batch,
        path: ModelPath,
        penalty: VoltagePenalty,
    ) -> Result<LossTerms, String> {
        penalty.validate()?;
        let active_penalty = (penalty.coefficient > 0.0).then_some(penalty);
        let (cross_entropy, measured_penalty) =
            self.loss_components(batch, path, active_penalty)?;
        let (voltage_penalty, total) = match measured_penalty {
            Some(value) => {
                let total = cross_entropy.clone() + value.clone() * penalty.coefficient;
                (value, total)
            }
            None => (Tensor::zeros([1], &self.device()), cross_entropy.clone()),
        };
        Ok(LossTerms {
            cross_entropy,
            voltage_penalty,
            total,
        })
    }

    /// One shared recurrence/reduction loop for pure CE and the training loss.
    /// None performs no auxiliary voltage operations, preserving the CE graph.
    fn loss_components(
        &self,
        batch: &Batch,
        path: ModelPath,
        penalty: Option<VoltagePenalty>,
    ) -> Result<(Tensor<1>, Option<Tensor<1>>), String> {
        // Besides shape checks, guard the width and its inverse before dividing.
        self.config.validate()?;
        let [rows, steps] = batch.inputs.dims();
        if rows == 0
            || steps == 0
            || batch.targets.dims() != [rows, steps]
            || batch.mask.dims() != [rows, steps]
            || batch.valid_targets == 0
            || batch.valid_targets > rows.saturating_mul(steps)
        {
            return Err(
                "batch needs matching nonempty [batch,time] shapes and valid targets".into(),
            );
        }
        let device = self.device();
        let mut state = State::new(rows, self.config.hidden_size, &device);
        let mut total = Tensor::<1>::zeros([1], &device);
        let mut penalty_total: Option<Tensor<1>> = None;
        for time in 0..steps {
            let tokens = batch
                .inputs
                .clone()
                .slice([0..rows, time..time + 1])
                .reshape([rows]);
            let targets = batch.targets.clone().slice([0..rows, time..time + 1]);
            let mask = batch.mask.clone().slice([0..rows, time..time + 1]);
            let (logits, trace) = self.step_traced_for_path(tokens, &mut state, path)?;
            let log_probability = log_softmax(logits, 1).gather(1, targets);
            total = total - (log_probability * mask.clone()).sum();
            if let Some(penalty) = penalty {
                let cost = (squared_voltage_excess(
                    trace.voltage,
                    self.config.threshold,
                    self.config.surrogate_width,
                    penalty.band_widths,
                ) * mask)
                    .sum();
                penalty_total = Some(match penalty_total {
                    Some(previous) => previous + cost,
                    None => cost,
                });
            }
        }
        let cross_entropy = total / batch.valid_targets as f64;
        let voltage_penalty = penalty_total
            .map(|total| total / (batch.valid_targets as f64 * self.config.hidden_size as f64));
        Ok((cross_entropy, voltage_penalty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig {
            vocab_size: 3,
            hidden_size: 2,
            threshold: 0.8,
            surrogate_width: 1.3,
        }
    }
    /// Uniform decays .65/.8/.9 keep the hand-derived expectations simple;
    /// the gradient oracle perturbs them to distinct per-neuron values.
    fn parameters() -> Vec<f32> {
        let mut p = vec![
            0.7, 0.3, -0.2, 0.9, 0.4, -0.1, // E
            0.2, -0.15, 0.1, 0.25, // R
            0.2, 0.15, // b
            0.3, -0.4, -0.1, 0.6, 0.2, 0.1, // O
            0.05, -0.1, 0.02, // output bias
            0.3, -0.2, 0.1, 0.4, // C
            0.2, 0.1, -0.15, 0.25, // A
            0.4, -0.1, 0.2, 0.3, // B
            0.25, -0.2, 0.1, 0.3, // T
            0.05, -0.02, // a
            0.2, -0.1, -0.3, 0.25, 0.15, 0.4, // D
        ];
        for decay in [0.65, 0.8, 0.9] {
            p.extend([decay_logit(decay) as f32; 2]);
        }
        p.extend([0.1, -0.3, 0.25, 0.05, -0.2, 0.35]); // P
        p
    }
    fn fixture(device: &Device) -> Model {
        Model::from_parameters(config(), parameters(), device).unwrap()
    }
    fn values<const D: usize>(tensor: &Tensor<D>) -> Vec<f32> {
        tensor.try_to_vec_as().unwrap()
    }
    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}; tolerance {tolerance}"
        );
    }

    #[test]
    fn triangular_surrogate_binary_forward_peak_support_and_boundaries() {
        let device = Device::flex().autodiff();
        let voltage = Tensor::<2>::from_floats([[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]], &device)
            .require_grad();
        let spikes = surrogate_spike(voltage.clone(), 0.0, 1.0);
        assert_eq!(values(&spikes), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let gradients = spikes.sum().backward();
        assert_eq!(
            values(&voltage.grad(&gradients).unwrap()),
            vec![0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
        );
        let voltage = Tensor::<2>::from_floats([[0.8, 1.45, 0.15]], &device).require_grad();
        let gradients = surrogate_spike(voltage.clone(), 0.8, 1.3).sum().backward();
        for (actual, expected) in values(&voltage.grad(&gradients).unwrap()).into_iter().zip([
            1.0 / 1.3,
            0.5 / 1.3,
            0.5 / 1.3,
        ]) {
            close(actual as f64, expected, 2e-7);
        }
    }

    #[test]
    fn recurrence_leak_and_subtractive_reset_follow_equations() {
        let device = Device::flex();
        let model = fixture(&device);
        let mut state = State {
            membrane: Tensor::from_floats([[0.4, -0.2]], &device),
            spikes: Tensor::from_floats([[1.0, 0.0]], &device),
            trace: Tensor::from_floats([[0.1, 0.4]], &device),
            semantic: Tensor::from_floats([[0.3, -0.2]], &device),
        };
        let logits = model
            .step(Tensor::from_data([0_i32], &device), &mut state)
            .unwrap();
        // Learned decays reproduce .65/.8/.9 up to f32 sigmoid rounding.
        assert_eq!(values(&state.spikes), vec![1.0, 0.0]);
        for (actual, expected) in values(&state.membrane).into_iter().zip([0.675, 0.39]) {
            close(actual as f64, expected, 5e-7);
        }
        for (actual, expected) in values(&state.trace).into_iter().zip([0.28, 0.32]) {
            close(actual as f64, expected, 5e-7);
        }
        let context = [0.27 + 0.1 * 0.32_f64.tanh(), -0.18 + 0.1 * 0.227_f64.tanh()];
        for (actual, expected) in values(&state.semantic).into_iter().zip(context) {
            close(actual as f64, expected, 5e-7);
        }
        // Spike readout + bias, trace readout P*[.28,.32], then memory readout.
        let expected = [
            0.35 + (0.1 * 0.28 - 0.3 * 0.32) + 0.2 * context[0] - 0.1 * context[1],
            -0.2 + (0.25 * 0.28 + 0.05 * 0.32) - 0.3 * context[0] + 0.25 * context[1],
            0.22 + (-0.2 * 0.28 + 0.35 * 0.32) + 0.15 * context[0] + 0.4 * context[1],
        ];
        for (actual, expected) in values(&logits).into_iter().zip(expected) {
            close(actual as f64, expected, 5e-7);
        }
        let mut cfg = config();
        cfg.threshold = 1.0;
        let mut p = parameters();
        p[0] = 0.75;
        p[1] = 3.0;
        p[10] = 0.25;
        p[11] = 0.25;
        let model = Model::from_parameters(cfg, p, &device).unwrap();
        let mut state = State::new(1, 2, &device);
        model
            .step(Tensor::from_data([0_i32], &device), &mut state)
            .unwrap();
        assert_eq!(values(&state.spikes), vec![1.0, 1.0]);
        assert_eq!(values(&state.membrane), vec![0.0, 2.25]);
    }

    #[test]
    fn batched_rows_are_independent_and_forward_is_causal() {
        let device = Device::flex();
        let model = fixture(&device);
        let rows = [[0_i32, 1, 2, 0], [2, 0, 0, 1]];
        let mut state = State::new(2, 2, &device);
        let mut singles = [State::new(1, 2, &device), State::new(1, 2, &device)];
        for (&first, &second) in rows[0].iter().zip(&rows[1]) {
            let tokens = [first, second];
            let output = values(
                &model
                    .step(Tensor::from_data(tokens, &device), &mut state)
                    .unwrap(),
            );
            for (row, token) in tokens.into_iter().enumerate() {
                let single = values(
                    &model
                        .step(Tensor::from_data([token], &device), &mut singles[row])
                        .unwrap(),
                );
                for (actual, expected) in output[row * 3..row * 3 + 3].iter().zip(single) {
                    close(*actual as f64, expected as f64, 1e-7);
                }
            }
        }
        let mut repeated = State::new(1, 2, &device);
        for token in rows[0] {
            model
                .step(Tensor::from_data([token], &device), &mut repeated)
                .unwrap();
        }
        assert_eq!(values(&repeated.membrane), values(&singles[0].membrane));
        assert_eq!(values(&repeated.trace), values(&singles[0].trace));
        assert_eq!(values(&repeated.semantic), values(&singles[0].semantic));
    }

    #[test]
    fn feedback_interfaces_are_causal_and_each_mode_keeps_all_gradient_paths() {
        let device = Device::flex().autodiff();
        let mut states = Vec::new();
        for mode in [
            FeedbackMode::Both,
            FeedbackMode::TopDown,
            FeedbackMode::BottomUp,
            FeedbackMode::Neither,
        ] {
            assert_eq!(FeedbackMode::parse(mode.as_str()).unwrap(), mode);
            let mut model = fixture(&device);
            model.feedback = mode;
            let mut state = State {
                membrane: Tensor::from_floats([[0.4, -0.2]], &device),
                spikes: Tensor::from_floats([[1.0, 0.0]], &device),
                trace: Tensor::from_floats([[0.1, 0.4]], &device),
                semantic: Tensor::from_floats([[0.3, -0.2]], &device),
            };
            model
                .step(Tensor::from_data([0_i32], &device), &mut state)
                .unwrap();
            states.push((values(&state.membrane), values(&state.semantic)));
            let gradients = model
                .loss(&batch(&device, [[1, 2, 0], [0, 1, 2]], 4))
                .unwrap()
                .backward();
            for (id, tensor) in model.parameter_tensors() {
                let gradient = values(
                    &tensor
                        .grad(&gradients)
                        .expect("all interfaces remain in graph"),
                );
                if (id == model.semantic_to_spike.id
                    && matches!(mode, FeedbackMode::BottomUp | FeedbackMode::Neither))
                    || (id == model.spike_to_semantic.id
                        && matches!(mode, FeedbackMode::TopDown | FeedbackMode::Neither))
                {
                    assert!(gradient.iter().all(|value| *value == 0.0));
                }
            }
        }
        // Bottom-up changes the context after this spike step, not its voltage.
        assert_eq!(states[0].0, states[1].0);
        assert_eq!(states[2].0, states[3].0);
        assert_ne!(states[0].1, states[1].1);
        // Prior context changes membrane dynamics through the top-down interface.
        assert_ne!(states[0].0, states[2].0);
        assert!(FeedbackMode::parse("unknown").is_err());

        // Starting at zero, top-down feedback cannot change the first spike step;
        // it can affect the next step after context has actually been computed.
        let mut both = fixture(&device);
        both.feedback = FeedbackMode::Both;
        let mut bottom = both.clone();
        bottom.feedback = FeedbackMode::BottomUp;
        let mut both_state = State::new(1, 2, &device);
        let mut bottom_state = both_state.clone();
        for time in 0..2 {
            both.step(Tensor::from_data([0_i32], &device), &mut both_state)
                .unwrap();
            bottom
                .step(Tensor::from_data([0_i32], &device), &mut bottom_state)
                .unwrap();
            if time == 0 {
                assert_eq!(values(&both_state.membrane), values(&bottom_state.membrane));
                assert_eq!(values(&both_state.semantic), values(&bottom_state.semantic));
            } else {
                assert_ne!(values(&both_state.membrane), values(&bottom_state.membrane));
            }
        }
    }

    #[test]
    fn traced_full_path_matches_wrappers_and_exposes_attached_pre_reset_voltage() {
        let device = Device::flex().autodiff();
        let model = fixture(&device);
        let mut ordinary = State::new(1, 2, &device);
        let mut explicit = ordinary.clone();
        let mut traced = ordinary.clone();
        for token in [0_i32, 1, 2, 0] {
            let output = model
                .step(Tensor::from_data([token], &device), &mut ordinary)
                .unwrap();
            let path_output = model
                .step_for_path(
                    Tensor::from_data([token], &device),
                    &mut explicit,
                    ModelPath::Full,
                )
                .unwrap();
            let (trace_output, trace) = model
                .step_traced_for_path(
                    Tensor::from_data([token], &device),
                    &mut traced,
                    ModelPath::Full,
                )
                .unwrap();
            assert_eq!(values(&output), values(&path_output));
            assert_eq!(values(&output), values(&trace_output));
            for (unreset, (reset, spike)) in values(&trace.voltage).into_iter().zip(
                values(&traced.membrane)
                    .into_iter()
                    .zip(values(&traced.spikes)),
            ) {
                assert!(spike == 0.0 || spike == 1.0);
                assert_eq!(spike, f32::from(unreset >= model.config.threshold as f32));
                close(
                    reset as f64,
                    unreset as f64 - model.config.threshold * spike as f64,
                    1e-7,
                );
            }
            for state in [&explicit, &traced] {
                assert_eq!(values(&ordinary.membrane), values(&state.membrane));
                assert_eq!(values(&ordinary.spikes), values(&state.spikes));
                assert_eq!(values(&ordinary.trace), values(&state.trace));
                assert_eq!(values(&ordinary.semantic), values(&state.semantic));
            }
        }
        let input = batch(&device, [[1, 2, 0], [0, 1, 2]], 4);
        assert_eq!(
            values(&model.loss(&input).unwrap()),
            values(&model.loss_for_path(&input, ModelPath::Full).unwrap())
        );

        let (_, trace) = model
            .step_traced_for_path(
                Tensor::from_data([0_i32], &device),
                &mut State::new(1, 2, &device),
                ModelPath::Full,
            )
            .unwrap();
        for (actual, expected) in values(&trace.voltage).into_iter().zip([0.9, 0.45]) {
            close(actual as f64, expected, 1e-7);
        }
        let gradients = trace.voltage.sum().backward();
        assert_eq!(
            values(&model.embedding.val().grad(&gradients).unwrap()),
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn component_paths_isolate_outputs_and_preserve_zero_gradient_groups() {
        let device = Device::flex().autodiff();
        let model = fixture(&device);
        for path in [ModelPath::MemoryOnly, ModelPath::SpikesOnly] {
            assert_eq!(ModelPath::parse(path.as_str()).unwrap(), path);
            let mut baseline = State::new(1, 2, &device);
            let mut changed = baseline.clone();
            // The spike trace feeds the spike-path readout P, so only memory
            // is irrelevant to the spike path.
            if path == ModelPath::MemoryOnly {
                changed.membrane = Tensor::from_floats([[7.0, -2.0]], &device);
                changed.spikes = Tensor::from_floats([[1.0, 1.0]], &device);
                changed.trace = Tensor::from_floats([[0.7, 0.9]], &device);
            } else {
                changed.semantic = Tensor::from_floats([[0.8, -0.4]], &device);
            }
            for token in [0_i32, 1, 2, 0] {
                let first = model
                    .step_for_path(Tensor::from_data([token], &device), &mut baseline, path)
                    .unwrap();
                let second = model
                    .step_for_path(Tensor::from_data([token], &device), &mut changed, path)
                    .unwrap();
                assert_eq!(values(&first), values(&second));
            }
            let loss = model
                .loss_for_path(&batch(&device, [[1, 2, 0], [0, 1, 2]], 4), path)
                .unwrap();
            let gradients = loss.backward();
            for (index, (_, tensor)) in model.parameter_tensors().into_iter().enumerate() {
                let gradient = values(
                    &tensor
                        .grad(&gradients)
                        .expect("all component paths stay attached"),
                );
                // 11-14: membrane, trace, memory decays and trace readout.
                let disabled = match path {
                    ModelPath::MemoryOnly => matches!(index, 1 | 2 | 3 | 7 | 8 | 11 | 12 | 14),
                    ModelPath::SpikesOnly => matches!(index, 5..=10 | 13),
                    ModelPath::Full => unreachable!(),
                };
                if disabled {
                    assert!(
                        gradient.iter().all(|&value| value == 0.0),
                        "group {index} must be inactive in {}",
                        path.as_str()
                    );
                } else {
                    assert!(
                        gradient.iter().any(|&value| value != 0.0),
                        "group {index} must learn in {}",
                        path.as_str()
                    );
                }
            }
            let mut malformed = State::new(1, 2, &device);
            malformed.trace = Tensor::zeros([1, 3], &device);
            assert!(
                model
                    .step_traced_for_path(Tensor::from_data([0_i32], &device), &mut malformed, path)
                    .is_err()
            );
        }
        assert_eq!(ModelPath::parse("full").unwrap(), ModelPath::Full);
        assert!(ModelPath::parse("unknown").is_err());
    }

    fn batch(device: &Device, targets: [[i32; 3]; 2], valid_targets: usize) -> Batch {
        Batch {
            inputs: Tensor::from_data([[0_i32, 1, 2], [2, 0, 0]], device),
            targets: Tensor::from_data(targets, device),
            mask: Tensor::from_floats([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], device),
            valid_targets,
        }
    }

    #[test]
    fn masked_loss_normalizes_actual_targets_and_padding_has_no_gradient() {
        let device = Device::flex().autodiff();
        let model = fixture(&device);
        let first = batch(&device, [[1, 2, 0], [0, 1, 2]], 4);
        let second = batch(&device, [[1, 2, 0], [0, 2, 0]], 4);
        let loss = model.loss(&first).unwrap();
        let modified = model.loss(&second).unwrap();
        assert_eq!(values(&loss), values(&modified));
        let gradients = loss.backward();
        let modified_gradients = modified.backward();
        for (_, tensor) in model.parameter_tensors() {
            assert_eq!(
                values(&tensor.grad(&gradients).unwrap()),
                values(&tensor.grad(&modified_gradients).unwrap())
            );
        }
        let mut state = State::new(1, 2, &device);
        let mut expected = 0.0;
        for (input, target) in [(0_i32, 1_usize), (1, 2), (2, 0)] {
            expected -= values(&log_softmax(
                model
                    .step(Tensor::from_data([input], &device), &mut state)
                    .unwrap(),
                1,
            ))[target] as f64;
        }
        let mut state = State::new(1, 2, &device);
        expected -= values(&log_softmax(
            model
                .step(Tensor::from_data([2_i32], &device), &mut state)
                .unwrap(),
            1,
        ))[0] as f64;
        close(
            values(&model.loss(&first).unwrap())[0] as f64,
            expected / 4.0,
            2e-7,
        );
    }

    #[test]
    fn voltage_excess_analytic_gradient_is_zero_inside_and_at_band_boundaries() {
        let device = Device::flex().autodiff();
        // theta=1,w=2,k=3 has the allowed voltage band [-5,7].
        let voltage = Tensor::<2>::from_floats([[-9.0, -5.0, -1.0, 1.0, 5.0, 7.0, 11.0]], &device)
            .require_grad();
        let penalty = squared_voltage_excess(voltage.clone(), 1.0, 2.0, 3.0);
        assert_eq!(values(&penalty), [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
        let gradients = penalty.sum().backward();
        assert_eq!(
            values(&voltage.grad(&gradients).unwrap()),
            [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
        );
    }

    #[test]
    fn training_voltage_penalty_masks_padding_and_normalizes_targets_and_neurons() {
        let device = Device::flex().autodiff();
        let mut cfg = config();
        cfg.threshold = 1.0;
        cfg.surrogate_width = 2.0;
        let mut p = vec![0.0; cfg.checked_parameter_count().unwrap()];
        p[..6].copy_from_slice(&[10.0, -8.0, 1.0, 1.0, 100.0, -100.0]);
        // Membrane decay ~4e-18: no carried voltage, as with a zero decay.
        p[45..47].copy_from_slice(&[-40.0, -40.0]);
        let model = Model::from_parameters(cfg, p, &device).unwrap();
        let batch = Batch {
            inputs: Tensor::from_data([[0_i32, 1], [0, 2]], &device),
            targets: Tensor::from_data([[1_i32, 2], [1, 0]], &device),
            mask: Tensor::from_floats([[1.0, 1.0], [1.0, 0.0]], &device),
            valid_targets: 3,
        };
        let terms = model
            .training_loss_for_path(
                &batch,
                ModelPath::Full,
                VoltagePenalty {
                    coefficient: 0.001,
                    band_widths: 3.0,
                },
            )
            .unwrap();
        // Two valid occurrences each contribute 2*(1.5^2); divide by 3*H=6.
        assert_eq!(values(&terms.voltage_penalty), [1.5]);
        close(values(&terms.cross_entropy)[0] as f64, 3.0_f64.ln(), 1e-7);
        close(values(&terms.total)[0] as f64, 3.0_f64.ln() + 0.0015, 1e-7);
        let gradients = terms.total.backward();
        let embedding_gradient = values(&model.embedding.val().grad(&gradients).unwrap());
        // CE is independent of E with zero output weights. Auxiliary credit
        // reaches E even though these voltages lie outside surrogate support.
        for (&actual, expected) in embedding_gradient
            .iter()
            .zip([0.0005, -0.0005, 0.0, 0.0, 0.0, 0.0])
        {
            close(actual as f64, expected, 1e-9);
        }
        // Only the masked suffix contains token 2; its large voltage contributes
        // neither penalty nor gradient. Changing its target is also irrelevant.
        let mut changed = batch.clone();
        changed.targets = Tensor::from_data([[1_i32, 2], [1, 2]], &device);
        let changed_terms = model
            .training_loss_for_path(
                &changed,
                ModelPath::Full,
                VoltagePenalty {
                    coefficient: 0.001,
                    band_widths: 3.0,
                },
            )
            .unwrap();
        assert_eq!(values(&changed_terms.voltage_penalty), [1.5]);
        let changed_gradients = changed_terms.total.backward();
        assert_eq!(
            embedding_gradient,
            values(&model.embedding.val().grad(&changed_gradients).unwrap())
        );
    }

    #[test]
    fn disabled_voltage_penalty_keeps_ce_and_every_parameter_gradient_exact() {
        let device = Device::flex().autodiff();
        let mut cfg = config();
        // Normalized voltage squares would overflow f32 if the disabled
        // auxiliary graph were accidentally evaluated, while spikes stay safe.
        cfg.surrogate_width = 1e-20;
        let model = Model::from_parameters(cfg, parameters(), &device).unwrap();
        for path in [
            ModelPath::Full,
            ModelPath::MemoryOnly,
            ModelPath::SpikesOnly,
        ] {
            let batch = batch(&device, [[1, 2, 0], [0, 1, 2]], 4);
            let plain = model.loss_for_path(&batch, path).unwrap();
            let terms = model
                .training_loss_for_path(&batch, path, VoltagePenalty::default())
                .unwrap();
            assert_eq!(values(&plain), values(&terms.cross_entropy));
            assert_eq!(values(&plain), values(&terms.total));
            assert_eq!(values(&terms.voltage_penalty), [0.0]);
            let plain_gradients = plain.backward();
            let disabled_gradients = terms.total.backward();
            for (_, parameter) in model.parameter_tensors() {
                assert_eq!(
                    values(&parameter.grad(&plain_gradients).unwrap()),
                    values(&parameter.grad(&disabled_gradients).unwrap())
                );
            }
        }
    }

    #[test]
    fn voltage_penalty_configuration_and_training_width_are_guarded() {
        VoltagePenalty::default().validate().unwrap();
        VoltagePenalty {
            coefficient: 0.001,
            band_widths: 3.0,
        }
        .validate()
        .unwrap();
        for coefficient in [-1.0, f64::NAN, f64::INFINITY, 1e-100, 1e100] {
            assert!(
                VoltagePenalty {
                    coefficient,
                    band_widths: 3.0
                }
                .validate()
                .is_err()
            );
        }
        for band_widths in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-100, 1e100] {
            assert!(
                VoltagePenalty {
                    coefficient: 0.0,
                    band_widths
                }
                .validate()
                .is_err()
            );
        }
        let device = Device::flex();
        let mut model = fixture(&device);
        model.config.surrogate_width = 1e-40;
        assert!(
            model
                .training_loss_for_path(
                    &batch(&device, [[1, 2, 0], [0, 1, 2]], 4),
                    ModelPath::Full,
                    VoltagePenalty::default()
                )
                .is_err()
        );
    }

    #[test]
    fn initialization_round_trip_validation_and_module_training_state() {
        let device = Device::flex();
        let model = Model::new(config(), 42, &device).unwrap();
        let again = Model::new(config(), 42, &device).unwrap();
        let p = model.parameter_values().unwrap();
        assert_eq!(p.len(), model.parameter_count());
        assert_eq!(p.len(), 57);
        assert_eq!(model.parameter_tensors().len(), 15);
        // Initial time constants lie in [2, 32] steps: decays in [e^-1/2, e^-1/32].
        for &logit in &p[45..51] {
            let decay = MAX_DECAY / (1.0 + (-f64::from(logit)).exp());
            assert!(((-0.5_f64).exp() - 1e-6..=(-1.0 / 32.0_f64).exp() + 1e-6).contains(&decay));
        }
        assert_eq!(p, again.parameter_values().unwrap());
        let restored = Model::from_parameters(config(), p.clone(), &device).unwrap();
        assert_eq!(p, restored.parameter_values().unwrap());
        let training = restored.train();
        assert!(training.embedding.val().is_require_grad());
        let validation = training.valid();
        assert!(!validation.embedding.val().is_require_grad());
        assert!(validation.train().embedding.val().is_require_grad());
        assert!(Model::from_parameters(config(), vec![0.0], &device).is_err());
        let mut cfg = config();
        cfg.hidden_size = usize::MAX;
        assert!(cfg.validate().is_err());
        assert!(
            model
                .step(
                    Tensor::from_data([0_i32, 1], &device),
                    &mut State::new(2, 3, &device)
                )
                .is_err()
        );
        for invalid_trace in [true, false] {
            let mut state = State::new(1, 2, &device);
            if invalid_trace {
                state.trace = Tensor::zeros([1, 3], &device);
            } else {
                state.semantic = Tensor::zeros([1, 3], &device);
            }
            assert!(
                model
                    .step(Tensor::from_data([0_i32], &device), &mut state)
                    .is_err()
            );
        }
        assert!(
            model
                .loss(&batch(&device, [[1, 2, 0], [0, 1, 2]], 0))
                .is_err()
        );
    }

    // Independent forward-only scalar tangent oracle, confined to tests.
    struct OperatingPoint {
        voltage: [f64; 2],
        spikes: [f64; 2],
        slope: [f64; 2],
    }
    fn scalar_loss(
        p: &[f64],
        inputs: &[usize],
        targets: &[usize],
        frozen: Option<&[OperatingPoint]>,
        mode: FeedbackMode,
        penalty: VoltagePenalty,
    ) -> (f64, Vec<OperatingPoint>) {
        let cfg = config();
        let mut membrane = [0.2, -0.1];
        let mut previous_spikes = [1.0, 0.0];
        let mut trace = [0.1, 0.4];
        let mut semantic = [0.3, -0.2];
        let (top_down, bottom_up) = mode.gates();
        // Decay logits: membrane p[45..47], trace p[47..49], memory p[49..51].
        let decay = |index: usize| MAX_DECAY / (1.0 + (-p[index]).exp());
        let mut trajectory = Vec::new();
        let mut total = 0.0;
        for (time, &token) in inputs.iter().enumerate() {
            let mut voltage = [0.0; 2];
            let mut spikes = [0.0; 2];
            let mut slope = [0.0; 2];
            for neuron in 0..2 {
                voltage[neuron] = decay(45 + neuron) * membrane[neuron]
                    + p[token * 2 + neuron]
                    + p[10 + neuron]
                    + (0..2)
                        .map(|source| p[6 + neuron * 2 + source] * previous_spikes[source])
                        .sum::<f64>()
                    + top_down
                        * (0..2)
                            .map(|source| p[33 + neuron * 2 + source] * semantic[source])
                            .sum::<f64>();
                slope[neuron] = (1.0
                    - ((voltage[neuron] - cfg.threshold) / cfg.surrogate_width).abs())
                .max(0.0)
                    / cfg.surrogate_width;
                spikes[neuron] = if let Some(frozen) = frozen {
                    frozen[time].spikes[neuron]
                        + frozen[time].slope[neuron]
                            * (voltage[neuron] - frozen[time].voltage[neuron])
                } else {
                    f64::from(voltage[neuron] >= cfg.threshold)
                };
                membrane[neuron] = voltage[neuron] - cfg.threshold * spikes[neuron];
                let rho = decay(47 + neuron);
                trace[neuron] = rho * trace[neuron] + (1.0 - rho) * spikes[neuron];
            }
            let previous_semantic = semantic;
            for neuron in 0..2 {
                let candidate = p[37 + neuron]
                    + (0..2)
                        .map(|source| {
                            p[25 + neuron * 2 + source] * previous_semantic[source]
                                + p[21 + neuron * 2 + source] * p[token * 2 + source]
                                + bottom_up * p[29 + neuron * 2 + source] * trace[source]
                        })
                        .sum::<f64>();
                let gamma = decay(49 + neuron);
                semantic[neuron] =
                    gamma * previous_semantic[neuron] + (1.0 - gamma) * candidate.tanh();
            }
            let logits: Vec<f64> = (0..3)
                .map(|token| {
                    p[18 + token]
                        + (0..2)
                            .map(|neuron| {
                                p[12 + token * 2 + neuron] * spikes[neuron]
                                    + p[51 + token * 2 + neuron] * trace[neuron]
                            })
                            .sum::<f64>()
                        + (0..2)
                            .map(|neuron| p[39 + token * 2 + neuron] * semantic[neuron])
                            .sum::<f64>()
                })
                .collect();
            let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            total += ((logits.iter().map(|x| (x - maximum).exp()).sum::<f64>()).ln() + maximum
                - logits[targets[time]])
                / inputs.len() as f64;
            if penalty.coefficient > 0.0 {
                total += penalty.coefficient
                    * voltage
                        .iter()
                        .map(|value| {
                            (((value - cfg.threshold) / cfg.surrogate_width).abs()
                                - penalty.band_widths)
                                .max(0.0)
                                .powi(2)
                        })
                        .sum::<f64>()
                    / (inputs.len() * 2) as f64;
            }
            trajectory.push(OperatingPoint {
                voltage,
                spikes,
                slope,
            });
            previous_spikes = spikes;
        }
        (total, trajectory)
    }

    #[test]
    fn every_bptt_parameter_matches_independent_frozen_tangent_finite_differences() {
        for penalty in [
            VoltagePenalty::default(),
            VoltagePenalty {
                coefficient: 0.17,
                band_widths: 0.25,
            },
        ] {
            for inactive in [false, true] {
                let device = Device::flex().autodiff();
                let mut p = parameters();
                // Distinct per-neuron membrane, trace, and memory decays.
                p[45] += 0.4;
                p[48] -= 0.5;
                p[50] += 0.3;
                if inactive {
                    p[4] = -3.0;
                }
                let model = Model::from_parameters(config(), p.clone(), &device).unwrap();
                let p: Vec<f64> = p.into_iter().map(f64::from).collect();
                let inputs = [0_usize, 1, 2, 0, 1];
                let targets = [1_usize, 2, 0, 2, 0];
                let other_inputs = [2_usize, 0, 1, 0, 0];
                let other_targets = [1_usize, 2, 0, 0, 0];
                // Both lanes have a fixed nonzero boundary state. The second lane's
                // last two positions are padding, with no objective contribution.
                let mut state = State {
                    membrane: Tensor::from_floats([[0.2, -0.1], [0.2, -0.1]], &device),
                    spikes: Tensor::from_floats([[1.0, 0.0], [1.0, 0.0]], &device),
                    trace: Tensor::from_floats([[0.1, 0.4], [0.1, 0.4]], &device),
                    semantic: Tensor::from_floats([[0.3, -0.2], [0.3, -0.2]], &device),
                };
                let mut loss = Tensor::<1>::zeros([1], &device);
                for time in 0..inputs.len() {
                    let (logits, trace) = model
                        .step_traced_for_path(
                            Tensor::from_data(
                                [inputs[time] as i32, other_inputs[time] as i32],
                                &device,
                            ),
                            &mut state,
                            ModelPath::Full,
                        )
                        .unwrap();
                    let target = Tensor::from_data(
                        [[targets[time] as i32], [other_targets[time] as i32]],
                        &device,
                    );
                    let mask =
                        Tensor::from_floats([[1.0], [if time < 3 { 1.0 } else { 0.0 }]], &device);
                    loss = loss
                        - (log_softmax(logits, 1).gather(1, target) * mask.clone()).sum() / 8.0;
                    if penalty.coefficient > 0.0 {
                        loss = loss
                            + (squared_voltage_excess(
                                trace.voltage,
                                model.config.threshold,
                                model.config.surrogate_width,
                                penalty.band_widths,
                            ) * mask)
                                .sum()
                                * (penalty.coefficient / 16.0);
                    }
                }
                let (scalar, frozen) =
                    scalar_loss(&p, &inputs, &targets, None, FeedbackMode::Both, penalty);
                let (other_scalar, other_frozen) = scalar_loss(
                    &p,
                    &other_inputs[..3],
                    &other_targets[..3],
                    None,
                    FeedbackMode::Both,
                    penalty,
                );
                close(
                    values(&loss)[0] as f64,
                    (scalar * 5.0 + other_scalar * 3.0) / 8.0,
                    3e-7,
                );
                assert_eq!(
                    frozen.iter().any(|step| step.slope.contains(&0.0)),
                    inactive
                );
                let gradients = loss.backward();
                let mut analytic = Vec::new();
                for (_, tensor) in model.parameter_tensors() {
                    analytic.extend(values(
                        &tensor.grad(&gradients).expect("parameter gradient exists"),
                    ));
                }
                for index in 0..p.len() {
                    let epsilon = 1e-5;
                    let mut plus = p.clone();
                    let mut minus = p.clone();
                    plus[index] += epsilon;
                    minus[index] -= epsilon;
                    let tangent = |p: &[f64]| {
                        (scalar_loss(
                            p,
                            &inputs,
                            &targets,
                            Some(&frozen),
                            FeedbackMode::Both,
                            penalty,
                        )
                        .0 * 5.0
                            + scalar_loss(
                                p,
                                &other_inputs[..3],
                                &other_targets[..3],
                                Some(&other_frozen),
                                FeedbackMode::Both,
                                penalty,
                            )
                            .0 * 3.0)
                            / 8.0
                    };
                    let numerical = (tangent(&plus) - tangent(&minus)) / (2.0 * epsilon);
                    close(analytic[index] as f64, numerical, 3e-6);
                }
            }
        }
    }
}
