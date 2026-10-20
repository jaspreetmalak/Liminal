//! Bounded observations of the production text model, without parameter updates.
//! Output-component scores reuse the actual full-path states and are not
//! retrained controls or interventions on the recurrent dynamics.

use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::data::Tokenizer;
use crate::model::{Model, ModelPath, PARAMETER_NAMES, State};
use crate::training;

const MAX_WINDOWS: usize = 16;
const MAX_TARGETS_PER_WINDOW: usize = 64;

fn values<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>, String> {
    let values = tensor
        .try_into_vec_as::<f32>()
        .map_err(|error| error.to_string())?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err("text activity diagnostic encountered non-finite tensor values".into());
    }
    Ok(values)
}

fn parameter_hash(model: &Model) -> Result<String, String> {
    let bytes: Vec<_> = model
        .parameter_values()?
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[derive(Default)]
struct Magnitudes {
    count: usize,
    squares: f64,
    maximum_absolute: f64,
}

impl Magnitudes {
    fn add(&mut self, values: &[f32]) {
        self.count += values.len();
        for &value in values {
            self.squares += (value as f64).powi(2);
            self.maximum_absolute = self.maximum_absolute.max((value as f64).abs());
        }
    }

    fn json(&self) -> Value {
        json!({
            "values": self.count, "global_l2_norm": self.squares.sqrt(),
            "rms": (self.squares / self.count as f64).sqrt(),
            "maximum_absolute": self.maximum_absolute,
        })
    }
}

#[derive(Default)]
struct LogitMagnitudes {
    raw: Magnitudes,
    positions: usize,
    centered_squares: f64,
    per_position_norm_sum: f64,
}

impl LogitMagnitudes {
    fn add(&mut self, logits: &[f32]) {
        self.raw.add(logits);
        self.positions += 1;
        let mean = logits.iter().map(|&value| value as f64).sum::<f64>() / logits.len() as f64;
        self.centered_squares += logits
            .iter()
            .map(|&value| (value as f64 - mean).powi(2))
            .sum::<f64>();
        self.per_position_norm_sum += logits
            .iter()
            .map(|&value| (value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
    }

    fn json(&self) -> Value {
        let mut report = self.raw.json();
        report["centered_vocab_rms"] =
            json!((self.centered_squares / self.raw.count as f64).sqrt());
        report["mean_per_position_l2_norm"] =
            json!(self.per_position_norm_sum / self.positions as f64);
        report["positions"] = json!(self.positions);
        report
    }
}

#[derive(Default)]
struct Score {
    targets: usize,
    nll: f64,
    target_probability: f64,
    correct: usize,
}

impl Score {
    fn add(&mut self, logits: &[f32], target: usize) {
        let mut prediction = 0;
        for token in 1..logits.len() {
            if logits[token] > logits[prediction] {
                prediction = token;
            }
        }
        let maximum = logits[prediction] as f64;
        let log_partition = maximum
            + logits
                .iter()
                .map(|&value| (value as f64 - maximum).exp())
                .sum::<f64>()
                .ln();
        let loss = log_partition - logits[target] as f64;
        self.nll += loss;
        self.target_probability += (-loss).exp();
        self.correct += usize::from(prediction == target);
        self.targets += 1;
    }

    fn json(&self) -> Value {
        json!({
            "targets": self.targets, "mean_next_character_nll_nats": self.nll / self.targets as f64,
            "top_one_accuracy": self.correct as f64 / self.targets as f64,
            "mean_target_probability": self.target_probability / self.targets as f64,
            "temperature": 1.0,
        })
    }
}

fn gradient_diagnostic(model: &Model, windows: &[Vec<usize>]) -> Result<Value, String> {
    let chosen: Vec<_> = windows.iter().take(2).map(Vec::as_slice).collect();
    let length = chosen
        .iter()
        .map(|tokens| tokens.len() - 1)
        .max()
        .ok_or("no gradient windows")?;
    let model = model.clone().train();
    let batch = training::build_batch(&chosen, length, &model.device())?;
    let targets = batch.valid_targets;
    let loss = model.loss_for_path(&batch, ModelPath::Full)?;
    let nll = loss.clone().into_scalar::<f32>() as f64;
    if !nll.is_finite() {
        return Err("text activity gradient loss is non-finite".into());
    }
    let gradients = loss.backward();
    let parameters = model.parameter_tensors();
    if parameters.len() != PARAMETER_NAMES.len() {
        return Err("text activity parameter names no longer match model wire order".into());
    }
    let mut groups = Vec::new();
    let mut global_squares = 0.0;
    for ((_, tensor), name) in parameters.into_iter().zip(PARAMETER_NAMES) {
        let shape = tensor.dims();
        let gradient = values(
            tensor
                .grad(&gradients)
                .ok_or_else(|| format!("missing text activity gradient for {name}"))?,
        )?;
        let mut magnitudes = Magnitudes::default();
        magnitudes.add(&gradient);
        global_squares += magnitudes.squares;
        let zero_count = gradient.iter().filter(|&&value| value == 0.0).count();
        groups.push(json!({
            "name": name, "shape": shape, "gradient": magnitudes.json(),
            "exact_zero_values": zero_count,
            "exact_zero_fraction": zero_count as f64 / gradient.len() as f64,
        }));
    }
    Ok(json!({
        "windows": chosen.len(), "sequence_length": length, "valid_targets": targets,
        "window_indices": (0..chosen.len()).collect::<Vec<_>>(),
        "mean_next_character_nll_nats": nll, "global_gradient_l2_norm": global_squares.sqrt(),
        "parameter_groups": groups, "optimizer_updates": 0,
        "interpretation": "Gradients of the masked mean full-path next-character loss on the first at most two supplied windows. Repeated tokens aggregate into embedding rows, so these group norms do not identify credit at a particular input position or a temporal Jacobian.",
    }))
}

/// Every window is inputs plus its final shifted next-character target.
/// This bounded diagnostic uses the existing forward and backward equations.
pub(crate) fn run(
    model: &Model,
    tokenizer: &Tokenizer,
    windows: &[Vec<usize>],
) -> Result<Value, String> {
    if windows.is_empty() || windows.len() > MAX_WINDOWS {
        return Err("text activity diagnostic requires 1..=16 windows".into());
    }
    if tokenizer.vocab_size() != model.config.vocab_size {
        return Err("text activity tokenizer and model vocabularies differ".into());
    }
    for window in windows {
        if !(2..=MAX_TARGETS_PER_WINDOW + 1).contains(&window.len())
            || window
                .iter()
                .any(|&token| token >= model.config.vocab_size || i32::try_from(token).is_err())
        {
            return Err("text activity windows need 2..=65 in-vocabulary token IDs".into());
        }
    }
    model.validate()?;
    let before = parameter_hash(model)?;
    let inference = model.clone().valid();
    let device = inference.device();
    let hidden = inference.config.hidden_size;
    let vocabulary = inference.config.vocab_size;
    let threshold = inference.config.threshold as f32 as f64;
    let width = inference.config.surrogate_width as f32 as f64;
    let mut positions = 0;
    let mut voltage_min = f32::INFINITY;
    let mut voltage_max = f32::NEG_INFINITY;
    let mut neuron_firings = vec![0_usize; hidden];
    let mut support_count = 0;
    let mut slope_sum = 0.0;
    let mut memory_high_magnitude_count = 0;
    let mut membrane_magnitudes = Magnitudes::default();
    let mut trace_magnitudes = Magnitudes::default();
    let mut memory_magnitudes = Magnitudes::default();
    let mut component_magnitudes: [LogitMagnitudes; 4] =
        std::array::from_fn(|_| LogitMagnitudes::default());
    let mut scores: [Score; 4] = std::array::from_fn(|_| Score::default());
    let mut reconstruction_max_error = 0.0_f64;
    let mut unknown_inputs = 0;
    let mut unknown_targets = 0;
    for window in windows {
        let mut state = State::new(1, hidden, &device);
        for adjacent in window.windows(2) {
            let token = adjacent[0];
            let target = adjacent[1];
            unknown_inputs += usize::from(token == 0);
            unknown_targets += usize::from(target == 0);
            let (logits, trace) = inference.step_traced_for_path(
                Tensor::<1, Int>::from_data(TensorData::new(vec![token as i32], [1]), &device),
                &mut state,
                ModelPath::Full,
            )?;
            // These projections use the actual full-path states. In particular,
            // do not use component paths, which also change feedback dynamics.
            // The spike projection includes the spike-trace readout P.
            let spike_logits = state
                .spikes
                .clone()
                .matmul(inference.readout.val().transpose())
                + state
                    .trace
                    .clone()
                    .matmul(inference.trace_readout.val().transpose());
            let memory_logits = state
                .semantic
                .clone()
                .matmul(inference.semantic_readout.val().transpose());
            let bias = inference.output_bias.val();
            let reconstructed = spike_logits.clone() + memory_logits.clone() + bias.clone();
            let logit_rows = values(Tensor::cat(
                vec![
                    logits,
                    spike_logits.clone() + bias.clone(),
                    memory_logits.clone() + bias.clone(),
                    bias.clone(),
                    spike_logits,
                    memory_logits,
                    reconstructed,
                ],
                0,
            ))?;
            for (score, row) in scores.iter_mut().zip(logit_rows.chunks(vocabulary).take(4)) {
                score.add(row, target);
            }
            for (magnitudes, row) in component_magnitudes.iter_mut().zip([
                &logit_rows[..vocabulary],
                &logit_rows[4 * vocabulary..5 * vocabulary],
                &logit_rows[5 * vocabulary..6 * vocabulary],
                &logit_rows[3 * vocabulary..4 * vocabulary],
            ]) {
                magnitudes.add(row);
            }
            for (&actual, &reconstructed) in logit_rows[..vocabulary]
                .iter()
                .zip(&logit_rows[6 * vocabulary..])
            {
                reconstruction_max_error =
                    reconstruction_max_error.max((actual as f64 - reconstructed as f64).abs());
            }
            let state_rows = values(Tensor::cat(
                vec![
                    trace.voltage,
                    state.spikes.clone(),
                    state.membrane.clone(),
                    state.trace.clone(),
                    state.semantic.clone(),
                ],
                0,
            ))?;
            for &voltage in &state_rows[..hidden] {
                voltage_min = voltage_min.min(voltage);
                voltage_max = voltage_max.max(voltage);
                let distance = (voltage as f64 - threshold).abs();
                support_count += usize::from(distance < width);
                slope_sum += (1.0 - distance / width).max(0.0) / width;
            }
            for (count, &spike) in neuron_firings
                .iter_mut()
                .zip(&state_rows[hidden..2 * hidden])
            {
                if spike != 0.0 && spike != 1.0 {
                    return Err("text activity observed a non-binary spike".into());
                }
                *count += usize::from(spike == 1.0);
            }
            membrane_magnitudes.add(&state_rows[2 * hidden..3 * hidden]);
            trace_magnitudes.add(&state_rows[3 * hidden..4 * hidden]);
            let memory = &state_rows[4 * hidden..];
            memory_magnitudes.add(memory);
            memory_high_magnitude_count +=
                memory.iter().filter(|&&value| value.abs() > 0.95).count();
            positions += 1;
        }
    }
    let gradients = gradient_diagnostic(model, windows)?;
    let after = parameter_hash(model)?;
    if before != after {
        return Err("text activity diagnostic unexpectedly changed parameters".into());
    }
    let total_hidden_values = positions * hidden;
    Ok(json!({
        "kind": "bounded production-model text activity diagnostic",
        "feedback_mode": model.feedback.as_str(), "model_path": "full",
        "windows": windows.len(), "window_target_counts": windows.iter().map(|window| window.len() - 1).collect::<Vec<_>>(),
        "scored_next_character_targets": positions, "hidden_size": hidden, "vocabulary_size": vocabulary,
        "unknown_input_tokens": unknown_inputs, "unknown_target_tokens": unknown_targets,
        "precision": "f32 forward/backward; f64 host summary arithmetic using rounded f32 threshold and width",
        "state_reset": "zero all states separately at each supplied window, without processing the final target as an input",
        "pre_reset_voltage_minimum": voltage_min, "pre_reset_voltage_maximum": voltage_max,
        "spikes": {
            "binary_forward_verified": true,
            "firing_fraction": neuron_firings.iter().sum::<usize>() as f64 / total_hidden_values as f64,
            "per_neuron_firing_fraction": neuron_firings.iter().map(|&count| count as f64 / positions as f64).collect::<Vec<_>>(),
            "never_firing_neuron_fraction": neuron_firings.iter().filter(|&&count| count == 0).count() as f64 / hidden as f64,
            "always_firing_neuron_fraction": neuron_firings.iter().filter(|&&count| count == positions).count() as f64 / hidden as f64,
            "neurons_firing_above_95_percent": neuron_firings.iter().filter(|&&count| (count as f64 / positions as f64) > 0.95).count(),
            "neurons_firing_below_5_percent": neuron_firings.iter().filter(|&&count| (count as f64 / positions as f64) < 0.05).count(),
        },
        "triangular_surrogate": {
            "threshold": threshold, "width": width,
            "strict_support_fraction": support_count as f64 / total_hidden_values as f64,
            "mean_slope_including_zeros": slope_sum / total_hidden_values as f64,
        },
        "state_magnitudes": {
            "membrane": membrane_magnitudes.json(), "trace": trace_magnitudes.json(),
            "memory": memory_magnitudes.json(),
            "memory_state_abs_above_0_95_fraction": memory_high_magnitude_count as f64 / total_hidden_values as f64,
            "interpretation": "High-magnitude occupancy of the leaky continuous memory state only; the tanh candidate is not exposed, so this is not a measurement of candidate saturation.",
        },
        "logit_components": {
            "full": component_magnitudes[0].json(), "spike_projection": component_magnitudes[1].json(),
            "memory_projection": component_magnitudes[2].json(), "output_bias": component_magnitudes[3].json(),
            "maximum_full_reconstruction_absolute_error": reconstruction_max_error,
            "centered_vocab_rms_definition": "For each scored position subtract that component's vocabulary mean, then take RMS across all vocabulary entries and scored positions; common offsets do not change softmax probabilities",
        },
        "same_state_output_scores": {
            "full": scores[0].json(), "spike_plus_bias": scores[1].json(),
            "memory_plus_bias": scores[2].json(), "bias_only": scores[3].json(),
            "interpretation": "Observational output recombinations O*s+o, D*c+o, and o on the same actual full-path states. Feedback and state dynamics stay unchanged. These are neither retrained baselines nor isolated causal effects on recurrent computation.",
        },
        "gradients": gradients,
        "parameters_f32_le_sha256_before": before, "parameters_f32_le_sha256_after": after,
        "parameters_unchanged": true, "optimizer_updates": 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use burn::tensor::Device;

    fn fixture() -> (Model, Tokenizer) {
        let tokenizer = Tokenizer::from_text("ab");
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
        (model, tokenizer)
    }

    #[test]
    fn observes_actual_states_scores_and_gradients_without_parameter_updates() {
        let (model, tokenizer) = fixture();
        let windows = vec![tokenizer.encode("abab"), tokenizer.encode("bab")];
        let before = parameter_hash(&model).unwrap();
        let report = run(&model, &tokenizer, &windows).unwrap();
        assert_eq!(report["scored_next_character_targets"], 5);
        assert_eq!(report["spikes"]["binary_forward_verified"], true);
        assert_eq!(
            report["spikes"]["per_neuron_firing_fraction"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            report["gradients"]["parameter_groups"]
                .as_array()
                .unwrap()
                .len(),
            PARAMETER_NAMES.len()
        );
        assert_eq!(report["gradients"]["valid_targets"], 5);
        assert_eq!(parameter_hash(&model).unwrap(), before);
        assert_eq!(report["parameters_unchanged"], true);
        assert_eq!(report["optimizer_updates"], 0);
        let samples: Vec<_> = windows.iter().map(Vec::as_slice).collect();
        let batch = training::build_batch(&samples, 3, &model.device()).unwrap();
        let expected = model
            .clone()
            .valid()
            .loss(&batch)
            .unwrap()
            .into_scalar::<f32>() as f64;
        let observed = report["same_state_output_scores"]["full"]["mean_next_character_nll_nats"]
            .as_f64()
            .unwrap();
        assert!((expected - observed).abs() < 5e-7);
        assert!(
            report["logit_components"]["maximum_full_reconstruction_absolute_error"]
                .as_f64()
                .unwrap()
                < 1e-6
        );
        assert_eq!(
            report["logit_components"]["output_bias"]["centered_vocab_rms"]
                .as_f64()
                .unwrap(),
            0.0
        );
        fn finite_numbers(value: &Value) {
            match value {
                Value::Number(number) => assert!(number.as_f64().unwrap().is_finite()),
                Value::Array(values) => values.iter().for_each(finite_numbers),
                Value::Object(values) => values.values().for_each(finite_numbers),
                _ => {}
            }
        }
        finite_numbers(&report);
    }

    #[test]
    fn rejects_unbounded_or_invalid_windows() {
        let (model, tokenizer) = fixture();
        for windows in [
            vec![],
            vec![vec![1]],
            vec![vec![1; 66]],
            vec![vec![3, 1]],
            vec![vec![1, 2]; 17],
        ] {
            assert!(run(&model, &tokenizer, &windows).is_err());
        }
        assert!(run(&model, &Tokenizer::from_text("abc"), &[vec![1, 2]]).is_err());
    }
}
