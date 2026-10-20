//! A model-only checkpoint: configuration, Unicode vocabulary, and Burn weights.
//! The f64 wire envelope exactly stores the model's f32 tensor values.
//! Prior model-only artifacts are imported into the current model: missing
//! semantic weights and the trace readout are zero, and their fixed decays
//! become uniform decay logits. This preserves predictions (up to f32 sigmoid
//! rounding of the decays) without a second runtime model.
//! Optimizer moments and the random-window sampler are not persisted.

use burn::tensor::Device;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::data::Tokenizer;
use crate::model::{FeedbackMode, MAX_DECAY, Model, ModelConfig, decay_logit};

/// Header: vocabulary (base characters + merges), hidden size, threshold,
/// surrogate width, feedback mode, merge count. Payload: base characters as u32,
/// merges as u32 token pairs in rank order, parameters as f64, checksum.
const MAGIC: &[u8; 8] = b"LIMINAL\x03";
/// Learned decays, character vocabulary only (no merge count or merges).
const CHARACTER_MAGIC: &[u8; 8] = b"LIMINAL\x02";
/// Fixed membrane decay in the header, fixed trace/memory decays, 11 groups.
const FIXED_DECAY_MAGIC: &[u8; 8] = b"LIMINAL\x01";
/// Fixed membrane decay, no feedback mode, and only the first five groups.
const PRE_SEMANTIC_MAGIC: &[u8; 8] = b"LIMINAL\0";
const PRE_SEMANTIC_FIXED_BYTES: usize = 8 + 8 + 8 + 3 * 8 + 8;
const FIXED_DECAY_FIXED_BYTES: usize = PRE_SEMANTIC_FIXED_BYTES + 8;
const CHARACTER_FIXED_BYTES: usize = 8 + 8 + 8 + 2 * 8 + 8 + 8;
const FIXED_BYTES: usize = CHARACTER_FIXED_BYTES + 8;
/// Trace and memory decays every pre-learned-decay artifact was trained with.
const FIXED_TRACE_DECAY: f64 = 0.8;
const FIXED_SEMANTIC_DECAY: f64 = 0.9;

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Current,
    Character,
    FixedDecay,
    PreSemantic,
}

pub fn save(path: &Path, model: &Model, tokenizer: &Tokenizer) -> Result<(), String> {
    model.validate()?;
    if tokenizer.vocab_size() != model.config.vocab_size {
        return Err("checkpoint vocabulary does not match the model".into());
    }
    let parameters = model.parameter_values()?;
    if parameters.iter().any(|value| !value.is_finite()) {
        return Err("cannot save non-finite model parameters".into());
    }
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(model.config.vocab_size as u64).to_le_bytes());
    bytes.extend_from_slice(&(model.config.hidden_size as u64).to_le_bytes());
    for value in [model.config.threshold, model.config.surrogate_width] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mode: u64 = match model.feedback {
        FeedbackMode::Both => 0,
        FeedbackMode::TopDown => 1,
        FeedbackMode::BottomUp => 2,
        FeedbackMode::Neither => 3,
    };
    bytes.extend_from_slice(&mode.to_le_bytes());
    bytes.extend_from_slice(&(tokenizer.merges().len() as u64).to_le_bytes());
    for &character in tokenizer.vocab() {
        bytes.extend_from_slice(&(character as u32).to_le_bytes());
    }
    for &(left, right) in tokenizer.merges() {
        for token in [left, right] {
            let token = u32::try_from(token).map_err(|_| "merge token exceeds u32")?;
            bytes.extend_from_slice(&token.to_le_bytes());
        }
    }
    for parameter in parameters {
        bytes.extend_from_slice(&f64::from(parameter).to_le_bytes());
    }
    let checksum = checksum(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());

    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create checkpoint directory: {error}"))?;
    }
    let temporary = temporary_path(path)?;
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("cannot create temporary checkpoint: {error}"))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("cannot write checkpoint: {error}"))?;
        drop(file);
        fs::rename(&temporary, path)
            .map_err(|error| format!("cannot finish checkpoint {}: {error}", path.display()))
    })();
    if result.is_err() {
        // Best-effort cleanup of only the uniquely named temporary file.
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn load(path: &Path, device: &Device) -> Result<(Model, Tokenizer), String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read checkpoint {}: {error}", path.display()))?;
    decode(&bytes, device)
}

fn decode(bytes: &[u8], device: &Device) -> Result<(Model, Tokenizer), String> {
    let format = match bytes.get(..8) {
        _ if bytes.len() < CHARACTER_FIXED_BYTES.min(PRE_SEMANTIC_FIXED_BYTES) => None,
        Some(magic) if magic == MAGIC => Some(Format::Current),
        Some(magic) if magic == CHARACTER_MAGIC => Some(Format::Character),
        Some(magic) if magic == FIXED_DECAY_MAGIC => Some(Format::FixedDecay),
        Some(magic) if magic == PRE_SEMANTIC_MAGIC => Some(Format::PreSemantic),
        _ => None,
    }
    .ok_or("not a Liminal checkpoint or checkpoint is truncated")?;
    let payload_end = bytes.len() - 8;
    let stored_checksum = u64::from_le_bytes(
        bytes[payload_end..]
            .try_into()
            .map_err(|_| "checkpoint checksum is truncated")?,
    );
    if checksum(&bytes[..payload_end]) != stored_checksum {
        return Err("checkpoint checksum mismatch".into());
    }
    let mut reader = Reader {
        bytes: &bytes[..payload_end],
        offset: 8,
    };
    let vocab_size = usize::try_from(reader.u64()?).map_err(|_| "vocabulary size is too large")?;
    let hidden_size = usize::try_from(reader.u64()?).map_err(|_| "hidden size is too large")?;
    let fixed_decay = if matches!(format, Format::Current | Format::Character) {
        None
    } else {
        let decay = reader.f64()?;
        if !(decay > 0.0 && decay < MAX_DECAY) {
            return Err(format!(
                "imported membrane decay must lie in (0, {MAX_DECAY})"
            ));
        }
        Some(decay)
    };
    let config = ModelConfig {
        vocab_size,
        hidden_size,
        threshold: reader.f64()?,
        surrogate_width: reader.f64()?,
    };
    config.validate()?;
    let feedback = if format == Format::PreSemantic {
        FeedbackMode::Both
    } else {
        match reader.u64()? {
            0 => FeedbackMode::Both,
            1 => FeedbackMode::TopDown,
            2 => FeedbackMode::BottomUp,
            3 => FeedbackMode::Neither,
            _ => return Err("checkpoint contains an invalid feedback mode".into()),
        }
    };
    let merge_count = if format == Format::Current {
        usize::try_from(reader.u64()?).map_err(|_| "merge count is too large")?
    } else {
        0
    };
    let characters = vocab_size
        .checked_sub(merge_count)
        .filter(|&characters| characters > 0)
        .ok_or("checkpoint merges exceed its vocabulary")?;
    let (count, fixed_bytes) = match format {
        Format::Current => (config.checked_parameter_count()?, FIXED_BYTES),
        Format::Character => (config.checked_parameter_count()?, CHARACTER_FIXED_BYTES),
        Format::FixedDecay => (fixed_decay_parameters(&config)?, FIXED_DECAY_FIXED_BYTES),
        Format::PreSemantic => (pre_semantic_parameters(&config)?, PRE_SEMANTIC_FIXED_BYTES),
    };
    let expected_length = characters
        .checked_mul(4)
        .and_then(|size| size.checked_add(merge_count.checked_mul(8)?))
        .and_then(|size| size.checked_add(count.checked_mul(8)?))
        .and_then(|size| size.checked_add(fixed_bytes))
        .ok_or("checkpoint dimensions overflow")?;
    // Validate the exact payload size before allocations from stored dimensions.
    if bytes.len() != expected_length {
        return Err("checkpoint dimensions do not match its payload length".into());
    }
    let mut vocabulary = Vec::with_capacity(characters);
    for _ in 0..characters {
        vocabulary.push(
            char::from_u32(reader.u32()?).ok_or("checkpoint contains an invalid Unicode scalar")?,
        );
    }
    let mut merges = Vec::with_capacity(merge_count);
    for _ in 0..merge_count {
        merges.push((reader.u32()? as usize, reader.u32()? as usize));
    }
    let tokenizer = Tokenizer::from_parts(vocabulary, merges)?;
    let mut parameters = Vec::with_capacity(count);
    for _ in 0..count {
        parameters.push(reader.f64()? as f32);
    }
    if let Some(decay) = fixed_decay {
        // Missing semantic groups are zero; then uniform learned-decay logits
        // reproduce the fixed decays, and a zero trace readout adds nothing.
        parameters.resize(fixed_decay_parameters(&config)?, 0.0);
        for value in [decay, FIXED_TRACE_DECAY, FIXED_SEMANTIC_DECAY] {
            parameters.extend(std::iter::repeat_n(decay_logit(value) as f32, hidden_size));
        }
        parameters.resize(config.checked_parameter_count()?, 0.0);
    }
    let mut model = Model::from_parameters(config, parameters, device)?;
    model.feedback = feedback;
    Ok((model, tokenizer))
}

/// The 11-group weight count of fixed-decay artifacts.
fn fixed_decay_parameters(config: &ModelConfig) -> Result<usize, String> {
    let (v, h) = (config.vocab_size, config.hidden_size);
    v.checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_add(h.checked_mul(h)?.checked_mul(5)?))
        .and_then(|n| n.checked_add(h.checked_mul(2)?))
        .and_then(|n| n.checked_add(v))
        .ok_or_else(|| "model dimensions overflow".into())
}

/// Only the prior artifact's weight count; no prior model execution is retained.
fn pre_semantic_parameters(config: &ModelConfig) -> Result<usize, String> {
    let vocabulary_hidden = config.vocab_size.checked_mul(config.hidden_size);
    let recurrent = config.hidden_size.checked_mul(config.hidden_size);
    vocabulary_hidden
        .and_then(|size| size.checked_mul(2))
        .and_then(|size| recurrent.and_then(|recurrent| size.checked_add(recurrent)))
        .and_then(|size| size.checked_add(config.hidden_size))
        .and_then(|size| size.checked_add(config.vocab_size))
        .ok_or_else(|| "model dimensions overflow".into())
}

fn checksum(bytes: &[u8]) -> u64 {
    // FNV-1a detects accidental damage; this is not cryptographic authentication.
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn temporary_path(path: &Path) -> Result<PathBuf, String> {
    let name = path.file_name().ok_or("checkpoint path needs a filename")?;
    let tick = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before the Unix epoch")?
        .as_nanos();
    let mut filename = name.to_os_string();
    filename.push(format!(".{}.{}.tmp", std::process::id(), tick));
    Ok(path.with_file_name(filename))
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or("checkpoint offset overflow")?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or("checkpoint is truncated")?
            .try_into()
            .map_err(|_| "checkpoint field has wrong length")?;
        self.offset = end;
        Ok(value)
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::State;
    use burn::module::Param;
    use burn::tensor::{Int, Tensor};

    #[test]
    fn checkpoint_round_trip_preserves_vocabulary_weights_and_predictions() {
        let device = Device::flex();
        let text = "hello world! é hello world";
        let tokenizer = Tokenizer::from_text(text)
            .with_merges(&[text.to_string()], 4)
            .unwrap();
        assert_eq!(tokenizer.merges().len(), 4);
        let mut model = Model::new(
            ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                hidden_size: 4,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            13,
            &device,
        )
        .unwrap();
        model.feedback = FeedbackMode::BottomUp;
        assert!(
            model
                .semantic_readout
                .val()
                .try_into_vec_as::<f32>()
                .unwrap()
                .iter()
                .any(|&value| value != 0.0)
        );
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-artifacts")
            .join(format!("checkpoint-{}.bin", std::process::id()));
        save(&path, &model, &tokenizer).unwrap();
        let (loaded, vocabulary) = load(&path, &device).unwrap();
        assert_eq!(vocabulary, tokenizer);
        assert_eq!(loaded.feedback, model.feedback);
        assert_eq!(
            model.parameter_values().unwrap(),
            loaded.parameter_values().unwrap()
        );
        let tokens = tokenizer.encode("hello");
        let mut first = State::new(1, 4, &device);
        let mut second = State::new(1, 4, &device);
        for token in tokens {
            assert_eq!(
                model
                    .step(
                        Tensor::<1, Int>::from_data([token as i32], &device),
                        &mut first
                    )
                    .unwrap()
                    .try_into_vec_as::<f32>()
                    .unwrap(),
                loaded
                    .step(
                        Tensor::<1, Int>::from_data([token as i32], &device),
                        &mut second
                    )
                    .unwrap()
                    .try_into_vec_as::<f32>()
                    .unwrap()
            );
        }
        // Replacing the same checkpoint is supported on the development platform.
        save(&path, &loaded, &vocabulary).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], MAGIC);
        // Reject invalid modes even when the envelope checksum is valid.
        let mut bad_mode = bytes.clone();
        bad_mode[40..48].copy_from_slice(&4_u64.to_le_bytes());
        let payload_end = bad_mode.len() - 8;
        let hash = checksum(&bad_mode[..payload_end]);
        bad_mode[payload_end..].copy_from_slice(&hash.to_le_bytes());
        assert!(
            decode(&bad_mode, &device)
                .unwrap_err()
                .contains("feedback mode")
        );
        bytes[24] ^= 1;
        assert!(decode(&bytes, &device).is_err());
        assert!(decode(&bytes[..12], &device).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn prior_wire_import_preserves_predictions_and_zeros_semantic_parameters() {
        let tokenizer = Tokenizer::from_text("ab");
        let decay = 0.65;
        let config = ModelConfig {
            vocab_size: 3,
            hidden_size: 2,
            threshold: 0.8,
            surrogate_width: 1.3,
        };
        let original = vec![
            0.7_f32, 0.3, -0.2, 0.9, 0.4, -0.1, 0.2, -0.15, 0.1, 0.25, 0.2, 0.15, 0.3, -0.4, -0.1,
            0.6, 0.2, 0.1, 0.05, -0.1, 0.02,
        ];
        let mut bytes = PRE_SEMANTIC_MAGIC.to_vec();
        bytes.extend_from_slice(&(config.vocab_size as u64).to_le_bytes());
        bytes.extend_from_slice(&(config.hidden_size as u64).to_le_bytes());
        for value in [decay, config.threshold, config.surrogate_width] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for &character in tokenizer.vocab() {
            bytes.extend_from_slice(&(character as u32).to_le_bytes());
        }
        for &parameter in &original {
            bytes.extend_from_slice(&f64::from(parameter).to_le_bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());

        let device = Device::flex();
        let (model, vocabulary) = decode(&bytes, &device).unwrap();
        assert_eq!(vocabulary.vocab(), tokenizer.vocab());
        assert_eq!(model.feedback, FeedbackMode::Both);
        // Groups: 21 prior weights, 24 zero semantic weights (to 45), six
        // decay logits (to 51), and a zero 3x2 trace readout (to 57).
        let parameters = model.parameter_values().unwrap();
        assert_eq!(&parameters[..original.len()], &original);
        assert!(
            parameters[original.len()..45]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(parameters[51..].iter().all(|&value| value == 0.0));
        let mut state = State::new(1, 2, &device);
        let mut membrane = [0.0_f32; 2];
        let mut spikes = [0.0_f32; 2];
        // A test-only two-neuron reference for the prior artifact's equations.
        for token in [0_usize, 1, 2, 0, 1, 0] {
            let previous_spikes = spikes;
            for neuron in 0..2 {
                let voltage = decay as f32 * membrane[neuron]
                    + original[token * 2 + neuron]
                    + original[6 + neuron * 2] * previous_spikes[0]
                    + original[7 + neuron * 2] * previous_spikes[1]
                    + original[10 + neuron];
                spikes[neuron] = if voltage >= config.threshold as f32 {
                    1.0
                } else {
                    0.0
                };
                membrane[neuron] = voltage - config.threshold as f32 * spikes[neuron];
            }
            let actual = model
                .step(
                    Tensor::<1, Int>::from_data([token as i32], &device),
                    &mut state,
                )
                .unwrap()
                .try_into_vec_as::<f32>()
                .unwrap();
            for symbol in 0..3 {
                let expected = original[12 + symbol * 2] * spikes[0]
                    + original[13 + symbol * 2] * spikes[1]
                    + original[18 + symbol];
                assert!((actual[symbol] - expected).abs() < 1e-7);
            }
            assert!(
                state
                    .semantic
                    .clone()
                    .try_into_vec_as::<f32>()
                    .unwrap()
                    .iter()
                    .all(|&value| value == 0.0)
            );
        }
    }

    #[test]
    fn fixed_decay_import_maps_decays_to_uniform_logits_and_zero_trace_readout() {
        let tokenizer = Tokenizer::from_text("ab");
        let (decay, threshold, width) = (0.9_f64, 1.0_f64, 1.0_f64);
        let stored: Vec<f32> = (0..45).map(|index| index as f32 / 100.0 - 0.2).collect();
        let mut bytes = FIXED_DECAY_MAGIC.to_vec();
        for value in [3_u64, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [decay, threshold, width] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&2_u64.to_le_bytes()); // bottom-up
        for &character in tokenizer.vocab() {
            bytes.extend_from_slice(&(character as u32).to_le_bytes());
        }
        for &parameter in &stored {
            bytes.extend_from_slice(&f64::from(parameter).to_le_bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());

        let (model, _) = decode(&bytes, &Device::flex()).unwrap();
        assert_eq!(model.feedback, FeedbackMode::BottomUp);
        let mut expected = stored;
        for value in [decay, FIXED_TRACE_DECAY, FIXED_SEMANTIC_DECAY] {
            expected.extend([decay_logit(value) as f32; 2]);
        }
        expected.extend([0.0; 6]);
        assert_eq!(model.parameter_values().unwrap(), expected);

        // A decay the learned parameterization cannot represent is rejected.
        bytes[24..32].copy_from_slice(&0.999_f64.to_le_bytes());
        let payload_end = bytes.len() - 8;
        let hash = checksum(&bytes[..payload_end]);
        bytes[payload_end..].copy_from_slice(&hash.to_le_bytes());
        assert!(
            decode(&bytes, &Device::flex())
                .unwrap_err()
                .contains("decay")
        );
    }

    #[test]
    fn character_format_without_merge_count_still_loads() {
        let device = Device::flex();
        let tokenizer = Tokenizer::from_text("ab");
        let model = Model::new(
            ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                hidden_size: 2,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            5,
            &device,
        )
        .unwrap();
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-artifacts")
            .join(format!("character-format-{}.bin", std::process::id()));
        save(&path, &model, &tokenizer).unwrap();
        let current = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        // Version 2 is version 3 without the merge-count field after the mode.
        let mut bytes = CHARACTER_MAGIC.to_vec();
        bytes.extend_from_slice(&current[8..48]);
        bytes.extend_from_slice(&current[56..current.len() - 8]);
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        let (loaded, vocabulary) = decode(&bytes, &device).unwrap();
        assert_eq!(vocabulary, tokenizer);
        assert_eq!(
            loaded.parameter_values().unwrap(),
            model.parameter_values().unwrap()
        );
    }

    #[test]
    fn invalid_checkpoint_does_not_panic() {
        for length in 0..100 {
            assert!(decode(&vec![0; length], &Device::flex()).is_err());
        }
    }

    #[test]
    fn save_rejects_mispartitioned_parameter_groups_before_writing() {
        let tokenizer = Tokenizer::from_text("ab");
        let device = Device::flex();
        let mut model = Model::new(
            ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                hidden_size: 2,
                threshold: 1.0,
                surrogate_width: 1.0,
            },
            1,
            &device,
        )
        .unwrap();
        let original_count = model.parameter_count();
        model.hidden_bias = Param::from_tensor(Tensor::zeros([1, 1], &device));
        model.output_bias =
            Param::from_tensor(Tensor::zeros([1, tokenizer.vocab_size() + 1], &device));
        assert_eq!(model.parameter_count(), original_count);
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-artifacts/invalid-parameters.bin");
        assert!(save(&path, &model, &tokenizer).is_err());
    }
}
