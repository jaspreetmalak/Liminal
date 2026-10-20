//! Frozen, teacher-forced continuation scoring through the maintained model.

use crate::data::Tokenizer;
use crate::model::{Model, State};
use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData, activation::log_softmax};

const MAX_SPAN: usize = 256;
const MAX_BATCH_SIZE: usize = 64;

#[derive(Debug, Clone)]
pub(crate) struct Continuation {
    pub prefix: Vec<usize>,
    pub targets: Vec<usize>,
}

impl Continuation {
    /// Preserve the tokenizer's full-text segmentation across the scoring
    /// boundary. Controlled fixtures reject unknown characters and boundaries
    /// where adding an answer changes an already encoded prompt token.
    pub(crate) fn from_text(
        tokenizer: &Tokenizer,
        prefix: &str,
        targets: &str,
    ) -> Result<Self, String> {
        let full = tokenizer.encode(&format!("{prefix}{targets}"));
        let prefix = tokenizer.encode(prefix);
        if prefix.is_empty() || targets.is_empty() {
            return Err("controlled continuation needs a nonempty prefix and targets".into());
        }
        if prefix.contains(&0) || full.contains(&0) {
            return Err("controlled continuation contains an unknown character".into());
        }
        if !full.starts_with(&prefix) {
            return Err(
                "unstable BPE scoring boundary: the answer changes prompt tokenization; end the prefix at a complete chunk boundary".into(),
            );
        }
        let targets = full[prefix.len()..].to_vec();
        if targets.is_empty() {
            return Err("controlled continuation has no target tokens".into());
        }
        Ok(Self { prefix, targets })
    }
}

/// Returns summed target log probabilities, excluding all prefix predictions.
/// Each row starts from zero state; padding has no contribution to its score.
/// The last prefix input predicts the first target, and subsequent targets are
/// teacher forced. There is no sampling, gradient update, or state across rows.
pub(crate) fn score(
    model: &Model,
    rows: &[Continuation],
    batch_size: usize,
) -> Result<Vec<f64>, String> {
    if rows.is_empty() {
        return Err("continuation scoring needs at least one row".into());
    }
    if !(1..=MAX_BATCH_SIZE).contains(&batch_size) {
        return Err(format!(
            "continuation batch size must be in 1..={MAX_BATCH_SIZE}"
        ));
    }
    for (index, row) in rows.iter().enumerate() {
        if row.prefix.is_empty() || row.targets.is_empty() {
            return Err(format!(
                "continuation row {index} needs a prefix and targets"
            ));
        }
        let span = row
            .prefix
            .len()
            .checked_add(row.targets.len())
            .ok_or("continuation span overflow")?;
        if span > MAX_SPAN {
            return Err(format!(
                "continuation row {index} exceeds {MAX_SPAN} tokens including targets"
            ));
        }
        if row
            .prefix
            .iter()
            .chain(&row.targets)
            .any(|&token| token >= model.config.vocab_size || i32::try_from(token).is_err())
        {
            return Err(format!("continuation row {index} has an invalid token ID"));
        }
    }
    model.validate()?;
    let model = model.clone().valid();
    let device = model.device();
    let mut batches = Vec::with_capacity(rows.len().div_ceil(batch_size));
    for chunk in rows.chunks(batch_size) {
        let count = chunk.len();
        let steps = chunk
            .iter()
            .map(|row| row.prefix.len() + row.targets.len() - 1)
            .max()
            .ok_or("empty continuation batch")?;
        let mut state = State::new(count, model.config.hidden_size, &device);
        let mut total = Tensor::<2>::zeros([count, 1], &device);
        for time in 0..steps {
            let mut inputs = Vec::with_capacity(count);
            let mut targets = Vec::with_capacity(count);
            let mut mask = Vec::with_capacity(count);
            for row in chunk {
                let last_input = row.prefix.len() + row.targets.len() - 1;
                let input = if time < row.prefix.len() {
                    row.prefix[time]
                } else if time < last_input {
                    row.targets[time - row.prefix.len()]
                } else {
                    0
                };
                inputs.push(input as i32);
                let answer_start = row.prefix.len() - 1;
                if (answer_start..last_input).contains(&time) {
                    targets.push(row.targets[time - answer_start] as i32);
                    mask.push(1.0_f32);
                } else {
                    targets.push(0);
                    mask.push(0.0_f32);
                }
            }
            let logits = model.step(
                Tensor::<1, Int>::from_data(TensorData::new(inputs, [count]), &device),
                &mut state,
            )?;
            let targets =
                Tensor::<2, Int>::from_data(TensorData::new(targets, [count, 1]), &device);
            let mask = Tensor::<2>::from_data(TensorData::new(mask, [count, 1]), &device);
            total = total + log_softmax(logits, 1).gather(1, targets) * mask;
        }
        batches.push(total);
    }
    // One result transfer after all rows have been scored on the device.
    let values = Tensor::cat(batches, 0)
        .try_into_vec_as::<f32>()
        .map_err(|error| format!("reading continuation scores: {error}"))?;
    if values.len() != rows.len() || values.iter().any(|value| !value.is_finite()) {
        return Err("continuation scores must be finite and match the row count".into());
    }
    Ok(values.into_iter().map(f64::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Batch, ModelConfig};
    use burn::tensor::Device;

    #[test]
    fn controlled_text_preserves_complete_bpe_segmentation_at_a_delimiter() {
        let documents = vec!["word: word. word word word. word: word.".into()];
        let tokenizer = Tokenizer::from_documents(&documents)
            .with_merges(&documents, 12)
            .unwrap();
        assert!(!tokenizer.merges().is_empty());
        let row = Continuation::from_text(&tokenizer, "word:", " word.").unwrap();
        let full: Vec<_> = row.prefix.iter().chain(&row.targets).copied().collect();
        assert_eq!(full, tokenizer.encode("word: word."));
        assert_eq!(tokenizer.decode(&row.prefix).unwrap(), "word:");
        assert_eq!(tokenizer.decode(&row.targets).unwrap(), " word.");
        assert_eq!(tokenizer.decode(&full).unwrap(), "word: word.");
        for (prefix, targets) in [("word: ", "word."), ("wor", "d.")] {
            let error = Continuation::from_text(&tokenizer, prefix, targets).unwrap_err();
            assert!(error.contains("unstable BPE scoring boundary"), "{error}");
        }
    }

    #[test]
    fn controlled_character_text_keeps_original_scores_and_rejects_unknowns_or_empty_spans() {
        let tokenizer = Tokenizer::from_text("Ann wears red. Bob wears tan.");
        let prefix = "Ann wears";
        let targets = " red.";
        let row = Continuation::from_text(&tokenizer, prefix, targets).unwrap();
        assert_eq!(row.prefix, tokenizer.encode(prefix));
        assert_eq!(row.targets, tokenizer.encode(targets));
        for (prefix, targets) in [
            ("", " red."),
            ("Ann wears", ""),
            ("Ω", " red."),
            ("Ann", " Ω"),
        ] {
            assert!(Continuation::from_text(&tokenizer, prefix, targets).is_err());
        }
    }

    fn model() -> Model {
        Model::new(
            ModelConfig {
                vocab_size: 5,
                hidden_size: 3,
                threshold: 0.8,
                surrogate_width: 1.3,
            },
            42,
            &Device::flex(),
        )
        .unwrap()
    }

    fn rows() -> Vec<Continuation> {
        vec![
            Continuation {
                prefix: vec![1, 2, 3],
                targets: vec![4, 1],
            },
            Continuation {
                prefix: vec![3],
                targets: vec![2, 4, 1, 0],
            },
            Continuation {
                prefix: vec![4, 2],
                targets: vec![0],
            },
        ]
    }

    // Independent scoring reduction: a single row, no masks/padding/gather,
    // host f64 stable log-sum-exp, and an explicit prefix/answer boundary.
    fn scalar_log_probability(model: &Model, row: &Continuation) -> f64 {
        let model = model.clone().valid();
        let device = model.device();
        let mut state = State::new(1, model.config.hidden_size, &device);
        let mut logits = None;
        for &input in &row.prefix {
            logits = Some(
                model
                    .step(
                        Tensor::<1, Int>::from_data([input as i32], &device),
                        &mut state,
                    )
                    .unwrap(),
            );
        }
        let mut total = 0.0;
        for (index, &target) in row.targets.iter().enumerate() {
            let values = logits.take().unwrap().try_into_vec_as::<f32>().unwrap();
            let maximum = values
                .iter()
                .map(|&value| f64::from(value))
                .fold(f64::NEG_INFINITY, f64::max);
            let normalizer = maximum
                + values
                    .iter()
                    .map(|&value| (f64::from(value) - maximum).exp())
                    .sum::<f64>()
                    .ln();
            total += f64::from(values[target]) - normalizer;
            if index + 1 < row.targets.len() {
                logits = Some(
                    model
                        .step(
                            Tensor::<1, Int>::from_data([target as i32], &device),
                            &mut state,
                        )
                        .unwrap(),
                );
            }
        }
        total
    }

    #[test]
    fn variable_length_scores_match_independent_step_oracle() {
        let model = model();
        let rows = rows();
        let expected: Vec<_> = rows
            .iter()
            .map(|row| scalar_log_probability(&model, row))
            .collect();
        for batch_size in [1, 2, 3, 64] {
            let actual = score(&model, &rows, batch_size).unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).abs() < 3e-6, "{actual} != {expected}");
                assert!(*actual < 0.0);
            }
        }
    }

    #[test]
    fn summed_scores_equal_target_only_masked_production_loss() {
        let model = model();
        let rows = rows();
        // Written explicitly to catch accidentally scoring prefix predictions,
        // scoring the last target twice, or including right padding.
        let batch = Batch {
            inputs: Tensor::<2, Int>::from_data(
                [[1, 2, 3, 4], [3, 2, 4, 1], [4, 2, 0, 0]],
                &model.device(),
            ),
            targets: Tensor::<2, Int>::from_data(
                [[0, 0, 4, 1], [2, 4, 1, 0], [0, 0, 0, 0]],
                &model.device(),
            ),
            mask: Tensor::<2>::from_data(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                &model.device(),
            ),
            valid_targets: 7,
        };
        let loss = model
            .clone()
            .valid()
            .loss(&batch)
            .unwrap()
            .try_into_vec_as::<f32>()
            .unwrap()[0];
        let scored_loss = -score(&model, &rows, 2).unwrap().iter().sum::<f64>() / 7.0;
        assert!((f64::from(loss) - scored_loss).abs() < 2e-6);
    }

    #[test]
    fn rows_are_independent_of_order_duplicates_and_batch_boundaries() {
        let model = model();
        let before = model.parameter_values().unwrap();
        let mut rows = rows();
        // The first two rows are competing equal-length answers to one prefix.
        rows[1].prefix = rows[0].prefix.clone();
        rows[1].targets = vec![1, 4];
        let original = score(&model, &rows, 2).unwrap();
        rows.push(rows[0].clone());
        rows.reverse();
        let reversed = score(&model, &rows, 3).unwrap();
        for (actual, expected) in
            reversed
                .iter()
                .zip([original[0], original[2], original[1], original[0]])
        {
            assert!((actual - expected).abs() < 2e-6);
        }
        assert_eq!(before, model.parameter_values().unwrap());
    }

    #[test]
    fn scorer_validates_bounds_and_accepts_the_span_limit() {
        let model = model();
        let valid = rows();
        assert!(score(&model, &[], 1).is_err());
        for batch_size in [0, 65, usize::MAX] {
            assert!(score(&model, &valid, batch_size).is_err());
        }
        for row in [
            Continuation {
                prefix: vec![],
                targets: vec![1],
            },
            Continuation {
                prefix: vec![1],
                targets: vec![],
            },
            Continuation {
                prefix: vec![5],
                targets: vec![1],
            },
            Continuation {
                prefix: vec![1],
                targets: vec![usize::MAX],
            },
            Continuation {
                prefix: vec![1; 256],
                targets: vec![1],
            },
        ] {
            assert!(score(&model, &[row], 1).is_err());
        }
        let edge = Continuation {
            prefix: vec![1; 255],
            targets: vec![2],
        };
        assert!(score(&model, &[edge], 1).unwrap()[0].is_finite());
        let mut nonfinite = model.clone();
        let mut parameters = model.parameter_values().unwrap();
        parameters[0] = f32::NAN;
        // from_parameters rejects nonfinite values, so exercise validation of
        // an externally modified tensor instead.
        nonfinite.embedding = burn::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(parameters[..15].to_vec(), [5, 3]),
            &model.device(),
        ));
        assert!(score(&nonfinite, &valid, 1).is_err());
    }
}
