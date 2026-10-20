//! Explicit hardware check; normal tests use the same Burn model on Flex CPU.

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};
use liminal::model::{
    Batch, Model, ModelConfig, ModelPath, State, VoltagePenalty, surrogate_spike,
};

fn config() -> ModelConfig {
    ModelConfig {
        vocab_size: 3,
        hidden_size: 2,
        threshold: 0.8,
        surrogate_width: 1.3,
    }
}

fn batch(device: &Device) -> Batch {
    Batch {
        inputs: Tensor::<2, Int>::from_data(
            TensorData::new(vec![0_i32, 1, 2, 0, 1, 2, 0, 0], [2, 4]),
            device,
        ),
        targets: Tensor::<2, Int>::from_data(
            TensorData::new(vec![1_i32, 2, 0, 1, 2, 0, 0, 0], [2, 4]),
            device,
        ),
        mask: Tensor::from_data([[1.0_f32, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], device),
        valid_targets: 6,
    }
}

fn loss_and_gradients(model: Model, device: &Device, path: ModelPath) -> (f32, Vec<f32>) {
    let model = model.train();
    let loss = model.loss_for_path(&batch(device), path).unwrap();
    let value = loss.clone().try_into_vec_as::<f32>().unwrap()[0];
    let gradients = loss.backward();
    let mut flat = Vec::new();
    for (_, parameter) in model.parameter_tensors() {
        flat.extend(
            parameter
                .grad(&gradients)
                .unwrap()
                .try_into_vec_as::<f32>()
                .unwrap(),
        );
    }
    (value, flat)
}

fn regularized_loss_and_gradients(model: Model, device: &Device) -> (Vec<f32>, Vec<f32>) {
    let model = model.train();
    // A narrow test band makes the auxiliary term active in this small fixture.
    let terms = model
        .training_loss_for_path(
            &batch(device),
            ModelPath::Full,
            VoltagePenalty {
                coefficient: 0.17,
                band_widths: 0.25,
            },
        )
        .unwrap();
    let values = Tensor::cat(
        vec![
            terms.cross_entropy,
            terms.voltage_penalty,
            terms.total.clone(),
        ],
        0,
    )
    .try_into_vec_as::<f32>()
    .unwrap();
    let gradients = terms.total.backward();
    let mut flat = Vec::new();
    for (_, parameter) in model.parameter_tensors() {
        flat.extend(
            parameter
                .grad(&gradients)
                .unwrap()
                .try_into_vec_as::<f32>()
                .unwrap(),
        );
    }
    (values, flat)
}

#[test]
#[ignore = "requires a discrete GPU; run explicitly with cargo test --test gpu -- --ignored --nocapture"]
fn wgpu_matches_cpu_for_binary_spikes_masked_loss_and_all_gradients() {
    let gpu = liminal::runtime::device("gpu").unwrap();
    println!("Actual GPU identity: {:?}", gpu.identity());
    let cpu = Device::flex();
    let mut parameters = vec![
        0.7, 0.3, -0.2, 0.9, 0.4, -0.1, 0.2, -0.15, 0.1, 0.25, 0.2, 0.15, 0.3, -0.4, -0.1, 0.6,
        0.2, 0.1, 0.05, -0.1, 0.02,
    ];
    // Exercise both feedback directions and the semantic readout, rather than
    // merely matching a current model with all new pathways disabled by zeros.
    parameters.extend([
        0.18, -0.07, 0.11, 0.22, // token embedding -> semantic state
        0.21, 0.08, -0.13, 0.17, // previous semantic state -> semantic state
        0.24, -0.09, 0.14, 0.16, // spike trace -> semantic state
        0.15, 0.06, -0.12, 0.19, // previous semantic state -> spike current
        0.04, -0.03, // semantic bias
        0.2, -0.1, -0.14, 0.23, 0.08, 0.16, // semantic readout
        1.2, 0.4, // membrane decay logits (distinct per neuron)
        1.6, 0.9, // spike trace decay logits
        2.1, 1.7, // memory decay logits
        0.1, -0.3, 0.25, 0.05, -0.2, 0.35, // spike trace readout
    ]);
    let cpu_model = Model::from_parameters(config(), parameters.clone(), &cpu).unwrap();
    let gpu_model = Model::from_parameters(config(), parameters, &gpu).unwrap();

    for path in [
        ModelPath::Full,
        ModelPath::MemoryOnly,
        ModelPath::SpikesOnly,
    ] {
        let mut cpu_state = State::new(2, 2, &cpu);
        let mut gpu_state = State::new(2, 2, &gpu);
        for tokens in [[0, 1], [1, 2], [2, 0], [0, 1]] {
            let (cpu_logits, cpu_trace) = cpu_model
                .step_traced_for_path(
                    Tensor::<1, Int>::from_data(tokens, &cpu),
                    &mut cpu_state,
                    path,
                )
                .unwrap();
            let (gpu_logits, gpu_trace) = gpu_model
                .step_traced_for_path(
                    Tensor::<1, Int>::from_data(tokens, &gpu),
                    &mut gpu_state,
                    path,
                )
                .unwrap();
            let expected_spikes = cpu_state.spikes.clone().try_into_vec_as::<f32>().unwrap();
            let actual_spikes = gpu_state.spikes.clone().try_into_vec_as::<f32>().unwrap();
            assert!(
                actual_spikes
                    .iter()
                    .all(|&value| value == 0.0 || value == 1.0)
            );
            assert_eq!(expected_spikes, actual_spikes);
            for (expected, actual) in [
                (cpu_trace.voltage, gpu_trace.voltage),
                (cpu_state.membrane.clone(), gpu_state.membrane.clone()),
                (cpu_state.trace.clone(), gpu_state.trace.clone()),
                (cpu_state.semantic.clone(), gpu_state.semantic.clone()),
                (cpu_logits, gpu_logits),
            ] {
                for (expected, actual) in expected
                    .try_into_vec_as::<f32>()
                    .unwrap()
                    .into_iter()
                    .zip(actual.try_into_vec_as::<f32>().unwrap())
                {
                    assert!(
                        (expected - actual).abs() < 2e-5,
                        "{} state/logit parity: {expected}/{actual}",
                        path.as_str()
                    );
                }
            }
        }
    }
    // Exact threshold, center, support endpoints, and inactive regions on GPU.
    let differentiable = gpu.clone().autodiff();
    let voltage =
        Tensor::<2>::from_data([[-1.0_f32, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]], &differentiable)
            .require_grad();
    let spikes = surrogate_spike(voltage.clone(), 1.0, 1.0);
    assert_eq!(
        spikes.clone().try_into_vec_as::<f32>().unwrap(),
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    );
    let gradients = spikes.sum().backward();
    let derivative = voltage
        .grad(&gradients)
        .unwrap()
        .try_into_vec_as::<f32>()
        .unwrap();
    for (actual, expected) in derivative
        .into_iter()
        .zip([0.0_f32, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
    {
        assert!((actual - expected).abs() < 2e-6);
    }
    for path in [
        ModelPath::Full,
        ModelPath::MemoryOnly,
        ModelPath::SpikesOnly,
    ] {
        let (cpu_loss, cpu_gradient) = loss_and_gradients(cpu_model.clone(), &cpu, path);
        let (gpu_loss, gpu_gradient) = loss_and_gradients(gpu_model.clone(), &gpu, path);
        assert_eq!(cpu_gradient.len(), 57);
        assert_eq!(gpu_gradient.len(), cpu_gradient.len());
        // The fixture supplies nonzero active gradients, while component modes
        // retain exact zero gradients for their explicitly disconnected groups.
        for (index, range) in [
            0..6,
            6..10,
            10..12,
            12..18,
            18..21,
            21..25,
            25..29,
            29..33,
            33..37,
            37..39,
            39..45,
            45..47,
            47..49,
            49..51,
            51..57,
        ]
        .into_iter()
        .enumerate()
        {
            let disabled = match path {
                ModelPath::Full => false,
                ModelPath::MemoryOnly => matches!(index, 1 | 2 | 3 | 7 | 8 | 11 | 12 | 14),
                ModelPath::SpikesOnly => matches!(index, 5..=10 | 13),
            };
            if disabled {
                assert!(
                    cpu_gradient[range.clone()]
                        .iter()
                        .all(|&value| value == 0.0)
                );
                assert!(gpu_gradient[range].iter().all(|&value| value == 0.0));
            } else {
                assert!(
                    cpu_gradient[range.clone()]
                        .iter()
                        .any(|&value| value.abs() > 1e-9),
                    "{} active group {index} needs a gradient",
                    path.as_str()
                );
                assert!(gpu_gradient[range].iter().any(|&value| value.abs() > 1e-9));
            }
        }
        assert!((cpu_loss - gpu_loss).abs() < 2e-5);
        let maximum_error = cpu_gradient
            .iter()
            .zip(&gpu_gradient)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum_error < 3e-5, "GPU gradient error: {maximum_error}");
        println!(
            "{} masked CPU/GPU loss: {cpu_loss:.8}/{gpu_loss:.8}; maximum all-parameter gradient difference {maximum_error:e}",
            path.as_str()
        );
    }
    let (cpu_terms, cpu_gradient) = regularized_loss_and_gradients(cpu_model, &cpu);
    let (gpu_terms, gpu_gradient) = regularized_loss_and_gradients(gpu_model, &gpu);
    assert!(cpu_terms[1] > 0.0 && gpu_terms[1] > 0.0);
    assert!((cpu_terms[2] - cpu_terms[0] - 0.17 * cpu_terms[1]).abs() < 2e-6);
    assert_eq!(cpu_gradient.len(), 57);
    assert_eq!(gpu_gradient.len(), 57);
    for (expected, actual) in cpu_terms.iter().zip(&gpu_terms) {
        assert!((expected - actual).abs() < 2e-5);
    }
    let maximum_error = cpu_gradient
        .iter()
        .zip(&gpu_gradient)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum_error < 3e-5,
        "regularized GPU gradient error: {maximum_error}"
    );
    println!(
        "regularized CPU/GPU [CE,raw P,total]: {cpu_terms:?}/{gpu_terms:?}; maximum all-parameter gradient difference {maximum_error:e}"
    );
}
