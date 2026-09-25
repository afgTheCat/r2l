use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    tensor::Tensor as BurnTensor,
};
use candle_core::{DType, Device, Tensor as CandleTensor};
use candle_nn::VarMap;
use r2l_core::networks::{MlpConfig, NetworkConfig};
use r2l_core::{
    env::Space,
    models::{ActivationFunction, Policy, ToSafetensors},
    tensor::{R2lTensor, VecTensor},
};
use r2l_distributions::learning_modules::burn_lm::BurnDistributionKind;
use r2l_distributions::{
    learning_modules::candle_lm::CandleDistributionKind,
    networks::{burn::NetworkKind, candle_mlp::Mlp, seeded_var_builder},
};
use safetensors::SafeTensors;

type BurnBatch = BurnTensor<NdArray, 2>;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    actual.iter().zip(expected).for_each(|(actual, expected)| {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    });
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert_eq!(remainder.len(), 0);
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn linear_parameters(bytes: &[u8], input_size: usize, output_size: usize) -> (Vec<f32>, Vec<f32>) {
    let tensors = SafeTensors::deserialize(bytes).unwrap();
    let weight = tensors
        .names()
        .iter()
        .find_map(|name| {
            name.ends_with("weight")
                .then(|| tensors.tensor(name).unwrap())
        })
        .unwrap();
    let bias = tensors
        .names()
        .iter()
        .find_map(|name| {
            name.ends_with("bias")
                .then(|| tensors.tensor(name).unwrap())
        })
        .unwrap();
    assert_eq!(
        weight.shape().iter().product::<usize>(),
        input_size * output_size
    );
    assert_eq!(bias.shape(), [output_size]);
    (f32_values(weight.data()), f32_values(bias.data()))
}

fn reference_logits(
    weights: &[f32],
    bias: &[f32],
    state: &[f32],
    weight_shape: &[usize],
) -> Vec<f32> {
    let input_size = state.len();
    let output_size = bias.len();
    if weight_shape == [output_size, input_size] {
        (0..output_size)
            .map(|output| {
                bias[output]
                    + (0..input_size)
                        .map(|input| weights[output * input_size + input] * state[input])
                        .sum::<f32>()
            })
            .collect()
    } else {
        assert_eq!(weight_shape, [input_size, output_size]);
        (0..output_size)
            .map(|output| {
                bias[output]
                    + (0..input_size)
                        .map(|input| weights[input * output_size + output] * state[input])
                        .sum::<f32>()
            })
            .collect()
    }
}

fn reference_log_probs_and_entropy(
    bytes: &[u8],
    states: &[Vec<f32>],
    actions: &[usize],
) -> (Vec<f32>, f32) {
    let tensors = SafeTensors::deserialize(bytes).unwrap();
    let weight = tensors
        .names()
        .iter()
        .find_map(|name| {
            name.ends_with("weight")
                .then(|| tensors.tensor(name).unwrap())
        })
        .unwrap();
    let (weights, bias) = linear_parameters(bytes, states[0].len(), 3);
    let mut entropy = 0.0;
    let log_probs = states
        .iter()
        .zip(actions)
        .map(|(state, action)| {
            let logits = reference_logits(&weights, &bias, state, weight.shape());
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp = logits
                .iter()
                .map(|logit| (logit - max).exp())
                .collect::<Vec<_>>();
            let sum = exp.iter().sum::<f32>();
            let probabilities = exp.iter().map(|value| value / sum).collect::<Vec<_>>();
            entropy += probabilities
                .iter()
                .filter(|probability| **probability > 0.0)
                .map(|probability| -probability * probability.ln())
                .sum::<f32>();
            probabilities[*action].ln()
        })
        .collect();
    (log_probs, entropy / states.len() as f32)
}

fn candle_distribution(space: Space<VecTensor>, log_std: f32) -> CandleDistributionKind {
    let varmap = VarMap::new();
    let vb = seeded_var_builder(&varmap, DType::F32, &Device::Cpu);
    CandleDistributionKind::from_space(
        space,
        &mut |prefix, width| Mlp::build(&[2, width], ActivationFunction::Tanh, &vb.pp(prefix)),
        &mut |prefix, width| {
            Ok(vb.pp(prefix).get_with_hints(
                (1, width),
                "log_std",
                candle_nn::Init::Const(f64::from(log_std)),
            )?)
        },
    )
    .unwrap()
}

fn burn_distribution(space: Space<VecTensor>, log_std: f32) -> BurnDistributionKind<NdArray> {
    let config = NetworkConfig::Mlp(MlpConfig {
        hidden_layers: vec![],
        activation: ActivationFunction::Tanh,
    });
    BurnDistributionKind::from_space(
        space,
        &mut |_, width| NetworkKind::build(&config, &[2].into(), width, &NdArrayDevice::Cpu),
        &mut |_, width| {
            Ok(burn::module::Param::from_tensor(BurnTensor::full(
                [1, width],
                log_std,
                &NdArrayDevice::Cpu,
            )))
        },
    )
    .unwrap()
}

#[test]
fn categorical_backends_match_independent_probability_calculations() {
    let states = vec![vec![1.0, -0.5], vec![-2.0, 0.25]];
    let actions = vec![0, 2];

    let candle = candle_distribution(Space::Discrete(3), 0.0);
    let candle_bytes = candle.to_safetensors().unwrap();
    let (expected_log_probs, expected_entropy) =
        reference_log_probs_and_entropy(&candle_bytes, &states, &actions);
    let candle_states = states
        .iter()
        .map(|state| CandleTensor::from_slice(state, (1, state.len()), &Device::Cpu).unwrap())
        .collect::<Vec<_>>();
    let candle_actions = actions
        .iter()
        .map(|action| CandleTensor::from_slice(&[*action as f32], (1, 1), &Device::Cpu).unwrap())
        .collect::<Vec<_>>();
    assert_close(
        &candle
            .log_probs(
                CandleTensor::cat(&candle_states, 0).unwrap(),
                CandleTensor::cat(&candle_actions, 0).unwrap(),
            )
            .unwrap()
            .to_vec()
            .unwrap(),
        &expected_log_probs,
        1e-5,
    );
    assert_close(
        &candle
            .entropy(CandleTensor::cat(&candle_states, 0).unwrap())
            .unwrap()
            .to_vec()
            .unwrap(),
        &[expected_entropy],
        1e-5,
    );

    let burn = burn_distribution(Space::Discrete(3), 0.0);
    let burn_bytes = burn.to_safetensors().unwrap();
    let (expected_log_probs, expected_entropy) =
        reference_log_probs_and_entropy(&burn_bytes, &states, &actions);
    let burn_states = states
        .iter()
        .map(|state| BurnBatch::from_slice_and_shape(state, vec![1, state.len()]).unwrap())
        .collect::<Vec<_>>();
    let burn_actions = actions
        .iter()
        .map(|action| BurnBatch::from_slice_and_shape(&[*action as f32], vec![1, 1]).unwrap())
        .collect::<Vec<_>>();
    assert_close(
        &burn
            .log_probs(
                BurnTensor::cat(burn_states.clone(), 0),
                BurnTensor::cat(burn_actions, 0),
            )
            .unwrap()
            .to_vec()
            .unwrap(),
        &expected_log_probs,
        1e-5,
    );
    assert_close(
        &burn
            .entropy(BurnTensor::cat(burn_states, 0))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[expected_entropy],
        1e-5,
    );
}

#[test]
fn diagonal_gaussian_backends_agree_on_std_and_entropy() {
    let action_space = Space::<VecTensor>::Box {
        min: None,
        max: None,
        shape: vec![2].into(),
    };
    let log_std = -0.7;
    let candle = candle_distribution(action_space.clone(), log_std);
    let burn = burn_distribution(action_space, log_std);
    let expected_entropy = 2.0 * (log_std + f32::midpoint((2.0 * std::f32::consts::PI).ln(), 1.0));

    assert_close(&[candle.std().unwrap().unwrap()], &[log_std.exp()], 1e-6);
    assert_close(&[burn.std().unwrap().unwrap()], &[log_std.exp()], 1e-6);
    assert_close(
        &candle
            .entropy(CandleTensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap()
            .to_vec()
            .unwrap(),
        &[expected_entropy],
        1e-5,
    );
    assert_close(
        &burn
            .entropy(BurnTensor::zeros([1, 2], &NdArrayDevice::Cpu))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[expected_entropy],
        1e-5,
    );
}
