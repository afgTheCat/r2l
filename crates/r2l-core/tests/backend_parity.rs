use burn::{backend::NdArray, tensor::Tensor as BurnTensor};
use candle_core::Tensor as CandleTensor;
use r2l_core::tensor::R2lTensor;

type BurnVector = BurnTensor<NdArray, 1>;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    actual.iter().zip(expected).for_each(|(actual, expected)| {
        let tolerance = 1e-5_f32.max(expected.abs() * 1e-5);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    });
}

fn candle(values: &[f32]) -> CandleTensor {
    <CandleTensor as R2lTensor>::from_slice_and_shape(values, vec![values.len()]).unwrap()
}

fn burn(values: &[f32]) -> BurnVector {
    <BurnVector as R2lTensor>::from_slice_and_shape(values, vec![values.len()]).unwrap()
}

#[test]
fn elementwise_tensor_operations_match_reference_and_each_other() {
    let left = [1.0, -2.0, 0.5, 4.0];
    let right = [3.0, 0.25, -2.0, 2.0];
    let candle_left = candle(&left);
    let candle_right = candle(&right);
    let burn_left = burn(&left);
    let burn_right = burn(&right);

    let cases = [
        (
            R2lTensor::add(&candle_left, &candle_right).unwrap(),
            R2lTensor::add(&burn_left, &burn_right)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![4.0, -1.75, -1.5, 6.0],
        ),
        (
            R2lTensor::sub(&candle_left, &candle_right).unwrap(),
            R2lTensor::sub(&burn_left, &burn_right)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![-2.0, -2.25, 2.5, 2.0],
        ),
        (
            R2lTensor::mul(&candle_left, &candle_right).unwrap(),
            R2lTensor::mul(&burn_left, &burn_right)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![3.0, -0.5, -1.0, 8.0],
        ),
        (
            R2lTensor::minimum(&candle_left, &candle_right).unwrap(),
            R2lTensor::minimum(&burn_left, &burn_right)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![1.0, -2.0, -2.0, 2.0],
        ),
    ];

    for (candle_result, burn_result, expected) in cases {
        let candle_result = candle_result.to_vec().unwrap();
        assert_close(&candle_result, &expected);
        assert_close(&burn_result, &expected);
        assert_close(&candle_result, &burn_result);
    }
}

#[test]
fn nonlinear_operations_and_reductions_have_backend_parity() {
    let values = [-2.0, -0.5, 1.0, 3.0];
    let candle = candle(&values);
    let burn = burn(&values);

    let candle_exp = R2lTensor::exp(&candle).unwrap().to_vec().unwrap();
    let burn_exp = R2lTensor::exp(&burn).unwrap().to_vec().unwrap();
    let expected_exp = values.map(f32::exp);
    assert_close(&candle_exp, &expected_exp);
    assert_close(&burn_exp, &expected_exp);

    let candle_clamped = R2lTensor::clamp(&candle, -1.0, 2.0)
        .unwrap()
        .to_vec()
        .unwrap();
    let burn_clamped = R2lTensor::clamp(&burn, -1.0, 2.0)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_close(&candle_clamped, &[-1.0, -0.5, 1.0, 2.0]);
    assert_close(&burn_clamped, &candle_clamped);

    let candle_mean = R2lTensor::mean(&candle).unwrap().to_vec().unwrap();
    let burn_mean = R2lTensor::mean(&burn).unwrap().to_vec().unwrap();
    assert_close(&candle_mean, &[0.375]);
    assert_close(&burn_mean, &candle_mean);
}

#[test]
fn aggregate_statistics_have_backend_parity() {
    let candle_samples = [
        candle(&[1.0, 2.0]),
        candle(&[3.0, 6.0]),
        candle(&[5.0, 4.0]),
    ];
    let burn_samples = [burn(&[1.0, 2.0]), burn(&[3.0, 6.0]), burn(&[5.0, 4.0])];

    let candle_mean = CandleTensor::mean_tensors(&candle_samples)
        .unwrap()
        .to_vec()
        .unwrap();
    let burn_mean = BurnVector::mean_tensors(&burn_samples)
        .unwrap()
        .to_vec()
        .unwrap();
    let candle_var = CandleTensor::var_tensors(&candle_samples)
        .unwrap()
        .to_vec()
        .unwrap();
    let burn_var = BurnVector::var_tensors(&burn_samples)
        .unwrap()
        .to_vec()
        .unwrap();

    assert_close(&candle_mean, &[3.0, 4.0]);
    assert_close(&burn_mean, &candle_mean);
    assert_close(&candle_var, &[8.0 / 3.0, 8.0 / 3.0]);
    assert_close(&burn_var, &candle_var);
}

#[test]
fn both_backends_reject_shape_mismatches() {
    let candle_left = candle(&[1.0, 2.0]);
    let candle_right = candle(&[1.0]);
    let burn_left = burn(&[1.0, 2.0]);
    let burn_right = burn(&[1.0]);

    assert!(R2lTensor::add(&candle_left, &candle_right).is_err());
    assert!(R2lTensor::add(&burn_left, &burn_right).is_err());
}
