use r2l_core::{
    running_mean::{RunningMeanStd, RunningMeanStdF32},
    tensor::{R2lTensor, VecTensor},
};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    actual.iter().zip(expected).for_each(|(actual, expected)| {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    });
}

#[test]
fn vector_running_statistics_match_direct_population_statistics() {
    let samples = [
        VecTensor::from_vec(vec![1.0, 2.0]),
        VecTensor::from_vec(vec![3.0, 4.0]),
        VecTensor::from_vec(vec![5.0, 6.0]),
    ];
    let mut stats = RunningMeanStd::<VecTensor>::new(vec![2]).unwrap();
    stats.update(&samples).unwrap();

    assert_close(&stats.mean.to_vec().unwrap(), &[3.0, 4.0]);
    assert_close(&stats.var.to_vec().unwrap(), &[8.0 / 3.0, 8.0 / 3.0]);
    assert_eq!(stats.count, 3.0);
}

#[test]
fn incremental_updates_match_one_combined_batch() {
    let samples = [
        VecTensor::from_vec(vec![1.0, 7.0]),
        VecTensor::from_vec(vec![2.0, 5.0]),
        VecTensor::from_vec(vec![8.0, 3.0]),
        VecTensor::from_vec(vec![9.0, 1.0]),
    ];
    let mut incremental = RunningMeanStd::<VecTensor>::new(vec![2]).unwrap();
    incremental.update(&samples[..2]).unwrap();
    incremental.update(&samples[2..]).unwrap();
    let mut combined = RunningMeanStd::<VecTensor>::new(vec![2]).unwrap();
    combined.update(&samples).unwrap();

    assert_close(
        &incremental.mean.to_vec().unwrap(),
        &combined.mean.to_vec().unwrap(),
    );
    assert_close(
        &incremental.var.to_vec().unwrap(),
        &combined.var.to_vec().unwrap(),
    );
    assert_eq!(incremental.count, combined.count);
}

#[test]
fn scalar_running_statistics_match_direct_population_statistics() {
    let mut stats = RunningMeanStdF32::with_epsilon(0.0);
    stats.update(&[1.0, 2.0]);
    stats.update(&[3.0, 4.0]);

    assert_close(&[stats.mean], &[2.5]);
    assert_close(&[stats.var], &[1.25]);
}

#[test]
fn empty_scalar_update_is_a_noop() {
    let mut stats = RunningMeanStdF32::with_epsilon(0.0);
    stats.update(&[]);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.var, 1.0);
}
