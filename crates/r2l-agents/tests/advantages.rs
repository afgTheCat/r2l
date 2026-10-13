use r2l_agents::on_policy_algorithms::Advantages;

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
fn advantage_normalization_spans_all_trajectory_batches() {
    let mut advantages = Advantages(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    advantages.normalize();

    let scale = 1.25_f32.sqrt();
    assert_close(
        &advantages.iter().flatten().copied().collect::<Vec<_>>(),
        &[-1.5 / scale, -0.5 / scale, 0.5 / scale, 1.5 / scale],
    );
}

#[test]
fn constant_advantages_normalize_to_zero() {
    let mut advantages = Advantages(vec![vec![3.0, 3.0], vec![3.0]]);
    advantages.normalize();
    assert_close(
        &advantages.iter().flatten().copied().collect::<Vec<_>>(),
        &[0.0, 0.0, 0.0],
    );
}
