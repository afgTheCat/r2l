use r2l_agents::on_policy_algorithms::{Advantages, batches_advantages_and_returns};
use r2l_core::{
    buffers::{Memory, buffer::TrajectoryBuffer},
    error::Result,
    models::ValueFunction,
    tensor::{R2lTensor, VecTensor},
};

struct IdentityValueFunction;

impl ValueFunction for IdentityValueFunction {
    type Tensor = VecTensor;

    fn values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor> {
        let values = observations
            .iter()
            .map(|observation| observation.to_vec().map(|values| values[0]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(VecTensor::from_vec(values))
    }
}

fn transition(
    state_value: f32,
    next_value: f32,
    reward: f32,
    terminated: bool,
    truncated: bool,
) -> Memory<VecTensor> {
    Memory {
        state: VecTensor::from_vec(vec![state_value]),
        next_state: VecTensor::from_vec(vec![next_value]),
        action: VecTensor::from_vec(vec![0.0]),
        reward,
        terminated,
        truncated,
    }
}

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
fn gae_matches_hand_calculated_trajectory() -> Result<()> {
    let mut batch = TrajectoryBuffer::default();
    batch.push(transition(1.0, 2.0, 0.5, false, false));
    batch.push(transition(2.0, 3.0, 1.0, false, false));
    batch.push(transition(3.0, 4.0, -1.0, true, false));

    let view = batch.to_trajectory_view();
    let (advantages, returns) =
        batches_advantages_and_returns(&[view], &IdentityValueFunction, 0.9, 0.8, Clone::clone)?;

    assert_close(&advantages[0], &[0.4504, -1.18, -4.0]);
    assert_close(&returns[0], &[1.4504, 0.82, -1.0]);
    Ok(())
}

#[test]
fn lambda_zero_reduces_gae_to_one_step_td_errors() -> Result<()> {
    let mut batch = TrajectoryBuffer::default();
    batch.push(transition(1.0, 2.0, 0.5, false, false));
    batch.push(transition(2.0, 3.0, 1.0, false, false));

    let view = batch.to_trajectory_view();
    let (advantages, returns) =
        batches_advantages_and_returns(&[view], &IdentityValueFunction, 0.9, 0.0, Clone::clone)?;

    assert_close(&advantages[0], &[1.3, 1.7]);
    assert_close(&returns[0], &[2.3, 3.7]);
    Ok(())
}

#[test]
fn termination_drops_bootstrap_but_truncation_keeps_it() -> Result<()> {
    let mut terminated = TrajectoryBuffer::default();
    terminated.push(transition(2.0, 10.0, 1.0, true, false));
    let mut truncated = TrajectoryBuffer::default();
    truncated.push(transition(2.0, 10.0, 1.0, false, true));

    let views = [
        terminated.to_trajectory_view(),
        truncated.to_trajectory_view(),
    ];
    let (advantages, returns) =
        batches_advantages_and_returns(&views, &IdentityValueFunction, 0.5, 0.9, Clone::clone)?;

    assert_close(&advantages[0], &[-1.0]);
    assert_close(&returns[0], &[1.0]);
    assert_close(&advantages[1], &[4.0]);
    assert_close(&returns[1], &[6.0]);
    Ok(())
}

#[test]
fn empty_trajectory_is_rejected() {
    let batch = TrajectoryBuffer::<VecTensor>::default();
    let view = batch.to_trajectory_view();
    assert!(
        batches_advantages_and_returns(&[view], &IdentityValueFunction, 0.9, 0.8, Clone::clone,)
            .is_err()
    );
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
