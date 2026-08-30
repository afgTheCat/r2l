mod support;

use std::{
    sync::{Arc, atomic::AtomicUsize, mpsc::channel},
    time::Duration,
};

use r2l_core::{env::EnvBuilder, on_policy::algorithm::Sampler, tensor::R2lTensor};
use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};
use support::{ConstantActor, EpisodeEnd, OneBoundHook, OwnedTrajectory, TestEnv};

fn env_builder(
    episode_len: usize,
    episode_end: EpisodeEnd,
    fail_at_step: Option<usize>,
    reset_count: Arc<AtomicUsize>,
) -> Arc<dyn EnvBuilder<Env = TestEnv>> {
    Arc::new(move || {
        Ok(TestEnv::new(
            episode_len,
            episode_end,
            fail_at_step,
            reset_count.clone(),
        ))
    })
}

fn direct_trajectories(
    execution_mode: SamplerExecutionMode,
    episode_end: EpisodeEnd,
) -> Vec<OwnedTrajectory> {
    let resets = Arc::new(AtomicUsize::new(0));
    let mut sampler = DirectSampler::build_from_env_builder(
        env_builder(2, episode_end, None, resets),
        2,
        OneBoundHook::steps(5),
        execution_mode,
    )
    .unwrap();
    sampler.collect_rollouts(ConstantActor).unwrap();
    sampler
        .trajectory_views()
        .as_ref()
        .iter()
        .map(OwnedTrajectory::from_view)
        .collect()
}

fn staged_trajectories(
    execution_mode: SamplerExecutionMode,
    episode_end: EpisodeEnd,
) -> Vec<OwnedTrajectory> {
    let resets = Arc::new(AtomicUsize::new(0));
    let mut sampler = StagedSampler::build_from_env_builder(
        env_builder(2, episode_end, None, resets),
        2,
        OneBoundHook::steps(5),
        execution_mode,
        None,
    )
    .unwrap();
    sampler.collect_rollouts(ConstantActor).unwrap();
    sampler
        .trajectory_views()
        .as_ref()
        .iter()
        .map(OwnedTrajectory::from_view)
        .collect()
}

fn expected_terminated_trajectory() -> OwnedTrajectory {
    OwnedTrajectory {
        states: vec![0.0, 1.0, 0.0, 1.0, 0.0],
        next_states: vec![1.0, 2.0, 1.0, 2.0, 1.0],
        rewards: vec![1.0, 2.0, 1.0, 2.0, 1.0],
        terminated: vec![false, true, false, true, false],
        truncated: vec![false; 5],
    }
}

#[test]
fn direct_sampler_resets_after_termination_in_both_execution_modes() {
    let expected = vec![
        expected_terminated_trajectory(),
        expected_terminated_trajectory(),
    ];
    assert_eq!(
        direct_trajectories(SamplerExecutionMode::SingleThreaded, EpisodeEnd::Terminated),
        expected
    );
    assert_eq!(
        direct_trajectories(SamplerExecutionMode::MultiThreaded, EpisodeEnd::Terminated),
        expected
    );
}

#[test]
fn staged_sampler_resets_after_termination_in_both_execution_modes() {
    let expected = vec![
        expected_terminated_trajectory(),
        expected_terminated_trajectory(),
    ];
    assert_eq!(
        staged_trajectories(SamplerExecutionMode::SingleThreaded, EpisodeEnd::Terminated),
        expected
    );
    assert_eq!(
        staged_trajectories(SamplerExecutionMode::MultiThreaded, EpisodeEnd::Terminated),
        expected
    );
}

#[test]
fn truncation_ends_and_resets_episodes_without_marking_termination() {
    for trajectories in [
        direct_trajectories(SamplerExecutionMode::SingleThreaded, EpisodeEnd::Truncated),
        direct_trajectories(SamplerExecutionMode::MultiThreaded, EpisodeEnd::Truncated),
        staged_trajectories(SamplerExecutionMode::SingleThreaded, EpisodeEnd::Truncated),
        staged_trajectories(SamplerExecutionMode::MultiThreaded, EpisodeEnd::Truncated),
    ] {
        for trajectory in &trajectories {
            assert_eq!(trajectory.states, [0.0, 1.0, 0.0, 1.0, 0.0]);
            assert_eq!(trajectory.terminated, [false; 5]);
            assert_eq!(trajectory.truncated, [false, true, false, true, false]);
        }
    }
}

#[test]
fn episode_bounds_collect_the_requested_number_of_complete_episodes() {
    for execution_mode in [
        SamplerExecutionMode::SingleThreaded,
        SamplerExecutionMode::MultiThreaded,
    ] {
        let resets = Arc::new(AtomicUsize::new(0));
        let mut sampler = DirectSampler::build_from_env_builder(
            env_builder(3, EpisodeEnd::Terminated, None, resets),
            2,
            OneBoundHook::episodes(2),
            execution_mode,
        )
        .unwrap();
        sampler.collect_rollouts(ConstantActor).unwrap();
        let trajectories = sampler.trajectory_views();
        for trajectory in trajectories.as_ref() {
            assert_eq!(trajectory.rewards.len(), 6);
            assert_eq!(trajectory.episode_terminations(), 2);
        }
        drop(trajectories);
    }
}

#[test]
fn reset_all_envs_clears_active_episode_state() {
    let resets = Arc::new(AtomicUsize::new(0));
    let mut sampler = DirectSampler::build_from_env_builder(
        env_builder(10, EpisodeEnd::Terminated, None, resets),
        1,
        OneBoundHook::steps(1),
        SamplerExecutionMode::MultiThreaded,
    )
    .unwrap();

    sampler.collect_rollouts(ConstantActor).unwrap();
    sampler.reset_all_envs().unwrap();
    sampler.collect_rollouts(ConstantActor).unwrap();
    let trajectories = sampler.trajectory_views();
    assert_eq!(trajectories.as_ref()[0].states[0].to_vec().unwrap(), [0.0]);
    drop(trajectories);
}

#[test]
fn environment_errors_propagate_instead_of_being_silently_ignored() {
    for execution_mode in [
        SamplerExecutionMode::SingleThreaded,
        SamplerExecutionMode::MultiThreaded,
    ] {
        let resets = Arc::new(AtomicUsize::new(0));
        let mut sampler = DirectSampler::build_from_env_builder(
            env_builder(10, EpisodeEnd::Terminated, Some(2), resets),
            2,
            OneBoundHook::steps(3),
            execution_mode,
        )
        .unwrap();
        assert!(sampler.collect_rollouts(ConstantActor).is_err());
    }
}

#[test]
fn dropping_multithreaded_samplers_releases_worker_environments() {
    for staged in [false, true] {
        let (drop_tx, drop_rx) = channel();
        let resets = Arc::new(AtomicUsize::new(0));
        let builder: Arc<dyn EnvBuilder<Env = TestEnv>> = Arc::new(move || {
            Ok(
                TestEnv::new(2, EpisodeEnd::Terminated, None, resets.clone())
                    .with_drop_notifier(drop_tx.clone()),
            )
        });

        if staged {
            let sampler = StagedSampler::build_from_env_builder(
                builder,
                2,
                OneBoundHook::steps(1),
                SamplerExecutionMode::MultiThreaded,
                None,
            )
            .unwrap();
            drop(sampler);
        } else {
            let sampler = DirectSampler::build_from_env_builder(
                builder,
                2,
                OneBoundHook::steps(1),
                SamplerExecutionMode::MultiThreaded,
            )
            .unwrap();
            drop(sampler);
        }

        for _ in 0..2 {
            drop_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("worker environment was not released when its sampler was dropped");
        }
    }
}
