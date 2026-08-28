use std::sync::{Mutex, mpsc};

use r2l::{LearningRateSchedule, PPOBuilder, PPOMinibatchStats, PPORolloutStats, TrainingLimit};

const SEED: u64 = 0;
static TRAINING_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug)]
enum LearningBackend {
    Burn,
    Candle,
}

#[derive(Clone, Copy)]
struct EnvironmentConfig {
    name: &'static str,
    n_envs: usize,
    n_timesteps: usize,
    n_steps: usize,
    batch_size: usize,
    gae_lambda: f32,
    gamma: f32,
    n_epochs: usize,
    learning_rate: LearningRateSchedule,
    clip_range: f32,
}

const CARTPOLE: EnvironmentConfig = EnvironmentConfig {
    name: "CartPole-v1",
    n_envs: 8,
    n_timesteps: 100_000,
    n_steps: 32,
    batch_size: 256,
    gae_lambda: 0.8,
    gamma: 0.98,
    n_epochs: 20,
    learning_rate: LearningRateSchedule::Linear(0.001),
    clip_range: 0.2,
};

const PENDULUM: EnvironmentConfig = EnvironmentConfig {
    name: "Pendulum-v1",
    n_envs: 4,
    n_timesteps: 100_000,
    n_steps: 1_024,
    batch_size: 64,
    gae_lambda: 0.95,
    gamma: 0.9,
    n_epochs: 10,
    learning_rate: LearningRateSchedule::Constant(0.001),
    clip_range: 0.2,
};

#[derive(Debug, PartialEq, Eq)]
struct RolloutStats {
    total_rollouts: Option<usize>,
    rollout_idx: usize,
    minibatches: Vec<MinibatchStats>,
    std: Option<u32>,
    average_reward: u32,
    learning_rate: u64,
    clip_range: u32,
}

impl From<PPORolloutStats> for RolloutStats {
    fn from(stats: PPORolloutStats) -> Self {
        Self {
            total_rollouts: stats.total_rollouts,
            rollout_idx: stats.rollout_idx,
            minibatches: stats
                .minibatch_stats
                .into_iter()
                .map(MinibatchStats::from)
                .collect(),
            std: stats.std.map(f32::to_bits),
            average_reward: stats.average_reward.to_bits(),
            learning_rate: stats.learning_rate.to_bits(),
            clip_range: stats.clip_range.to_bits(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct MinibatchStats {
    clip_fraction: u32,
    entropy_loss: u32,
    policy_loss: u32,
    approx_kl: u32,
    value_loss: u32,
}

impl From<PPOMinibatchStats> for MinibatchStats {
    fn from(stats: PPOMinibatchStats) -> Self {
        Self {
            clip_fraction: stats.clip_fraction.to_bits(),
            entropy_loss: stats.entropy_loss.to_bits(),
            policy_loss: stats.policy_loss.to_bits(),
            approx_kl: stats.approx_kl.to_bits(),
            value_loss: stats.value_loss.to_bits(),
        }
    }
}

fn learn(config: EnvironmentConfig, backend: LearningBackend) -> anyhow::Result<Vec<RolloutStats>> {
    let (reporter, reports) = mpsc::channel();
    let builder = PPOBuilder::gym(config.name, config.n_envs)?
        .with_rollout_steps(config.n_steps)
        .with_training_limit(TrainingLimit::steps(config.n_timesteps))
        .with_observation_normalizer(None)?
        .with_lambda(config.gae_lambda)
        .with_gamma(config.gamma)
        .with_total_epochs(config.n_epochs)
        .with_entropy_coefficient(0.0)
        .with_sample_size(config.batch_size)
        .with_learning_rate_schedule(Some(config.learning_rate))
        .with_clip_range(config.clip_range)
        .with_log_std_init(0.0)
        .with_value_loss_coefficient(Some(0.5))
        .with_seed(SEED)
        .with_gradient_clipping(Some(0.5))
        .with_rollout_reporter(Some(reporter))
        .with_log_progress(false);

    match backend {
        LearningBackend::Burn => builder.with_burn().build()?.train()?,
        LearningBackend::Candle => builder.build()?.train()?,
    }

    Ok(reports.into_iter().map(RolloutStats::from).collect())
}

fn assert_learning_is_deterministic(
    config: EnvironmentConfig,
    backend: LearningBackend,
) -> anyhow::Result<()> {
    let (first_run, second_run) = {
        let _training_guard = TRAINING_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first_run = learn(config, backend)?;
        let second_run = learn(config, backend)?;
        (first_run, second_run)
    };

    assert_eq!(
        first_run.len(),
        second_run.len(),
        "{backend:?} {} produced a different number of rollout reports",
        config.name,
    );
    for (rollout_idx, (first, second)) in first_run.iter().zip(second_run.iter()).enumerate() {
        assert_eq!(
            first.total_rollouts, second.total_rollouts,
            "{backend:?} {} reported a different rollout total at rollout {}",
            config.name, rollout_idx,
        );
        assert_eq!(
            first.rollout_idx, second.rollout_idx,
            "{backend:?} {} reported a different rollout index at rollout {}",
            config.name, rollout_idx,
        );
        assert_eq!(
            first.std, second.std,
            "{backend:?} {} reported a different standard deviation at rollout {}",
            config.name, rollout_idx,
        );
        assert_eq!(
            first.average_reward, second.average_reward,
            "{backend:?} {} reported a different average reward at rollout {}",
            config.name, rollout_idx,
        );
        assert_eq!(
            first.learning_rate, second.learning_rate,
            "{backend:?} {} reported a different learning rate at rollout {}",
            config.name, rollout_idx,
        );
        assert_eq!(
            first.clip_range, second.clip_range,
            "{backend:?} {} reported a different clip range at rollout {}",
            config.name, rollout_idx,
        );
        assert_eq!(
            first.minibatches.len(),
            second.minibatches.len(),
            "{backend:?} {} reported a different minibatch count at rollout {}",
            config.name,
            rollout_idx,
        );
        for (minibatch_idx, (first, second)) in first
            .minibatches
            .iter()
            .zip(second.minibatches.iter())
            .enumerate()
        {
            assert_eq!(
                first, second,
                "{backend:?} {} diverged at rollout {}, minibatch {}",
                config.name, rollout_idx, minibatch_idx,
            );
        }
    }

    Ok(())
}

#[test]
fn cartpole_learning_is_deterministic() -> anyhow::Result<()> {
    assert_learning_is_deterministic(CARTPOLE, LearningBackend::Burn)
}

#[test]
fn pendulum_learning_is_deterministic() -> anyhow::Result<()> {
    assert_learning_is_deterministic(PENDULUM, LearningBackend::Burn)
}

#[test]
fn cartpole_candle_learning_is_deterministic() -> anyhow::Result<()> {
    assert_learning_is_deterministic(CARTPOLE, LearningBackend::Candle)
}

#[test]
fn pendulum_candle_learning_is_deterministic() -> anyhow::Result<()> {
    assert_learning_is_deterministic(PENDULUM, LearningBackend::Candle)
}
