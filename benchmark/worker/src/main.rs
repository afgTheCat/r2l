//! Worker process for a single benchmark task.

use std::{env::var, process::Command};

use anyhow::{Context, bail};
use r2l::{LearningRateSchedule, PPOBuilder, TrainingArtifactsConfig, TrainingLimit};
use r2l_benchmark_task::{Backend, BenchmarkTask, RlZooSchedule};

const SB3_SCRIPT_PATH: &str = "/opt/r2l/sb3/ppo.py";
const TASK_ENV_VAR: &str = "R2L_TASK";

fn run(task: &BenchmarkTask) -> anyhow::Result<()> {
    match task.backend {
        Backend::Burn | Backend::Candle => train_r2l(task),
        Backend::Sb3 => train_sb3(task),
    }
}

fn train_r2l(task: &BenchmarkTask) -> anyhow::Result<()> {
    let config = &task.rl_zoo_env_config;
    let obs_clip = config.normalize.norm_obs().then_some(10.0);
    let artifacts_config = TrainingArtifactsConfig::new(&task.output_dir);
    let mut builder = PPOBuilder::gym(task.env_name.clone(), config.n_envs)?
        .with_rollout_steps(config.n_steps)
        .with_training_limit(TrainingLimit::steps(config.n_timesteps))
        .with_training_artifacts(artifacts_config)
        .with_observation_normalizer(obs_clip)?
        .with_lambda(config.gae_lambda)
        .with_gamma(config.gamma)
        .with_total_epochs(config.n_epochs)
        .with_entropy_coefficient(config.ent_coef)
        .with_sample_size(config.batch_size)
        .with_learning_rate_schedule(Some(learning_rate_schedule(config.learning_rate)))
        .with_clip_range(config.clip_range.initial_value() as f32)
        .with_log_std_init(config.log_std_init)
        .with_value_loss_coefficient(Some(config.vf_coef))
        .with_seed(0) // TODO: should we keep this?
        .with_gradient_clipping(Some(config.max_grad_norm));
    if config.normalize.norm_reward() {
        builder = builder.with_reward_normalizer(config.gamma, 10.0);
    }
    match task.backend {
        Backend::Burn => builder.with_burn().build()?.train()?,
        Backend::Candle => builder.build()?.train()?,
        Backend::Sb3 => unreachable!(),
    }
    Ok(())
}

fn train_sb3(task: &BenchmarkTask) -> anyhow::Result<()> {
    let status = Command::new("python")
        .arg(SB3_SCRIPT_PATH)
        .arg(&task.env_name)
        .status()
        .with_context(|| format!("failed to run {SB3_SCRIPT_PATH}"))?;
    if !status.success() {
        bail!("SB3 evaluation for {} exited with {status}", task.env_name);
    }
    Ok(())
}

fn learning_rate_schedule(schedule: RlZooSchedule) -> LearningRateSchedule {
    match schedule {
        RlZooSchedule::Constant(value) => LearningRateSchedule::Constant(value),
        RlZooSchedule::Linear(value) => LearningRateSchedule::Linear(value),
    }
}

fn main() -> anyhow::Result<()> {
    let task = var(TASK_ENV_VAR).context(format!("{TASK_ENV_VAR} was not set"))?;
    let task: BenchmarkTask = serde_json::from_str(&task)
        .context(format!("{TASK_ENV_VAR} was not a valid task specification"))?;
    run(&task)
}
