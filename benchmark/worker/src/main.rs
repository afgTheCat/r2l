//! Worker process for a single benchmark task.

use std::{env::var, process::Command};

use anyhow::{Context, bail};
use r2l::{
    AdamWConfig, GradientClippingConfig, OptimizerConfig, PPOBuilder, TrainingArtifactsConfig,
    TrainingLimit,
};
use r2l_benchmark_task::{Backend, BenchmarkTask};

const SB3_SCRIPT_PATH: &str = "/opt/r2l/sb3/ppo.py";
const TASK_ENV_VAR: &str = "R2L_TASK";

fn train_r2l(task: &BenchmarkTask) -> anyhow::Result<()> {
    let config = &task.rl_zoo_env_config;
    let normalizer_config = config
        .normalize
        .to_normalizer_config()
        .with_clip(Some(10.0));
    let artifacts_config = TrainingArtifactsConfig::new(&task.output_dir);
    let mut builder = PPOBuilder::gym(task.env_name.clone(), config.n_envs)?
        .with_rollout_steps(config.n_steps)
        .with_training_limit(TrainingLimit::steps(config.n_timesteps))
        .with_training_artifacts(artifacts_config)
        .with_observation_normalizer(normalizer_config)
        .with_lambda(config.gae_lambda)
        .with_gamma(config.gamma)
        .with_total_epochs(config.n_epochs)
        .with_entropy_coefficient(config.ent_coef)
        .with_sample_size(config.batch_size)
        .with_optimizer(OptimizerConfig::Joint(AdamWConfig {
            learning_rate: config.learning_rate.into_learning_rate_schedule(),
            gradient_clipping: GradientClippingConfig::Norm(config.max_grad_norm),
            ..Default::default()
        }))
        .with_clip_range_schedule(config.clip_range.into_clip_range_schedule())
        .with_log_std_init(config.log_std_init)
        .with_value_loss_coefficient(config.vf_coef)
        .with_seed(0); // TODO: should we keep this?
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

fn run(task: &BenchmarkTask) -> anyhow::Result<()> {
    match task.backend {
        Backend::Burn | Backend::Candle => train_r2l(task),
        Backend::Sb3 => train_sb3(task),
    }
}

fn main() -> anyhow::Result<()> {
    let task = var(TASK_ENV_VAR).context(format!("{TASK_ENV_VAR} was not set"))?;
    let task: BenchmarkTask = serde_json::from_str(&task)
        .context(format!("{TASK_ENV_VAR} was not a valid task specification"))?;
    run(&task)
}
