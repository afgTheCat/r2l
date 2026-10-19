//! Worker process for a single benchmark task.

use std::{env::var, process::Command};

use anyhow::{Context, bail};
use r2l::{AlgoBackendConfig, BackendConfig, EnvBuilderType, GymEnvBuilder, OnPolicyBuilder};
use r2l_benchmark_task::{BenchmarkTask, R2lTask};

const SB3_SCRIPT_PATH: &str = "/opt/r2l/sb3/ppo.py";
const TASK_ENV_VAR: &str = "R2L_TASK";

fn train_r2l(task: R2lTask) -> anyhow::Result<()> {
    let env_builder =
        EnvBuilderType::homogeneous(GymEnvBuilder::new(&task.env_name), task.n_envs.get())?;
    match task.backend {
        BackendConfig::Burn(backend) => OnPolicyBuilder::from_config(
            task.config,
            AlgoBackendConfig {
                algo: task.ppo,
                backend,
            },
            env_builder,
        )
        .build()?
        .train()?,
        BackendConfig::Candle(backend) => OnPolicyBuilder::from_config(
            task.config,
            AlgoBackendConfig {
                algo: task.ppo,
                backend,
            },
            env_builder,
        )
        .build()?
        .train()?,
    }
    Ok(())
}

fn train_sb3(env_name: &str) -> anyhow::Result<()> {
    let status = Command::new("python")
        .arg(SB3_SCRIPT_PATH)
        .arg(env_name)
        .status()
        .with_context(|| format!("failed to run {SB3_SCRIPT_PATH}"))?;
    if !status.success() {
        bail!("SB3 evaluation for {env_name} exited with {status}");
    }
    Ok(())
}

fn run(task: BenchmarkTask) -> anyhow::Result<()> {
    match task {
        BenchmarkTask::R2l(task) => train_r2l(*task),
        BenchmarkTask::Sb3 { env_name } => train_sb3(&env_name),
    }
}

fn main() -> anyhow::Result<()> {
    let task = var(TASK_ENV_VAR).context(format!("{TASK_ENV_VAR} was not set"))?;
    let task: BenchmarkTask = serde_json::from_str(&task)
        .context(format!("{TASK_ENV_VAR} was not a valid task specification"))?;
    run(task)
}
