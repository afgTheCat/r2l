// The main reponsibiliy of this code is to:
// - run extensive test suits if we need be
//  -- with seeds etc (repro test)
//  -- with seeds not set
// - generate statistics, generate figures etc

mod zoo_parser;

use std::{path::PathBuf, process::Command};

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};

use crate::zoo_parser::ZooConfig;

const SEED: u64 = 0;
const CONFIG_PATH: &str = "/home/gabor/projects/r2l/assets/ppo.yaml";
const LOG_DIR: &str = "/home/gabor/projects/r2l/logs";
const SMALL_ENVIRONMENTS: [&str; 8] = [
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "BipedalWalker-v3",
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
];

#[derive(Parser)]
#[command(about = "Evaluate r2l against Stable Baselines3 Zoo configurations")]
struct Args {
    #[command(subcommand)]
    command: Option<Cli>,
}

#[derive(Subcommand)]
enum Cli {
    /// Trains and evaluates one environment in this process.
    Evaluate {
        /// Gymnasium environment ID.
        env: Vec<String>,
    },
}

fn evaluate_all() -> anyhow::Result<()> {
    let executable = std::env::current_exe().context("failed to locate evaluator executable")?;
    let mut children = Vec::with_capacity(SMALL_ENVIRONMENTS.len());
    for env in SMALL_ENVIRONMENTS {
        let command = Command::new(&executable).args(["evaluate", env]).spawn();
        match command {
            Ok(child) => children.push((env, child)),
            Err(error) => {
                for (_, child) in &mut children {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return Err(error).with_context(|| format!("failed to start evaluation for {env}"));
            }
        }
    }
    let mut failures = Vec::new();
    for (env, mut child) in children {
        match child.wait() {
            Ok(status) if status.success() => {}
            Ok(status) => failures.push(format!("{env} exited with {status}")),
            Err(error) => failures.push(format!("failed to wait for {env}: {error}")),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        bail!("evaluation failures: {}", failures.join("; "))
    }
}

fn evaluate(envs: Vec<String>) -> anyhow::Result<()> {
    let config_path = PathBuf::from(CONFIG_PATH);
    let zoo_config = ZooConfig::parse_rl_zoo_config(config_path);
    for env in envs {
        let log_file = PathBuf::from(LOG_DIR).join(format!("{env}.csv"));
        let env_config = zoo_config.supported_envs.get(&env).unwrap();
        println!("Evaluating {env}");
        let mut algorithm = env_config
            .build_burn_ppo_algorithm(&env, log_file, SEED)
            .unwrap();
        algorithm.train().unwrap();
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    match Args::parse().command {
        Some(Cli::Evaluate { env }) => evaluate(env),
        None => evaluate_all(),
    }
}

#[cfg(test)]
mod test {
    use crate::evaluate;

    #[test]
    fn pendulum_v1() {
        evaluate(vec!["Pendulum-v1".into()]).unwrap();
    }
}
