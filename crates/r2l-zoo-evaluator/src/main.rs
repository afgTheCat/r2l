// The main reponsibiliy of this code is to:
// - run extensive test suits if we need be
//  -- with seeds etc (repro test)
//  -- with seeds not set
// - generate statistics, generate figures etc

//! Command-line evaluator for comparing `r2l` agents with RL Zoo configurations.

mod zoo_parser;

use std::{path::PathBuf, process::Command};

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use pyo3::Python;

use crate::zoo_parser::ZooConfig;

const SEED: u64 = 0;
const CONFIG_PATH: &str = "../../assets/ppo.yaml";
const LOG_DIR: &str = "../../logs";
const SMALL_ENVIRONMENTS: [&str; 10] = [
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "BipedalWalker-v3",
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
    "VizdoomBasic-MultiBinary-v1",
    "popgym-BattleshipEasy-v0",
];

#[derive(Clone, Copy)]
pub(crate) enum Backend {
    Burn,
    Candle,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Burn => "burn",
            Self::Candle => "candle",
        }
    }
}

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
        /// Gymnasium environment IDs and backend (`burn` or `candle`) in any order.
        args: Vec<String>,
    },
}

fn evaluate_all() -> anyhow::Result<()> {
    let executable = std::env::current_exe().context("failed to locate evaluator executable")?;
    let mut children = Vec::with_capacity(SMALL_ENVIRONMENTS.len());
    for env in SMALL_ENVIRONMENTS {
        let command = Command::new(&executable)
            .args(["evaluate", "burn", env])
            .spawn();
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

fn evaluate(envs: Vec<String>, backend: Backend) -> anyhow::Result<()> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config_path = crate_dir.join(CONFIG_PATH);
    let zoo_config = ZooConfig::parse_rl_zoo_config(config_path);
    for env in envs {
        let registration_module = match env.as_str() {
            "VizdoomBasic-MultiBinary-v1" => Some("vizdoom.gymnasium_wrapper"),
            "popgym-BattleshipEasy-v0" => Some("popgym"),
            _ => None,
        };
        if let Some(module) = registration_module {
            Python::with_gil(|py| py.import(module).map(|_| ()))?;
        }
        let Some(env_config) = zoo_config.supported_envs.get(&env) else {
            if zoo_config.unsupported_envs.contains_key(&env) {
                bail!("{env} uses an unsupported RL Zoo policy");
            }
            bail!("{env} is not present in the RL Zoo configuration");
        };
        let output_dir = crate_dir.join(LOG_DIR).join(backend.name()).join(&env);
        println!("Evaluating {env} with {}", backend.name());
        env_config.train_ppo_algorithm(backend, &env, output_dir, SEED)?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    match Args::parse().command {
        Some(Cli::Evaluate { mut args }) => {
            let backend_index = args
                .iter()
                .position(|arg| arg == "burn" || arg == "candle")
                .unwrap();
            let backend = match args.remove(backend_index).as_str() {
                "burn" => Backend::Burn,
                "candle" => Backend::Candle,
                _ => unreachable!(),
            };
            evaluate(args, backend)
        }
        None => evaluate_all(),
    }
}
