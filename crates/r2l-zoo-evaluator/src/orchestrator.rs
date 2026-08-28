use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use r2l::{PPOBuilder, TrainingArtifactsConfig, TrainingLimit};
use serde::{Deserialize, Serialize};

use crate::{Backend, zoo_parser::RlZooEnvironmentConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct R2lTask {
    backend: Backend,
    rl_zoo_env_config: RlZooEnvironmentConfig,
    output_dir: PathBuf,
    env_name: String,
}

impl std::str::FromStr for R2lTask {
    type Err = serde_json::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(value)
    }
}

impl R2lTask {
    pub fn train_ppo_algo(&self) -> anyhow::Result<()> {
        let obs_clip = self.rl_zoo_env_config.normalize.norm_obs().then_some(10.0);
        let artifacts_config = TrainingArtifactsConfig::new(&self.output_dir);
        let mut builder = PPOBuilder::gym(self.env_name.clone(), self.rl_zoo_env_config.n_envs)?
            .with_rollout_steps(self.rl_zoo_env_config.n_steps)
            .with_training_limit(TrainingLimit::steps(self.rl_zoo_env_config.n_timesteps))
            .with_training_artifacts(artifacts_config)
            .with_observation_normalizer(obs_clip)?
            .with_lambda(self.rl_zoo_env_config.gae_lambda)
            .with_gamma(self.rl_zoo_env_config.gamma)
            .with_total_epochs(self.rl_zoo_env_config.n_epochs)
            .with_entropy_coefficient(self.rl_zoo_env_config.ent_coef)
            .with_sample_size(self.rl_zoo_env_config.batch_size)
            .with_learning_rate_schedule(Some(self.rl_zoo_env_config.learning_rate.into()))
            .with_clip_range(self.rl_zoo_env_config.clip_range.initial_value() as f32)
            .with_log_std_init(self.rl_zoo_env_config.log_std_init)
            .with_value_loss_coefficient(Some(self.rl_zoo_env_config.vf_coef))
            .with_seed(0) // TODO: should we keep this?
            .with_gradient_clipping(Some(self.rl_zoo_env_config.max_grad_norm));
        if self.rl_zoo_env_config.normalize.norm_reward() {
            builder = builder.with_reward_normalizer(self.rl_zoo_env_config.gamma, 10.0);
        }
        match self.backend {
            Backend::Burn => builder.with_burn().build()?.train()?,
            Backend::Candle => builder.build()?.train()?,
        }
        Ok(())
    }
}

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    Submit,
    Run {
        #[arg(long)]
        task: R2lTask,
    },
}

fn gather_tasks() -> Vec<R2lTask> {
    todo!()
}

struct Scheduler {
    project: String,
    region: String,
    job_name: String,
    image_uri: String,
    service_account: String,
    bucket: String,
}

impl Scheduler {
    fn new() -> anyhow::Result<Self> {
        let project = std::env::var("PROJECT_ID").context("PROJECT_ID was not set")?;
        let region = std::env::var("REGION").context("REGION was not set")?;
        let job_name = std::env::var("JOB_NAME").context("JOB_NAME was not set")?;
        let image_uri = std::env::var("IMAGE_URI").context("IMAGE_URI was not set")?;
        let service_account =
            std::env::var("SERVICE_ACCOUNT").context("SERVICE_ACCOUNT was not set")?;
        let bucket = std::env::var("BUCKET").context("BUCKET was not set")?;
        Ok(Self {
            project,
            region,
            job_name,
            image_uri,
            service_account,
            bucket,
        })
    }

    fn submit_tasks(&self) -> anyhow::Result<()> {
        let tasks = gather_tasks();
        todo!()
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Submit => {
            let scheduler = Scheduler::new()?;
            scheduler.submit_tasks()
        }
        Command::Run { task } => task.train_ppo_algo(),
    }
}
