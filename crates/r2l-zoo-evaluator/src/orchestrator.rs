use std::{
    env::var,
    io::Write,
    path::PathBuf,
    process::{Command as ProcessCommand, Stdio},
};

use anyhow::{Context, bail};
use clap::Parser;
use r2l::{PPOBuilder, TrainingArtifactsConfig, TrainingLimit};
use serde::{Deserialize, Serialize};

use crate::{
    Backend,
    gcloud::{
        AllocationPolicy, BatchJob, Container, Environment, LogsPolicy, Runnable, ServiceAccount,
        TaskGroup, TaskSpec,
    },
    zoo_parser::RlZooEnvironmentConfig,
};

const TASK_ENV_VAR: &str = "R2L_TASK";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct R2lTask {
    backend: Backend,
    rl_zoo_env_config: RlZooEnvironmentConfig,
    output_dir: PathBuf,
    env_name: String,
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
    Run,
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
        let project = var("PROJECT_ID").context("PROJECT_ID was not set")?;
        let region = var("REGION").context("REGION was not set")?;
        let job_name = var("JOB_NAME").context("JOB_NAME was not set")?;
        let image_uri = var("IMAGE_URI").context("IMAGE_URI was not set")?;
        let service_account = var("SERVICE_ACCOUNT").context("SERVICE_ACCOUNT was not set")?;
        let bucket = var("BUCKET").context("BUCKET was not set")?;
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
        let task_environments = tasks
            .into_iter()
            .map(|task| {
                let task = serde_json::to_string(&task)?;
                Ok(Environment::from_variable(TASK_ENV_VAR, task))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let parallelism = task_environments.len() as u64;
        let container = Container::new(&self.image_uri, vec!["run".to_owned()]);
        let task_spec = TaskSpec::new(vec![Runnable::container(container)]);
        let task_group = TaskGroup::new(task_spec)
            .with_parallelism(parallelism)
            .with_task_environments(task_environments);
        let allocation_policy =
            AllocationPolicy::with_service_account(ServiceAccount::new(&self.service_account));
        let job = BatchJob::new(vec![task_group])
            .with_allocation_policy(allocation_policy)
            .with_logs_policy(LogsPolicy::cloud_logging());
        let job = serde_json::to_vec(&job).context("failed to serialize Batch job")?;
        let mut child = ProcessCommand::new("gcloud")
            .args([
                "batch",
                "jobs",
                "submit",
                &self.job_name,
                "--project",
                &self.project,
                "--location",
                &self.region,
                "--config=-",
                "--quiet",
            ])
            .stdin(Stdio::piped())
            .spawn()
            .context("failed to start gcloud")?;
        child
            .stdin
            .take()
            .context("failed to open gcloud stdin")?
            .write_all(&job)
            .context("failed to write Batch job to gcloud")?;
        let status = child.wait().context("failed to wait for gcloud")?;
        if !status.success() {
            bail!("gcloud failed to submit Batch job: {status}");
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Submit => {
            let scheduler = Scheduler::new()?;
            scheduler.submit_tasks()
        }
        Command::Run => {
            let task = var(TASK_ENV_VAR).context(format!("{TASK_ENV_VAR} was not set"))?;
            let task: R2lTask = serde_json::from_str(&task)
                .context(format!("{TASK_ENV_VAR} was not a valid task specification"))?;
            task.train_ppo_algo()
        }
    }
}
