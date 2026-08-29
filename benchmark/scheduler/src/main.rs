//! Submits benchmark tasks to Google Batch.

mod gcloud;

use std::{
    collections::BTreeMap,
    env::var,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{Context, bail, ensure};
use r2l_benchmark_task::{Backend, BenchmarkTask, RlZooEnvironmentConfig};
use yaml_serde::Value;

use crate::gcloud::{
    AllocationPolicy, BatchJob, ComputeResource, Container, Disk, Environment, InstancePolicy,
    InstancePolicyOrTemplate, LogsPolicy, ProvisioningModel, Runnable, ServiceAccount, TaskGroup,
    TaskSpec, Volume,
};

const CONFIG_PATH: &str = "../assets/ppo.yaml";
const LOG_DIR: &str = "/opt/r2l/logs";
const RESULTS_MOUNT_PATH: &str = "/mnt/disks/r2l-results";
const BOOT_DISK_TYPE: &str = "hyperdisk-balanced";
const MAX_RETRY_COUNT: u32 = 2;
const TASK_ENV_VAR: &str = "R2L_TASK";
const UNSUPPORTED_ENVIRONMENTS: [&str; 2] = ["MinitaurBulletEnv-v0", "MinitaurBulletDuckEnv-v0"];

fn uses_flat_obs_wrapper(value: &Value) -> bool {
    match value {
        Value::String(wrapper) => wrapper.ends_with("FlatObsWrapper"),
        Value::Sequence(wrappers) => wrappers.iter().any(uses_flat_obs_wrapper),
        _ => false,
    }
}

fn should_skip_environment(env_name: &str, value: &Value) -> bool {
    UNSUPPORTED_ENVIRONMENTS.contains(&env_name)
        || value.get("env_wrapper").is_some_and(uses_flat_obs_wrapper)
}

struct ZooConfig {
    supported_envs: BTreeMap<String, RlZooEnvironmentConfig>,
}

impl ZooConfig {
    fn parse(path: &Path) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path).with_context(|| {
            format!("failed to read RL Zoo configuration at {}", path.display())
        })?;
        let mut parsed_content: BTreeMap<String, Value> = yaml_serde::from_str(&content)
            .with_context(|| {
                format!("failed to parse RL Zoo configuration at {}", path.display())
            })?;
        parsed_content.remove("atari");
        let mut supported_envs = BTreeMap::new();
        for (env_name, value) in parsed_content {
            if should_skip_environment(&env_name, &value) {
                continue;
            }
            let config = yaml_serde::from_value::<RlZooEnvironmentConfig>(value)
                .with_context(|| format!("failed to parse RL Zoo configuration for {env_name}"))?;
            if config.supported() {
                supported_envs.insert(env_name, config);
            }
        }
        Ok(Self { supported_envs })
    }
}

fn gather_tasks() -> anyhow::Result<Vec<BenchmarkTask>> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let zoo_config = ZooConfig::parse(&crate_dir.join(CONFIG_PATH))?;
    let log_dir = PathBuf::from(LOG_DIR);

    Ok(zoo_config
        .supported_envs
        .into_iter()
        .flat_map(|(env_name, rl_zoo_env_config)| {
            [Backend::Burn, Backend::Candle, Backend::Sb3].map(|backend| BenchmarkTask {
                output_dir: log_dir.join(backend.name()).join(&env_name),
                rl_zoo_env_config: rl_zoo_env_config.clone(),
                env_name: env_name.clone(),
                backend,
            })
        })
        .collect())
}

struct Scheduler {
    project: String,
    region: String,
    job_name: String,
    image_uri: String,
    service_account: String,
    results_bucket: String,
    machine_type: String,
    parallelism: u64,
    cpu_milli_per_task: i64,
    memory_mib_per_task: i64,
}

impl Scheduler {
    fn from_environment() -> anyhow::Result<Self> {
        let project = var("PROJECT_ID").context("PROJECT_ID was not set")?;
        let region = var("REGION").context("REGION was not set")?;
        let job_name = var("JOB_NAME").context("JOB_NAME was not set")?;
        let image_uri = var("IMAGE_URI").context("IMAGE_URI was not set")?;
        let service_account = var("SERVICE_ACCOUNT").context("SERVICE_ACCOUNT was not set")?;
        let results_bucket = var("RESULTS_BUCKET").context("RESULTS_BUCKET was not set")?;
        let machine_type = var("BATCH_MACHINE_TYPE").context("BATCH_MACHINE_TYPE was not set")?;
        ensure!(
            !machine_type.trim().is_empty(),
            "BATCH_MACHINE_TYPE must not be empty"
        );
        let parallelism = var("BATCH_PARALLELISM")
            .context("BATCH_PARALLELISM was not set")?
            .parse()
            .context("BATCH_PARALLELISM must be a positive integer")?;
        ensure!(parallelism > 0, "BATCH_PARALLELISM must be positive");
        let cpu_milli_per_task = var("BATCH_CPU_MILLI_PER_TASK")
            .context("BATCH_CPU_MILLI_PER_TASK was not set")?
            .parse()
            .context("BATCH_CPU_MILLI_PER_TASK must be a positive integer")?;
        ensure!(
            cpu_milli_per_task > 0,
            "BATCH_CPU_MILLI_PER_TASK must be positive"
        );
        let memory_mib_per_task = var("BATCH_MEMORY_MIB_PER_TASK")
            .context("BATCH_MEMORY_MIB_PER_TASK was not set")?
            .parse()
            .context("BATCH_MEMORY_MIB_PER_TASK must be a positive integer")?;
        ensure!(
            memory_mib_per_task > 0,
            "BATCH_MEMORY_MIB_PER_TASK must be positive"
        );
        Ok(Self {
            project,
            region,
            job_name,
            image_uri,
            service_account,
            results_bucket,
            machine_type,
            parallelism,
            cpu_milli_per_task,
            memory_mib_per_task,
        })
    }

    fn submit_tasks(&self) -> anyhow::Result<()> {
        let tasks = gather_tasks()?;
        if tasks.is_empty() {
            bail!("no supported benchmark tasks were found");
        }
        let task_environments = tasks
            .into_iter()
            .map(|task| {
                let task = serde_json::to_string(&task)?;
                Ok(Environment::from_variable(TASK_ENV_VAR, task))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let parallelism = self.parallelism.min(task_environments.len() as u64);
        let results_path = format!(
            "{}/runs/{}",
            self.results_bucket.trim_end_matches('/'),
            self.job_name
        );
        let volume = Volume::gcs(results_path, RESULTS_MOUNT_PATH);
        let container = Container::new(&self.image_uri, Vec::new())
            .with_volume(format!("{RESULTS_MOUNT_PATH}:{LOG_DIR}"));
        let environment = Environment::from_variable("OPENBLAS_NUM_THREADS", "1")
            .with_variable("RAYON_NUM_THREADS", "1");
        let task_spec = TaskSpec::new(vec![Runnable::container(container)])
            .with_volumes(vec![volume])
            .with_compute_resource(ComputeResource::new(
                self.cpu_milli_per_task,
                self.memory_mib_per_task,
            ))
            .with_environment(environment)
            .with_max_retry_count(MAX_RETRY_COUNT);
        let task_group = TaskGroup::new(task_spec)
            .with_parallelism(parallelism)
            .with_task_environments(task_environments);
        let instance_policy = InstancePolicy::new(&self.machine_type, ProvisioningModel::Spot)
            .with_boot_disk(Disk::new(BOOT_DISK_TYPE));
        let allocation_policy =
            AllocationPolicy::with_service_account(ServiceAccount::new(&self.service_account))
                .with_instances(vec![InstancePolicyOrTemplate::new(instance_policy)]);
        let job = BatchJob::new(vec![task_group])
            .with_allocation_policy(allocation_policy)
            .with_logs_policy(LogsPolicy::cloud_logging());
        let job = serde_json::to_vec(&job).context("failed to serialize Batch job")?;
        let mut child = Command::new("gcloud")
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
    Scheduler::from_environment()?.submit_tasks()
}
