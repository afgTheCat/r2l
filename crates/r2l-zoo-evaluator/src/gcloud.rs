use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchJob {
    pub task_groups: Vec<TaskGroup>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocation_policy: Option<AllocationPolicy>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_policy: Option<LogsPolicy>,
}

impl BatchJob {
    pub fn new(task_groups: Vec<TaskGroup>) -> Self {
        Self {
            task_groups,
            allocation_policy: None,
            logs_policy: None,
        }
    }

    pub fn with_allocation_policy(mut self, allocation_policy: AllocationPolicy) -> Self {
        self.allocation_policy = Some(allocation_policy);
        self
    }

    pub fn with_logs_policy(mut self, logs_policy: LogsPolicy) -> Self {
        self.logs_policy = Some(logs_policy);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskGroup {
    pub task_spec: TaskSpec,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallelism: Option<u64>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub task_environments: Vec<Environment>,
}

impl TaskGroup {
    pub fn new(task_spec: TaskSpec) -> Self {
        Self {
            task_spec,
            parallelism: None,
            task_environments: Vec::new(),
        }
    }

    pub fn with_parallelism(mut self, parallelism: u64) -> Self {
        self.parallelism = Some(parallelism);
        self
    }

    pub fn with_task_environments(mut self, task_environments: Vec<Environment>) -> Self {
        self.task_environments = task_environments;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSpec {
    pub runnables: Vec<Runnable>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub volumes: Vec<Volume>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_resource: Option<ComputeResource>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<Environment>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_count: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_run_duration: Option<String>,
}

impl TaskSpec {
    pub fn new(runnables: Vec<Runnable>) -> Self {
        Self {
            runnables,
            volumes: Vec::new(),
            compute_resource: None,
            environment: None,
            max_retry_count: None,
            max_run_duration: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Runnable {
    pub container: Container,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<Environment>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_exit_status: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub always_run: Option<bool>,
}

impl Runnable {
    pub fn container(container: Container) -> Self {
        Self {
            container,
            environment: None,
            ignore_exit_status: None,
            background: None,
            always_run: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Container {
    pub image_uri: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub entrypoint: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub volumes: Vec<String>,
}

impl Container {
    pub fn new(image_uri: impl Into<String>, commands: Vec<String>) -> Self {
        Self {
            image_uri: image_uri.into(),
            commands,
            entrypoint: None,
            options: None,
            volumes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub variables: HashMap<String, String>,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub secret_variables: HashMap<String, String>,
}

impl Environment {
    pub fn from_variable(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            variables: HashMap::from([(key.into(), value.into())]),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Volume {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gcs: Option<GcsVolume>,

    pub mount_path: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mount_options: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GcsVolume {
    pub remote_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputeResource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_milli: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mib: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub boot_disk_mib: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AllocationPolicy {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_account: Option<ServiceAccount>,
}

impl AllocationPolicy {
    pub fn with_service_account(service_account: ServiceAccount) -> Self {
        Self {
            service_account: Some(service_account),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceAccount {
    pub email: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
}

impl ServiceAccount {
    pub fn new(email: impl Into<String>) -> Self {
        Self {
            email: email.into(),
            scopes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogsPolicy {
    pub destination: LogsDestination,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_path: Option<String>,
}

impl LogsPolicy {
    pub fn cloud_logging() -> Self {
        Self {
            destination: LogsDestination::CloudLogging,
            logs_path: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LogsDestination {
    DestinationUnspecified,
    CloudLogging,
    Path,
}
