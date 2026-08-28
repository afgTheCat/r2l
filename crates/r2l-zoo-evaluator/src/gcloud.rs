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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskGroup {
    pub task_spec: TaskSpec,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallelism: Option<u64>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub task_environments: Vec<Environment>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub variables: HashMap<String, String>,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub secret_variables: HashMap<String, String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceAccount {
    pub email: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogsPolicy {
    pub destination: LogsDestination,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_path: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LogsDestination {
    DestinationUnspecified,
    CloudLogging,
    Path,
}
