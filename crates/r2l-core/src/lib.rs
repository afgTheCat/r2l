pub mod buffers;
pub mod env;
pub mod error;
pub mod models;
pub mod on_policy;
pub mod rng;
pub mod tensor;
mod utils;

/// Common imports for implementing environments, policies, agents, samplers,
/// and learning modules.
pub mod prelude {
    pub use crate::buffers::{
        EditableTrajectoryContainer, ExpandableTrajectoryContainer, Memory, TrajectoryContainer,
        fix_sized::FixedSizeStateBuffer, variable_sized::VariableSizedStateBuffer,
    };
    pub use crate::env::{Env, EnvBuilder, EnvBuilderTrait, EnvDescription, Space};
    pub use crate::models::{Actor, LearningModule, Policy, ValueFunction};
    pub use crate::on_policy::algorithm::{
        Agent, OnPolicyAlgorithm, OnPolicyAlgorithmHooks, Sampler,
    };
    pub use crate::on_policy::learning_module::OnPolicyLearningModule;
    pub use crate::on_policy::losses::FromPolicyValueLosses;
    pub use crate::tensor::{R2lTensor, R2lTensorMath, TensorData};
}
