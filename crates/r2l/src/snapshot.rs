//! Shared snapshot configuration and progress, with backend-specific learner state.

use r2l_core::error::Result;
use r2l_core::{
    env::Env,
    on_policy::algorithm::{OnPolicyAlgorithm, OnPolicyRuntime},
};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearnerSnapshot;

use crate::{
    OnPolicyBuilder, OnPolicyHooks, PPOBurn,
    backend::{BackendConfig, BurnBackendConfig},
    builders::{
        BuilderResources,
        algorithm::{AlgoConfig, PPOConfig},
        learner::{LearnerConfig, LearningHookConfig},
        on_policy_hook::OnPolicyHookConfig,
        sampler::SamplerConfiguration,
    },
    hooks::progress::TrainingProgress,
};

struct SnapshotAlgo {}

impl SnapshotAlgo {
    fn get_builder<A, B, E: Env + 'static>(&self) -> Result<OnPolicyBuilder<A, B, E>> {
        todo!()
    }

    fn get_snapshot(&self) -> PolicyValueLearnerSnapshot {
        todo!()
    }

    fn restore<E: Env + 'static>(self) -> Result<OnPolicyAlgorithm<OnPolicyHooks<PPOBurn, E>>> {
        let builder: OnPolicyBuilder<PPOConfig, BurnBackendConfig, E> = self.get_builder()?;
        let mut algo = builder.build()?;
        let snapshot = self.get_snapshot();
        // load the snapshot
        algo.runtime.agent.lm = algo.runtime.agent.lm.load_snapshot(snapshot);
        // obs normalizer, reward normalizer
        Ok(algo)
    }
}
