use anyhow::{Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use r2l_core::{
    env::Space,
    models::{ActivationFunction, Actor, Policy, PolicyMetadata},
    tensor::R2lTensor,
};
use safetensors::serialize as st_serialize;

use crate::distributions::CandlePolicyKind;

/// Composite Candle policy for tuple and dict action spaces.
#[derive(Clone, Debug)]
pub struct CompositeDistribution {
    policies: Vec<CandlePolicyKind>,
    action_sizes: Vec<usize>,
    observation_size: usize,
    device: Device,
    activation: ActivationFunction,
}

impl CompositeDistribution {
    /// Builds one child policy per nested action space.
    ///
    /// # Errors
    ///
    /// Returns an error if any child policy cannot be built.
    pub fn build<T: R2lTensor>(
        action_spaces: Vec<Space<T>>,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        observation_size: usize,
        activation: ActivationFunction,
        log_std_init: f32,
        prefix: &str,
    ) -> Result<Self> {
        let mut policies = Vec::new();
        let mut action_sizes = Vec::new();
        for (idx, action_space) in action_spaces.into_iter().enumerate() {
            let action_size = action_space.size();
            let child_prefix = format!("{prefix}.{idx}");
            policies.push(CandlePolicyKind::build_with_prefix(
                action_space,
                policy_varbuilder,
                hidden_layers,
                observation_size,
                activation,
                log_std_init,
                &child_prefix,
            )?);
            action_sizes.push(action_size);
        }
        Ok(Self {
            policies,
            action_sizes,
            observation_size,
            device: policy_varbuilder.device().clone(),
            activation,
        })
    }

    /// Returns the Candle device used by this policy.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Returns the flattened observation size expected by this policy.
    #[must_use]
    pub fn observation_size(&self) -> usize {
        self.observation_size
    }

    pub(crate) fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.policies
            .iter()
            .enumerate()
            .flat_map(|(index, policy)| policy.named_tensors(&format!("{prefix}.{index}")))
            .collect()
    }
}

impl Actor for CompositeDistribution {
    type Tensor = Tensor;
    fn action(&self, observation: Tensor) -> Result<Tensor> {
        let mut actions = Vec::new();
        for policy in &self.policies {
            actions.push(policy.action(observation.clone())?);
        }
        Ok(Tensor::cat(&actions, 0)?.detach())
    }

    fn mode_action(&self, observation: Tensor) -> Result<Tensor> {
        let mut actions = Vec::new();
        for policy in &self.policies {
            actions.push(policy.mode_action(observation.clone())?);
        }
        Ok(Tensor::cat(&actions, 0)?.detach())
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        let metadata = PolicyMetadata {
            activation: self.activation,
        }
        .to_safetensors_metadata();
        st_serialize(self.named_tensors("policy"), Some(metadata)).ok()
    }
}

impl Policy for CompositeDistribution {
    fn log_probs(&self, states: &[Tensor], actions: &[Tensor]) -> Result<Tensor> {
        let mut offset = 0;
        let mut log_probs = Vec::new();
        for (policy, action_size) in self.policies.iter().zip(&self.action_sizes) {
            let mut child_actions = Vec::new();
            for action in actions {
                child_actions.push(action.narrow(0, offset, *action_size)?);
            }
            log_probs.push(policy.log_probs(states, &child_actions)?);
            offset += action_size;
        }
        Ok(Tensor::stack(&log_probs, 0)?.sum(0)?)
    }

    fn entropy(&self, states: &[Tensor]) -> Result<Tensor> {
        let mut entropies = Vec::new();
        for policy in &self.policies {
            entropies.push(policy.entropy(states)?);
        }
        Ok(Tensor::stack(&entropies, 0)?.sum_all()?)
    }

    fn std(&self) -> Result<f32> {
        bail!("standard deviation is not defined for composite distributions")
    }

    fn resample_noise(&mut self) -> Result<()> {
        for policy in &mut self.policies {
            policy.resample_noise()?;
        }
        Ok(())
    }
}
