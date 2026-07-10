use anyhow::{Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use r2l_core::{
    env::ActionSpaceType,
    models::{ActivationFunction, Actor, Policy},
};

use crate::distributions::CandlePolicyKind;

/// Composite Candle policy for tuple and dict action spaces.
#[derive(Clone, Debug)]
pub struct CompositeDistribution {
    policies: Vec<CandlePolicyKind>,
    action_sizes: Vec<usize>,
    observation_size: usize,
    device: Device,
}

impl CompositeDistribution {
    /// Builds one child policy per nested action space.
    pub fn build(
        action_spaces: Vec<ActionSpaceType>,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        observation_size: usize,
        activation: ActivationFunction,
        prefix: &str,
    ) -> Result<Self> {
        let mut policies = Vec::new();
        let mut action_sizes = Vec::new();
        for (idx, action_space) in action_spaces.into_iter().enumerate() {
            let action_size = action_space.tensor_size();
            let child_prefix = format!("{prefix}.{idx}");
            policies.push(CandlePolicyKind::build_with_prefix(
                action_space,
                policy_varbuilder,
                hidden_layers,
                observation_size,
                activation,
                &child_prefix,
            )?);
            action_sizes.push(action_size);
        }
        Ok(Self {
            policies,
            action_sizes,
            observation_size,
            device: policy_varbuilder.device().clone(),
        })
    }

    /// Returns the Candle device used by this policy.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Returns the flattened observation size expected by this policy.
    pub fn observation_size(&self) -> usize {
        self.observation_size
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
