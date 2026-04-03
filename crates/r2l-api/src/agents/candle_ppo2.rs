use crate::{
    builders::{distribution::DistributionBuilder, learning_module::LearningModuleBuilder},
    hooks::ppo::{PPOHook, PPOHookBuilder, PPOStats},
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use r2l_agents::{
    candle_agents::ActorCriticKind,
    ppo2::{NewPPO, NewPPOParams, PPOModule2},
};
use r2l_candle_lm::{
    distributions::DistributionKind,
    learning_module::{PolicyValuesLosses, SequentialValueFunction},
};
use r2l_core::{
    env_builder::{EnvBuilder, EnvBuilderTrait},
    policies::LearningModule,
};
use std::sync::mpsc::Sender;

pub struct R2lCandleLearningModule {
    pub policy: DistributionKind,
    pub actor_critic: ActorCriticKind,
    pub value_function: SequentialValueFunction,
}

impl R2lCandleLearningModule {
    pub fn set_grad_clipping(&mut self, gradient_clipping: Option<f32>) {
        self.actor_critic.set_grad_clipping(gradient_clipping);
    }

    pub fn policy_learning_rate(&self) -> f64 {
        self.actor_critic.policy_learning_rate()
    }
}

// NOTE: I super don't like this, but whatever.
impl PPOModule2 for R2lCandleLearningModule {
    type Tensor = Tensor;
    type InferenceTensor = Tensor;
    type Policy = DistributionKind;
    type InferencePolicy = DistributionKind;
    type ValueFunction = SequentialValueFunction;
    type Losses = PolicyValuesLosses;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.policy.clone()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        self.actor_critic.update(losses)
    }

    fn value_func(&self) -> &Self::ValueFunction {
        &self.value_function
    }

    // this is also not PPO
    fn tensor_from_slice(&self, slice: &[f32]) -> Self::Tensor {
        Tensor::from_slice(slice, slice.len(), &candle_core::Device::Cpu).unwrap()
    }

    // this is very not PPO
    fn lifter(t: &Self::InferenceTensor) -> Self::Tensor {
        t.clone()
    }

    // this needs to be a constraint on the losses
    fn get_losses(policy_loss: Self::Tensor, value_loss: Self::Tensor) -> Self::Losses {
        PolicyValuesLosses::new(policy_loss, value_loss)
    }
}

pub struct PPOCandleAgentBuilder {
    pub device: Device,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: PPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub ppo_params: NewPPOParams,
}

impl PPOCandleAgentBuilder {
    fn build_lm(&mut self) -> anyhow::Result<R2lCandleLearningModule> {
        let distribution_varmap = VarMap::new();
        let distribution_var_builder =
            VarBuilder::from_varmap(&distribution_varmap, DType::F32, &self.device);
        let policy = self
            .distribution_builder
            .build(&distribution_var_builder, &self.device)?;
        let (value_function, learning_module) = self.actor_critic_type.build(
            distribution_varmap,
            distribution_var_builder,
            &self.device,
        )?;
        let learning_module = R2lCandleLearningModule {
            policy,
            actor_critic: learning_module,
            value_function,
        };
        Ok(learning_module)
    }

    fn build(
        mut self,
        tx: Sender<PPOStats>,
    ) -> anyhow::Result<NewPPO<R2lCandleLearningModule, PPOHook<R2lCandleLearningModule>>> {
        let lm = self.build_lm()?;
        let hooks = self.hook_builder.build(tx);
        let params = self.ppo_params;
        Ok(NewPPO { lm, hooks, params })
    }
}

// Goal: PPOBuilder::new()
struct PPOBuilder<EB: EnvBuilderTrait> {
    pub ppo_params: NewPPOParams,
    pub env_builder: EnvBuilder<EB>,
}

impl<EB: EnvBuilderTrait> PPOBuilder<EB> {
    fn new(builder: EB, n_envs: usize) -> Self {
        let env_builder = EnvBuilder::homogenous(builder, n_envs);
        let ppo_params = NewPPOParams::default();
        Self {
            ppo_params,
            env_builder,
        }
    }
}
