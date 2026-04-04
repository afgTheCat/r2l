use crate::agents::PPOCandleLearningModuleBuilder;
use crate::{
    agents::AgentBuilder,
    builders::{
        distribution::{ActionSpaceType, DistributionBuilder, DistributionType},
        learning_module::{LearningModuleBuilder, LearningModuleType},
    },
    hooks::ppo::{PPOHook, PPOHookBuilder, PPOStats},
    sampler::SamplerBuilder,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::{
    candle_agents::ActorCriticKind,
    ppo2::{NewPPO, NewPPOParams, PPOModule2},
};
use r2l_candle_lm::{
    distributions::CandleDistributionKind,
    learning_module::{PolicyValuesLosses, SequentialValueFunction},
};
use r2l_core::{
    agents::Agent,
    env::Space,
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
    policies::{LearningModule, ValueFunction},
    sampler::{
        FinalSampler, Location,
        buffer::{StepTrajectoryBound, TrajectoryBound, TrajectoryContainer},
    },
};
use std::sync::mpsc::Sender;

pub struct R2lCandleLearningModule {
    pub policy: CandleDistributionKind,
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

impl ValueFunction for R2lCandleLearningModule {
    type Tensor = Tensor;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        self.value_function.calculate_values(observations)
    }
}

impl LearningModule for R2lCandleLearningModule {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        self.actor_critic.update(losses)
    }
}

// NOTE: I super don't like this, but whatever.
impl PPOModule2 for R2lCandleLearningModule {
    type LearningTensor = Tensor;
    type InferenceTensor = Tensor;
    type Policy = CandleDistributionKind;
    type InferencePolicy = CandleDistributionKind;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.policy.clone()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.policy
    }

    // this is also not PPO
    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_slice(slice, slice.len(), &candle_core::Device::Cpu).unwrap()
    }

    // this is very not PPO
    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone()
    }

    // this needs to be a constraint on the losses
    fn get_losses(
        policy_loss: Self::LearningTensor,
        value_loss: Self::LearningTensor,
    ) -> <Self as LearningModule>::Losses {
        PolicyValuesLosses::new(policy_loss, value_loss)
    }
}

pub struct CandlePPO(pub NewPPO<R2lCandleLearningModule, PPOHook<R2lCandleLearningModule>>);

impl Agent for CandlePPO {
    type Tensor = candle_core::Tensor;
    type Policy = CandleDistributionKind;

    fn policy(&self) -> Self::Policy {
        self.0.policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}
