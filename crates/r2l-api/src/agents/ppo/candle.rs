use crate::agents::AgentBuilder;
use crate::hooks::ppo::StandardPPOHook;
use crate::learning_module::CandleActorCriticKind;
use crate::{
    builders::{
        distribution::{ActionSpaceType, DistributionBuilder, DistributionType},
        learning_module::{LearningModuleBuilder, LearningModuleType},
    },
    hooks::ppo::StandardPPOHookBuilder,
};
use candle_core::Tensor;
use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};
use r2l_candle_lm::{
    distributions::CandleDistributionKind,
    learning_module::{CandlePolicyValuesLosses, SequentialValueFunction},
};
use r2l_core::policies::OnPolicyLearningModule;
use r2l_core::{
    agents::Agent,
    policies::{LearningModule, ValueFunction},
    sampler::buffer::TrajectoryContainer,
};

pub struct R2lCandleLearningModule {
    pub policy: CandleDistributionKind,
    pub actor_critic: CandleActorCriticKind,
    pub value_function: SequentialValueFunction,
    pub device: Device,
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
    type Losses = CandlePolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        self.actor_critic.update(losses)
    }
}

impl OnPolicyLearningModule for R2lCandleLearningModule {
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
        Tensor::from_slice(slice, slice.len(), &self.device).unwrap()
    }

    // this is very not PPO
    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone()
    }
}

// TODO: this is the preferred way
// pub type CandlePPO = NewPPO<R2lCandleLearningModule, PPOHook<R2lCandleLearningModule>>;

pub struct CandlePPO(pub PPO<R2lCandleLearningModule, StandardPPOHook<R2lCandleLearningModule>>);

impl Agent for CandlePPO {
    type Tensor = candle_core::Tensor;
    type Actor = CandleDistributionKind;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
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

// NOTE: experimantally implementing it here. in the future this should not depend on candle
pub struct PPOCandleLearningModuleBuilder {
    pub ppo_params: PPOParams,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: StandardPPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub device: Device,
}

impl Default for PPOCandleLearningModuleBuilder {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            distribution_builder: DistributionBuilder {
                hidden_layers: vec![64, 64],
                distribution_type: DistributionType::Dynamic,
            },
            actor_critic_type: LearningModuleBuilder {
                learning_module_type: LearningModuleType::Paralell {
                    value_layers: vec![64, 64],
                    max_grad_norm: None,
                },
                params: ParamsAdamW {
                    lr: 3e-4,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-5,
                    weight_decay: 1e-4,
                },
            },
            hook_builder: StandardPPOHookBuilder::default(),
            ppo_params: PPOParams::default(),
        }
    }
}

impl PPOCandleLearningModuleBuilder {
    fn build_lm(
        &mut self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<R2lCandleLearningModule> {
        let distribution_varmap = VarMap::new();
        let distr_var_builder =
            VarBuilder::from_varmap(&distribution_varmap, DType::F32, &self.device);
        let policy = self.distribution_builder.build_candle(
            &distr_var_builder,
            &self.device,
            observation_size,
            action_size,
            action_space,
        )?;
        let (value_function, learning_module) = self.actor_critic_type.build_candle(
            distribution_varmap,
            distr_var_builder,
            observation_size,
            &self.device,
        )?;
        let learning_module = R2lCandleLearningModule {
            policy,
            actor_critic: learning_module,
            value_function,
            device: self.device.clone(),
        };
        Ok(learning_module)
    }
}

impl AgentBuilder for PPOCandleLearningModuleBuilder {
    type Agent = CandlePPO;

    fn build(
        mut self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        let lm = self.build_lm(observation_size, action_size, action_space)?;
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(CandlePPO(PPO { lm, hooks, params }))
    }
}
