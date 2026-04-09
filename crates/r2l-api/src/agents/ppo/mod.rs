use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    tensor::backend::AutodiffBackend,
};
use candle_core::Tensor;
use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};
use r2l_burn_lm::{distributions::DistributionKind, learning_module::BurnActorCriticLMKind};
use r2l_candle_lm::{
    distributions::CandleDistributionKind,
    learning_module::{CandlePolicyValuesLosses, SequentialValueFunction},
};
use r2l_core::policies::OnPolicyLearningModule;
use r2l_core::{
    agents::Agent,
    distributions::Actor,
    policies::{LearningModule, ValueFunction},
    sampler::buffer::{TrajectoryContainer, wrapper::BufferWrapper},
    tensor::R2lBuffer,
};

use crate::{
    agents::AgentBuilder,
    builders::{
        distribution::{ActionSpaceType, DistributionBuilder, DistributionType},
        learning_module::{LearningModuleBuilder, LearningModuleType},
    },
    hooks::ppo::{StandardPPOHook, StandardPPOHookBuilder},
    learning_module::CandleActorCriticKind,
};

pub type BurnBackend = Autodiff<NdArray>;

pub struct BurnPPO<B: AutodiffBackend>(
    pub  PPO<
        BurnActorCriticLMKind<B, DistributionKind<B>>,
        StandardPPOHook<BurnActorCriticLMKind<B, DistributionKind<B>>>,
    >,
);

impl<B: AutodiffBackend> Agent for BurnPPO<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <DistributionKind<B> as AutodiffModule<B>>::InnerModule;

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

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_slice(slice, slice.len(), &self.device).unwrap()
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone()
    }
}

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

pub enum BurnOrCandlePPOActor {
    Burn(DistributionKind<NdArray>),
    Candle(CandleDistributionKind),
}

impl Actor for BurnOrCandlePPOActor {
    type Tensor = R2lBuffer;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Burn(d) => {
                let observation = observation.into();
                let action = d.get_action(observation)?;
                Ok(action.into())
            }
            Self::Candle(d) => {
                let observation = observation.into();
                let action = d.get_action(observation)?;
                Ok(action.into())
            }
        }
    }
}

pub enum BurnOrCandlePPO {
    Burn(BurnPPO<BurnBackend>),
    Candle(CandlePPO),
}

impl Agent for BurnOrCandlePPO {
    type Tensor = R2lBuffer;
    type Actor = BurnOrCandlePPOActor;

    fn actor(&self) -> Self::Actor {
        match self {
            Self::Burn(ppo) => BurnOrCandlePPOActor::Burn(ppo.actor()),
            Self::Candle(ppo) => BurnOrCandlePPOActor::Candle(ppo.actor()),
        }
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        match self {
            Self::Burn(ppo) => {
                let buffers = buffers
                    .as_ref()
                    .iter()
                    .map(BufferWrapper::new)
                    .collect::<Vec<_>>();
                ppo.learn(&buffers)
            }
            Self::Candle(ppo) => {
                let buffers = buffers
                    .as_ref()
                    .iter()
                    .map(BufferWrapper::new)
                    .collect::<Vec<_>>();
                ppo.learn(&buffers)
            }
        }
    }

    fn shutdown(&mut self) {
        match self {
            Self::Burn(ppo) => ppo.shutdown(),
            Self::Candle(ppo) => ppo.shutdown(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PPOBackend {
    Burn,
    Candle(Device),
}

impl Default for PPOBackend {
    fn default() -> Self {
        Self::Burn
    }
}

pub struct PPOAgentBuilder {
    pub ppo_params: PPOParams,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: StandardPPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub backend: PPOBackend,
}

impl Default for PPOAgentBuilder {
    fn default() -> Self {
        Self {
            hook_builder: StandardPPOHookBuilder::default(),
            ppo_params: PPOParams::default(),
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
            backend: PPOBackend::default(),
        }
    }
}

impl PPOAgentBuilder {
    pub fn with_backend(mut self, backend: PPOBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_candle_device(mut self, device: Device) -> Self {
        self.backend = PPOBackend::Candle(device);
        self
    }

    fn build_candle_lm(
        &self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: &Device,
    ) -> anyhow::Result<R2lCandleLearningModule> {
        let distribution_varmap = VarMap::new();
        let distr_var_builder = VarBuilder::from_varmap(&distribution_varmap, DType::F32, device);
        let policy = self.distribution_builder.build_candle(
            &distr_var_builder,
            device,
            observation_size,
            action_size,
            action_space,
        )?;
        let (value_function, learning_module) = self.actor_critic_type.build_candle(
            distribution_varmap,
            distr_var_builder,
            observation_size,
            device,
        )?;
        Ok(R2lCandleLearningModule {
            policy,
            actor_critic: learning_module,
            value_function,
            device: device.clone(),
        })
    }

    fn build_candle_with_device(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: Device,
    ) -> anyhow::Result<CandlePPO> {
        let lm = self.build_candle_lm(observation_size, action_size, action_space, &device)?;
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(CandlePPO(PPO { lm, hooks, params }))
    }

    fn build_burn_agent(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<BurnPPO<BurnBackend>> {
        let distr = self.distribution_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let lm = self.actor_critic_type.build_burn::<BurnBackend, _>(distr);
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(BurnPPO(PPO { lm, hooks, params }))
    }
}

impl AgentBuilder for PPOAgentBuilder {
    type Agent = BurnOrCandlePPO;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        match self.backend.clone() {
            PPOBackend::Burn => Ok(BurnOrCandlePPO::Burn(self.build_burn_agent(
                observation_size,
                action_size,
                action_space,
            )?)),
            PPOBackend::Candle(device) => Ok(BurnOrCandlePPO::Candle(
                self.build_candle_with_device(observation_size, action_size, action_space, device)?,
            )),
        }
    }
}

pub struct PPOBurnLearningModuleBuilder {
    pub ppo_params: PPOParams,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: StandardPPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
}

impl Default for PPOBurnLearningModuleBuilder {
    fn default() -> Self {
        let builder = PPOAgentBuilder::default().with_backend(PPOBackend::Burn);
        Self {
            ppo_params: builder.ppo_params,
            distribution_builder: builder.distribution_builder,
            hook_builder: builder.hook_builder,
            actor_critic_type: builder.actor_critic_type,
        }
    }
}

impl AgentBuilder for PPOBurnLearningModuleBuilder {
    type Agent = BurnPPO<BurnBackend>;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            distribution_builder: self.distribution_builder,
            hook_builder: self.hook_builder,
            actor_critic_type: self.actor_critic_type,
            backend: PPOBackend::Burn,
        }
        .build_burn_agent(observation_size, action_size, action_space)
    }
}

pub struct PPOCandleLearningModuleBuilder {
    pub ppo_params: PPOParams,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: StandardPPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub device: Device,
}

impl Default for PPOCandleLearningModuleBuilder {
    fn default() -> Self {
        let builder = PPOAgentBuilder::default().with_candle_device(Device::Cpu);
        let device = match builder.backend {
            PPOBackend::Candle(device) => device,
            PPOBackend::Burn => Device::Cpu,
        };
        Self {
            ppo_params: builder.ppo_params,
            distribution_builder: builder.distribution_builder,
            hook_builder: builder.hook_builder,
            actor_critic_type: builder.actor_critic_type,
            device,
        }
    }
}

impl AgentBuilder for PPOCandleLearningModuleBuilder {
    type Agent = CandlePPO;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            distribution_builder: self.distribution_builder,
            hook_builder: self.hook_builder,
            actor_critic_type: self.actor_critic_type,
            backend: PPOBackend::Candle(self.device.clone()),
        }
        .build_candle_with_device(observation_size, action_size, action_space, self.device)
    }
}
