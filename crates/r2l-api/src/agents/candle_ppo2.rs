use crate::{
    builders::{
        distribution::{ActionSpaceType, DistributionBuilder},
        learning_module::LearningModuleBuilder,
    },
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
    agents::Agent,
    env::Space,
    env_builder::{EnvBuilder, EnvBuilderTrait},
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
    policies::LearningModule,
    sampler::{
        FinalSampler, Location,
        buffer::{StepTrajectoryBound, TrajectoryBound, TrajectoryContainer},
    },
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

pub struct DefaultPPO(NewPPO<R2lCandleLearningModule, PPOHook<R2lCandleLearningModule>>);

impl Agent for DefaultPPO {
    type Tensor = candle_core::Tensor;
    type Policy = DistributionKind;

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

pub struct PPOCandleAgentBuilder {
    pub device: Device,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: PPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub ppo_params: NewPPOParams,
}

impl Default for PPOCandleAgentBuilder {
    fn default() -> Self {
        todo!()
    }
}

impl PPOCandleAgentBuilder {
    fn build_lm(
        &mut self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<R2lCandleLearningModule> {
        let distribution_varmap = VarMap::new();
        let distr_var_builder =
            VarBuilder::from_varmap(&distribution_varmap, DType::F32, &self.device);
        let policy = self.distribution_builder.build(
            &distr_var_builder,
            &self.device,
            observation_size,
            action_size,
            action_space,
        )?;
        let (value_function, learning_module) = self.actor_critic_type.build(
            distribution_varmap,
            distr_var_builder,
            observation_size,
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
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        tx: Sender<PPOStats>,
    ) -> anyhow::Result<DefaultPPO> {
        let lm = self.build_lm(observation_size, action_size, action_space)?;
        let hooks = self.hook_builder.build(tx);
        let params = self.ppo_params;
        Ok(DefaultPPO(NewPPO { lm, hooks, params }))
    }
}

// Goal: PPOBuilder::new()
pub struct PPOBuilder2<
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor> = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>,
> {
    pub ppo_params: NewPPOParams,
    pub env_builder: EnvBuilder<EB>,
    pub trajectory_bound: BD,
    pub agent_builder: PPOCandleAgentBuilder,
    pub location: Location,
    pub learning_schedule: LearningSchedule,
}

impl<EB: EnvBuilderTrait> PPOBuilder2<EB> {
    pub fn new(builder: impl Into<EB>, n_envs: usize) -> Self {
        let env_builder = EnvBuilder::homogenous(builder, n_envs);
        let ppo_params = NewPPOParams::default();
        Self {
            ppo_params,
            env_builder: env_builder.into(),
            trajectory_bound: StepTrajectoryBound::new(1024),
            agent_builder: PPOCandleAgentBuilder::default(),
            location: Location::Vec,
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}

impl<EB: EnvBuilderTrait, BD: TrajectoryBound<Tensor = EB::Tensor>> PPOBuilder2<EB, BD> {
    pub fn with_bound<BD2: TrajectoryBound<Tensor = EB::Tensor>>(
        self,
        trajectory_bound: BD2,
    ) -> PPOBuilder2<EB, BD2> {
        let Self {
            ppo_params,
            env_builder,
            agent_builder,
            location,
            learning_schedule,
            ..
        } = self;
        PPOBuilder2 {
            ppo_params,
            env_builder,
            trajectory_bound,
            agent_builder,
            location,
            learning_schedule,
        }
    }

    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_total_epochs(total_epochs);
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_entropy_coeff(entropy_coeff);
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.agent_builder.hook_builder = self.agent_builder.hook_builder.with_vf_coeff(vf_coeff);
        self
    }

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.agent_builder.hook_builder = self.agent_builder.hook_builder.with_target_kl(target_kl);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.ppo_params.clip_range = clip_range;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.ppo_params.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.ppo_params.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.ppo_params.sample_size = sample_size;
        self
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }

    // TODO: too much. Also not generic enough
    pub fn build(
        self,
        tx: Sender<PPOStats>,
    ) -> anyhow::Result<
        OnPolicyAlgorithm<
            DefaultPPO,
            FinalSampler<EB::Env, BD>,
            DefaultOnPolicyAlgorightmsHooks<DefaultPPO, FinalSampler<EB::Env, BD>>,
        >,
    > {
        let env_description = self.env_builder.env_description()?;
        let sampler =
            FinalSampler::build(self.env_builder, self.trajectory_bound, None, self.location);
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space = match env_description.action_space {
            Space::Discrete(_) => ActionSpaceType::Discrete,
            Space::Continous { .. } => ActionSpaceType::Continous,
        };
        let agent = self
            .agent_builder
            .build(observation_size, action_size, action_space, tx)?;
        let hooks = DefaultOnPolicyAlgorightmsHooks::new(LearningSchedule::RolloutBound {
            total_rollouts: 300,
            current_rollout: 0,
        });
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
        })
    }
}
