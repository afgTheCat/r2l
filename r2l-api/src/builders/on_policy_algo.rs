use crate::{
    builders::{
        a2c::A2CBuilder,
        env_pool::EnvBuilderTrait,
        env_pool::EnvPoolBuilder,
        ppo::{PPOBuilder, PPOBuilder2},
    },
    hooks::on_policy_algo_hooks::LoggerTrainingHook,
};
use candle_core::{Device, Result};
use r2l_agents::AgentKind;
use r2l_core::{
    env::{EnvPool, EnvPoolType, RolloutMode},
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm, OnPolicyHooks},
};

pub enum AgentType {
    PPO(PPOBuilder),
    PPO2(PPOBuilder2),
    A2C(A2CBuilder),
}

// Currently only works if a gym env is set,.
// TODO: proc macro for setters
pub struct OnPolicyAlgorithmBuilder {
    pub device: Device,
    pub env_pool_builder: EnvPoolBuilder,
    pub normalize_env: bool,
    pub hooks: OnPolicyHooks,
    pub rollout_mode: RolloutMode,
    pub learning_schedule: LearningSchedule,
    pub agent_type: AgentType,
}

impl Default for OnPolicyAlgorithmBuilder {
    fn default() -> Self {
        let mut hooks = OnPolicyHooks::default();
        hooks.add_training_hook(LoggerTrainingHook::default());
        Self {
            device: Device::Cpu,
            env_pool_builder: EnvPoolBuilder::default(),
            normalize_env: false,
            hooks,
            rollout_mode: RolloutMode::StepBound { n_steps: 0 },
            learning_schedule: LearningSchedule::TotalStepBound {
                total_steps: 0,
                current_step: 0,
            },
            agent_type: AgentType::PPO(PPOBuilder::default()),
        }
    }
}

impl OnPolicyAlgorithmBuilder {
    pub fn build<EB: EnvBuilderTrait>(
        mut self,
        env_builder: EB,
    ) -> Result<OnPolicyAlgorithm<EnvPoolType<EB::Env>, AgentKind>>
    where
        EB::Env: Sync + 'static,
    {
        let env_pool = self.env_pool_builder.build(&self.device, env_builder)?;
        let env_description = env_pool.env_description();
        let agent = match &mut self.agent_type {
            AgentType::PPO(builder) => {
                let ppo = builder.build(&self.device, &env_description)?;
                AgentKind::PPO(ppo)
            }
            AgentType::A2C(builder) => {
                let a2c = builder.build(&self.device, &env_description)?;
                AgentKind::A2C(a2c)
            }
            AgentType::PPO2(builder) => {
                let ppo = builder.build(&self.device, &env_description)?;
                AgentKind::PPO2(ppo)
            }
        };
        Ok(OnPolicyAlgorithm {
            env_pool,
            agent,
            learning_schedule: self.learning_schedule,
            rollout_mode: self.rollout_mode,
            hooks: self.hooks,
        })
    }

    pub fn set_learning_schedule(&mut self, learning_schedule: LearningSchedule) {
        self.learning_schedule = learning_schedule;
    }

    // TODO: once we have some better policies, we should add it here. That would correspond to the stable baselines API
    pub fn ppo() -> Self {
        let env_pool_builder = EnvPoolBuilder::default();
        let agent_type = AgentType::PPO(PPOBuilder::default());
        Self {
            env_pool_builder,
            rollout_mode: RolloutMode::StepBound { n_steps: 2048 },
            agent_type,
            ..Default::default()
        }
    }

    pub fn ppo2() -> Self {
        let env_pool_builder = EnvPoolBuilder::default();
        let agent_type = AgentType::PPO2(PPOBuilder2::default());
        Self {
            env_pool_builder,
            rollout_mode: RolloutMode::StepBound { n_steps: 2048 },
            agent_type,
            ..Default::default()
        }
    }

    pub fn a2c() -> Self {
        let env_pool_builder = EnvPoolBuilder::default();
        let agent_type = AgentType::A2C(A2CBuilder::default());
        Self {
            env_pool_builder,
            rollout_mode: RolloutMode::StepBound { n_steps: 5 },
            agent_type,
            ..Default::default()
        }
    }
}
