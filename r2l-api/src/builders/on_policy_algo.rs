use crate::{
    builders::{env_pool::EnvPoolBuilder, ppo::PPOBuilder},
    hooks::on_policy_algo_hooks::LoggerTrainingHook,
};
use candle_core::{Device, Result};
use r2l_agents::AgentKind;
use r2l_core::{
    env::{EnvPool, EnvPoolType, RolloutMode},
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm, OnPolicyHooks},
};
use r2l_gym::GymEnv;

pub enum AgentType {
    PPO(PPOBuilder),
    A2C,
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
    pub fn build(mut self) -> Result<OnPolicyAlgorithm<EnvPoolType<GymEnv>, AgentKind>> {
        let env_pool = self.env_pool_builder.build(&self.device);
        let observation_space = env_pool.observation_space();
        let action_space = env_pool.action_space();
        let agent = match &mut self.agent_type {
            AgentType::PPO(builder) => {
                builder
                    .policy_builder
                    .set_io_dim((observation_space.size(), action_space.size()));
                let ppo = builder.build(&self.device)?;
                AgentKind::PPO(ppo)
            }
            _ => todo!(),
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
    pub fn ppo(gym_env_name: String) -> Self {
        let env_pool_builder = EnvPoolBuilder {
            gym_env_name: Some(gym_env_name),
            ..Default::default()
        };
        Self {
            env_pool_builder,
            rollout_mode: RolloutMode::StepBound { n_steps: 2048 },
            ..Default::default()
        }
    }
}
