use crate::{
    builders::{
        agents::{a2c::A2CBuilder, ppo::PPOBuilder},
        env_pool::{self, EnvBuilderTrait, VecPoolType},
    },
    hooks::on_policy_algo_hooks::LoggerTrainingHook,
};
use candle_core::{Device, Result, Tensor};
use r2l_agents::AgentKind;
use r2l_core::{
    env::{self, RolloutMode, Sampler},
    env_pools::{R2lEnvHolder, R2lEnvPool},
    numeric::Buffer,
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm, OnPolicyHooks},
};

pub enum AgentType {
    PPO(PPOBuilder),
    A2C(A2CBuilder),
}

pub struct OnPolicyAlgorithmBuilder {
    pub device: Device,
    pub env_pool_type: VecPoolType,
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
            env_pool_type: VecPoolType::default(),
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
    // TODO: THIS RETURN TYPE IS INSANITY
    pub fn build<EB: EnvBuilderTrait>(
        mut self,
        env_builder: EB,
        n_envs: usize,
    ) -> Result<OnPolicyAlgorithm<R2lEnvPool<R2lEnvHolder<EB::Env>>, AgentKind>>
    where
        EB::Env: Sync + 'static,
    {
        let env_pool =
            self.env_pool_type
                .build(&self.device, env_builder, n_envs, self.rollout_mode)?;
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
        };
        Ok(OnPolicyAlgorithm {
            env_pool,
            agent,
            learning_schedule: self.learning_schedule,
            hooks: self.hooks,
        })
    }

    pub fn set_learning_schedule(&mut self, learning_schedule: LearningSchedule) {
        self.learning_schedule = learning_schedule;
    }

    // TODO: once we have some better policies, we should add it here. That would correspond to the stable baselines API
    pub fn ppo() -> Self {
        let env_pool_builder = VecPoolType::default();
        let agent_type = AgentType::PPO(PPOBuilder::default());
        Self {
            env_pool_type: env_pool_builder,
            rollout_mode: RolloutMode::StepBound { n_steps: 2048 },
            agent_type,
            ..Default::default()
        }
    }

    pub fn a2c() -> Self {
        let env_pool_builder = VecPoolType::default();
        let agent_type = AgentType::A2C(A2CBuilder::default());
        Self {
            env_pool_type: env_pool_builder,
            rollout_mode: RolloutMode::StepBound { n_steps: 5 },
            agent_type,
            ..Default::default()
        }
    }
}
