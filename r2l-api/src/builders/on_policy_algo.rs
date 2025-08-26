use crate::builders::{
    agents::{a2c::A2CBuilder, ppo::PPOBuilder},
    env_pool::{EnvBuilderTrait, VecPoolType},
};
use candle_core::{Device, Result};
use r2l_agents::AgentKind;
use r2l_core::{
    env::RolloutMode,
    env_pools::{R2lEnvHolder, R2lEnvPool},
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
};

pub enum AgentType {
    PPO(PPOBuilder),
    A2C(A2CBuilder),
}

pub struct OnPolicyAlgorithmBuilder {
    pub device: Device,
    pub env_pool_type: VecPoolType,
    pub normalize_env: bool,
    // pub hooks: OnPolicyHooks,
    pub rollout_mode: RolloutMode,
    pub learning_schedule: LearningSchedule,
    pub agent_type: AgentType,
}

impl Default for OnPolicyAlgorithmBuilder {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            env_pool_type: VecPoolType::default(),
            normalize_env: false,
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
    ) -> Result<
        OnPolicyAlgorithm<
            R2lEnvPool<R2lEnvHolder<EB::Env>>,
            AgentKind,
            DefaultOnPolicyAlgorightmsHooks,
        >,
    >
    where
        EB::Env: Sync + 'static,
    {
        let sampler =
            self.env_pool_type
                .build(&self.device, env_builder, n_envs, self.rollout_mode)?;
        let env_description = sampler.env_description();
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
        let hooks = DefaultOnPolicyAlgorightmsHooks::new(self.learning_schedule);
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
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
