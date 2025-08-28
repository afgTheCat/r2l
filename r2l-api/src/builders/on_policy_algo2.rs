use crate::builders::{
    agents::{a2c::A2CBuilder, ppo::PPOBuilder},
    env::EnvBuilderTrait,
    on_policy_algo::AgentType,
    sampler::{EnvBuilderType, EnvPoolType, SamplerType},
    sampler_hooks2::EvaluatorNormalizerOptions2,
};
use candle_core::{Device, Result};
use r2l_agents::AgentKind;
use r2l_core::{
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
    sampler::samplers::NewSampler,
};
use std::sync::Arc;

pub struct OnPolicyAlgorithmBuilder2 {
    pub device: Device,
    pub sampler_type: SamplerType,
    pub agent_type: AgentType,
    pub learning_schedule: LearningSchedule,
}

impl Default for OnPolicyAlgorithmBuilder2 {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            sampler_type: SamplerType {
                capacity: 2048,
                hook_options: Default::default(),
                env_pool_type: Default::default(),
            },
            learning_schedule: LearningSchedule::TotalStepBound {
                total_steps: 0,
                current_step: 0,
            },
            agent_type: AgentType::PPO(PPOBuilder::default()),
        }
    }
}

impl OnPolicyAlgorithmBuilder2 {
    pub fn build<EB: EnvBuilderTrait>(
        &self,
        env_builder: EB,
        n_envs: usize,
    ) -> Result<OnPolicyAlgorithm<NewSampler<EB::Env>, AgentKind, DefaultOnPolicyAlgorightmsHooks>>
    {
        let sampler = self.sampler_type.build_with_builder_type(
            EnvBuilderType::EnvBuilder {
                builder: Arc::new(env_builder),
                n_envs,
            },
            &self.device,
        );
        let env_description = sampler.env_description();
        let agent = match &self.agent_type {
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

    pub fn ppo() -> Self {
        let agent_type = AgentType::PPO(PPOBuilder::default());
        let sampler_type = SamplerType {
            capacity: 2048,
            hook_options: EvaluatorNormalizerOptions2::default(),
            env_pool_type: EnvPoolType::VecStep,
        };
        Self {
            agent_type,
            sampler_type,
            ..Default::default()
        }
    }

    pub fn a2c() -> Self {
        let agent_type = AgentType::A2C(A2CBuilder::default());
        let sampler_type = SamplerType {
            capacity: 5,
            hook_options: EvaluatorNormalizerOptions2::default(),
            env_pool_type: EnvPoolType::VecStep,
        };
        Self {
            agent_type,
            sampler_type,
            ..Default::default()
        }
    }

    pub fn set_env_pool_type(&mut self, env_pool_type: EnvPoolType) {
        self.sampler_type.env_pool_type = env_pool_type;
    }

    pub fn set_n_step(&mut self, n_step: usize) {
        self.sampler_type.capacity = n_step;
    }
}
