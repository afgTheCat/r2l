use crate::builders::ppo::PPOBuilder;
use candle_core::{Device, Result};
use r2l_agents::AgentKind;
use r2l_core::{
    env::{EnvPoolType, RolloutMode, dummy_vec_env::DummyVecEnv},
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm, OnPolicyHooks},
    utils::rollout_buffer::RolloutBuffer,
};
use r2l_gym::GymEnv;

pub enum VecEnvType {
    Dummy,
    Vec,
    Subprocessing,
}

struct EnvBuilder {
    n_envs: usize,
    env_type: VecEnvType,
}

pub enum AgentType {
    PPO(PPOBuilder),
    A2C,
}

// Currently only works if a gym env is set,.
pub struct OnPolicyAlgorithmBuilder {
    pub device: Device,
    pub gym_env_name: Option<String>,
    pub env_type: EnvBuilder,
    pub normalize_env: bool,
    pub hooks: OnPolicyHooks,
    pub rollout_mode: RolloutMode,
    pub learning_schedule: LearningSchedule,
    pub agent_type: AgentType,
}

impl Default for OnPolicyAlgorithmBuilder {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            gym_env_name: None,
            env_type: EnvBuilder {
                n_envs: 16,
                env_type: VecEnvType::Dummy,
            },
            normalize_env: false,
            hooks: OnPolicyHooks::default(),
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
    fn build(&mut self) -> Result<OnPolicyAlgorithm<EnvPoolType<GymEnv>, AgentKind>> {
        let Some(gym_env_name) = &self.gym_env_name else {
            todo!()
        };
        let EnvBuilder {
            n_envs,
            env_type: VecEnvType::Dummy,
        } = self.env_type
        else {
            todo!()
        };
        let env = (0..n_envs)
            .map(|_| GymEnv::new(&gym_env_name, None, &self.device).unwrap())
            .collect::<Vec<_>>();
        let agent = match &mut self.agent_type {
            AgentType::PPO(builder) => {
                builder.set_gym_env_io(&env[0]);
                let ppo = builder.build()?;
                AgentKind::PPO(ppo)
            }
            _ => todo!(),
        };
        let env_pool = EnvPoolType::Dummy(DummyVecEnv {
            buffers: vec![RolloutBuffer::default(); n_envs],
            env,
        });
        Ok(OnPolicyAlgorithm {
            env_pool,
            agent,
            learning_schedule: self.learning_schedule,
            rollout_mode: self.rollout_mode,
            hooks: OnPolicyHooks::default(),
        })
    }
}
