use crate::builders::{
    agents::{a2c::A2CBuilder, ppo::PPOBuilder},
    sampler::{EnvBuilderType, EnvPoolType, SamplerType},
    sampler_hooks2::EvaluatorNormalizerOptions,
};
use anyhow::Result;
use candle_core::Device;
use derive_more::{Deref, DerefMut};
use r2l_agents::{
    ActorCriticKind, GenericLearningModuleWithValueFunction,
    GenericLearningModuleWithValueFunction2, LearningModuleKind,
    candle_agents::{
        a2c::A2C,
        ppo::CandlePPO,
        ppo2::{CandlePPO2, PPOHooksTrait2},
    },
};
use r2l_candle_lm::distributions::DistributionKind;
use r2l_core::{
    env::{Env, EnvBuilderTrait},
    on_policy_algorithm::{
        DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm, OnPolicyAlgorithm2,
    },
    sampler::R2lSampler,
    sampler2::{R2lSampler2, env_pools::builder::BufferKind},
    tensor::R2lBuffer,
};
use r2l_gym::GymEnv;
use std::sync::Arc;

// TODO: this is pretty much a sampler builder at this point
pub struct OnPolicyAlgorithmBuilder {
    pub device: Device,
    pub sampler_type: SamplerType,
    pub learning_schedule: LearningSchedule,
}

impl Default for OnPolicyAlgorithmBuilder {
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
        }
    }
}

#[derive(Deref, DerefMut)]
pub struct A2CAlgoBuilder {
    #[deref]
    #[deref_mut]
    on_policy_builder: OnPolicyAlgorithmBuilder,
    a2c_builder: A2CBuilder,
}

impl Default for A2CAlgoBuilder {
    fn default() -> Self {
        let sampler_type = SamplerType {
            capacity: 5, // Default value in SB3
            hook_options: EvaluatorNormalizerOptions::default(),
            env_pool_type: EnvPoolType::VecStep,
        };
        Self {
            on_policy_builder: OnPolicyAlgorithmBuilder {
                sampler_type,
                ..Default::default()
            },
            a2c_builder: Default::default(),
        }
    }
}

impl A2CAlgoBuilder {
    pub fn build<E: Env<Tensor = R2lBuffer> + 'static, EB: EnvBuilderTrait<Env = E>>(
        &self,
        env_builder: EB,
        n_envs: usize,
    ) -> Result<
        OnPolicyAlgorithm<
            R2lSampler<EB::Env>,
            A2C<LearningModuleKind>,
            DefaultOnPolicyAlgorightmsHooks,
        >,
    > {
        let sampler = self.on_policy_builder.sampler_type.build_with_builder_type(
            EnvBuilderType::EnvBuilder {
                builder: Arc::new(env_builder),
                n_envs,
            },
        );
        let env_description = sampler.env_description();
        let agent = self
            .a2c_builder
            .build(&self.on_policy_builder.device, &env_description)?;
        let hooks = DefaultOnPolicyAlgorightmsHooks::new(self.on_policy_builder.learning_schedule);
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
        })
    }
}

#[derive(Deref, DerefMut)]
pub struct PPOAlgoBuilder2 {
    #[deref]
    #[deref_mut]
    on_policy_builder: OnPolicyAlgorithmBuilder,
    ppo_builder: PPOBuilder,
}

// impl PPOAlgoBuilder2 {
//     pub fn build<E: Env<Tensor = R2lBuffer> + 'static, EB: EnvBuilderTrait<Env = E>>(
//         &self,
//     ) -> OnPolicyAlgorithm2<
//         BufferKind<GymEnv>,
//         DistributionKind,
//         R2lSampler2<GymEnv>,
//         CandlePPO2<
//             GenericLearningModuleWithValueFunction<DistributionKind, ActorCriticKind>,
//             PPOHook2,
//         >,
//         DefaultOnPolicyAlgorightmsHooks,
//     > {
//         todo!()
//     }
// }

#[derive(Deref, DerefMut)]
pub struct PPOAlgoBuilder {
    #[deref]
    #[deref_mut]
    on_policy_builder: OnPolicyAlgorithmBuilder,
    ppo_builder: PPOBuilder,
}

impl PPOAlgoBuilder {
    pub fn build<E: Env<Tensor = R2lBuffer> + 'static, EB: EnvBuilderTrait<Env = E>>(
        &self,
        env_builder: EB,
        n_envs: usize,
    ) -> Result<
        OnPolicyAlgorithm<
            R2lSampler<EB::Env>,
            CandlePPO<LearningModuleKind>,
            DefaultOnPolicyAlgorightmsHooks,
        >,
    > {
        let sampler = self.on_policy_builder.sampler_type.build_with_builder_type(
            EnvBuilderType::EnvBuilder {
                builder: Arc::new(env_builder),
                n_envs,
            },
        );
        let env_description = sampler.env_description();
        let agent = self
            .ppo_builder
            .build(&self.on_policy_builder.device, &env_description)?;
        let hooks = DefaultOnPolicyAlgorightmsHooks::new(self.on_policy_builder.learning_schedule);
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
        })
    }
}

impl Default for PPOAlgoBuilder {
    fn default() -> Self {
        let sampler_type = SamplerType {
            capacity: 2048, // Default value in SB3
            hook_options: EvaluatorNormalizerOptions::default(),
            env_pool_type: EnvPoolType::VecStep,
        };
        Self {
            on_policy_builder: OnPolicyAlgorithmBuilder {
                sampler_type,
                ..Default::default()
            },
            ppo_builder: PPOBuilder::default(),
        }
    }
}

impl OnPolicyAlgorithmBuilder {
    pub fn set_learning_schedule(&mut self, learning_schedule: LearningSchedule) {
        self.learning_schedule = learning_schedule;
    }

    pub fn set_env_pool_type(&mut self, env_pool_type: EnvPoolType) {
        self.sampler_type.env_pool_type = env_pool_type;
    }

    pub fn set_n_step(&mut self, n_step: usize) {
        self.sampler_type.capacity = n_step;
    }

    pub fn set_hook_options(&mut self, hook_options: EvaluatorNormalizerOptions) {
        self.sampler_type.hook_options = hook_options;
    }
}
