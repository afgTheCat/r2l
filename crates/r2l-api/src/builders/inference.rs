use anyhow::{Result, bail};
use burn::{backend::NdArray, prelude::Backend};
use r2l_burn::distributions::BurnPolicyKind;
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    ActorWrapper,
    buffers::Memory,
    env::{
        Env, EnvBuilder, Snapshot,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    models::{ActivationFunction, Actor},
    rng::{sample_u64, set_seed},
    tensor::R2lTensor,
};

use crate::{
    BurnBackend,
    builders::{
        agent::{BurnBackendConfig, CandleBackend},
        policy::PolicyBuilder,
    },
};

/// Stateful coupling of one environment and one inference actor.
pub struct InferenceRunner<E: Env, A: Actor<Tensor = E::Tensor>> {
    env: E,
    actor: A,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    current_observation: Option<E::Tensor>,
    episode_done: bool,
    initial_seed: Option<u64>,
}
/// Candle-backed inference runner produced by [`InferenceRunnerBuilder`].
pub type CandleInferenceRunner<EB> = InferenceRunner<
    <EB as EnvBuilder>::Env,
    ActorWrapper<CandlePolicyKind, <<EB as EnvBuilder>::Env as Env>::Tensor>,
>;

/// Burn-backed inference runner produced by [`InferenceRunnerBuilder`].
pub type BurnInferenceRunner<EB> = InferenceRunner<
    <EB as EnvBuilder>::Env,
    ActorWrapper<BurnPolicyKind<NdArray>, <<EB as EnvBuilder>::Env as Env>::Tensor>,
>;

impl<E: Env<Tensor: R2lTensor>, A: Actor<Tensor = E::Tensor>> InferenceRunner<E, A> {
    fn new(
        env: E,
        actor: A,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
        initial_seed: Option<u64>,
    ) -> Self {
        Self {
            env,
            actor,
            obs_normalizer,
            current_observation: None,
            episode_done: false,
            initial_seed,
        }
    }

    /// Resets the environment and returns its raw initial observation.
    ///
    /// The builder seed is used for the first reset. Later resets use a fresh
    /// sampled seed unless [`reset_with_seed`](Self::reset_with_seed) is used.
    pub fn reset(&mut self) -> Result<E::Tensor> {
        let seed = self.initial_seed.take().unwrap_or_else(sample_u64);
        self.reset_with_seed(seed)
    }

    /// Resets the environment with an explicit seed.
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<E::Tensor> {
        let observation = self.env.reset(seed)?;
        self.current_observation = Some(observation.clone());
        self.episode_done = false;
        Ok(observation)
    }

    /// Chooses an action and advances the environment by one step.
    pub fn step(&mut self) -> Result<Memory<E::Tensor>> {
        if self.episode_done {
            bail!("the episode has ended; reset the inference runner before stepping again");
        }
        let observation = self
            .current_observation
            .clone()
            .ok_or_else(|| anyhow::anyhow!("reset the inference runner before stepping"))?;
        let mut actor_observation = vec![observation.clone()];
        if let Some(normalizer) = &self.obs_normalizer {
            normalizer.apply_in_place(&mut actor_observation);
        }
        let action = self.actor.action(actor_observation.pop().unwrap())?;
        let Snapshot {
            state: next_observation,
            reward,
            terminated,
            truncated,
        } = self.env.step(action.clone())?;
        self.current_observation = Some(next_observation.clone());
        self.episode_done = terminated || truncated;
        Ok(Memory {
            state: observation,
            action,
            next_state: next_observation,
            reward,
            terminated,
            truncated,
        })
    }

    /// Returns the current raw observation, if the runner has been reset.
    pub fn current_observation(&self) -> Option<&E::Tensor> {
        self.current_observation.as_ref()
    }

    /// Returns the inference actor.
    pub fn actor(&self) -> &A {
        &self.actor
    }

    /// Returns the environment.
    pub fn env(&self) -> &E {
        &self.env
    }
}

/// Builds an [`InferenceRunner`] from an environment and policy configuration.
pub struct InferenceRunnerBuilder<
    EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>,
    Backend = CandleBackend,
> {
    env_builder: EB,
    policy_builder: PolicyBuilder,
    backend: Backend,
    obs_normalizer: Option<ClippedNormalizer<<<EB as EnvBuilder>::Env as Env>::Tensor>>,
    seed: Option<u64>,
}

impl<EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>> InferenceRunnerBuilder<EB, CandleBackend> {
    /// Creates an inference builder using a CPU Candle policy.
    pub fn new(env_builder: EB) -> Self {
        Self {
            env_builder,
            policy_builder: PolicyBuilder::default(),
            backend: CandleBackend {
                device: candle_core::Device::Cpu,
            },
            obs_normalizer: None,
            seed: None,
        }
    }
}

impl<EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>, Backend> InferenceRunnerBuilder<EB, Backend> {
    /// Replaces the policy architecture configuration.
    pub fn with_policy_builder(mut self, policy_builder: PolicyBuilder) -> Self {
        self.policy_builder = policy_builder;
        self
    }

    /// Sets the policy hidden layer sizes.
    pub fn with_policy_hidden_layers(mut self, hidden_layers: Vec<usize>) -> Self {
        self.policy_builder.hidden_layers = hidden_layers;
        self
    }

    /// Sets the policy activation function.
    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.policy_builder.activation_function = activation_function;
        self
    }

    /// Sets the initial log standard deviation for Gaussian policies.
    pub fn with_log_std_init(mut self, log_std_init: f32) -> Self {
        self.policy_builder.log_std_init = log_std_init;
        self
    }

    /// Installs trained observation statistics in read-only mode.
    pub fn with_obs_normalizer(
        mut self,
        obs_normalizer: ClippedNormalizer<<<EB as EnvBuilder>::Env as Env>::Tensor>,
    ) -> Self {
        self.obs_normalizer = Some(obs_normalizer.with_mode(NormalizerMode::ReadOnly));
        self
    }

    /// Sets the seed used for policy construction and the first environment reset.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Switches policy construction to Candle on `device`.
    pub fn with_candle(
        self,
        device: candle_core::Device,
    ) -> InferenceRunnerBuilder<EB, CandleBackend> {
        InferenceRunnerBuilder {
            env_builder: self.env_builder,
            policy_builder: self.policy_builder,
            backend: CandleBackend { device },
            obs_normalizer: self.obs_normalizer,
            seed: self.seed,
        }
    }

    /// Switches policy construction to the default Burn inference backend.
    pub fn with_burn(self) -> InferenceRunnerBuilder<EB, BurnBackendConfig> {
        InferenceRunnerBuilder {
            env_builder: self.env_builder,
            policy_builder: self.policy_builder,
            backend: BurnBackendConfig,
            obs_normalizer: self.obs_normalizer,
            seed: self.seed,
        }
    }
}

impl<EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>> InferenceRunnerBuilder<EB, CandleBackend> {
    /// Builds a Candle-backed inference runner.
    pub fn build(self) -> Result<CandleInferenceRunner<EB>> {
        if let Some(seed) = self.seed {
            set_seed(seed);
            self.backend.seed(seed);
        }
        let env = self.env_builder.build_env()?;
        let env_description = env.env_description();
        let actor = self.policy_builder.build_candle(
            env_description.observation_space.size(),
            env_description.action_space,
            &self.backend.device,
        )?;
        Ok(InferenceRunner::new(
            env,
            ActorWrapper::new(actor),
            self.obs_normalizer,
            self.seed,
        ))
    }
}

impl<EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>> InferenceRunnerBuilder<EB, BurnBackendConfig> {
    /// Builds a Burn-backed inference runner.
    pub fn build(self) -> Result<BurnInferenceRunner<EB>> {
        if let Some(seed) = self.seed {
            set_seed(seed);
            BurnBackend::seed(&Default::default(), seed);
        }
        let env = self.env_builder.build_env()?;
        let env_description = env.env_description();
        let actor = self.policy_builder.build_burn::<NdArray, _>(
            env_description.observation_space.size(),
            env_description.action_space,
        );
        Ok(InferenceRunner::new(
            env,
            ActorWrapper::new(actor),
            self.obs_normalizer,
            self.seed,
        ))
    }
}
