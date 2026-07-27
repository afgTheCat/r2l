use anyhow::{Result, bail};
use burn::{backend::NdArray, prelude::Backend};
use r2l_core::{
    ActorWrapper,
    buffers::Memory,
    env::{
        Env, EnvBuilder, EnvBuilderType,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    models::{ActivationFunction, Actor},
    rng::set_seed,
    tensor::R2lTensor,
};
use r2l_sampler::{
    SamplerExecutionMode,
    staged2::{NormalizedPool, WorkerPool2},
};

use crate::{
    BurnBackend,
    builders::{
        agent::{BurnBackendConfig, CandleBackend},
        policy::PolicyBuilder,
    },
};

enum InferencePool<E: Env> {
    Raw(WorkerPool2<E>),
    Normalized(NormalizedPool<E>),
}

impl<E: Env> InferencePool<E> {
    fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, actor: A) {
        match self {
            Self::Raw(pool) => pool.set_policy(actor),
            Self::Normalized(pool) => pool.set_policy(actor),
        }
    }

    fn step(&mut self) -> Memory<E::Tensor> {
        let memories = match self {
            Self::Raw(pool) => pool.step(),
            Self::Normalized(pool) => pool.step(),
        };
        memories
            .into_iter()
            .next()
            .expect("an inference pool always contains one environment")
    }

    fn reset(&mut self) -> E::Tensor {
        match self {
            Self::Raw(pool) => {
                pool.reset_all();
                pool.current_states()
            }
            Self::Normalized(pool) => {
                pool.reset_all();
                pool.current_states()
            }
        }
        .pop()
        .expect("an inference pool always contains one environment")
    }

    fn current_observation(&mut self) -> E::Tensor {
        match self {
            Self::Raw(pool) => pool.current_states(),
            Self::Normalized(pool) => pool.current_states(),
        }
        .pop()
        .expect("an inference pool always contains one environment")
    }
}

/// Stateful single-environment inference facade.
///
/// Environment interaction is delegated to the same raw or normalized pools
/// used by staged sampling. Terminal observations remain available until the
/// caller explicitly resets the runner.
pub struct InferenceRunner<E: Env> {
    pool: InferencePool<E>,
    episode_done: bool,
}

/// Candle-backed inference runner produced by [`InferenceRunnerBuilder`].
pub type CandleInferenceRunner<EB> = InferenceRunner<<EB as EnvBuilder>::Env>;

/// Burn-backed inference runner produced by [`InferenceRunnerBuilder`].
pub type BurnInferenceRunner<EB> = InferenceRunner<<EB as EnvBuilder>::Env>;

impl<E: Env> InferenceRunner<E> {
    fn new(pool: InferencePool<E>) -> Self {
        Self {
            pool,
            episode_done: false,
        }
    }

    /// Chooses an action and advances the environment by one step.
    pub fn step(&mut self) -> Result<Memory<E::Tensor>> {
        if self.episode_done {
            bail!("the episode has ended; reset the inference runner before stepping again");
        }
        let memory = self.pool.step();
        self.episode_done = memory.is_done();
        Ok(memory)
    }

    /// Resets the environment and returns its current actor observation.
    ///
    /// The returned observation is normalized when the runner was built with
    /// an observation normalizer.
    pub fn reset(&mut self) -> E::Tensor {
        let observation = self.pool.reset();
        self.episode_done = false;
        observation
    }

    /// Clones the current actor observation.
    pub fn current_observation(&mut self) -> E::Tensor {
        self.pool.current_observation()
    }

    /// Returns whether the current episode has ended.
    pub fn episode_done(&self) -> bool {
        self.episode_done
    }
}

/// Builds a single-environment [`InferenceRunner`].
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

    /// Sets the seed used for policy construction and initial environment state.
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

    fn build_pool(self) -> InferencePool<<EB as EnvBuilder>::Env> {
        let env_builder = EnvBuilderType::homogeneous(self.env_builder, 1);
        match self.obs_normalizer {
            Some(obs_normalizer) => InferencePool::Normalized(NormalizedPool::build(
                env_builder,
                SamplerExecutionMode::SingleThreaded,
                obs_normalizer,
            )),
            None => InferencePool::Raw(WorkerPool2::build(
                env_builder,
                SamplerExecutionMode::SingleThreaded,
            )),
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
        let env_description = self.env_builder.env_description()?;
        let actor = self.policy_builder.build_candle(
            env_description.observation_space.size(),
            env_description.action_space,
            &self.backend.device,
        )?;
        let mut pool = self.build_pool();
        pool.set_policy(ActorWrapper::new(actor));
        Ok(InferenceRunner::new(pool))
    }
}

impl<EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>> InferenceRunnerBuilder<EB, BurnBackendConfig> {
    /// Builds a Burn-backed inference runner.
    pub fn build(self) -> Result<BurnInferenceRunner<EB>> {
        if let Some(seed) = self.seed {
            set_seed(seed);
            BurnBackend::seed(&Default::default(), seed);
        }
        let env_description = self.env_builder.env_description()?;
        let actor = self.policy_builder.build_burn::<NdArray, _>(
            env_description.observation_space.size(),
            env_description.action_space,
        );
        let mut pool = self.build_pool();
        pool.set_policy(ActorWrapper::new(actor));
        Ok(InferenceRunner::new(pool))
    }
}
