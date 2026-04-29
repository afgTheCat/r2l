use r2l_core::env::{EnvBuilder, EnvBuilderType, TensorOfEnvBuilder};
use r2l_sampler::{R2lSampler, SamplerExecutionMode, StepTrajectoryBound, TrajectoryBound};

/// Builder for [`R2lSampler`] instances.
///
/// This builder configures how environments are instantiated, how long each
/// collected trajectory may grow before yielding, and where sampler execution
/// takes place.
///
/// By default, [`new`](Self::new) creates a homogeneous vectorized sampler
/// using `n_envs` copies of the same environment builder, a
/// [`StepTrajectoryBound`] of `1024`, and
/// [`SamplerExecutionMode::Vec`].
pub struct SamplerBuilder<
    EB: EnvBuilder,
    BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>> = StepTrajectoryBound<
        TensorOfEnvBuilder<EB>,
    >,
> {
    pub(crate) env_builder: EnvBuilderType<EB>,
    pub(crate) trajectory_bound: BD,
    pub(crate) location: SamplerExecutionMode,
}

impl<EB: EnvBuilder> SamplerBuilder<EB> {
    /// Creates a sampler builder from a single environment builder and count.
    ///
    /// The provided builder is replicated into a homogeneous environment set
    /// with `n_envs` copies.
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let env_builder = EnvBuilderType::homogenous(builder.into(), n_envs);
        Self {
            env_builder,
            trajectory_bound: StepTrajectoryBound::new(1024),
            location: SamplerExecutionMode::Vec,
        }
    }
}

impl<EB: EnvBuilder, BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>>> SamplerBuilder<EB, BD> {
    /// Replaces the full environment builder configuration.
    ///
    /// This is useful when you need heterogeneous environments or when the
    /// default homogeneous setup created by [`new`](Self::new) is not enough.
    pub fn with_env_builder(mut self, env_builder: EnvBuilderType<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

    /// Replaces the trajectory bound used by the sampler.
    ///
    /// This changes the bound type carried by the builder, allowing callers to
    /// swap the default [`StepTrajectoryBound`] for another
    /// [`TrajectoryBound`] implementation.
    pub fn with_bound<BD2: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>>>(
        self,
        trajectory_bound: BD2,
    ) -> SamplerBuilder<EB, BD2> {
        let Self {
            env_builder,
            location,
            ..
        } = self;
        SamplerBuilder {
            env_builder,
            trajectory_bound,
            location,
        }
    }

    /// Sets where the sampler should execute.
    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.location = location;
        self
    }

    /// Builds the configured sampler.
    pub fn build(self) -> R2lSampler<EB::Env, BD> {
        R2lSampler::build(self.env_builder, self.trajectory_bound, self.location)
    }
}
