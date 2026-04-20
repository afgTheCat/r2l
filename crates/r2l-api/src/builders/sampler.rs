use r2l_core::env::{EnvBuilder, EnvBuilderType, TensorOfEnvBuilder};
use r2l_sampler::{FinalSampler, Location, StepTrajectoryBound, TrajectoryBound};

pub struct SamplerBuilder<
    EB: EnvBuilder,
    BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>> = StepTrajectoryBound<
        TensorOfEnvBuilder<EB>,
    >,
> {
    pub env_builder: EnvBuilderType<EB>,
    pub trajectory_bound: BD,
    pub location: Location,
}

impl<EB: EnvBuilder> SamplerBuilder<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let env_builder = EnvBuilderType::homogenous(builder.into(), n_envs);
        Self {
            env_builder,
            trajectory_bound: StepTrajectoryBound::new(1024),
            location: Location::Vec,
        }
    }
}

impl<EB: EnvBuilder, BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>>> SamplerBuilder<EB, BD> {
    pub fn with_env_builder(mut self, env_builder: EnvBuilderType<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

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

    pub fn with_location(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    pub fn build(self) -> FinalSampler<EB::Env, BD> {
        FinalSampler::build(self.env_builder, self.trajectory_bound, self.location)
    }
}
