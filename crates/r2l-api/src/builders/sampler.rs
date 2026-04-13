use r2l_core::env_builder::{EnvBuilder, EnvBuilderTrait};
use r2l_sampler::{FinalSampler, Location, StepTrajectoryBound, TrajectoryBound};

pub struct SamplerBuilder<
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor> = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>,
> {
    pub env_builder: EnvBuilder<EB>,
    pub trajectory_bound: BD,
    pub location: Location,
}

impl<EB: EnvBuilderTrait> SamplerBuilder<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let env_builder = EnvBuilder::homogenous(builder.into(), n_envs);
        Self {
            env_builder,
            trajectory_bound: StepTrajectoryBound::new(1024),
            location: Location::Vec,
        }
    }
}

impl<EB: EnvBuilderTrait, BD: TrajectoryBound<Tensor = EB::Tensor>> SamplerBuilder<EB, BD> {
    pub fn with_env_builder(mut self, env_builder: EnvBuilder<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

    pub fn with_bound<BD2: TrajectoryBound<Tensor = EB::Tensor>>(
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
