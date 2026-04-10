use r2l_core::{
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::OnPolicyAlgorithm,
    sampler::{FinalSampler, buffer::StepTrajectoryBound},
};

use crate::{
    BurnBackend,
    agents::a2c::{BurnA2C, CandleA2C},
    hooks::on_policy::DefaultOnPolicyAlgorightmsHooks,
};

pub type A2CBurnAlgorithm<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorithm<
        BurnA2C<BurnBackend>,
        FinalSampler<<EB as EnvBuilderTrait>::Env, BD>,
        DefaultOnPolicyAlgorightmsHooks<
            BurnA2C<BurnBackend>,
            FinalSampler<<EB as EnvBuilderTrait>::Env, BD>,
        >,
    >;

pub type A2CCandleAlgorithm<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorithm<
        CandleA2C,
        FinalSampler<<EB as EnvBuilderTrait>::Env, BD>,
        DefaultOnPolicyAlgorightmsHooks<CandleA2C, FinalSampler<<EB as EnvBuilderTrait>::Env, BD>>,
    >;
