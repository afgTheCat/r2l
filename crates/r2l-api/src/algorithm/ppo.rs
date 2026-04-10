use crate::{
    BurnBackend,
    agents::ppo::{BurnPPO, CandlePPO},
    hooks::on_policy::DefaultOnPolicyAlgorightmsHooks,
};
use r2l_core::{
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::OnPolicyAlgorithm,
    sampler::{FinalSampler, buffer::StepTrajectoryBound},
};

pub type PPOBurnAlgorithm<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorithm<
        BurnPPO<BurnBackend>,
        FinalSampler<<EB as EnvBuilderTrait>::Env, BD>,
        DefaultOnPolicyAlgorightmsHooks<
            BurnPPO<BurnBackend>,
            FinalSampler<<EB as EnvBuilderTrait>::Env, BD>,
        >,
    >;

pub type PPOCandleAlgorithm<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorithm<
        CandlePPO,
        FinalSampler<<EB as EnvBuilderTrait>::Env, BD>,
        DefaultOnPolicyAlgorightmsHooks<CandlePPO, FinalSampler<<EB as EnvBuilderTrait>::Env, BD>>,
    >;

pub type PPOAlgorithm<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    PPOCandleAlgorithm<EB, BD>;
