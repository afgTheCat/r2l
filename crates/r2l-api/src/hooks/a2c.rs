use std::marker::PhantomData;

use burn::tensor::backend::AutodiffBackend;
use r2l_agents::{HookResult, on_policy_algorithms::a2c::A2CHook};
use r2l_burn::learning_module::{BurnActorCriticLMKind, BurnPolicy};
use r2l_candle::learning_module::R2lCandleLearningModule;

pub struct DefaultA2CHook<T = ()> {
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> A2CHook<BurnActorCriticLMKind<B, D>>
    for DefaultA2CHook<BurnActorCriticLMKind<B, D>>
{
    fn before_learning_hook<C: r2l_core::sampler::buffer::TrajectoryContainer<Tensor = <BurnActorCriticLMKind<B, D> as r2l_core::policies::OnPolicyLearningModule>::InferenceTensor>>(
            &mut self,
            _params: &mut r2l_agents::on_policy_algorithms::a2c::A2CParams,
            _module: &mut BurnActorCriticLMKind<B, D>,
            _buffers: &[C],
            _advantages: &mut r2l_core::utils::rollout_buffer::Advantages,
            _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> anyhow::Result<r2l_agents::HookResult>{
        // TODO: should finish this
        Ok(HookResult::Continue)
    }
}

impl A2CHook<R2lCandleLearningModule> for DefaultA2CHook<R2lCandleLearningModule> {
    fn before_learning_hook<B: r2l_core::sampler::buffer::TrajectoryContainer<Tensor = <R2lCandleLearningModule as r2l_core::policies::OnPolicyLearningModule>::InferenceTensor>>(
            &mut self,
            _params: &mut r2l_agents::on_policy_algorithms::a2c::A2CParams,
            _module: &mut R2lCandleLearningModule,
            _buffers: &[B],
            _advantages: &mut r2l_core::utils::rollout_buffer::Advantages,
            _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> anyhow::Result<HookResult>{
        // TODO: should finish this
        Ok(HookResult::Continue)
    }
}
