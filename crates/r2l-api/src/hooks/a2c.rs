use std::marker::PhantomData;

use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use r2l_agents::{
    HookResult,
    on_policy_algorithms::{
        Advantages, Returns,
        a2c::{A2CHook, A2CParams},
    },
};
use r2l_burn::learning_module::{BurnActorCriticLMKind, BurnPolicy};
use r2l_candle::learning_module::R2lCandleLearningModule;
use r2l_core::buffers::TrajectoryContainer;

pub struct DefaultA2CHook<T = ()> {
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> A2CHook<BurnActorCriticLMKind<B, D>>
    for DefaultA2CHook<BurnActorCriticLMKind<B, D>>
{
    fn before_learning_hook<C: TrajectoryContainer<Tensor = <BurnActorCriticLMKind<B, D> as r2l_core::policies::OnPolicyLearningModule>::InferenceTensor>>(
            &mut self,
            _params: &mut A2CParams,
            _module: &mut BurnActorCriticLMKind<B, D>,
            _buffers: &[C],
            _advantages: &mut Advantages,
            _returns: &mut Returns,
    ) -> Result<HookResult>{
        // TODO: should finish this
        Ok(HookResult::Continue)
    }
}

impl A2CHook<R2lCandleLearningModule> for DefaultA2CHook<R2lCandleLearningModule> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = <R2lCandleLearningModule as r2l_core::policies::OnPolicyLearningModule>::InferenceTensor>>(
            &mut self,
            _params: &mut A2CParams,
            _module: &mut R2lCandleLearningModule,
            _buffers: &[B],
            _advantages: &mut Advantages,
            _returns: &mut Returns,
    ) -> Result<HookResult>{
        // TODO: should finish this
        Ok(HookResult::Continue)
    }
}
