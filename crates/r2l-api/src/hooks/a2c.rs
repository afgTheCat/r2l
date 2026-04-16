use std::marker::PhantomData;

use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use candle_core::Tensor;
use r2l_agents::{
    HookResult,
    on_policy_algorithms::{
        Advantages, Returns,
        a2c::{A2CBatchData, A2CHook, A2CParams},
    },
};
use r2l_burn::learning_module::{BurnActorCriticLMKind, BurnPolicy, BurnPolicyValuesLosses};
use r2l_candle::learning_module::{CandlePolicyValuesLosses, R2lCandleLearningModule};
use r2l_core::{
    buffers::TrajectoryContainer, models::Policy,
    on_policy::learning_module::OnPolicyLearningModule,
};

pub struct DefaultA2CHook<T = ()> {
    pub normalize_advantage: bool,
    pub entropy_coeff: f32,
    pub vf_coeff: Option<f32>,
    pub gradient_clipping: Option<f32>,
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> A2CHook<BurnActorCriticLMKind<B, D>>
    for DefaultA2CHook<BurnActorCriticLMKind<B, D>>
{
    fn before_learning_hook<
        C: TrajectoryContainer<
            Tensor = <BurnActorCriticLMKind<B, D> as OnPolicyLearningModule>::InferenceTensor,
        >,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnActorCriticLMKind<B, D>,
        _buffers: &[C],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        if self.normalize_advantage {
            advantages.normalize();
        }
        if let Some(max_grad_norm) = self.gradient_clipping {
            module.set_grad_clipping(burn::grad_clipping::GradientClipping::Norm(max_grad_norm));
        }
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnActorCriticLMKind<B, D>,
        losses: &mut BurnPolicyValuesLosses<B>,
        data: &A2CBatchData<burn::Tensor<B, 1>>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        if self.entropy_coeff != 0. {
            let entropy = module.get_policy().entropy(&data.observations)?;
            losses.apply_entropy(entropy.neg() * self.entropy_coeff);
        }
        Ok(HookResult::Continue)
    }
}

impl A2CHook<R2lCandleLearningModule> for DefaultA2CHook<R2lCandleLearningModule> {
    fn before_learning_hook<
        B: TrajectoryContainer<
            Tensor = <R2lCandleLearningModule as OnPolicyLearningModule>::InferenceTensor,
        >,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut R2lCandleLearningModule,
        _buffers: &[B],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        if self.normalize_advantage {
            advantages.normalize();
        }
        module.set_grad_clipping(self.gradient_clipping);
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut R2lCandleLearningModule,
        losses: &mut CandlePolicyValuesLosses,
        data: &A2CBatchData<candle_core::Tensor>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        if self.entropy_coeff != 0. {
            let entropy = module.get_policy().entropy(&data.observations)?;
            let device = entropy.device();
            let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
            losses.apply_entropy(entropy_loss)?;
        }
        Ok(HookResult::Continue)
    }
}
