use crate::burn_agents::ppo::HookResult;
use burn::{
    module::AutodiffModule,
    prelude::Backend,
    tensor::{Tensor as BurnTensor, backend::AutodiffBackend},
};
use r2l_burn_lm::{
    learning_module::{BurnPolicy, ParalellActorCriticLM},
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
};
use r2l_core::utils::rollout_buffer::{Advantages, Returns};
use r2l_core::{agents::Agent5, sampler5::buffer::TrajectoryContainer};

pub struct PPOBatchData<B: Backend> {
    pub logp: Logp<B>,
    pub values_pred: ValuesPred<B>,
    pub logp_diff: LogpDiff<B>,
    pub ratio: BurnTensor<B, 1>,
}

pub trait BurnPPOHooksTrait<B: AutodiffBackend, D: BurnPolicy<B>> {
    fn before_learning_hook<T: TrajectoryContainer<Tensor = BurnTensor<B, 1>>>(
        &mut self,
        _agent: &mut BurnPPOCore<B, D>,
        _rollout_buffers: &[T],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<T: TrajectoryContainer<Tensor = BurnTensor<B, 1>>>(
        &mut self,
        _agent: &mut BurnPPOCore<B, D>,
        _rollout_buffers: &[T],
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _agent: &mut BurnPPOCore<B, D>,
        _policy_loss: &mut PolicyLoss<B>,
        _value_loss: &mut ValueLoss<B>,
        _data: &PPOBatchData<B>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct BurnPPOCore<B: AutodiffBackend, D: BurnPolicy<B>> {
    pub lm: ParalellActorCriticLM<B, D>,
    pub clip_range: f32,
    pub sample_size: usize,
    pub gamma: f32,
    pub lambda: f32,
}

pub struct BurnPPO<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnPPOHooksTrait<B, D>> {
    pub core: BurnPPOCore<B, D>,
    pub hooks: H,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnPPOHooksTrait<B, D>> Agent5 for BurnPPO<B, D, H> {
    type Tensor = BurnTensor<B::InnerBackend, 1>;

    type Policy = D::InnerModule;

    fn policy(&self) -> Self::Policy {
        self.core.lm.model.distr.valid()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        _buffers: &[C],
    ) -> anyhow::Result<()> {
        todo!()
    }
}
