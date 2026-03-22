use crate::Policy;
use burn::{
    module::{AutodiffModule, ModuleDisplay},
    tensor::{Tensor as BurnTensor, backend::AutodiffBackend},
};
use r2l_burn_lm::learning_module::{BurnPolicy, ParalellActorCriticLM};
use r2l_core::{agents::Agent5, sampler5::buffer::TrajectoryContainer};
use r2l_core::{agents::TensorOfAgent, policies::LearningModule};

// impl<B: AutodiffBackend, D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = BurnTensor<B, 1>>>
//     BurnPPPHooksTrait<B, D> for EmptyBurnPPOHooks
// where
//     <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
// {
// }

pub struct BurnPPOCore<B: AutodiffBackend, D: BurnPolicy<B>> {
    pub lm: ParalellActorCriticLM<B, D>,
    pub clip_range: f32,
    pub sample_size: usize,
    pub gamma: f32,
    pub lambda: f32,
}

pub struct BurnPPO<B: AutodiffBackend, D: BurnPolicy<B>> {
    pub core: BurnPPOCore<B, D>,
    // pub hooks: Box<dyn BurnPPPHooksTrait<B, D>>,
}

impl<B: AutodiffBackend<InnerBackend = B>, D: BurnPolicy<B>> Agent5 for BurnPPO<B, D> {
    type Tensor = BurnTensor<B, 1>;

    type Policy = D::InnerModule;

    fn policy(&self) -> Self::Policy {
        todo!()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        todo!()
    }
}
