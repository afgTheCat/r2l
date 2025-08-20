use burn::tensor::{Tensor, backend::AutodiffBackend};
use r2l_core2::thread_safe_sequential::ThreadSafeSequential;
use r2l_distributions::DistribnutionKind;

use crate::LearningModule;

pub struct ParalellActorCritic<B: AutodiffBackend> {
    distribution: DistribnutionKind<B>,
    value_net: ThreadSafeSequential<B>,
    // optimizer: OptimizerAdaptor<AdamW>,
}

pub struct ParalellActorCriticLosses<B: AutodiffBackend> {
    policy_loss: Tensor<B, 2>,
    value_loss: Tensor<B, 2>,
}

pub struct PolicyValueLosses<B: AutodiffBackend> {
    pub policy_loss: Tensor<B, 1>,
    pub value_loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend> LearningModule for ParalellActorCritic<B> {
    type Losses = PolicyValueLosses<B>;

    fn update(&mut self, losses: Self::Losses) {
        todo!()
    }
}
