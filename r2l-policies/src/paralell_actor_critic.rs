use burn::tensor::{Tensor, backend::AutodiffBackend};
use r2l_core2::thread_safe_sequential::ThreadSafeSequential;
use r2l_distributions::DistribnutionKind;

pub struct ParalellActorCritic<B: AutodiffBackend> {
    distribution: DistribnutionKind<B>,
    value_net: ThreadSafeSequential<B>,
    // optimizer: OptimizerAdaptor<AdamW>,
}

pub struct ParalellActorCriticLosses<B: AutodiffBackend> {
    policy_loss: Tensor<B, 2>,
    value_loss: Tensor<B, 2>,
}

// impl<B: AutodiffBackend, O: Observation, A: Action> Policy<O, A> for ParalellActorCritic<B> {
//     type Dist = DistribnutionKind<B>;
//     type Losses = ParalellActorCriticLosses<B>;
//
//     fn distribution(&self) -> Self::Dist {
//         self.distribution.clone()
//     }
//
//     fn update(&mut self, loss: ParalellActorCriticLosses<B>) {
//         // let loss = policy_loss + value_loss;
//         // let grads = loss.backward();
//         // self.optimizer.step(0.01, loss, grads, None);
//         todo!()
//     }
// }
