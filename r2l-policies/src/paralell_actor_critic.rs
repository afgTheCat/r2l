use burn::tensor::{Tensor, backend::AutodiffBackend};
use r2l_core2::{
    policies::{Policy, PolicyWithValueFunction},
    thread_safe_sequential::ThreadSafeSequential,
};
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

impl<B: AutodiffBackend> Policy for ParalellActorCritic<B> {
    type Obs = Tensor<B, 2>;
    type Act = Tensor<B, 2>;
    type Logp = Tensor<B, 2>;
    type Losses = ();
    type Dist = DistribnutionKind<B>;

    fn distribution(&self) -> Self::Dist {
        self.distribution.clone()
    }

    fn update(&mut self, losses: Self::Losses) {
        todo!()
    }
}

impl<B: AutodiffBackend> PolicyWithValueFunction for ParalellActorCritic<B> {
    fn calculate_value(&self, observation: <Self as Policy>::Obs) -> f32 {
        todo!()
    }
}
