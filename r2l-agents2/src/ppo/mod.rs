use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
};
use r2l_core2::{
    agent::Agent,
    distributions::Distribution,
    env::SnapShot,
    policies::{Policy, PolicyWithValueFunction},
    utils::rollout_buffers::RolloutBufferV2,
};
use r2l_policies::ParalellActorCritic;

trait PPOBurnPolicy:
    PolicyWithValueFunction<
        Obs: From<Tensor<Self::Back, 2>> + Into<Tensor<Self::Back, 2>>,
        Act: From<Tensor<Self::Back, 2>> + Into<Tensor<Self::Back, 2>>,
    >
{
    type Back: Backend;
}

impl<B: AutodiffBackend> PPOBurnPolicy for ParalellActorCritic<B> {
    type Back = B;
}

pub struct PPO<P: Policy> {
    pub policy: P,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
}

impl<P: PPOBurnPolicy> PPO<P> {
    fn batch_loop(&mut self, rb: &mut RolloutBufferV2<P::Obs, P::Act>) {
        while let Some(batch) = rb.get_batch() {
            let distribution = self.distribution();
            let logp = distribution.log_probs(&batch.observations, &batch.actions);
        }
    }
}

impl<P: PPOBurnPolicy> Agent for PPO<P> {
    type Obs = P::Obs;
    type Act = P::Act;
    type Policy = P;

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn learn<O: r2l_core2::env::Observation, A: r2l_core2::env::Action>(
        &mut self,
        snapshots: Vec<SnapShot<O, A>>,
    ) where
        Self::Obs: From<O>,
        Self::Act: From<A>,
    {
        todo!()
    }
}
