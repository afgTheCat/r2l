use burn::{
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_core2::{
    agent::Agent,
    env::SnapShot,
    policies::{Policy, PolicyWithValueFunction},
};
use r2l_policies::ParalellActorCritic;

trait PPOBurnPolicy:
    PolicyWithValueFunction<
        Obs: From<Tensor<Self::B, 2>> + Into<Tensor<Self::B, 2>>,
        Act: From<Tensor<Self::B, 2>> + Into<Tensor<Self::B, 2>>,
    >
{
    type B: Backend;
}

impl<B: AutodiffBackend> PPOBurnPolicy for ParalellActorCritic<B> {
    type B = B;
}

pub struct PPO<P: Policy> {
    pub policy: P,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
}

impl<P: PPOBurnPolicy> PPO<P> {}

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
