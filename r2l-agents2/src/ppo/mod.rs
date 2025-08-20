use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
};
use r2l_core2::{
    agent::Agent,
    distributions::Distribution,
    env::SnapShot,
    // policies::{Policy, PolicyWithValueFunction},
    utils::rollout_buffers::RolloutBufferV2,
};
use r2l_policies::{LearningModule, ParalellActorCritic};

pub struct PPO<D: Distribution, LM: LearningModule> {
    pub distribution: D,
    pub learning_module: LM,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
}

impl<D: Distribution, LM: LearningModule> PPO<D, LM> {
    fn batching_loop(&mut self) {
        todo!()
    }
}
