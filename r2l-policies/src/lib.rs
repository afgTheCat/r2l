mod paralell_actor_critic;
pub use paralell_actor_critic::ParalellActorCritic;

pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses);
}
