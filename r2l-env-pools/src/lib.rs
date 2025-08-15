mod vec_env_holder;

#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}
