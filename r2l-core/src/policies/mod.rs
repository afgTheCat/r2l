use anyhow::Result;

// convinience trait
pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

pub trait ValueFunction {
    type Tensor: Clone;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor>;
}
