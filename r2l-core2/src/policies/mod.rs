use crate::{
    distributions::Distribution,
    env::{Action, Logp, Observation},
};

pub trait Policy {
    type Obs: Observation;
    type Act: Action;
    type Logp: Logp;
    type Dist: Distribution<Self::Obs, Self::Act, Self::Logp>;

    // The losses. We will need a trait for that, maybe
    type Losses;

    // retrieves the underlying distribution
    fn distribution(&self) -> Self::Dist;

    // updates the policy according to the losses! there are a couple of issues with thi
    fn update(&mut self, losses: Self::Losses);
}

pub trait PolicyWithValueFunction: Policy {
    fn calculate_value(&self, observation: <Self as Policy>::Obs) -> f32;
}
