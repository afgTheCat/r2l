use crate::policies::PolicyWithValueFunction;
use candle_core::{Result, Tensor};

#[derive(Debug, Default)]
pub struct RolloutBuffer {
    pub states: Vec<Tensor>,
    pub actions: Vec<Tensor>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub advantages: Option<Vec<f32>>,
    pub returns: Option<Vec<f32>>,
    // TODO: this should not be an option
    pub logps: Vec<f32>,
}

impl RolloutBuffer {
    pub fn push_step(&mut self, state: Tensor, action: Tensor, reward: f32, done: bool, logp: f32) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.dones.push(done);
        self.logps.push(logp);
    }

    pub fn push_state(&mut self, state: Tensor) {
        self.states.push(state);
    }

    pub fn calculate_advantages_and_returns<P: PolicyWithValueFunction>(
        &mut self,
        policy: &P,
        gamma: f32,
        lambda: f32,
    ) -> Result<()> {
        let states = Tensor::stack(&self.states, 0)?;
        let values: Vec<f32> = policy.calculate_values(&states)?.to_vec1()?;
        let total_steps = self.rewards.len();
        let mut advantages: Vec<f32> = vec![0.; total_steps];
        let mut returns: Vec<f32> = vec![0.; total_steps];
        let mut last_gae_lam: f32 = 0.;
        for i in (0..total_steps).rev() {
            let next_non_terminal = if self.dones[i] {
                last_gae_lam = 0.;
                0f32
            } else {
                1.
            };
            let delta = self.rewards[i] + next_non_terminal * gamma * values[i + 1] - values[i];
            last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
            advantages[i] = last_gae_lam;
            returns[i] = last_gae_lam + values[i];
        }
        self.advantages = Some(advantages);
        self.returns = Some(returns);
        Ok(())
    }

    pub fn sample_point(&self, index: usize) -> (&Tensor, &Tensor, Option<f32>, Option<f32>, f32) {
        (
            &self.states[index],
            &self.actions[index],
            self.advantages.as_ref().map(|adv| adv[index]),
            self.returns.as_ref().map(|ret| ret[index]),
            self.logps[index],
        )
    }

    pub fn normalize_advantage(&mut self) {
        let advantage = self.advantages.as_mut().unwrap();
        let mean = advantage.iter().sum::<f32>() / advantage.len() as f32;
        let variance =
            advantage.iter().map(|x| (*x - mean).powi(2)).sum::<f32>() / advantage.len() as f32;
        let std = variance.sqrt() + 1e-8;
        for x in advantage.iter_mut() {
            *x = (*x - mean) / std;
        }
    }
}

pub struct RolloutBatch {
    pub observations: Tensor,
    pub actions: Tensor,
    pub returns: Tensor,
    pub advantages: Tensor,
    pub logp_old: Tensor,
}
