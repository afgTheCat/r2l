use crate::{
    distributions::Policy,
    policies::ValueFunction,
    sampler3::buffers::{
        BatchIndexIterator, Buffer, VariableSizedStateBuffer, calculate_advantages_and_returns,
    },
    tensor::R2lTensor,
    utils::rollout_buffer::{Advantages, Logps, Returns},
};

pub struct BufferStack3<T: R2lTensor> {
    buffs: Vec<VariableSizedStateBuffer<T>>,
}

impl<T: R2lTensor> BufferStack3<T> {
    pub fn total_rewards(&self) -> f32 {
        self.buffs
            .iter()
            .map(|s| s.rewards.iter().sum::<f32>())
            .sum::<f32>()
    }

    pub fn total_episodes(&self) -> usize {
        self.buffs
            .iter()
            // TODO: there is no need to clone things here
            .flat_map(|s| s.dones())
            .filter(|d| *d)
            .count()
    }

    pub fn total_steps(&self) -> usize {
        self.buffs.iter().map(|b| b.states.len()).sum()
    }

    pub fn new(buffs: Vec<VariableSizedStateBuffer<T>>) -> Self {
        Self { buffs }
    }

    pub fn sample(&self, indicies: &[(usize, usize)]) -> (Vec<T>, Vec<T>) {
        let mut observations = vec![];
        let mut actions = vec![];
        for (buffer_idx, idx) in indicies {
            let observation = self.buffs[*buffer_idx].states[*idx].clone();
            let action = self.buffs[*buffer_idx].actions[*idx].clone();
            observations.push(observation);
            actions.push(action);
        }
        (observations, actions)
    }

    pub fn index_iterator(&self, sample_size: usize) -> BatchIndexIterator {
        let buf_refs = self.buffs.iter().collect::<Vec<_>>();
        BatchIndexIterator::new(&buf_refs, sample_size)
    }

    pub fn advantages_and_returns(
        &self,
        value_func: &impl ValueFunction<Tensor = T>,
        gamma: f32,
        lambda: f32,
    ) -> (Advantages, Returns) {
        let buf_refs = self.buffs.iter().collect::<Vec<_>>();
        calculate_advantages_and_returns(&buf_refs, value_func, gamma, lambda)
    }

    pub fn logps(&self, policy: &impl Policy<Tensor = T>) -> Logps {
        let mut logps = vec![];
        for buff in &self.buffs {
            let logp = policy
                .log_probs(&buff.states, &buff.actions)
                .map(|t| t.to_vec())
                .unwrap();
            logps.push(logp);
        }
        Logps(logps)
    }
}
