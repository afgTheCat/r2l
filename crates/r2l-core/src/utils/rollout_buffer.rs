use derive_more::Deref;

#[derive(Debug, Clone)]
pub struct RolloutBuffer<T: Clone> {
    pub states: Vec<T>,
    pub actions: Vec<T>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
}

impl<T: Clone> RolloutBuffer<T> {
    pub fn convert<U: Clone>(self) -> RolloutBuffer<U>
    where
        T: Into<U>,
    {
        let RolloutBuffer {
            states,
            actions,
            rewards,
            dones,
        } = self;
        RolloutBuffer {
            states: states.into_iter().map(|s| s.into()).collect(),
            actions: actions.into_iter().map(|s| s.into()).collect(),
            rewards,
            dones,
        }
    }
}

impl<T: Clone> Default for RolloutBuffer<T> {
    fn default() -> Self {
        Self {
            states: vec![],
            actions: vec![],
            rewards: vec![],
            dones: vec![],
        }
    }
}

impl<T: Clone> RolloutBuffer<T> {
    // TODO: this should be the last state
    pub fn set_last_state(&mut self, state: T) {
        self.states.push(state.clone());
    }

    pub fn sample_point(&self, index: usize) -> (&T, &T) {
        (&self.states[index], &self.actions[index])
    }
}

#[derive(Deref, Debug)]
pub struct Advantages(pub Vec<Vec<f32>>);

impl Advantages {
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        todo!()
    }

    pub fn normalize(&mut self) {
        for advantage in self.0.iter_mut() {
            let mean = advantage.iter().sum::<f32>() / advantage.len() as f32;
            let variance =
                advantage.iter().map(|x| (*x - mean).powi(2)).sum::<f32>() / advantage.len() as f32;
            let std = variance.sqrt() + 1e-8;
            for x in advantage.iter_mut() {
                *x = (*x - mean) / std;
            }
        }
    }
}

#[derive(Deref, Debug)]
pub struct Returns(pub Vec<Vec<f32>>);

impl Returns {
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        todo!()
    }
}

#[derive(Deref, Debug)]
pub struct Logps(pub Vec<Vec<f32>>);

impl Logps {
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        todo!()
    }
}
