use derive_more::Deref;

#[derive(Deref, Debug)]
pub struct Advantages(pub Vec<Vec<f32>>);

impl Advantages {
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        indicies
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
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
        indicies
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
    }
}

#[derive(Deref, Debug)]
pub struct Logps(pub Vec<Vec<f32>>);

impl Logps {
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        indicies
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
    }
}
