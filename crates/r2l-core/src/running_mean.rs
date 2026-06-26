use crate::tensor::R2lTensor;

// super simplified running mean, only for vectors
pub struct RunningMeanStd3<T: R2lTensor> {
    pub mean: T,
    pub var: T,
    count: f32,
}

// mega simplified view
impl<T: R2lTensor> RunningMeanStd3<T> {
    pub fn new(shape: Vec<usize>) -> Self {
        let mean = T::zeros(shape.clone());
        let var = T::zeros(shape);
        Self {
            mean,
            var,
            count: 0.,
        }
    }

    fn update_from_moments2(
        &mut self,
        batch_mean: T,
        batch_var: T,
        batch_count: f32,
    ) -> anyhow::Result<()> {
        if batch_count == 0.0 {
            return Ok(());
        }
        let tot_count = self.count + batch_count;
        let delta = batch_mean.sub(&self.mean).unwrap();
        self.mean = self
            .mean
            .add(&delta.mul_scalar(batch_count / tot_count).unwrap())
            .unwrap();
        let m_a = self.var.mul_scalar(self.count).unwrap();
        let m_b = batch_var.mul_scalar(batch_count).unwrap();
        let m_2 = m_a
            .add(&m_b)
            .unwrap()
            .add(
                &delta
                    .sqr()
                    .unwrap()
                    .mul_scalar(self.count * batch_count / tot_count)
                    .unwrap(),
            )
            .unwrap();
        self.var = m_2.mul_scalar(1.0 / tot_count).unwrap();
        self.count = tot_count;
        Ok(())
    }

    pub fn update(&mut self, t: &[T]) {
        let mean = T::mean_tensors(t);
        let var = T::var_tensors(t);
        self.update_from_moments2(mean, var, t.len() as f32);
    }
}
