use crate::tensor::RunningMeanTensor;

#[derive(Clone)]
pub struct RunningMeanStd2<T: RunningMeanTensor> {
    pub mean: T,
    pub var: T,
    pub count: f32,
}

impl<T: RunningMeanTensor> RunningMeanStd2<T> {
    pub fn new(shape: Vec<usize>) -> Self {
        let mean = T::zeros(shape.clone());
        let var = T::zeros(shape.clone());
        Self {
            mean,
            var,
            count: 0.,
        }
    }

    pub fn update(&mut self, arr: &T) -> anyhow::Result<()> {
        let batch_mean = arr.batch_mean()?;
        let batch_var = arr.biased_var()?;
        let batch_count = arr.batch_count()?;
        self.update_from_moments(batch_mean, batch_var, batch_count)?;
        Ok(())
    }

    fn update_from_moments(
        &mut self,
        batch_mean: T,
        batch_var: T,
        batch_count: f32,
    ) -> anyhow::Result<()> {
        if batch_count == 0.0 {
            return Ok(());
        }
        let tot_count = self.count + batch_count;
        let delta = batch_mean.sub(&self.mean)?;
        self.mean = self.mean.add(&delta.mul_scalar(batch_count / tot_count)?)?;
        let m_a = self.var.mul_scalar(self.count)?;
        let m_b = batch_var.mul_scalar(batch_count)?;
        let m_2 = m_a.add(&m_b)?.add(
            &delta
                .sqr()?
                .mul_scalar(self.count * batch_count / tot_count)?,
        )?;
        self.var = m_2.mul_scalar(1.0 / tot_count)?;
        self.count = tot_count;
        Ok(())
    }
}
