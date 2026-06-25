use crate::tensor::R2lTensor;

#[derive(Clone, Debug)]
pub struct RunningMeanStd2<T: R2lTensor> {
    pub mean: T,
    pub var: T,
    pub count: f32,
}

impl<T: R2lTensor> RunningMeanStd2<T> {
    pub fn new(shape: Vec<usize>) -> Self {
        let mean = T::zeros(shape.clone());
        let var = T::zeros(shape.clone());
        Self {
            mean,
            var,
            count: 0.,
        }
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

    pub fn update(&mut self, arr: &T) -> anyhow::Result<()> {
        let batch_mean = arr.batch_mean()?;
        let batch_var = arr.biased_var()?;
        let batch_count = arr.batch_count()?;
        self.update_from_moments(batch_mean, batch_var, batch_count)?;
        Ok(())
    }
}

// super simplified running mean, only for vectors
pub struct RunningMeanStd3 {
    mean: Vec<f32>,
    var: Vec<f32>,
    count: f32,
}

fn vector_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.into_iter()
        .zip(b.into_iter())
        .map(|(a, b)| *a - *b)
        .collect()
}

fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.into_iter()
        .zip(b.into_iter())
        .map(|(a, b)| *a + *b)
        .collect()
}

fn vector_mul_scalar(a: &[f32], c: f32) -> Vec<f32> {
    a.into_iter().map(|a| *a * c).collect()
}

fn vector_sqr(a: &[f32]) -> Vec<f32> {
    a.into_iter().map(|a| a * a).collect()
}

// mega simplified view
impl RunningMeanStd3 {
    pub fn new(size: usize) -> Self {
        let mean = vec![0.; size];
        let var = vec![0.; size];
        Self {
            mean,
            var,
            count: 0.,
        }
    }

    fn update_from_moments(
        &mut self,
        batch_mean: Vec<f32>,
        batch_var: Vec<f32>,
        batch_count: f32,
    ) -> anyhow::Result<()> {
        if batch_count == 0.0 {
            return Ok(());
        }
        let tot_count = self.count + batch_count;
        let delta = vector_sub(&batch_mean, &self.mean);
        self.mean = vector_add(
            &self.mean,
            &vector_mul_scalar(&delta, batch_count / tot_count),
        );
        let m_a = vector_mul_scalar(&self.var, self.count);
        let m_b = vector_mul_scalar(&batch_var, batch_count);
        let m_2 = vector_add(
            &vector_add(&m_a, &m_b),
            &vector_mul_scalar(&vector_sqr(&delta), self.count * batch_count / tot_count),
        );
        self.var = vector_mul_scalar(&m_2, 1.0 / tot_count);
        self.count = tot_count;
        Ok(())
    }

    // NOTE: we should only accept
    pub fn update<T: R2lTensor>(&mut self, t: &[T]) -> anyhow::Result<()> {
        let mut datas = vec![];
        let mut shapes = vec![];
        for t in t {
            let (data, shape) = t.to_vec_and_shape();
            anyhow::ensure!(shape.len() == 1, "RunningMeanStd3 only accepts vectors");
            anyhow::ensure!(
                data.len() == self.mean.len(),
                "observation size does not match running mean size"
            );
            datas.push(data);
            shapes.push(shape);
        }

        let batch_count = datas.len() as f32;
        if batch_count == 0.0 {
            return self.update_from_moments(
                vec![0.0; self.mean.len()],
                vec![0.0; self.var.len()],
                batch_count,
            );
        }
        anyhow::ensure!(
            shapes.iter().all(|shape| shape == &shapes[0]),
            "all observations must have the same shape"
        );

        let mut batch_mean = vec![0.0; self.mean.len()];
        for data in &datas {
            for (mean, value) in batch_mean.iter_mut().zip(data) {
                *mean += value;
            }
        }
        for mean in &mut batch_mean {
            *mean /= batch_count;
        }

        let mut batch_var = vec![0.0; self.var.len()];
        for data in &datas {
            for ((var, value), mean) in batch_var.iter_mut().zip(data).zip(&batch_mean) {
                let delta = value - mean;
                *var += delta * delta;
            }
        }
        for var in &mut batch_var {
            *var /= batch_count;
        }

        self.update_from_moments(batch_mean, batch_var, batch_count)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{running_mean::RunningMeanStd2, tensor::TensorData};

    #[test]
    fn test_normalize() {
        // 2d observation
        let mut rm = RunningMeanStd2::new(vec![2]);
        let t1 = TensorData::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
        rm.update(&t1).unwrap();
        println!("{:?}", rm);
    }
}
