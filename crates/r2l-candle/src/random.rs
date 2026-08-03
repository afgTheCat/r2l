use std::collections::hash_map::Entry;

use candle_core::{DType, Device, Result, Shape, Tensor, Var};
use candle_nn::{Init, VarBuilder, VarMap, init::NormalOrUniform, var_builder::SimpleBackend};
use rand::distr::{Distribution, Uniform};
use rand_distr::{Normal, StandardNormal};

struct SeededVarMap(VarMap);

impl SimpleBackend for SeededVarMap {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        init: Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mut variables = self.0.data().lock().unwrap();
        let variable = match variables.entry(name.to_string()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(initialized_var(&shape, init, dtype, device)?),
        };
        if variable.shape() != &shape {
            candle_core::bail!(
                "shape mismatch on {name}: {shape:?} <> {:?}",
                variable.shape()
            );
        }
        Ok(variable.as_tensor().clone())
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.0.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }
}

/// Creates a variable builder that initializes new variables with r2l's seeded RNG.
pub fn seeded_var_builder(varmap: &VarMap, dtype: DType, device: &Device) -> VarBuilder<'static> {
    VarBuilder::from_backend(
        Box::new(SeededVarMap(varmap.clone())),
        dtype,
        device.clone(),
    )
}

pub(crate) fn standard_normal(shape: &Shape, device: &Device) -> Result<Tensor> {
    let values = random_values(shape.elem_count(), StandardNormal);
    Tensor::from_vec(values, shape.clone(), device)
}

fn initialized_var(shape: &Shape, init: Init, dtype: DType, device: &Device) -> Result<Var> {
    let count = shape.elem_count();
    let values = match init {
        Init::Const(_) => return init.var(shape.clone(), dtype, device),
        Init::Uniform { lo, up } => {
            random_values(count, Uniform::new(lo as f32, up as f32).unwrap())
        }
        Init::Randn { mean, stdev } => {
            random_values(count, Normal::new(mean as f32, stdev as f32).unwrap())
        }
        Init::Kaiming {
            dist,
            fan,
            non_linearity,
        } => {
            let std = non_linearity.gain() / (fan.for_shape(shape) as f64).sqrt();
            match dist {
                NormalOrUniform::Normal => {
                    random_values(count, Normal::new(0.0, std as f32).unwrap())
                }
                NormalOrUniform::Uniform => {
                    let bound = (3f64.sqrt() * std) as f32;
                    random_values(count, Uniform::new(-bound, bound).unwrap())
                }
            }
        }
    };
    let tensor = Tensor::from_vec(values, shape.clone(), device)?.to_dtype(dtype)?;
    Var::from_tensor(&tensor)
}

fn random_values<D: Distribution<f32>>(count: usize, distribution: D) -> Vec<f32> {
    r2l_core::rng::with_rng(|rng| distribution.sample_iter(rng).take(count).collect())
}
