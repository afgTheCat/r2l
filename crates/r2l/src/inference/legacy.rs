//! Import the unversioned actor layout used before the distribution migration.

use std::collections::HashMap;

use r2l_core::{
    env::Space,
    error::{Error, Result},
    networks::NetworkConfig,
    tensor::VecTensor,
};
use safetensors::{SafeTensors, tensor::TensorView};

use super::{InferenceBackend, InferenceConfig};

fn leaves<'a>(
    space: &'a Space<VecTensor>,
    path: Vec<usize>,
    output: &mut Vec<(Vec<usize>, &'a Space<VecTensor>)>,
) {
    let children: Vec<_> = match space {
        Space::Tuple(spaces) => spaces.iter().collect(),
        Space::Dict(spaces) => spaces.values().collect(),
        _ => {
            output.push((path, space));
            return;
        }
    };
    for (index, child) in children.into_iter().enumerate() {
        let mut child_path = path.clone();
        child_path.push(index);
        leaves(child, child_path, output);
    }
}

pub(super) fn upgrade_actor(bytes: &[u8], config: &InferenceConfig) -> Result<Vec<u8>> {
    let stored = SafeTensors::deserialize(bytes).map_err(Error::wrap)?;
    let mut policies = Vec::new();
    leaves(
        &config.policy_builder.action_space,
        Vec::new(),
        &mut policies,
    );
    let candle = matches!(config.backend, InferenceBackend::Candle(_));
    let mut candle_names = HashMap::new();
    let mut burn_prefixes = Vec::new();
    for (flat_index, (path, space)) in policies.iter().enumerate() {
        if candle {
            let NetworkConfig::Mlp(network) = &config.policy_builder.network else {
                return Err(Error::Unsupported {
                    operation: "load legacy Candle actor".into(),
                    details: "Candle CNN artifacts are unsupported".into(),
                });
            };
            let prefix = path.iter().fold("policy".to_owned(), |prefix, index| {
                format!("{prefix}.{index}")
            });
            for layer in 0..=network.hidden_layers.len() {
                for parameter in ["weight", "bias"] {
                    candle_names.insert(
                        format!("{prefix}{layer}.{parameter}"),
                        format!("{prefix}.{layer}.{parameter}"),
                    );
                }
            }
        } else {
            let variant = match space {
                Space::Discrete(_) => "Categorical",
                Space::Box { .. } => "Diag",
                Space::MultiBinary { .. } => "MultiBernoulli",
                Space::MultiDiscrete { .. } => "MultiCategorical",
                Space::Tuple(_) | Space::Dict(_) => unreachable!("only leaf spaces are collected"),
            };
            // The old Burn container flattened nested action spaces and included enum names.
            let old = if path.is_empty() {
                String::new()
            } else {
                format!("policies.{flat_index}.{variant}.")
            };
            let new = path.iter().fold(String::new(), |prefix, index| {
                format!("{prefix}policies.{index}.")
            });
            burn_prefixes.push((old, new));
        }
    }
    let tensors = stored
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            let new_name = if candle {
                candle_names
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| name.clone())
            } else {
                burn_prefixes
                    .iter()
                    .find_map(|(old, new)| {
                        name.strip_prefix(old).map(|suffix| {
                            let suffix = suffix
                                .strip_prefix("mu_net.")
                                .map_or_else(|| suffix.to_owned(), |tail| format!("mean.{tail}"));
                            format!("{new}{suffix}")
                        })
                    })
                    .unwrap_or_else(|| name.clone())
            };
            let mut shape = tensor.shape().to_vec();
            if candle && name.ends_with(".log_std") && shape.len() == 1 {
                shape.insert(0, 1);
            }
            Ok((
                new_name,
                TensorView::new(tensor.dtype(), shape, tensor.data()).map_err(Error::wrap)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize(tensors, None).map_err(Error::wrap)
}
