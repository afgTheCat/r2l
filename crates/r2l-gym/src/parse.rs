use pyo3::{
    Bound, IntoPyObjectExt, PyResult, Python,
    types::{PyAny, PyAnyMethods, PyModule},
};
use r2l_core::{env::Space, tensor::TensorData};

pub(crate) fn parse_gym_space(
    space: &Bound<'_, PyAny>,
    gym_spaces: &Bound<'_, PyModule>,
) -> PyResult<Space<TensorData>> {
    let is_space = |name| space.is_instance(&gym_spaces.getattr(name)?);
    if is_space("Discrete")? {
        let size = space.getattr("n")?.extract()?;
        Ok(Space::Discrete(size))
    } else if is_space("Box")? {
        let shape: Vec<usize> = space.getattr("shape")?.extract()?;
        let low: Vec<f32> = space.getattr("low")?.extract()?;
        let high: Vec<f32> = space.getattr("high")?.extract()?;
        Ok(Space::Continuous {
            min: Some(TensorData::new(low, shape.clone())),
            max: Some(TensorData::new(high, shape.clone())),
            shape,
        })
    } else if is_space("MultiDiscrete")? {
        let nvec = space.getattr("nvec")?;
        let shape: Vec<usize> = nvec.getattr("shape")?.extract()?;
        let nvec: Vec<usize> = nvec
            .call_method0("flatten")?
            .call_method0("tolist")?
            .extract()?;
        let nvec = nvec.into_iter().map(|n| n as f32).collect();
        Ok(Space::multi_discrete(
            TensorData::new(nvec, shape.clone()),
            shape,
        ))
    } else if is_space("MultiBinary")? {
        let shape = space.getattr("shape")?.extract()?;
        Ok(Space::multi_binary(shape))
    } else if is_space("Tuple")? {
        let spaces = space.getattr("spaces")?;
        let mut parsed = Vec::new();
        for space in spaces.try_iter()? {
            parsed.push(parse_gym_space(&space?, gym_spaces)?);
        }
        Ok(Space::tuple(parsed))
    } else if is_space("Dict")? {
        let spaces = space.getattr("spaces")?;
        let mut parsed = std::collections::BTreeMap::new();
        for item in spaces.call_method0("items")?.try_iter()? {
            let item = item?;
            let key = item.get_item(0)?.extract()?;
            let space = item.get_item(1)?;
            parsed.insert(key, parse_gym_space(&space, gym_spaces)?);
        }
        Ok(Space::dict(parsed))
    } else {
        todo!();
    }
}

pub(crate) fn parse_action<'py>(
    py: Python<'py>,
    action: TensorData,
    space: &Space<TensorData>,
) -> PyResult<Bound<'py, PyAny>> {
    match space {
        Space::Continuous {
            min: Some(min),
            max: Some(max),
            ..
        } => action.clamp(min, max).into_vec().into_bound_py_any(py),
        Space::Continuous { .. } => action.into_vec().into_bound_py_any(py),
        Space::Discrete(_) => {
            let action = action.into_vec();
            // TODO: remove unwrap
            action
                .iter()
                .position(|i| *i > 0.)
                .unwrap()
                .into_bound_py_any(py)
        }
        Space::MultiDiscrete { .. } => {
            todo!();
        }
        Space::MultiBinary { .. } => {
            todo!();
        }
        Space::Tuple(_) => {
            todo!();
        }
        Space::Dict(_) => {
            todo!();
        }
    }
}

pub(crate) fn parse_observation(
    observation: &Bound<'_, PyAny>,
    space: &Space<TensorData>,
) -> PyResult<TensorData> {
    match space {
        Space::Discrete(_) => Ok(TensorData::new(
            vec![observation.extract::<f32>()?],
            vec![1],
        )),
        Space::Continuous { shape, .. }
        | Space::MultiDiscrete { shape, .. }
        | Space::MultiBinary { shape } => parse_tensor_observation(observation, shape),
        Space::Tuple(spaces) => parse_tuple_observation(observation, spaces),
        Space::Dict(spaces) => parse_dict_observation(observation, spaces),
    }
}

fn parse_tensor_observation(
    observation: &Bound<'_, PyAny>,
    shape: &[usize],
) -> PyResult<TensorData> {
    let values = observation
        .call_method0("flatten")?
        .call_method0("tolist")?;
    let values: Vec<f32> = values.extract()?;
    Ok(TensorData::new(values, shape.to_vec()))
}

fn parse_tuple_observation(
    observation: &Bound<'_, PyAny>,
    spaces: &[Space<TensorData>],
) -> PyResult<TensorData> {
    let mut data = Vec::new();
    for (idx, space) in spaces.iter().enumerate() {
        let value = observation.get_item(idx)?;
        data.extend(parse_observation(&value, space)?.into_vec());
    }
    Ok(TensorData::from_vec(data))
}

fn parse_dict_observation(
    observation: &Bound<'_, PyAny>,
    spaces: &std::collections::BTreeMap<String, Space<TensorData>>,
) -> PyResult<TensorData> {
    let mut data = Vec::new();
    for (key, space) in spaces {
        let value = observation.get_item(key)?;
        data.extend(parse_observation(&value, space)?.into_vec());
    }
    Ok(TensorData::from_vec(data))
}
