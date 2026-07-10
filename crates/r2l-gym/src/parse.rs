use pyo3::{
    Bound, IntoPyObjectExt, PyResult, Python,
    types::{PyAny, PyAnyMethods, PyDict, PyDictMethods, PyModule, PyTuple},
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
    action: &[f32],
    space: &Space<TensorData>,
) -> PyResult<Bound<'py, PyAny>> {
    match space {
        Space::Continuous {
            min: Some(min),
            max: Some(max),
            shape,
            ..
        } => TensorData::new(action.to_vec(), shape.clone())
            .clamp(min, max)
            .into_vec()
            .into_bound_py_any(py),
        Space::Continuous { .. } => action.to_vec().into_bound_py_any(py),
        Space::Discrete(_) => {
            // TODO: remove unwrap
            action
                .iter()
                .position(|i| *i > 0.)
                .unwrap()
                .into_bound_py_any(py)
        }
        Space::MultiDiscrete { .. } => {
            let action: Vec<usize> = action.iter().map(|value| *value as usize).collect();
            action.into_bound_py_any(py)
        }
        Space::MultiBinary { .. } => {
            let action: Vec<usize> = action.iter().map(|value| (*value > 0.) as usize).collect();
            action.into_bound_py_any(py)
        }
        Space::Tuple(spaces) => {
            let actions = parse_child_actions(py, action, spaces)?;
            Ok(PyTuple::new(py, actions)?.into_any())
        }
        Space::Dict(spaces) => {
            let parsed_actions = parse_child_actions(py, action, spaces.values())?;
            let actions = PyDict::new(py);
            for (key, action) in spaces.keys().zip(parsed_actions) {
                actions.set_item(key, action)?;
            }
            Ok(actions.into_any())
        }
    }
}

fn parse_child_actions<'py, 'space>(
    py: Python<'py>,
    action: &[f32],
    spaces: impl IntoIterator<Item = &'space Space<TensorData>>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut offset = 0;
    let mut actions = Vec::new();
    for space in spaces {
        let end = offset + space.size();
        actions.push(parse_action(py, &action[offset..end], space)?);
        offset = end;
    }
    Ok(actions)
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
        Space::Tuple(spaces) => parse_fields(
            spaces
                .iter()
                .enumerate()
                .map(|(idx, space)| Ok((observation.get_item(idx)?, space))),
        ),
        Space::Dict(spaces) => parse_fields(
            spaces
                .iter()
                .map(|(key, space)| Ok((observation.get_item(key)?, space))),
        ),
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

fn parse_fields<'py>(
    fields: impl IntoIterator<Item = PyResult<(Bound<'py, PyAny>, &'py Space<TensorData>)>>,
) -> PyResult<TensorData> {
    let mut data = Vec::new();
    for field in fields {
        let (value, space) = field?;
        data.extend(parse_observation(&value, space)?.into_vec());
    }
    Ok(TensorData::from_vec(data))
}
