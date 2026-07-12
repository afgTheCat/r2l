use pyo3::{
    Bound, FromPyObject, IntoPyObjectExt, PyResult, Python,
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
        let low = flatten_extract(&space.getattr("low")?)?;
        let high = flatten_extract(&space.getattr("high")?)?;
        Ok(Space::Box {
            min: Some(TensorData::new(low, shape.clone())),
            max: Some(TensorData::new(high, shape.clone())),
            shape,
        })
    } else if is_space("MultiDiscrete")? {
        let nvec = space.getattr("nvec")?;
        let shape: Vec<usize> = nvec.getattr("shape")?.extract()?;
        let nvec: Vec<usize> = flatten_extract(&nvec)?;
        let nvec = nvec.into_iter().map(|n| n as f32).collect();
        Ok(Space::MultiDiscrete {
            nvec: TensorData::new(nvec, shape.clone()),
            shape,
        })
    } else if is_space("MultiBinary")? {
        let shape = space.getattr("shape")?.extract()?;
        Ok(Space::MultiBinary { shape })
    } else if is_space("Tuple")? {
        let spaces = space.getattr("spaces")?;
        let mut parsed = Vec::new();
        for space in spaces.try_iter()? {
            parsed.push(parse_gym_space(&space?, gym_spaces)?);
        }
        Ok(Space::Tuple(parsed))
    } else if is_space("Dict")? {
        let spaces = space.getattr("spaces")?;
        let mut parsed = std::collections::BTreeMap::new();
        for item in spaces.call_method0("items")?.try_iter()? {
            let item = item?;
            let key = item.get_item(0)?.extract()?;
            let space = item.get_item(1)?;
            parsed.insert(key, parse_gym_space(&space, gym_spaces)?);
        }
        Ok(Space::Dict(parsed))
    } else {
        unreachable!()
    }
}

pub(crate) fn parse_action<'py>(
    py: Python<'py>,
    action: &[f32],
    space: &Space<TensorData>,
) -> PyResult<Bound<'py, PyAny>> {
    match space {
        Space::Box {
            min: Some(min),
            max: Some(max),
            shape,
            ..
        } => action_array(
            py,
            TensorData::new(action.to_vec(), shape.clone())
                .clamp(min, max)
                .into_vec(),
            shape,
            "float32",
        ),
        Space::Box { shape, .. } => action_array(py, action.to_vec(), shape, "float32"),
        Space::Discrete(_) => {
            // TODO: remove unwrap
            action
                .iter()
                .position(|i| *i > 0.)
                .unwrap()
                .into_bound_py_any(py)
        }
        Space::MultiDiscrete { shape, .. } => action_array(py, action.to_vec(), shape, "int64"),
        Space::MultiBinary { shape } => action_array(
            py,
            action
                .iter()
                .map(|value| if *value > 0. { 1. } else { 0. })
                .collect(),
            shape,
            "int8",
        ),
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

fn action_array<'py>(
    py: Python<'py>,
    action: Vec<f32>,
    shape: &[usize],
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?
        .call_method1("asarray", (action,))?
        .call_method1("astype", (dtype,))?
        .call_method1("reshape", (shape,))
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

fn flatten_extract<'py, T: FromPyObject<'py>>(value: &Bound<'py, PyAny>) -> PyResult<Vec<T>> {
    value
        .call_method0("flatten")?
        .call_method0("tolist")?
        .extract()
}

pub(crate) fn parse_obs(
    observation: &Bound<'_, PyAny>,
    space: &Space<TensorData>,
) -> PyResult<TensorData> {
    match space {
        Space::Discrete(size) => {
            let idx: usize = observation.extract()?;
            let mut values = vec![0.; *size];
            values[idx] = 1.;
            Ok(TensorData::new(values, vec![*size]))
        }
        Space::Box { shape, .. }
        | Space::MultiDiscrete { shape, .. }
        | Space::MultiBinary { shape } => {
            let values = flatten_extract(observation)?;
            Ok(TensorData::new(values, shape.to_vec()))
        }
        Space::Tuple(spaces) => parse_obs_fields(
            spaces
                .iter()
                .enumerate()
                .map(|(idx, space)| Ok((observation.get_item(idx)?, space))),
        ),
        Space::Dict(spaces) => parse_obs_fields(
            spaces
                .iter()
                .map(|(key, space)| Ok((observation.get_item(key)?, space))),
        ),
    }
}

fn parse_obs_fields<'py>(
    fields: impl IntoIterator<Item = PyResult<(Bound<'py, PyAny>, &'py Space<TensorData>)>>,
) -> PyResult<TensorData> {
    let mut data = Vec::new();
    for field in fields {
        let (value, space) = field?;
        data.extend(parse_obs(&value, space)?.into_vec());
    }
    Ok(TensorData::from_vec(data))
}

#[cfg(test)]
mod tests {
    use pyo3::{PyResult, Python, types::PyAnyMethods};

    use super::{parse_action, parse_gym_space};

    #[test]
    fn fundamental_space_shapes_match_gymnasium() -> PyResult<()> {
        Python::with_gil(|py| {
            let spaces = py.import("gymnasium.spaces")?;
            let gym_spaces = [
                spaces.getattr("Discrete")?.call1((3,))?,
                spaces.getattr("Box")?.call1((-1., 1., (2, 3)))?,
                spaces.getattr("MultiDiscrete")?.call1((vec![2, 3],))?,
                spaces.getattr("MultiBinary")?.call1(((2, 3),))?,
            ];

            for gym_space in gym_spaces {
                let gym_shape: Vec<usize> = gym_space.getattr("shape")?.extract()?;
                let space = parse_gym_space(&gym_space, &spaces)?;
                assert_eq!(space.shape(), Some(gym_shape.as_slice()));
            }
            Ok(())
        })
    }

    #[test]
    fn multi_binary_actions_keep_gymnasium_shape() -> PyResult<()> {
        Python::with_gil(|py| {
            let spaces = py.import("gymnasium.spaces")?;
            let gym_space = spaces.getattr("MultiBinary")?.call1(((2, 3),))?;
            let space = parse_gym_space(&gym_space, &spaces)?;
            let action = parse_action(py, &[1., 0., 1., 0., 1., 0.], &space)?;
            let shape: Vec<usize> = action.getattr("shape")?.extract()?;

            assert_eq!(shape, vec![2, 3]);
            assert!(gym_space.call_method1("contains", (action,))?.is_truthy()?);
            Ok(())
        })
    }
}
