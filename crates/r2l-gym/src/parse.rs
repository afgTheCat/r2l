use pyo3::{
    Bound, PyResult,
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
        parse_box_space(space)
    } else if is_space("MultiDiscrete")? {
        parse_multi_discrete_space(space)
    } else if is_space("MultiBinary")? {
        parse_multi_binary_space(space)
    } else if is_space("Tuple")? {
        parse_tuple_space(space, gym_spaces)
    } else if is_space("Dict")? {
        parse_dict_space(space, gym_spaces)
    } else {
        todo!();
    }
}

fn parse_box_space(space: &Bound<'_, PyAny>) -> PyResult<Space<TensorData>> {
    let shape: Vec<usize> = space.getattr("shape")?.extract()?;
    let low: Vec<f32> = space.getattr("low")?.extract()?;
    let high: Vec<f32> = space.getattr("high")?.extract()?;

    Ok(Space::Continuous {
        min: Some(TensorData::new(low, shape.clone())),
        max: Some(TensorData::new(high, shape.clone())),
        shape,
    })
}

fn parse_multi_discrete_space(space: &Bound<'_, PyAny>) -> PyResult<Space<TensorData>> {
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
}

fn parse_multi_binary_space(space: &Bound<'_, PyAny>) -> PyResult<Space<TensorData>> {
    let shape: Vec<usize> = space.getattr("shape")?.extract()?;
    Ok(Space::multi_binary(shape))
}

fn parse_tuple_space(
    space: &Bound<'_, PyAny>,
    gym_spaces: &Bound<'_, PyModule>,
) -> PyResult<Space<TensorData>> {
    let spaces = space.getattr("spaces")?;
    let mut parsed = Vec::new();
    for space in spaces.try_iter()? {
        parsed.push(parse_gym_space(&space?, gym_spaces)?);
    }
    Ok(Space::tuple(parsed))
}

fn parse_dict_space(
    space: &Bound<'_, PyAny>,
    gym_spaces: &Bound<'_, PyModule>,
) -> PyResult<Space<TensorData>> {
    let spaces = space.getattr("spaces")?;
    let mut parsed = std::collections::BTreeMap::new();
    for item in spaces.call_method0("items")?.try_iter()? {
        let item = item?;
        let key = item.get_item(0)?.extract()?;
        let space = item.get_item(1)?;
        parsed.insert(key, parse_gym_space(&space, gym_spaces)?);
    }
    Ok(Space::dict(parsed))
}
