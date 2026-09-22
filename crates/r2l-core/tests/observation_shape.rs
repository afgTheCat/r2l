use std::collections::BTreeMap;

use r2l_core::{
    env::{EnvDescription, Space},
    tensor::VecTensor,
};

#[test]
fn observation_shapes_follow_the_encoded_space() {
    let cases = [
        (Space::Discrete(5), vec![5], 5),
        (
            Space::Box {
                min: None,
                max: None,
                shape: vec![8],
            },
            vec![8],
            8,
        ),
        (
            Space::Box {
                min: None,
                max: None,
                shape: vec![3, 84, 84],
            },
            vec![3, 84, 84],
            3 * 84 * 84,
        ),
        (
            Space::Box {
                min: None,
                max: None,
                shape: vec![],
            },
            vec![],
            1,
        ),
        (Space::MultiBinary { shape: vec![2, 3] }, vec![2, 3], 6),
        (
            Space::MultiDiscrete {
                nvec: VecTensor::new(vec![2., 3., 4., 5.], vec![2, 2]).unwrap(),
                shape: vec![2, 2],
            },
            vec![2, 2],
            4,
        ),
        (
            Space::Tuple(vec![
                Space::Discrete(3),
                Space::MultiBinary { shape: vec![2, 2] },
            ]),
            vec![7],
            7,
        ),
        (
            Space::Dict(BTreeMap::from([
                ("position".into(), Space::MultiBinary { shape: vec![2, 3] }),
                ("state".into(), Space::Tuple(vec![Space::Discrete(4)])),
            ])),
            vec![10],
            10,
        ),
    ];

    for (space, shape, size) in cases {
        assert_eq!(space.observation_shape(), shape);
        let description = EnvDescription::new(space, Space::Discrete(2));
        assert_eq!(description.observation_shape(), shape);
        assert_eq!(description.observation_size(), size);
        assert_eq!(shape.iter().product::<usize>(), size);
    }
}
