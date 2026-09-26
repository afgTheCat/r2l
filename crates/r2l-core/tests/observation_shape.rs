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
                shape: vec![8].into(),
            },
            vec![8],
            8,
        ),
        (
            Space::Box {
                min: None,
                max: None,
                shape: vec![3, 84, 84].into(),
            },
            vec![3, 84, 84],
            3 * 84 * 84,
        ),
        (
            Space::Box {
                min: None,
                max: None,
                shape: vec![].into(),
            },
            vec![],
            1,
        ),
        (
            Space::MultiBinary {
                shape: vec![2, 3].into(),
            },
            vec![2, 3],
            6,
        ),
        (
            Space::MultiDiscrete {
                nvec: VecTensor::new(vec![2., 3., 4., 5.], vec![2, 2]).unwrap(),
                shape: vec![2, 2].into(),
            },
            vec![2, 2],
            4,
        ),
        (
            Space::Tuple(vec![
                Space::Discrete(3),
                Space::MultiBinary {
                    shape: vec![2, 2].into(),
                },
            ]),
            vec![7],
            7,
        ),
        (
            Space::Dict(BTreeMap::from([
                (
                    "position".into(),
                    Space::MultiBinary {
                        shape: vec![2, 3].into(),
                    },
                ),
                ("state".into(), Space::Tuple(vec![Space::Discrete(4)])),
            ])),
            vec![10],
            10,
        ),
    ];

    for (space, shape, size) in cases {
        assert_eq!(space.observation_shape().dims(), shape.as_slice());
        let description = EnvDescription::new(space, Space::Discrete(2));
        assert_eq!(description.observation_shape().dims(), shape.as_slice());
        assert_eq!(description.observation_size(), size);
        assert_eq!(shape.iter().product::<usize>(), size);
    }
}
