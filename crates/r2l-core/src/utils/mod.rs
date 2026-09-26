pub(crate) mod actor_wrapper;
pub(crate) mod buffer_wrapper;

pub fn slice_mean(v: &[f32]) -> f32 {
    assert!(v.len() > 0, "Can only mean non zero vector");
    v.iter().sum::<f32>() / v.len() as f32
}
