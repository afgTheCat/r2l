pub(crate) mod actor_wrapper;
pub(crate) mod buffer_wrapper;

#[must_use]
pub fn slice_mean(v: &[f32]) -> f32 {
    assert!(!v.is_empty(), "Can only mean non zero vector");
    v.iter().sum::<f32>() / v.len() as f32
}
