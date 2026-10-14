pub(crate) mod actor_wrapper;
pub(crate) mod buffer_wrapper;

#[must_use]
/// Returns the arithmetic mean of the values.
///
/// # Panics
/// Panics if `v` is empty.
pub fn slice_mean(v: &[f32]) -> f32 {
    assert!(!v.is_empty(), "Can only mean non zero vector");
    v.iter().sum::<f32>() / v.len() as f32
}
