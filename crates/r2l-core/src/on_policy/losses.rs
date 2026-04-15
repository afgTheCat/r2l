pub trait PolicyValuesLosses<T> {
    fn losses(policy_loss: T, value_loss: T) -> Self;
}
