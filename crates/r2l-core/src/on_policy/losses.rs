pub trait FromPolicyValueLosses<T> {
    fn from_policy_value_losses(policy_loss: T, value_loss: T) -> Self;
}
