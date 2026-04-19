/// Constructs a backend-specific loss bundle from policy and value loss terms.
///
/// On-policy algorithms compute the policy and value losses generically, then
/// use this trait to package them into the `LearningModule::Losses` type.
pub trait FromPolicyValueLosses<T> {
    /// Builds a loss bundle from already-computed policy and value losses.
    fn from_policy_value_losses(policy_loss: T, value_loss: T) -> Self;
}
