pub mod a2c3;
pub mod hooks;

// This changes the contorlflow, returning on hook break
macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

// pub struct A2C<P: PolicyWithValueFunction> {
//     pub policy: P,
//     pub hooks: A2CHooks<P>,
//     pub device: Device,
//     pub gamma: f32,
//     pub lambda: f32,
//     pub sample_size: usize,
// }
//
// impl<P: PolicyWithValueFunction> A2C<P> {
//     fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
//         loop {
//             let Some(batch) = batch_iter.next() else {
//                 return Ok(());
//             };
//             let distribution = self.policy.distribution();
//             let logps = distribution.log_probs(&batch.observations, &batch.actions)?;
//             let values_pred = self.policy.calculate_values(&batch.observations)?;
//             let value_loss = &batch.returns.sub(&values_pred)?.sqr()?.mean_all()?;
//             let policy_loss = &batch.advantages.mul(&logps)?.neg()?.mean_all()?;
//             self.policy.update(policy_loss, value_loss)?;
//         }
//     }
// }
//
// impl<P: PolicyWithValueFunction> Agent for A2C<P> {
//     type Policy = P;
//
//     fn policy(&self) -> &Self::Policy {
//         &self.policy
//     }
//
//     fn learn(&mut self, mut rollouts: Vec<RolloutBuffer>) -> candle_core::Result<()> {
//         let (mut advantages, mut returns) =
//             calculate_advantages_and_returns(&rollouts, &self.policy, self.gamma, self.lambda);
//         let before_learning_hook_res = self.hooks.call_before_learning_hook(
//             &mut self.policy,
//             &mut rollouts,
//             &mut advantages,
//             &mut returns,
//         );
//         process_hook_result!(before_learning_hook_res);
//         let mut batch_iter = RolloutBatchIterator::new(
//             &rollouts,
//             &advantages,
//             &returns,
//             self.sample_size,
//             self.device.clone(),
//         );
//         self.batching_loop(&mut batch_iter)?;
//         Ok(())
//     }
// }
