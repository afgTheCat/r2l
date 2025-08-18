pub mod vpg3;

// use candle_core::{Device, Result};
// use r2l_core::{
//     agents::Agent,
//     policies::PolicyWithValueFunction,
//     utils::rollout_buffer::{
//         RolloutBatch, RolloutBatchIterator, RolloutBuffer, calculate_advantages_and_returns,
//     },
// };

// pub struct VPG<P: PolicyWithValueFunction> {
//     policy: P,
//     device: Device,
//     gamma: f32,
//     lambda: f32,
//     sample_size: usize,
// }
//
// impl<P: PolicyWithValueFunction> VPG<P> {
//     fn train_single_batch(&mut self, batch: RolloutBatch) -> Result<bool> {
//         let policy_loss = batch.advantages.mul(&batch.logp_old)?.neg()?.mean_all()?;
//         let values_pred = self.policy.calculate_values(&batch.observations)?;
//         let value_loss = batch.returns.sub(&values_pred)?.sqr()?.mean_all()?;
//         self.policy.update(&policy_loss, &value_loss)?;
//         Ok(true)
//     }
// }
//
// impl<P: PolicyWithValueFunction> Agent for VPG<P> {
//     type Policy = P;
//
//     fn policy(&self) -> &Self::Policy {
//         &self.policy
//     }
//
//     fn learn(&mut self, rollouts: Vec<RolloutBuffer>) -> candle_core::Result<()> {
//         let (advantages, returns) =
//             calculate_advantages_and_returns(&rollouts, &self.policy, self.gamma, self.lambda);
//         let rollout_batch_iter = RolloutBatchIterator::new(
//             &rollouts,
//             &advantages,
//             &returns,
//             self.sample_size,
//             self.device.clone(),
//         );
//         for batch in rollout_batch_iter {
//             self.train_single_batch(batch)?;
//         }
//         Ok(())
//     }
// }
