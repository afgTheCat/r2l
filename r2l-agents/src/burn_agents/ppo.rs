use anyhow::Result;
use burn::{
    module::{AutodiffModule, ModuleDisplay},
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_burn_lm::{
    burn_rollout_buffer::{RolloutBatch, RolloutBatchIterator},
    learning_module::ParalellActorCriticLM,
};
use r2l_core::distributions::Distribution;
use r2l_core::policies::ValueFunction;

struct BurnPPO<
    B: AutodiffBackend,
    D: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>,
> where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    lm: ParalellActorCriticLM<B, D>,
}

impl<B: AutodiffBackend, D: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>>
    BurnPPO<B, D>
where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator<B>) -> Result<()> {
        loop {
            let distr = &self.lm.model.distr;
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            // let logp = distr.log_probs(batch.observations.clone(), batch.actions.clone());
            // let values_pred = self.lm.calculate_values(batch.observations);
            // let logp_diff = LogpDiff((logp.deref() - &batch.logp_old)?);
            // let ratio = logp_diff.exp()?;
            // let clip_adv = (ratio.clamp(1. - ppo.clip_range, 1. + ppo.clip_range)?
            //     * batch.advantages.clone())?;
            // let mut policy_loss = PolicyLoss(
            //     Tensor::minimum(&(&ratio * &batch.advantages)?, &clip_adv)?
            //         .neg()?
            //         .mean_all()?,
            // );
            // let ppo_data = PPOBatchData {
            //     logp,
            //     values_pred,
            //     logp_diff,
            //     ratio,
            // };
            // let hook_result =
            //     self.hooks
            //         .batch_hook(ppo, &batch, &mut policy_loss, &mut value_loss, &ppo_data)?;
            // ppo.learning_module.update(PolicyValuesLosses {
            //     policy_loss,
            //     value_loss,
            // })?;
            // match hook_result {
            //     HookResult::Break => return Ok(()),
            //     HookResult::Continue => {}
            // }
        }
    }
}
