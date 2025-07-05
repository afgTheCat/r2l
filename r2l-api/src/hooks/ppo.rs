use candle_core::{Result, Tensor};
use r2l_agents::ppo::{
    hooks::{HookResult, PPOBatchData},
    ppo2::PPOHooksTrait,
};
use r2l_core::{
    distributions::Distribution,
    policies::PolicyWithValueFunction,
    tensors::{PolicyLoss, ValueLoss},
    utils::rollout_buffer::{Advantages, Returns, RolloutBatch, RolloutBuffer},
};
use std::marker::PhantomData;

pub struct DefaultPPOHooks<P: PolicyWithValueFunction> {
    ent_coeff: f32,
    vf_coeff: f32,
    target_kl: Option<f32>,
    total_epochs: usize,
    current_epoch: usize,
    p: PhantomData<P>,
}

impl<P: PolicyWithValueFunction> Default for DefaultPPOHooks<P> {
    fn default() -> Self {
        Self {
            ent_coeff: 0.,
            vf_coeff: 0.,
            target_kl: None,
            total_epochs: 10,
            current_epoch: 1,
            p: PhantomData,
        }
    }
}

impl<P: PolicyWithValueFunction> PPOHooksTrait<P> for DefaultPPOHooks<P> {
    fn before_learning_hook(
        &mut self,
        _policy: &mut P,
        _rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        advantages.normalize();
        Ok(HookResult::Continue)
    }

    fn rollout_hook(
        &mut self,
        _policy: &mut P,
        _rollout_buffers: &Vec<RolloutBuffer>,
    ) -> Result<r2l_agents::ppo::hooks::HookResult> {
        if self.current_epoch == self.total_epochs {
            self.current_epoch = 1;
            Ok(HookResult::Break)
        } else {
            self.current_epoch += 1;
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        policy: &mut P,
        _rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> Result<HookResult> {
        if self.ent_coeff != 0. {
            let entropy = policy.distribution().entropy()?;
            let entropy_loss =
                (Tensor::full(self.ent_coeff, (), &candle_core::Device::Cpu)? * entropy.neg()?)?;
            *policy_loss = PolicyLoss(policy_loss.add(&entropy_loss)?);
        }
        if self.vf_coeff != 0. {
            let vf_coeff = Tensor::full(self.vf_coeff, (), &candle_core::Device::Cpu)?;
            *value_loss = ValueLoss(value_loss.broadcast_mul(&vf_coeff)?);
        }
        if self.vf_coeff != 0. {}
        let approx_kl = -(data.logp_diff.mean_all()?.to_scalar::<f32>()?);
        if let Some(target_kl) = self.target_kl
            && approx_kl > 1.5 * target_kl
        {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }
}
