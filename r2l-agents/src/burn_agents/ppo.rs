use anyhow::Result;
use burn::{
    module::{AutodiffModule, ModuleDisplay},
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_burn_lm::{
    burn_rollout_buffer::{
        BurnRolloutBuffer, RolloutBatchIterator, calculate_advantages_and_returns,
    },
    learning_module::{ParalellActorCriticLM, PolicyValuesLosses},
};
use r2l_core::policies::ValueFunction;
use r2l_core::utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer};
use r2l_core::{agents::Agent, distributions::Distribution};
use r2l_core::{agents::TensorOfAgent, policies::LearningModule};

struct BurnPPO<
    B: AutodiffBackend,
    D: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>,
> where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    lm: ParalellActorCriticLM<B, D>,
    pub clip_range: f32,
    pub sample_size: usize,
    pub gamma: f32,
    pub lambda: f32,
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
            let logp = distr.log_probs(&batch.observations, &batch.actions)?;
            let values_pred = self.lm.calculate_values(&batch.observations)?;
            let value_loss = (values_pred.clone() * values_pred).mean();
            let logp_diff = logp - batch.logp_old;
            let ratio = logp_diff.exp();
            let clip_adv = ratio
                .clone()
                .clamp(1. - self.clip_range, 1. + self.clip_range)
                * batch.advantages.clone();
            let policy_loss = (-(ratio * batch.advantages).min_pair(clip_adv)).mean();
            // TODO: add hook result
            // let hook_result =
            //     self.hooks
            //         .batch_hook(ppo, &batch, &mut policy_loss, &mut value_loss, &ppo_data)?;
            self.lm.update(PolicyValuesLosses {
                policy_loss,
                value_loss,
            })?;
            // TODO: add matching on hook result
            // match hook_result {
            //     HookResult::Break => return Ok(()),
            //     HookResult::Continue => {}
            // }
        }
    }

    fn learning_loop(
        &mut self,
        rollouts: &Vec<BurnRolloutBuffer<B>>,
        advantages: &Advantages,
        returns: &Returns,
        logps: &Logps,
    ) -> Result<()> {
        let mut batch_iter =
            RolloutBatchIterator::new(rollouts, advantages, returns, logps, self.sample_size);
        self.batching_loop(&mut batch_iter)?;
        Ok(())
    }
}

fn uplift_tensor<const N: usize, B: AutodiffBackend>(
    t: Tensor<B::InnerBackend, N>,
) -> Tensor<B, N> {
    let device = Default::default();
    let data = t.into_data();
    Tensor::from_data(data, &device)
}

impl<B: AutodiffBackend, D: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>>
    Agent for BurnPPO<B, D>
where
    <D as AutodiffModule<B>>::InnerModule:
        ModuleDisplay + Distribution<Tensor = Tensor<B::InnerBackend, 1>>,
{
    type Dist = D::InnerModule;

    fn distribution(&self) -> Self::Dist {
        self.lm.model.distr.valid()
    }

    fn learn(&mut self, rollouts: Vec<RolloutBuffer<TensorOfAgent<Self>>>) -> Result<()> {
        let rollouts: Vec<RolloutBuffer<Tensor<B, 1>>> = rollouts
            .into_iter()
            .map(
                |RolloutBuffer {
                     states,
                     actions,
                     rewards,
                     dones,
                 }| {
                    RolloutBuffer {
                        states: states.into_iter().map(uplift_tensor).collect(),
                        actions: actions.into_iter().map(uplift_tensor).collect(),
                        rewards,
                        dones,
                    }
                },
            )
            .collect();

        let mut rollouts = rollouts
            .into_iter()
            .map(|r| BurnRolloutBuffer::new(r))
            .collect::<Vec<_>>();

        let (mut advantages, mut returns) =
            calculate_advantages_and_returns(&rollouts, &self.lm, self.gamma, self.lambda);
        let logps = rollouts
            .iter()
            .map(|roll| {
                let states = &roll.0.states[0..roll.0.states.len() - 1];
                let actions = &roll.0.actions;
                self.lm.model.distr.log_probs(states, actions).map(|t| {
                    let t: Tensor<B, 1> = t.squeeze(0);
                    t.to_data().to_vec().unwrap()
                })
            })
            .collect::<Result<Vec<Vec<f32>>>>()?;
        self.learning_loop(&rollouts, &advantages, &returns, &Logps(logps))?;
        Ok(())
    }
}
