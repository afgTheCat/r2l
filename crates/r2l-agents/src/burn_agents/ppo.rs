use anyhow::Result;
use burn::{
    module::{AutodiffModule, ModuleDisplay},
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_burn_lm::{
    burn_rollout_buffer::{
        BurnRolloutBuffer, RolloutBatch, RolloutBatchIterator, calculate_advantages_and_returns,
    },
    learning_module::{ParalellActorCriticLM, PolicyValuesLosses},
};
use r2l_core::policies::ValueFunction;
use r2l_core::utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer};
use r2l_core::{agents::Agent, distributions::Policy};
use r2l_core::{agents::TensorOfAgent, policies::LearningModule};

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub enum HookResult {
    Continue,
    Break,
}

pub struct BurnPPOCore<
    B: AutodiffBackend,
    D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = Tensor<B, 1>>,
> where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    pub lm: ParalellActorCriticLM<B, D>,
    pub clip_range: f32,
    pub sample_size: usize,
    pub gamma: f32,
    pub lambda: f32,
}

impl<B: AutodiffBackend, D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = Tensor<B, 1>>>
    BurnPPOCore<B, D>
where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    pub fn new(
        lm: ParalellActorCriticLM<B, D>,
        clip_range: f32,
        sample_size: usize,
        gamma: f32,
        lambda: f32,
    ) -> Self {
        Self {
            lm,
            clip_range,
            sample_size,
            gamma,
            lambda,
        }
    }
}

pub trait BurnPPPHooksTrait<
    B: AutodiffBackend,
    D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = Tensor<B, 1>>,
> where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    fn before_learning_hook(
        &mut self,
        agent: &mut BurnPPOCore<B, D>,
        rollout_buffers: &mut Vec<BurnRolloutBuffer<B>>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook(
        &mut self,
        agent: &mut BurnPPOCore<B, D>,
        rollout_buffers: &Vec<BurnRolloutBuffer<B>>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        agent: &mut BurnPPOCore<B, D>,
        rollout_batch: &RolloutBatch<B>,
        // policy_loss: &mut PolicyLoss,
        // value_loss: &mut ValueLoss,
        // data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct EmptyBurnPPOHooks;

impl<B: AutodiffBackend, D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = Tensor<B, 1>>>
    BurnPPPHooksTrait<B, D> for EmptyBurnPPOHooks
where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
}

pub struct BurnPPO<
    B: AutodiffBackend,
    D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = Tensor<B, 1>>,
> where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    pub core: BurnPPOCore<B, D>,
    pub hooks: Box<dyn BurnPPPHooksTrait<B, D>>,
}

impl<B: AutodiffBackend, D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = Tensor<B, 1>>>
    BurnPPO<B, D>
where
    <D as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    pub fn new(core: BurnPPOCore<B, D>, hooks: Box<dyn BurnPPPHooksTrait<B, D>>) -> Self {
        Self { core, hooks }
    }

    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator<B>) -> Result<()> {
        loop {
            let distr = &self.core.lm.model.distr;
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let logp = distr.log_probs(&batch.observations, &batch.actions)?;
            let values_pred = self.core.lm.calculate_values(&batch.observations)?;
            let value_loss = (values_pred.clone() * values_pred).mean();
            let logp_diff = logp - batch.logp_old.clone();
            let ratio = logp_diff.exp();
            let clip_adv = ratio
                .clone()
                .clamp(1. - self.core.clip_range, 1. + self.core.clip_range)
                * batch.advantages.clone();
            let policy_loss = (-(ratio * batch.advantages.clone()).min_pair(clip_adv)).mean();
            // TODO: add the rest of the hooks
            let hook_result = self.hooks.batch_hook(
                &mut self.core,
                &batch,
                // &mut policy_loss,
                // &mut value_loss,
                // &ppo_data,
            )?;
            self.core.lm.update(PolicyValuesLosses {
                policy_loss,
                value_loss,
            })?;
            match hook_result {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
        }
    }

    fn learning_loop(
        &mut self,
        rollouts: &Vec<BurnRolloutBuffer<B>>,
        advantages: &Advantages,
        returns: &Returns,
        logps: &Logps,
    ) -> Result<()> {
        loop {
            let mut batch_iter = RolloutBatchIterator::new(
                rollouts,
                advantages,
                returns,
                logps,
                self.core.sample_size,
            );
            self.batching_loop(&mut batch_iter)?;
            let rollout_hook_res = self.hooks.rollout_hook(&mut self.core, &rollouts);
            process_hook_result!(rollout_hook_res);
        }
    }
}

// TODO: is there a better way?
fn uplift_tensor<const N: usize, B: AutodiffBackend>(
    t: Tensor<B::InnerBackend, N>,
) -> Tensor<B, N> {
    let device = Default::default();
    let data = t.into_data();
    Tensor::from_data(data, &device)
}

impl<B: AutodiffBackend, D: AutodiffModule<B> + ModuleDisplay + Policy<Tensor = Tensor<B, 1>>> Agent
    for BurnPPO<B, D>
where
    <D as AutodiffModule<B>>::InnerModule:
        ModuleDisplay + Policy<Tensor = Tensor<B::InnerBackend, 1>>,
{
    type Dist = D::InnerModule;

    fn distribution(&self) -> Self::Dist {
        self.core.lm.model.distr.valid()
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
        let (mut advantages, mut returns) = calculate_advantages_and_returns(
            &rollouts,
            &self.core.lm,
            self.core.gamma,
            self.core.lambda,
        );
        let before_learning_hook_res = self.hooks.before_learning_hook(
            &mut self.core,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        );
        process_hook_result!(before_learning_hook_res);
        let logps = rollouts
            .iter()
            .map(|roll| {
                let states = &roll.0.states[0..roll.0.states.len() - 1];
                let actions = &roll.0.actions;
                self.core
                    .lm
                    .model
                    .distr
                    .log_probs(states, actions)
                    .map(|t| t.to_data().to_vec().unwrap())
            })
            .collect::<Result<Vec<Vec<f32>>>>()?;
        self.learning_loop(&rollouts, &advantages, &returns, &Logps(logps))?;
        Ok(())
    }
}
