use std::{marker::PhantomData, sync::mpsc::Sender};

use anyhow::Result;
use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::{
    HookResult,
    on_policy_algorithms::{
        Advantages, Returns,
        a2c::{A2CBatchData, A2CHook, A2CParams},
    },
};
use r2l_burn::learning_module::{
    BurnPolicy, PolicyValueLosses as BurnPolicyValueLosses,
    PolicyValueModule as BurnPolicyValueModule,
};
use r2l_candle::learning_module::{
    PolicyValueLosses as CandlePolicyValueLosses, PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{
    buffers::TrajectoryContainer, models::Policy,
    on_policy::learning_module::OnPolicyLearningModule,
};

#[derive(Debug, Clone)]
pub struct A2CBatchStats {
    pub entropy_loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
}

#[derive(Default, Debug, Clone)]
pub struct A2CStats {
    pub batch_stats: Vec<A2CBatchStats>,
    pub std: Option<f32>,
    pub average_reward: f32,
    pub learning_rate: f64,
}

impl A2CStats {
    pub fn collect_batch_data(&mut self, batch_stats: A2CBatchStats) {
        self.batch_stats.push(batch_stats);
    }
}

pub(crate) struct DefaultA2CHookReporter {
    report: A2CStats,
    tx: Sender<A2CStats>,
    unfinished_episode_rewards: Vec<f32>,
    latest_average_reward: f32,
}

impl DefaultA2CHookReporter {
    pub fn new(tx: Sender<A2CStats>, n_envs: usize) -> Self {
        Self {
            report: A2CStats::default(),
            tx,
            unfinished_episode_rewards: vec![0.; n_envs],
            latest_average_reward: 0.,
        }
    }
}

impl DefaultA2CHookReporter {
    fn update_average_reward<T: TrajectoryContainer>(&mut self, buffers: &[T]) {
        let mut completed_episode_rewards = vec![];
        for (running_reward, buffer) in self
            .unfinished_episode_rewards
            .iter_mut()
            .zip(buffers.iter())
        {
            for (reward, done) in buffer.rewards().zip(buffer.dones()) {
                *running_reward += reward;
                if done {
                    completed_episode_rewards.push(*running_reward);
                    *running_reward = 0.;
                }
            }
        }

        if !completed_episode_rewards.is_empty() {
            self.latest_average_reward = completed_episode_rewards.iter().sum::<f32>()
                / completed_episode_rewards.len() as f32;
        }
        self.report.average_reward = self.latest_average_reward;
    }

    fn send_report(&mut self) {
        self.tx.send(std::mem::take(&mut self.report)).unwrap();
        self.report.average_reward = self.latest_average_reward;
    }
}

pub struct DefaultA2CHook<T = ()> {
    pub(crate) normalize_advantage: bool,
    pub(crate) entropy_coeff: f32,
    pub(crate) vf_coeff: Option<f32>,
    pub(crate) gradient_clipping: Option<f32>,
    pub(crate) reporter: Option<DefaultA2CHookReporter>,
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> A2CHook<BurnPolicyValueModule<B, D>>
    for DefaultA2CHook<BurnPolicyValueModule<B, D>>
{
    fn before_learning_hook<
        C: TrajectoryContainer<
            Tensor = <BurnPolicyValueModule<B, D> as OnPolicyLearningModule>::InferenceTensor,
        >,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueModule<B, D>,
        _buffers: &[C],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        if self.normalize_advantage {
            advantages.normalize();
        }
        if let Some(max_grad_norm) = self.gradient_clipping {
            module.set_grad_clipping(GradientClipping::Norm(max_grad_norm));
        }
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueModule<B, D>,
        losses: &mut BurnPolicyValueLosses<B>,
        data: &A2CBatchData<burn::Tensor<B, 1>>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let entropy_loss = entropy.neg() * self.entropy_coeff;
        if let Some(DefaultA2CHookReporter { report, .. }) = &mut self.reporter {
            report.collect_batch_data(A2CBatchStats {
                policy_loss: losses.policy_loss.to_data().to_vec::<f32>().unwrap()[0],
                entropy_loss: entropy_loss.to_data().to_vec::<f32>().unwrap()[0],
                value_loss: losses.value_loss.to_data().to_vec::<f32>().unwrap()[0],
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss);
        }
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<
        C: TrajectoryContainer<
            Tensor = <BurnPolicyValueModule<B, D> as OnPolicyLearningModule>::InferenceTensor,
        >,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueModule<B, D>,
        buffers: &[C],
    ) -> Result<HookResult> {
        if let Some(reporter) = &mut self.reporter {
            reporter.update_average_reward(buffers);
            reporter.report.std = module.policy().std().ok();
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report();
        }
        Ok(HookResult::Continue)
    }
}

impl A2CHook<CandlePolicyValueModule> for DefaultA2CHook<CandlePolicyValueModule> {
    fn before_learning_hook<
        B: TrajectoryContainer<
            Tensor = <CandlePolicyValueModule as OnPolicyLearningModule>::InferenceTensor,
        >,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueModule,
        _buffers: &[B],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        if self.normalize_advantage {
            advantages.normalize();
        }
        module.set_grad_clipping(self.gradient_clipping);
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueModule,
        losses: &mut CandlePolicyValueLosses,
        data: &A2CBatchData<candle_core::Tensor>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
        if let Some(DefaultA2CHookReporter { report, .. }) = &mut self.reporter {
            report.collect_batch_data(A2CBatchStats {
                policy_loss: losses.policy_loss.to_scalar()?,
                entropy_loss: entropy_loss.to_scalar()?,
                value_loss: losses.value_loss.to_scalar()?,
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss)?;
        }
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<B: TrajectoryContainer<Tensor = candle_core::Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueModule,
        buffers: &[B],
    ) -> Result<HookResult> {
        if let Some(reporter) = &mut self.reporter {
            reporter.update_average_reward(buffers);
            reporter.report.std = module.policy().std().ok();
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report();
        }
        Ok(HookResult::Continue)
    }
}
