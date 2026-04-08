use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::AdamWConfig,
    tensor::backend::AutodiffBackend,
};
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};
use r2l_burn_lm::{
    distributions::DistributionKind,
    learning_module::{ActorCriticLMKind, ParalellActorCriticLM, ParalellActorModel},
};
use r2l_core::{agents::Agent, sampler::buffer::TrajectoryContainer};

use crate::{
    agents::AgentBuilder,
    builders::distribution::{ActionSpaceType, DistributionBuilder, DistributionType},
    hooks::ppo::{StandardPPOHook, StandardPPOHookBuilder},
};

// TODO: maybe make this generic?
pub type BurnBackend = Autodiff<NdArray>;

// TODO: a type alias would be prefered
pub struct BurnPPO<B: AutodiffBackend>(
    pub  PPO<
        ActorCriticLMKind<B, DistributionKind<B>>,
        StandardPPOHook<ActorCriticLMKind<B, DistributionKind<B>>>,
    >,
);

impl<B: AutodiffBackend> Agent for BurnPPO<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <DistributionKind<B> as AutodiffModule<B>>::InnerModule;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}

pub struct PPOBurnLearningModuleBuilder {
    pub ppo_params: PPOParams,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: StandardPPOHookBuilder,
}

impl Default for PPOBurnLearningModuleBuilder {
    fn default() -> Self {
        Self {
            hook_builder: StandardPPOHookBuilder::default(),
            ppo_params: PPOParams::default(),
            distribution_builder: DistributionBuilder {
                hidden_layers: vec![64, 64],
                distribution_type: DistributionType::Dynamic,
            },
        }
    }
}

impl AgentBuilder for PPOBurnLearningModuleBuilder {
    type Agent = BurnPPO<BurnBackend>;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<BurnPPO<BurnBackend>> {
        let value_layers = &[observation_size, 64, 64, 1];
        let distr = self.distribution_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let value_net = r2l_burn_lm::sequential::Sequential::build(value_layers);
        let model = ParalellActorModel::new(distr, value_net);
        let lm = ActorCriticLMKind::Paralell(ParalellActorCriticLM::new(
            model,
            AdamWConfig::new().init(),
        ));
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(BurnPPO(PPO { lm, hooks, params }))
    }
}
