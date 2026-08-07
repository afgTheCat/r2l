use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::a2c::A2C;
use r2l_burn::{
    distributions::BurnPolicyKind,
    learning_module::PolicyValueLearner as BurnPolicyValueLearner,
};
use r2l_candle::{
    distributions::CandlePolicyKind,
    learning_module::PolicyValueLearner as CandlePolicyValueLearner,
};
use r2l_core::{buffers::TrajectoryBatch, on_policy::algorithm::Agent};

use crate::hooks::a2c::DefaultA2CHook;

/// A2C agent specialized to the Burn backend.
///
/// This wraps the core [`A2C`](r2l_agents::on_policy_algorithms::a2c::A2C)
/// implementation with a Burn learner and the default A2C training
/// hook.
///
/// Use this type when you want an [`Agent`](r2l_core::on_policy::algorithm::Agent)
/// backed by Burn instead of the default Candle backend.
pub struct A2CBurnAgent<B: AutodiffBackend>(
    pub  A2C<
        BurnPolicyValueLearner<B>,
        DefaultA2CHook<BurnPolicyValueLearner<B>>,
    >,
);

impl<B: AutodiffBackend> Agent for A2CBurnAgent<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <BurnPolicyKind<B> as AutodiffModule<B>>::InnerModule;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.0.set_learning_rate(learning_rate);
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}

/// A2C agent specialized to the Candle backend.
///
/// This wraps the core [`A2C`](r2l_agents::on_policy_algorithms::a2c::A2C)
/// implementation with a Candle learner and the default A2C training
/// hook.
///
/// Use this type when you want an [`Agent`](r2l_core::on_policy::algorithm::Agent)
/// on the default Candle backend, optionally selecting a device through
/// [`with_candle`](crate::A2CAlgorithmBuilder::with_candle).
pub struct A2CCandleAgent(
    pub A2C<CandlePolicyValueLearner, DefaultA2CHook<CandlePolicyValueLearner>>,
);

impl Agent for A2CCandleAgent {
    type Tensor = candle_core::Tensor;
    type Actor = CandlePolicyKind;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.0.set_learning_rate(learning_rate);
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}
