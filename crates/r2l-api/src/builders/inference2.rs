use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    ActorWrapper,
    env::{Env, Snapshot, normalizer::ClippedNormalizer},
    models::Actor,
    on_policy::algorithm::{OnPolicyAlgorithm, OnPolicyAlgorithmHooks, Sampler},
    rng::sample_u64,
};
use serde::{Deserialize, Serialize};

use crate::{
    CandleBackend, PPOCandleAgent, PolicyBuilder, builders::normalizer::NormalizerBuilder,
};

pub struct Inference2<E: Env, A: Actor<Tensor = E::Tensor>> {
    env: E,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    actor: A,
    last_state: E::Tensor,
}

impl<E: Env, A: Actor<Tensor = E::Tensor>> Inference2<E, A> {
    pub fn new(mut env: E, obs_normalizer: Option<ClippedNormalizer<E::Tensor>>, actor: A) -> Self {
        let last_state = env.reset(sample_u64()).unwrap();
        Self {
            env,
            obs_normalizer,
            actor,
            last_state,
        }
    }

    pub fn reset(&mut self) {
        let last_state = self.env.reset(sample_u64()).unwrap();
        self.last_state = last_state;
    }

    pub fn step(&mut self) -> Snapshot<E::Tensor> {
        let action = self.actor.action(self.last_state.clone()).unwrap();
        let mut snapshot = self.env.step(action).unwrap();
        if let Some(obs_normalizer) = &mut self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut snapshot.state);
        }
        self.last_state = snapshot.state.clone();
        snapshot
    }
}

#[derive(Serialize, Deserialize)]
pub struct Inference2Builder<Backend = CandleBackend> {
    normalizer_builder: Option<NormalizerBuilder>,
    policy_builder: PolicyBuilder,
    backend: Backend,
}

impl Inference2Builder<CandleBackend> {
    fn to_inference<E: Env>(
        self,
        env: E,
    ) -> Inference2<E, ActorWrapper<CandlePolicyKind, E::Tensor>> {
        let obs_normalizer = self.normalizer_builder.map(|x| x.into_normalizer());
        let env_description = env.env_description();
        let actor = self
            .policy_builder
            .build_candle(
                env_description.observation_space.size(),
                env_description.action_space,
                &self.backend.device,
            )
            .unwrap();
        let actor = ActorWrapper::new(actor);
        Inference2::new(env, obs_normalizer, actor)
    }

    // maybe we do not want to do this? maybe we want to use a builder for this?
    fn from_algo<S: Sampler, H: OnPolicyAlgorithmHooks<A = PPOCandleAgent, S = S>>(
        _algo: OnPolicyAlgorithm<PPOCandleAgent, S, H>,
    ) {
    }
}
