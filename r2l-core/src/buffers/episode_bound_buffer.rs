// use r2l_core2::{
//     distributions::Distribution,
//     env::{Env, SnapShot},
// };
// use rand::{Rng, SeedableRng, rngs::StdRng};
// use ringbuffer::{AllocRingBuffer, RingBuffer};
// use std::cell::RefCell;
//
// use crate::env::Env;
//
// thread_local! {
//     pub static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
// }
//
// pub struct StepBoundBuffer<E: Env> {
//     env: E,
//     states: AllocRingBuffer<E::Obs>,
//     next_states: AllocRingBuffer<E::Obs>,
//     rewards: AllocRingBuffer<f32>,
//     action: AllocRingBuffer<E::Act>,
//     terminated: AllocRingBuffer<bool>,
//     trancuated: AllocRingBuffer<bool>,
// }
//
// impl<E: Env> StepBoundBuffer<E> {
//     pub fn push_snapshot(&mut self, snapshot: SnapShot<E::Obs, E::Act>) {
//         let SnapShot {
//             state,
//             next_state,
//             action,
//             reward,
//             terminated,
//             trancuated,
//         } = snapshot;
//         self.states.enqueue(state);
//         self.next_states.enqueue(next_state);
//         self.action.enqueue(action);
//         self.rewards.enqueue(reward);
//         self.terminated.enqueue(terminated);
//         self.trancuated.enqueue(trancuated);
//     }
//
//     pub fn step(
//         &mut self,
//         distr: &impl Distribution<Observation = E::Obs, Action = E::Act>,
//     ) -> SnapShot<E::Obs, E::Act> {
//         let state = if let Some(obs) = self.next_states.back() {
//             obs.clone()
//         } else {
//             // TODO: get the seed from core
//             let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
//             self.env.reset(seed)
//         };
//         let (action, _) = distr.get_action(state);
//         let SnapShot {
//             state,
//             mut next_state,
//             action,
//             reward,
//             terminated,
//             trancuated,
//         } = self.env.step(&action);
//         let done = terminated || trancuated;
//         if done {
//             let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
//             next_state = self.env.reset(seed);
//         }
//         SnapShot {
//             state,
//             next_state,
//             action,
//             reward,
//             terminated,
//             trancuated,
//         }
//     }
// }
