• Yes. If sampling and obs normalization are now close, I would look at PPO
defaults and rollout post-processing next. The biggest suspects are:

1. None does not mean the same thing in your test as in SB3.

In sb3/ppo.py, for MountainCar-v0:

- vf_coef=None means “do not pass it”, so SB3 uses its default vf_coef=0.5.
- max_grad_norm=None means “do not pass it”, so SB3 uses its default
  max_grad_norm=0.5.

In our builder:

- vf_coeff: None means value loss is not scaled, effectively coefficient 1.0.
- gradient_clipping: None means no gradient clipping.

That alone can make PPO much worse. I would first try explicitly matching SB3
with vf_coeff = Some(0.5) and gradient_clipping = Some(0.5) for MountainCar.

2. Advantage normalization differs.

Our Advantages::normalize() normalizes per rollout buffer, so with n_envs=16 and
n_steps=16, each env’s 16 advantages are normalized separately.

SB3 flattens rollout data and normalizes advantages over minibatches/rollout
samples, not independently per environment. For MountainCar, n_steps=16 is very
short, so per-env normalization is noisy and can distort the learning signal
quite a bit.

3. Timeout/truncation handling is probably different.

MountainCar often ends by time limit. SB3 handles timeout truncations specially
by bootstrapping from the terminal observation. Our current GAE path treats:

terminated || truncated

as fully done, with no value bootstrap. Since we are not storing terminal
observations right now, this is a real semantic difference from SB3. It may
matter a lot on MountainCar.

4. Policy/value network defaults are not SB3-like.

SB3 MlpPolicy defaults are roughly:

- 2x64 hidden layers
- Tanh activation
- orthogonal initialization
- small policy-output init gain
- Adam with eps=1e-5, no AdamW weight decay

Our Candle stack currently uses:

- 2x64 layers, yes
- ReLU
- Kaiming-style init
- AdamW with weight_decay=1e-4

That is not a small difference for PPO. PPO is sensitive to initialization and
optimizer details.

My ranking would be:

1. Match vf_coef=0.5 and max_grad_norm=0.5.
2. Fix advantage normalization semantics.
3. Handle truncation/time-limit bootstrapping.
4. Match SB3 network activation/init/optimizer more closely.

The first item is the easiest and most likely to produce an immediate
improvement.
