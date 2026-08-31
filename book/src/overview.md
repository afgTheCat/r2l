> [!WARNING]  
> **Pre-alpha:** This library is under active development. APIs may change
> between releases, and some planned features are not implemented yet.
> Basic familiarity with reinforcement learning concepts and rust knowledge is
> assumed.

## Why **r2l**

The goal of **r2l** is to be a customizable, ergonomic and easily embeddable
library. To be more exact:

- **Customizable**: users have control over _how_ agents are trained. **r2l**
  defines how components interact and exposes lifecycle hooks for custom
  behavior.
- **Ergonomic**: most users are not necessarily concerned with implementation
  details. High-level builders provide common configurations.
- **Embeddable**: **r2l** describes its backend requirements with traits.
  Candle and Burn implementations are currently available.

The near-term scope of **r2l** is a dependable, well-tested on-policy stack for
PPO and A2C. Stable Baselines3 is used as a benchmark reference, not as a
feature-parity target. The next planned extension is recurrent-policy support,
starting with an end-to-end recurrent PPO path across both tensor backends.
Broader capabilities will be added as independently tested vertical slices
rather than by trying to reproduce another library's entire algorithm catalog.

Potential longer-term work includes hyperparameter tuning, monitoring
integrations, additional persistence formats, and multi-agent support. These
directions do not currently have release commitments.

## About this book

This book will help you get up to speed with _using_ and _hacking_ **r2l**. In
particular:

- [User Guide](./user_guide.md): Introduces how environments are to be
  implemented and how to work with the higher level APIs. Most users should
  start here. Some basic examples are also shown.
- [On policy algorithms](./on_policy_algorithms.md): A detailed architectural
  overview of the components of on-policy algorithms, how the pieces
  fit together, and how to create your own custom hook system.
- [Off-policy algorithms](./off_policy_algorithms.md): Current support status.
