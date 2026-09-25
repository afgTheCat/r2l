These unversioned artifacts were exported with the old backend crates at commit
`7df91df`, using seed 7 and an MLP with one hidden layer of width 2.

The action space is a dictionary containing a categorical action and a nested tuple
of a two-dimensional Gaussian, two Bernoulli actions, and a multi-categorical action
with category counts `[2, 3]`. The observation is `[0.25, -0.5, 0.75, 1.0]`.
`expected.yaml` records each original backend's modal action before migration.

The fixtures exercise old Candle parameter names and one-dimensional `log_std`,
and old Burn flattened composites, enum prefixes, and `mu_net` names.
