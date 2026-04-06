# HierarchicalMogSampler.jl

`HierarchicalMogSampler.jl` is a focused Julia package for one specific model:

- a two-component Gaussian mixture for repeated measurements within each subject,
- subject-specific component means,
- subject-specific component scale multipliers,
- component-shared covariance matrices across subjects,
- Gibbs sampling for the latent allocation stage only.

This package is an extraction of the latent allocation sampler from the GLAM demo code in
this workspace. It does **not** include the downstream logistic regression stage, pooled
feature selection, simulation regimes, or arbitrary-`K` mixtures.

## What The Package Models

For subject `i`, repeated observation `j`, and component `k in {1, 2}`:

```math
x_{ij} \mid z_{ij}=k \sim \mathcal{N}(\mu_{ik}, \Sigma_k / \lambda_{ik})
```

with

- subject-specific component means `mu[i, k, :]`,
- subject-specific scale multipliers `lambda[i, k]`,
- shared component covariance matrices `Sigma[k]`,
- subject-specific mixture probability `component2_prob[i] = Pr(z_ij = 2)`.

The sampler updates:

1. subject-specific component-2 probabilities,
2. latent allocations,
3. subject-specific means,
4. subject-specific scale multipliers,
5. shared covariance matrices.

If `canonicalize_labels=true`, labels are globally re-oriented after every Gibbs sweep so
that component 1 has the smaller average first-feature mean across subjects. This is the
same practical anti-label-switching convention used in the original GLAM demo.

## What The Package Does Not Do

- It does not fit the regression stage from GLAM.
- It does not expose a generic `K`-component mixture API.
- It does not own the simulation code anymore; reproducible simulation lives in `GLAM.jl`.
- It does not marginalize over label switching; instead it uses optional global
  canonicalization.

## Installation

From Julia:

```julia
using Pkg
Pkg.develop(path="HierarchicalMogSampler.jl")
```

## Quick Demo

The example below is the same workflow used in
[`examples/readme_demo.jl`](examples/readme_demo.jl). It builds a small toy repeated-
measurement dataset with visibly separated components, fits the allocation sampler, and
inspects the saved posterior draws. For reproducible simulation regimes and end-to-end
benchmarks, use `GLAM.jl`.

```julia
using Random
using Statistics
using HierarchicalMogSampler

rng = MersenneTwister(20260406)

x = Matrix{Float64}[]
for _ in 1:8
    component1 = -2 .+ 0.20 .* randn(rng, 4, 2)
    component2 = 2 .+ 0.25 .* randn(rng, 5, 2)
    push!(x, vcat(component1, component2))
end

data = HierarchicalMogData(x)
cfg = SamplerConfig(n_features(data); prior_means = [-2.0 -2.0; 2.0 2.0])

result = sample_posterior(rng, data, cfg, 120; burnin = 60, thin = 20)

println("saved draws: ", length(result.samples))
println("last log posterior: ", round(result.final_sample.logposterior; digits = 2))
println(
    "mean cluster-1 occupancy: ",
    round(mean(result.cluster1_fraction_trace); digits = 3),
)

last_draw = result.samples[end]
println("subject 1 component-2 probability: ", round(last_draw.component2_prob[1]; digits = 3))
println(
    "component ordering check: ",
    round(mean(last_draw.mu[:, 1, 1]); digits = 3),
    " < ",
    round(mean(last_draw.mu[:, 2, 1]); digits = 3),
)
```

## Public API

### Types

- `HierarchicalMogData(x)`
  Stores the repeated-measurement matrices. Each element of `x` is one subject matrix with
  shape `n_i x p`.
- `SamplerConfig(p; kwargs...)`
  Hyperparameters and sampler behavior for the two-component hierarchical mixture.
- `PosteriorSample`
  One saved Gibbs draw containing `z`, `component2_prob`, `mu`, `lambda`, `Sigma`, and
  the joint `logposterior`.
- `SamplerResult`
  Full sampler output with saved draws, the per-iteration log-posterior trace,
  the component-1 occupancy trace, and the final Gibbs state.

### Functions

- `sample_posterior([rng], data, cfg, n_iters; burnin=0, thin=1)`
  Run the Gibbs sampler and return a `SamplerResult`.
- `logposterior(data, sample, cfg)`
  Recompute the joint log posterior for a saved `PosteriorSample`.
- `n_subjects(data)` and `n_features(data)`
  Convenience accessors for `HierarchicalMogData`.
- `cluster1_fraction(sample)` or `cluster1_fraction(z)`
  Compute the fraction of repeated observations assigned to component 1.

## Notes On Labels

The original GLAM demo stored latent allocations as `0/1` internally. This package uses
the cleaner public convention `1/2`:

- `z[i][j] == 1` means component 1,
- `z[i][j] == 2` means component 2,
- `component2_prob[i]` is the subject-specific weight for component 2.

When `canonicalize_labels=true`, component 1 is always the lower-first-feature component
on average across subjects. That is why component labels remain globally consistent across
patients and across saved posterior draws.

## Relation To The Original GLAM Demo

This package intentionally preserves the allocation model and Gibbs update order from the
original demo code while presenting a cleaner standalone API:

- same two-component hierarchical mixture,
- same conditional updates for `component2_prob`, `z`, `mu`, `lambda`, and `Sigma`,
- same optional global label canonicalization rule.

The meaningful cleanups are:

- the public label convention is `1/2`, not `0/1`,
- simulation code now lives in `GLAM.jl`, where benchmark regimes are defined.
