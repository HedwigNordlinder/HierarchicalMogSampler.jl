using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using HierarchicalMogSampler

rng = MersenneTwister(20260406)

sim = simulate_glam_style_data(
    rng,
    SimulationConfig(
        n_subjects = 32,
        n_features = 6,
        min_repeats = 12,
        max_repeats = 18,
    );
    active_indices = [1, 3],
)

cfg = SamplerConfig(
    n_features(sim.data);
    prior_means = sim.true_global_means,
    canonicalize_labels = true,
)

result = sample_posterior(rng, sim.data, cfg, 160; burnin = 80, thin = 20)

println("saved draws: ", length(result.samples))
println("last log posterior: ", round(result.final_sample.logposterior; digits = 2))
println("mean cluster-1 occupancy: ", round(mean(result.cluster1_fraction_trace); digits = 3))

last_draw = result.samples[end]
println("subject 1 component-2 probability: ", round(last_draw.component2_prob[1]; digits = 3))
println(
    "component ordering check: ",
    round(mean(last_draw.mu[:, 1, 1]); digits = 3),
    " < ",
    round(mean(last_draw.mu[:, 2, 1]); digits = 3),
)
