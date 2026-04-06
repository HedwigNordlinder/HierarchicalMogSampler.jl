using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

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
println("mean cluster-1 occupancy: ", round(mean(result.cluster1_fraction_trace); digits = 3))

last_draw = result.samples[end]
println("subject 1 component-2 probability: ", round(last_draw.component2_prob[1]; digits = 3))
println(
    "component ordering check: ",
    round(mean(last_draw.mu[:, 1, 1]); digits = 3),
    " < ",
    round(mean(last_draw.mu[:, 2, 1]); digits = 3),
)
