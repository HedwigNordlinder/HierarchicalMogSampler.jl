using Test
using Random
using Statistics

using HierarchicalMogSampler

@testset "HierarchicalMogSampler" begin
    rng = MersenneTwister(20260406)
    x = Matrix{Float64}[]
    for _ in 1:10
        component1 = -2 .+ 0.20 .* randn(rng, 4, 2)
        component2 = 2 .+ 0.20 .* randn(rng, 5, 2)
        push!(x, vcat(component1, component2))
    end
    data = HierarchicalMogData(x)
    cfg = SamplerConfig(n_features(data); prior_means = [-2.0 -2.0; 2.0 2.0], canonicalize_labels = true)

    result = sample_posterior(rng, data, cfg, 40; burnin = 10, thin = 10)

    @test length(result.samples) == 3
    @test length(result.logposterior_trace) == 40
    @test length(result.cluster1_fraction_trace) == 40
    @test all(isfinite, result.logposterior_trace)
    @test result.final_sample.logposterior ≈ logposterior(data, result.final_sample, cfg)
    @test mean(result.final_sample.mu[:, 1, 1]) <= mean(result.final_sample.mu[:, 2, 1])
    @test 0.0 <= mean(result.final_sample.component2_prob) <= 1.0
    @test 0.0 <= cluster1_fraction(result.final_sample) <= 1.0
    @test all(all(zij in (1, 2) for zij in zi) for zi in result.final_sample.z)
end
