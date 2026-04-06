using Test
using Random
using Statistics

using HierarchicalMogSampler

@testset "HierarchicalMogSampler" begin
    rng = MersenneTwister(20260406)
    sim = simulate_glam_style_data(
        rng,
        SimulationConfig(
            n_subjects = 18,
            n_features = 5,
            min_repeats = 8,
            max_repeats = 12,
        );
        active_indices = [1, 3],
    )

    cfg = SamplerConfig(
        n_features(sim.data);
        prior_means = sim.true_global_means,
        canonicalize_labels = true,
    )

    result = sample_posterior(rng, sim.data, cfg, 40; burnin = 10, thin = 10)

    @test length(result.samples) == 3
    @test length(result.logposterior_trace) == 40
    @test length(result.cluster1_fraction_trace) == 40
    @test all(isfinite, result.logposterior_trace)
    @test result.final_sample.logposterior ≈ logposterior(sim.data, result.final_sample, cfg)
    @test mean(result.final_sample.mu[:, 1, 1]) <= mean(result.final_sample.mu[:, 2, 1])
    @test 0.0 <= mean(result.final_sample.component2_prob) <= 1.0
    @test 0.0 <= cluster1_fraction(result.final_sample) <= 1.0
    @test all(all(zij in (1, 2) for zij in zi) for zi in result.final_sample.z)
end
