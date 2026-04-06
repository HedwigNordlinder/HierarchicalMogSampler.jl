module HierarchicalMogSampler

using Random
using LinearAlgebra
using Statistics
using Distributions
using PDMats

export HierarchicalMogData,
       SamplerConfig,
       PosteriorSample,
       SamplerResult,
       n_subjects,
       n_features,
       sample_posterior,
       logposterior,
       cluster1_fraction

"""
    HierarchicalMogData(x)

Observed repeated measurements for a two-component hierarchical Gaussian mixture.

`x` must be a vector of matrices. Each matrix corresponds to one subject or patient and
has shape `n_i x p`, where rows are repeated observations and columns are features. The
constructor validates that every subject matrix has the same number of features and at
least one row.
"""
struct HierarchicalMogData
    x::Vector{Matrix{Float64}}
    function HierarchicalMogData(x::Vector{<:AbstractMatrix{<:Real}})
        isempty(x) && throw(ArgumentError("HierarchicalMogData requires at least one subject matrix."))
        copied = Matrix{Float64}[]
        p = nothing
        for (i, Xi) in pairs(x)
            size(Xi, 1) > 0 || throw(ArgumentError("Subject $i has no repeated observations."))
            size(Xi, 2) > 0 || throw(ArgumentError("Subject $i has zero features."))
            if p === nothing
                p = size(Xi, 2)
            elseif size(Xi, 2) != p
                throw(ArgumentError("All subject matrices must have the same number of features."))
            end
            push!(copied, Matrix{Float64}(Xi))
        end
        return new(copied)
    end
end

"""
    n_subjects(data)

Return the number of subjects in `data`.
"""
n_subjects(data::HierarchicalMogData) = length(data.x)

"""
    n_features(data)

Return the number of features in each repeated-measurement row of `data`.
"""
n_features(data::HierarchicalMogData) = size(data.x[1], 2)

"""
    SamplerConfig(p; kwargs...)

Configuration for the hierarchical latent-allocation sampler.

The model is fixed to two Gaussian components. For subject `i` and component `k`,
observations are modeled as

`x_ij | z_ij = k ~ Normal(mu_ik, Sigma_k / lambda_ik)`

where `mu_ik` and `lambda_ik` are subject-specific and `Sigma_k` is shared across
subjects. `component2_prob[i]` is the mixture probability for component 2, so
`Pr(z_ij = 2) = component2_prob[i]`.

`prior_means` must be a `2 x p` matrix whose first row is the prior mean for component 1
and second row is the prior mean for component 2. `S0` must be a positive-definite `p x p`
matrix used by the inverse-Wishart prior on the shared component covariances.
"""
struct SamplerConfig
    a_pi::Float64
    b_pi::Float64
    a_lambda::Float64
    b_lambda::Float64
    kappa0::Float64
    nu0::Float64
    prior_means::Matrix{Float64}
    S0::Matrix{Float64}
    canonicalize_labels::Bool
    initialize_by_first_feature::Bool
end

function SamplerConfig(
    p::Integer;
    a_pi::Real = 1.5,
    b_pi::Real = 1.5,
    a_lambda::Real = 3.0,
    b_lambda::Real = 3.0,
    kappa0::Real = 1.0,
    nu0::Real = p + 4.0,
    prior_means::AbstractMatrix{<:Real} = vcat(fill(-1.0, 1, p), fill(1.0, 1, p)),
    S0::AbstractMatrix{<:Real} = Matrix{Float64}(I, p, p),
    canonicalize_labels::Bool = true,
    initialize_by_first_feature::Bool = true,
)
    p > 0 || throw(ArgumentError("The feature dimension p must be positive."))
    nu0 > p - 1 || throw(ArgumentError("nu0 must exceed p - 1 for the inverse-Wishart prior."))
    a_pi > 0 || throw(ArgumentError("a_pi must be positive."))
    b_pi > 0 || throw(ArgumentError("b_pi must be positive."))
    a_lambda > 0 || throw(ArgumentError("a_lambda must be positive."))
    b_lambda > 0 || throw(ArgumentError("b_lambda must be positive."))
    kappa0 > 0 || throw(ArgumentError("kappa0 must be positive."))

    prior = Matrix{Float64}(prior_means)
    size(prior) == (2, p) || throw(ArgumentError("prior_means must have size 2 x p."))

    scale = Matrix{Float64}(S0)
    size(scale) == (p, p) || throw(ArgumentError("S0 must have size p x p."))
    issymmetric(scale) || throw(ArgumentError("S0 must be symmetric."))
    isposdef(Symmetric(scale)) || throw(ArgumentError("S0 must be positive definite."))

    return SamplerConfig(
        Float64(a_pi),
        Float64(b_pi),
        Float64(a_lambda),
        Float64(b_lambda),
        Float64(kappa0),
        Float64(nu0),
        prior,
        scale,
        canonicalize_labels,
        initialize_by_first_feature,
    )
end

"""
    PosteriorSample

A saved Gibbs draw from the latent-allocation model.

Fields:
- `z`: subject-specific latent allocations with labels in `{1, 2}`.
- `component2_prob`: subject-specific probability of component 2.
- `mu`: array of shape `n_subjects x 2 x p` containing the subject-specific component means.
- `lambda`: matrix of shape `n_subjects x 2` containing subject-specific scale multipliers.
- `Sigma`: length-2 vector of shared `p x p` covariance matrices, one per component.
- `logposterior`: the joint log posterior density of this draw under `SamplerConfig`.
"""
struct PosteriorSample
    z::Vector{Vector{Int}}
    component2_prob::Vector{Float64}
    mu::Array{Float64, 3}
    lambda::Matrix{Float64}
    Sigma::Vector{Matrix{Float64}}
    logposterior::Float64
end

"""
    SamplerResult

Output from [`sample_posterior`](@ref).

Fields:
- `samples`: saved posterior draws after burn-in and thinning.
- `logposterior_trace`: joint log posterior at every Gibbs iteration.
- `cluster1_fraction_trace`: fraction of all observations currently assigned to component 1.
- `final_sample`: the last Gibbs state, whether or not it was saved by thinning.
"""
struct SamplerResult
    samples::Vector{PosteriorSample}
    logposterior_trace::Vector{Float64}
    cluster1_fraction_trace::Vector{Float64}
    final_sample::PosteriorSample
end

mutable struct AllocationState
    z::Vector{Vector{Int}}
    component2_prob::Vector{Float64}
    mu::Array{Float64, 3}
    lambda::Matrix{Float64}
    Sigma::Vector{Matrix{Float64}}
end

@inline _sigmoid(x::Float64) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))

@inline function _quadform(cholA::Cholesky{Float64, Matrix{Float64}}, v::AbstractVector{<:Real})
    y = cholA.L \ v
    return dot(y, y)
end

_copy_z(z::Vector{Vector{Int}}) = [copy(zi) for zi in z]

function _initialize_state(rng::AbstractRNG, data::HierarchicalMogData, cfg::SamplerConfig)
    n = n_subjects(data)
    p = n_features(data)
    z = Vector{Vector{Int}}(undef, n)
    component2_prob = zeros(n)
    mu = zeros(n, 2, p)
    lambda = ones(n, 2)

    first_feature = vcat([Xi[:, 1] for Xi in data.x]...)
    threshold = median(first_feature)

    for i in 1:n
        Xi = data.x[i]
        n_i = size(Xi, 1)
        zi = Vector{Int}(undef, n_i)
        for j in 1:n_i
            zi[j] = cfg.initialize_by_first_feature && Xi[j, 1] > threshold ? 2 : 1
        end

        if all(==(1), zi)
            zi[rand(rng, 1:n_i)] = 2
        elseif all(==(2), zi)
            zi[rand(rng, 1:n_i)] = 1
        end
        z[i] = zi

        n2 = count(==(2), zi)
        component2_prob[i] = (n2 + cfg.a_pi) / (n_i + cfg.a_pi + cfg.b_pi)
        for k in 1:2
            idx = findall(==(k), zi)
            if isempty(idx)
                mu[i, k, :] .= cfg.prior_means[k, :]
            else
                mu[i, k, :] .= vec(mean(Xi[idx, :], dims = 1))
            end
        end
    end

    centered = zeros(p, p)
    denom = 0
    for Xi in data.x
        xbar = vec(mean(Xi, dims = 1))
        for row in eachrow(Xi)
            v = row .- xbar
            centered .+= v * v'
            denom += 1
        end
    end
    base_cov = centered / max(denom, 1) + 0.2I
    Sigma = [copy(base_cov), copy(base_cov)]
    return AllocationState(z, component2_prob, mu, lambda, Sigma)
end

function _canonicalize_labels!(state::AllocationState)
    avg1 = mean(state.mu[:, 1, 1])
    avg2 = mean(state.mu[:, 2, 1])
    avg1 <= avg2 && return

    state.component2_prob .= 1 .- state.component2_prob
    for zi in state.z
        @inbounds for j in eachindex(zi)
            zi[j] = zi[j] == 1 ? 2 : 1
        end
    end

    mu1 = copy(state.mu[:, 1, :])
    lambda1 = copy(state.lambda[:, 1])
    Sigma1 = copy(state.Sigma[1])
    state.mu[:, 1, :] = state.mu[:, 2, :]
    state.lambda[:, 1] = state.lambda[:, 2]
    state.Sigma[1] = state.Sigma[2]
    state.mu[:, 2, :] = mu1
    state.lambda[:, 2] = lambda1
    state.Sigma[2] = Sigma1
    return nothing
end

function _sample_component2_prob!(rng::AbstractRNG, state::AllocationState, cfg::SamplerConfig)
    for i in eachindex(state.z)
        n2 = count(==(2), state.z[i])
        n1 = length(state.z[i]) - n2
        state.component2_prob[i] = rand(rng, Beta(cfg.a_pi + n2, cfg.b_pi + n1))
    end
    return nothing
end

function _sample_z!(rng::AbstractRNG, data::HierarchicalMogData, state::AllocationState)
    n = n_subjects(data)
    p = n_features(data)
    caches = Matrix{Cholesky{Float64, Matrix{Float64}}}(undef, n, 2)
    logdets = zeros(n, 2)

    for i in 1:n, k in 1:2
        cov = Symmetric(state.Sigma[k] / state.lambda[i, k] + 1e-9I)
        chol = cholesky(Matrix(cov))
        caches[i, k] = chol
        logdets[i, k] = 2sum(log, diag(chol.L))
    end

    constant = p * log(2π)
    for i in 1:n
        Xi = data.x[i]
        for j in 1:size(Xi, 1)
            xij = view(Xi, j, :)
            logps = zeros(2)
            logps[1] = log(max(1 - state.component2_prob[i], eps()))
            logps[2] = log(max(state.component2_prob[i], eps()))
            for k in 1:2
                delta = xij .- view(state.mu, i, k, :)
                q = _quadform(caches[i, k], delta)
                logps[k] += -0.5 * (constant + logdets[i, k] + q)
            end
            prob2 = _sigmoid(logps[2] - logps[1])
            state.z[i][j] = rand(rng) < prob2 ? 2 : 1
        end
    end
    return nothing
end

function _sample_mu!(
    rng::AbstractRNG,
    data::HierarchicalMogData,
    state::AllocationState,
    cfg::SamplerConfig,
)
    n = n_subjects(data)
    p = n_features(data)
    for i in 1:n
        Xi = data.x[i]
        for k in 1:2
            idx = findall(==(k), state.z[i])
            n_ik = length(idx)
            prior_mean = vec(view(cfg.prior_means, k, :))
            if n_ik == 0
                post_mean = copy(prior_mean)
            else
                xbar = vec(mean(Xi[idx, :], dims = 1))
                post_mean = (cfg.kappa0 .* prior_mean .+ n_ik .* xbar) ./ (cfg.kappa0 + n_ik)
            end
            post_cov = Symmetric(state.Sigma[k] / ((cfg.kappa0 + n_ik) * state.lambda[i, k]) + 1e-9I)
            draw = rand(rng, MvNormal(post_mean, post_cov))
            @inbounds for ell in 1:p
                state.mu[i, k, ell] = draw[ell]
            end
        end
    end
    return nothing
end

function _sample_lambda!(
    rng::AbstractRNG,
    data::HierarchicalMogData,
    state::AllocationState,
    cfg::SamplerConfig,
)
    n = n_subjects(data)
    p = n_features(data)
    chols = [cholesky(Symmetric(state.Sigma[k] + 1e-9I)) for k in 1:2]
    for i in 1:n
        Xi = data.x[i]
        for k in 1:2
            cholSigma = chols[k]
            muik = vec(view(state.mu, i, k, :))
            prior_mean = vec(view(cfg.prior_means, k, :))
            q = cfg.kappa0 * _quadform(cholSigma, muik .- prior_mean)
            n_ik = 0
            for j in 1:size(Xi, 1)
                if state.z[i][j] == k
                    q += _quadform(cholSigma, vec(view(Xi, j, :)) .- muik)
                    n_ik += 1
                end
            end
            shape = cfg.a_lambda + 0.5 * (n_ik + 1) * p
            rate = cfg.b_lambda + 0.5 * q
            state.lambda[i, k] = rand(rng, Gamma(shape, 1 / rate))
        end
    end
    return nothing
end

function _sample_Sigma!(
    rng::AbstractRNG,
    data::HierarchicalMogData,
    state::AllocationState,
    cfg::SamplerConfig,
)
    n = n_subjects(data)
    for k in 1:2
        nu_star = cfg.nu0
        S_star = copy(cfg.S0)
        prior_mean = vec(view(cfg.prior_means, k, :))
        for i in 1:n
            muik = vec(view(state.mu, i, k, :))
            delta_mu = muik .- prior_mean
            S_star .+= state.lambda[i, k] * cfg.kappa0 * (delta_mu * delta_mu')
            nu_star += 1
            Xi = data.x[i]
            for j in 1:size(Xi, 1)
                if state.z[i][j] == k
                    delta = vec(view(Xi, j, :)) .- muik
                    S_star .+= state.lambda[i, k] * (delta * delta')
                    nu_star += 1
                end
            end
        end
        draw = rand(rng, InverseWishart(nu_star, PDMat(Symmetric(S_star + 1e-8I))))
        state.Sigma[k] = Matrix(Symmetric(draw))
    end
    return nothing
end

function _logposterior(
    data::HierarchicalMogData,
    z::Vector{Vector{Int}},
    component2_prob::Vector{Float64},
    mu::Array{Float64, 3},
    lambda::Matrix{Float64},
    Sigma::Vector{Matrix{Float64}},
    cfg::SamplerConfig,
)
    n = n_subjects(data)
    lp = 0.0
    for k in 1:2
        lp += logpdf(InverseWishart(cfg.nu0, PDMat(Symmetric(cfg.S0))), Sigma[k])
    end

    for i in 1:n
        lp += logpdf(Beta(cfg.a_pi, cfg.b_pi), component2_prob[i])
        for j in eachindex(z[i])
            prob = z[i][j] == 2 ? component2_prob[i] : 1 - component2_prob[i]
            lp += log(max(prob, eps()))
        end
        for k in 1:2
            lp += logpdf(Gamma(cfg.a_lambda, 1 / cfg.b_lambda), lambda[i, k])
            prior_mean = vec(view(cfg.prior_means, k, :))
            muik = vec(view(mu, i, k, :))
            mu_cov = Symmetric(Sigma[k] / (cfg.kappa0 * lambda[i, k]) + 1e-9I)
            lp += logpdf(MvNormal(prior_mean, mu_cov), muik)
        end
        Xi = data.x[i]
        for j in 1:size(Xi, 1)
            k = z[i][j]
            muik = vec(view(mu, i, k, :))
            obs_cov = Symmetric(Sigma[k] / lambda[i, k] + 1e-9I)
            lp += logpdf(MvNormal(muik, obs_cov), vec(view(Xi, j, :)))
        end
    end
    return lp
end

function _posterior_sample(state::AllocationState, lp::Float64)
    return PosteriorSample(
        _copy_z(state.z),
        copy(state.component2_prob),
        copy(state.mu),
        copy(state.lambda),
        [copy(Sigma_k) for Sigma_k in state.Sigma],
        lp,
    )
end

"""
    logposterior(data, sample, cfg)

Evaluate the joint log posterior for a saved posterior draw under the hierarchical
allocation model.
"""
function logposterior(
    data::HierarchicalMogData,
    sample::PosteriorSample,
    cfg::SamplerConfig,
)
    return _logposterior(
        data,
        sample.z,
        sample.component2_prob,
        sample.mu,
        sample.lambda,
        sample.Sigma,
        cfg,
    )
end

"""
    cluster1_fraction(z)
    cluster1_fraction(sample)

Return the fraction of all repeated observations assigned to component 1.
"""
function cluster1_fraction(z::Vector{Vector{Int}})
    total = 0
    n1 = 0
    for zi in z
        total += length(zi)
        n1 += count(==(1), zi)
    end
    return n1 / max(total, 1)
end

cluster1_fraction(sample::PosteriorSample) = cluster1_fraction(sample.z)

"""
    sample_posterior([rng], data, cfg, n_iters; burnin=0, thin=1)

Run the Gibbs sampler for the hierarchical two-component latent allocation model.

The sampler updates, in order, the subject-specific component-2 probabilities,
allocations, subject-specific means, subject-specific scale multipliers, and shared
component covariances. If `cfg.canonicalize_labels` is `true`, labels are globally
re-oriented after each sweep so that component 1 has the smaller average first-feature
mean across subjects.

Saved samples satisfy the same rule as the original GLAM code: an iteration is saved when
`iter > burnin` and `(iter - burnin) % thin == 0`.
"""
function sample_posterior(
    rng::AbstractRNG,
    data::HierarchicalMogData,
    cfg::SamplerConfig,
    n_iters::Integer;
    burnin::Integer = 0,
    thin::Integer = 1,
)
    n_iters > 0 || throw(ArgumentError("n_iters must be positive."))
    0 <= burnin < n_iters || throw(ArgumentError("burnin must satisfy 0 <= burnin < n_iters."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    n_features(data) == size(cfg.prior_means, 2) ||
        throw(ArgumentError("The data dimension and config prior_means dimension do not match."))

    state = _initialize_state(rng, data, cfg)
    samples = PosteriorSample[]
    logpost_trace = Float64[]
    cluster1_trace = Float64[]

    for iter in 1:n_iters
        _sample_component2_prob!(rng, state, cfg)
        _sample_z!(rng, data, state)
        _sample_mu!(rng, data, state, cfg)
        _sample_lambda!(rng, data, state, cfg)
        _sample_Sigma!(rng, data, state, cfg)
        cfg.canonicalize_labels && _canonicalize_labels!(state)

        lp = _logposterior(
            data,
            state.z,
            state.component2_prob,
            state.mu,
            state.lambda,
            state.Sigma,
            cfg,
        )
        push!(logpost_trace, lp)
        push!(cluster1_trace, cluster1_fraction(state.z))

        if iter > burnin && ((iter - burnin) % thin == 0)
            push!(samples, _posterior_sample(state, lp))
        end
    end

    final_sample = _posterior_sample(state, logpost_trace[end])
    return SamplerResult(samples, logpost_trace, cluster1_trace, final_sample)
end

sample_posterior(data::HierarchicalMogData, cfg::SamplerConfig, n_iters::Integer; kwargs...) =
    sample_posterior(Random.default_rng(), data, cfg, n_iters; kwargs...)

end
