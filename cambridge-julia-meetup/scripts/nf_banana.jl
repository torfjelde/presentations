using DrWatson
quickactivate(@__DIR__)

using Bijectors
using Turing
using StatsFuns: softplus
using Plots, StatsPlots
using ForwardDiff
pyplot()

include(srcdir("leaky_relu.jl"))

const DTYPE = Float32

# Create data
function generate_samples(batch_size)
    x2_dist = Normal(DTYPE(0), DTYPE(4.0))
    x2_samples = rand(x2_dist, batch_size)

    x1 = MvNormal(DTYPE.(.25 * x2_samples.^2), ones(DTYPE, batch_size))
    x1_samples = rand(x1)

    x_samples = vcat(x1_samples', x2_samples')
end

BATCH_SIZE = 512
x_samples = generate_samples(BATCH_SIZE)

# Construct base distribution
base = MvNormal(DTYPE.(zeros(2)), DTYPE.(ones(2)))

# Affine transformation is simply a `Scale` composed with `Shift`
using Bijectors: Shift, Scale
Affine(W::AbstractMatrix, b::AbstractVector; dim::Val{N} = Val(1)) where {N} = Shift(b; dim = dim) ∘ Scale(W; dim = dim)

##################
# BIG UGLY STUFF #
##################
using LinearAlgebra

d, r = 2, 2
num_layers = 6

# num_params_per_layer = d + Int((d * (d + 1)) / 2) + 1
num_params_per_layer = 2d + 1
num_params_per_layer = d + d^2 + d + 1 + (d * r)
num_params_per_layer = d + d^2 + 1 + (d * r) # without diagonal

start = 0
shift_param_range = start:d - 1
start += d
scale1_param_range = start:(start + d^2 - 1)         # triangular matrix
start += d^2
# scale2_param_range = start:(start + d - 1)  # diagonal
# start += d
scale_perturb_range = start:(start + d * r - 1)
start += d * r
relu_param_idx = start  # <= should be equal to (num_params_per_level - 1)

@assert relu_param_idx == num_params_per_layer - 1

function construct_flow(θ)
    start = 1

    layers = []
    for i = 1:num_layers
        start = (i - 1) * num_params_per_layer + 1
        shift_params = θ[start .+ shift_param_range]

        # Constructing a Cholesky with positive diagonals
        # L_unit_diag = UnitLowerTriangular(reshape(θ[start .+ scale1_param_range], (d, d)))
        # L_diag = Diagonal(softplus.(θ[start .+ scale2_param_range]) .+ eps(T))
        # L = L_diag + L_unit_diag - Diagonal(ones(T, d))  # subtract 1 because `Lᵀ_unit_diag` has 1 along diagonal

        # @assert all(diag(L_diag) .> 0.0) "$L_diag"

        # Alternative: hope Cholesky factor has non-zero diagonals
        L = LowerTriangular(reshape(θ[start .+ scale1_param_range], (d, d)))

        # scale_perturb
        V = reshape(θ[start .+ scale_perturb_range], (d, r))

        # relu params
        relu_param = θ[start + relu_param_idx]
        
        aff = Affine(L + V * transpose(V), shift_params)
        # aff = Shift(shift_params) ∘ MatrixPlusLowRankScale(L, V)
        b = LeakyReLU(abs(relu_param) + DTYPE(0.01))

        if i == num_layers
            # drop the ReLU in last layer
            layer = aff
        else
            layer = b ∘ aff
        end

        push!(layers, layer)
    end

    if num_layers > 1
        return Bijectors.composel(layers...)
    else
        return layers[1]
    end
end

# Let's try constructing a flow
θ = DTYPE.(randn(num_layers * num_params_per_layer) ./ sqrt(num_layers * num_params_per_layer))
b = construct_flow(θ)
td = transformed(base, b)
logpdf(td, x_samples)

x = rand(td.dist, 10)
y = b(x)
@assert size(y, 2) == 10
@assert length(logabsdetjac(b, x)) == 10

x_ = inv(b)(y)
@assert size(x_, 2) == 10
@assert length(logabsdetjac(inv(b), y)) == 10


function f(θ)
    flow = construct_flow(θ)

    td = transformed(base, flow)
    x_samples = generate_samples(BATCH_SIZE)
    return - mean(logpdf(td, x_samples))
end

θ = DTYPE.(randn(num_layers * num_params_per_layer) ./ sqrt(num_layers * num_params_per_layer))
f(θ)

using ForwardDiff
ForwardDiff.gradient(f, θ)

# Running the below code we get the following benchmarks
# Float32: 6.203 ms (4641 allocations: 13.81 MiB)
# Float64: 11.168 ms (4641 allocations: 27.43 MiB)

# using BenchmarkTools
# @btime ForwardDiff.gradient($f, $θ)

############
# Training #
############
### Optimization
# We want to use ADAM for optimization so we need Flux.Optimise
using Flux.Optimise, ProgressMeter, Random
using Hyperopt

# # learning_rate = 1e-3
# ho = @hyperopt for  opt_idx = 10, learning_rate = LinRange(1e-6, 1e-1, 100), seed = [1, 2, 3, 4, 5]
#     println(opt_idx, "\t", learning_rate, "\t", seed)

#     # initialization
#     rng = Random.MersenneTwister(seed)
#     θ = randn(rng, num_params_per_layer * num_layers)
    
#     diff_result = DiffResults.GradientResult(θ)
#     nlls = []
    
#     opt = Optimise.ADAM(learning_rate)

#     ps = []

#     num_steps = Int(10e3)
#     prog = Progress(num_steps)

#     for i = 1:num_steps
#         # gradient
#         ForwardDiff.gradient!(diff_result, f, θ)

#         # compute update step
#         Δ = DiffResults.gradient(diff_result)
#         Optimise.apply!(opt, θ, Δ)

#         # perform update
#         @. θ = θ - Δ

#         # save logpdf
#         nll = DiffResults.value(diff_result)
#         push!(nlls, nll)

#         ProgressMeter.next!(prog; showvalues = [(:nll, nll), ])
#     end

#     # Estimate likelihood
#     flow = construct_flow(θ)
#     td = transformed(base, flow)
#     x_samples = generate_samples(5_000)
#     @show - mean(logpdf(td, x_samples))
# end

# plot(ho)
# learning_rate, seed = first(minimum(ho))

# # Now we do a finer search over the `learning_rate` using Bayesina optimization
# # with the "optimal" initialization obtained from the previous random search.
# ho = @hyperopt for  opt_idx = 10, sampler = GPSampler(Min), learning_rate = LinRange(1e-6, 1e-2, 100)
#     println(opt_idx, "\t", learning_rate, "\t", seed)

#     # initialization
#     global seed
#     rng = Random.MersenneTwister(seed)
#     θ = randn(rng, num_params_per_layer * num_layers)
    
#     diff_result = DiffResults.GradientResult(θ)
#     nlls = []
    
#     opt = Optimise.ADAM(learning_rate)

#     ps = []

#     num_steps = Int(5 * 10e3)
#     prog = Progress(num_steps)

#     for i = 1:num_steps
#         # gradient
#         ForwardDiff.gradient!(diff_result, f, θ)

#         # compute update step
#         Δ = DiffResults.gradient(diff_result)
#         Optimise.apply!(opt, θ, Δ)

#         # perform update
#         @. θ = θ - Δ

#         # save logpdf
#         nll = DiffResults.value(diff_result)
#         push!(nlls, nll)

#         ProgressMeter.next!(prog; showvalues = [(:nll, nll), ])
#     end

#     # Estimate likelihood
#     flow = construct_flow(θ)
#     td = transformed(base, flow)
#     x_samples = generate_samples(5_000)
#     @show - mean(logpdf(td, x_samples))
# end

# in case we haven't done the hyper-optimization
learning_rate, seed = isdefined(Main, :ho) ? (first(minimum(ho)), seed) : (0.002223, 3)
println(learning_rate, "\t", seed)
rng = Random.MersenneTwister(seed)
θ = randn(rng, num_params_per_layer * num_layers)

diff_result = DiffResults.GradientResult(θ)
nlls = []

opt = Optimise.ADAM(learning_rate)

ps = []

num_steps = Int(5 * 10e3)
prog = Progress(num_steps)

for i = 1:num_steps
    # gradient
    ForwardDiff.gradient!(diff_result, f, θ)

    # compute update step
    Δ = DiffResults.gradient(diff_result)
    Optimise.apply!(opt, θ, Δ)

    # perform update
    @. θ = θ - Δ

    # save logpdf
    nll = DiffResults.value(diff_result)
    push!(nlls, nll)

    ProgressMeter.next!(prog; showvalues = [(:nll, nll), ])

    if mod(i, 500) == 0
        flow = construct_flow(θ)
        td = transformed(base, flow)

        nf_samples = rand(td, 4000)
        x_samples = generate_samples(BATCH_SIZE)

        p2 = scatter(nf_samples[1, :], nf_samples[2, :], label = "transformed")
        scatter!(x_samples[1, :], x_samples[2, :], label = "x")

        xlims!(-10.0, 30.0)
        ylims!(-10.0, 15.0)

        push!(ps, p2)
    end
end

# Estimate likelihood
flow = construct_flow(θ)
td = transformed(base, flow)
x_samples = generate_samples(5_000)
@show - mean(logpdf(td, x_samples))

# plot(ps..., layout = (Int(length(ps) / 2), 2), size=(500, length(ps) * 100))

anim = @animate for i = 1:length(ps)
    p1 = plot(ps[i], size=(500, 500), legend = :topright)
    p2 = plot(Float64.(nlls[1:i]), label = "Average NLL")
    plot(p1, p2, layout = grid(2, 1, heights = [2 * 750 / 3, 750 / 3]), size = (500, 750))
end

gif(anim, "../figures/nf-banana-density-estimation.gif", fps=10)
