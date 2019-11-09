using Pkg; Pkg.activate("..")

using Bijectors
using Turing
using StatsFuns: softplus
using Plots, StatsPlots
using ForwardDiff
pyplot()

DTYPE = Float32

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

#############
# LeakyReLU #
#############
import Bijectors: logabsdetjac, forward

struct LeakyReLU{T} <: Bijector{1} # <= expects 1-dim input i.e. vector-input
    α::T
end

function (b::LeakyReLU{T1})(x::AbstractVecOrMat{T2}) where {T1, T2}
    return @. ifelse(x < 0.0, b.α * x, x)
end
function (ib::Inversed{<:LeakyReLU{T1}})(y::AbstractVecOrMat{T2}) where {T1, T2}
    return @. ifelse(y < 0.0, inv(ib.orig.α) * y, y)
end


logabsdetjac(b::LeakyReLU, x::AbstractVecOrMat) = - logabsdetjac(inv(b), b(x))
function logabsdetjac(ib::Inversed{<:LeakyReLU{T1}}, y::AbstractVecOrMat{T2}) where {T1, T2}
    T = T2
    J⁻¹ = @. ifelse(y < 0, inv(ib.orig.α), one(T)) # <= is really diagonal of jacobian

    # this is optimized away
    if y isa AbstractVector
        return sum(log.(abs.(J⁻¹)))
    elseif y isa AbstractMatrix
        return vec(sum(log.(abs.(J⁻¹)); dims = 1))  # sum along column
    end
end

# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU{T1}, x::AbstractVecOrMat{T2}) where {T1, T2}
    T = T2
    J = @. ifelse(x < 0, b.α, one(T)) # <= is really diagonal of jacobian
    
    if x isa AbstractVector
        logjac = sum(log.(abs.(J)))
    elseif x isa AbstractMatrix
        logjac = vec(sum(log.(abs.(J)); dims = 1))  # sum along column
    end
    
    y = J .* x
    return (rv=y, logabsdetjac=logjac)
end

b = LeakyReLU(DTYPE.(2))
z = ones(DTYPE, 2) - DTYPE.([0.0, 2.0])
y = b(z)
z_ = inv(b)(y)

@assert z_ == z

rv, logjac = forward(b, z)
logjac_forward = logabsdetjac(b, z)
logjac_inverse = logabsdetjacinv(b, y)

@assert logjac ≈ logjac_forward ≈ - logjac_inverse

b = LeakyReLU(DTYPE.(2))
z = ones(DTYPE, (2, 3)) - DTYPE.(repeat([0.0, 2.0]', 3)')
y = b(z)
z_ = inv(b)(y)

@assert z_ == z

rv, logjac = forward(b, z)
logjac_forward = logabsdetjac(b, z)
logjac_inverse = logabsdetjacinv(b, y)

@assert logjac ≈ logjac_forward ≈ - logjac_inverse
@assert all([y[:, i] == b(z[:, i]) for i = 1:size(z, 2)])


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
        # @info start
        # @info start:start + d - 1
        # @info start + d: start + 2d - 1
        # @info start + 2d
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
# Some functions for visualization purposes
function quadrant_masks(x)
    ll_mask = [(x[1, i] < 0) & (x[2, i] < 0) for i = 1:size(x, 2)]
    lr_mask = [(x[1, i] ≥ 0) & (x[2, i] < 0) for i = 1:size(x, 2)]

    ul_mask = [(x[1, i] < 0) & (x[2, i] ≥ 0) for i = 1:size(x, 2)]
    ur_mask = [(x[1, i] ≥ 0) & (x[2, i] ≥ 0) for i = 1:size(x, 2)]

    return ll_mask, ul_mask, ur_mask, lr_mask
end

function plot_samples(x; masks = nothing, colors = [:red, :blue, :black, :green])
    ll_mask, ul_mask, ur_mask, lr_mask = (masks === nothing) ? quadrant_masks(x) : masks
    
    p = scatter(x[1, ll_mask], x[2, ll_mask], markercolor = colors[1], label = "", markersize = 2, markerstrokewidth = 0)
    scatter!(x[1, ul_mask], x[2, ul_mask], markercolor = colors[2], label = "", markersize = 2, markerstrokewidth = 0)
    scatter!(x[1, ur_mask], x[2, ur_mask], markercolor = colors[3], label = "", markersize = 2, markerstrokewidth = 0)
    scatter!(x[1, lr_mask], x[2, lr_mask], markercolor = colors[4], label = "", markersize = 2, markerstrokewidth = 0)

    return p
end


function layerbylayer_plot(td::TransformedDistribution, x)
    y = x
    
    masks = quadrant_masks(y)
    transform_plots = [plot_samples(y; masks = masks)]

    b = td.transform

    for i = 1:length(b.ts)
        y = b.ts[i](y)
        # p = scatter(y[1, :], y[2, :], label = "y_$i")
        p = plot_samples(y)
        xlims!(-10.0, 30.0)
        ylims!(-10.0, 15.0)
        push!(transform_plots, p)
    end

    return plot(transform_plots..., layout = (1, 1 + length(b.ts)))
end

# We want to use ADAM for optimization so we need Flux.Optimise
using Flux.Optimise, ProgressMeter, Random

# Initializing using the same parameters as Tensorflow uses
V0 = [[-0.69470036  1.0377892 ]
 [-1.1973481  -0.45261383]]
shift0 = [-0.8440926,  0.4192394]
L0 = [[-0.3992703   0.        ]
 [ 0.7473383   0.39921927]]
alpha0 = -1.0906933546066284
V1 = [[ 0.4329685  -0.48924023]
 [-0.1613357   1.1891216 ]]
shift1 = [-0.53116786,  0.7054597 ]
L1 = [[0.22240257 0.        ]
 [0.33831072 0.35498   ]]
alpha1 = -1.6679620742797852
V2 = [[-0.8232367  -0.6679769 ]
 [ 0.4709369  -0.38397086]]
shift2 = [-0.32219,    -0.74817294]
L2 = [[0.71648455 0.        ]
 [0.28550506 0.11648226]]
alpha2 = -1.3658761978149414
V3 = [[ 0.44979656  0.5101937 ]
 [-0.4264843   0.97859585]]
shift3 = [-0.589462,    0.55738735]
L3 = [[0.05405951 0.        ]
 [0.4904921  0.37576723]]
alpha3 = 0.44315946102142334
V4 = [[ 1.0989882  -0.02543747]
 [ 0.9325644  -0.21350157]]
shift4 = [0.3811028,  0.11544478]
L4 = [[-0.6974654   0.        ]
 [-0.87845683 -0.3911724 ]]
alpha4 = 1.014729380607605
V5 = [[ 0.82150924 -1.1683286 ]
 [ 0.4476515  -0.4257114 ]]
shift5 = [1.2163152, 1.1666728]
L5 = [[ 0.49313235  0.        ]
 [-0.6587894  -0.47925997]]
alpha5 = -0.5069767236709595

@assert reshape(vec(L0), (d, d)) == L0

θ₀ = vcat(
    vcat(shift0, vec(L0), vec(V0), alpha0),
    vcat(shift1, vec(L1), vec(V1), alpha1),
    vcat(shift2, vec(L2), vec(V2), alpha2),
    vcat(shift3, vec(L3), vec(V3), alpha3),
    vcat(shift4, vec(L4), vec(V4), alpha4),
    vcat(shift5, vec(L5), vec(V5), alpha5)
)
θ = θ₀
@assert length(θ) == num_params_per_layer * num_layers

# Optimization

std(θ₀)
mean(θ₀)
inv(sqrt(length(θ₀)))


nlls = []

using Hyperopt

# learning_rate = 1e-3
ho = @hyperopt for  opt_idx = 10, learning_rate = LinRange(1e-6, 1e-1, 100), seed = [1, 2, 3, 4, 5]
    println(opt_idx, "\t", learning_rate, "\t", seed)

    # global θ₀
    # θ = copy(θ₀)
    rng = Random.MersenneTwister(seed)
    θ = randn(rng, num_params_per_layer * num_layers)
    
    diff_result = DiffResults.GradientResult(θ)
    nlls = []
    
    opt = Optimise.ADAM(learning_rate)
    # opt.eta = 0.001
    # θ .+= randn(length(θ)) * 0.1

    ps = []

    num_steps = Int(10e3)
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

        # if mod(i, 500) == 0
        #     flow = construct_flow(θ)
        #     td = transformed(base, flow)

        #     # base_samples = rand(td.dist, BATCH_SIZE)
        #     # nf_samples = hcat([td.transform(base_samples[:, i]) for i = 1:size(base_samples, 2)]...)
        #     nf_samples = rand(td, 4000)

        #     x_samples = generate_samples(BATCH_SIZE)

        #     p2 = scatter(nf_samples[1, :], nf_samples[2, :], label = "transformed")
        #     scatter!(x_samples[1, :], x_samples[2, :], label = "x")

        #     xlims!(-10.0, 30.0)
        #     ylims!(-10.0, 15.0)

        #     push!(ps, p2)
        # end
    end

    # Estimate likelihood
    flow = construct_flow(θ)
    td = transformed(base, flow)
    x_samples = generate_samples(5_000)
    @show - mean(logpdf(td, x_samples))
end

# good seeds: [4, ]

plot(ho)
learning_rate, seed = first(minimum(ho))

# Now we do a finer search over the `learning_rate` using Bayesina optimization
# with the "optimal" initialization obtained from the previous random search.
ho = @hyperopt for  opt_idx = 10, sampler = GPSampler(Min), learning_rate = LinRange(1e-6, 1e-2, 100)
    println(opt_idx, "\t", learning_rate, "\t", seed)

    global seed
    rng = Random.MersenneTwister(seed)
    θ = randn(rng, num_params_per_layer * num_layers)
    
    diff_result = DiffResults.GradientResult(θ)
    nlls = []
    
    opt = Optimise.ADAM(learning_rate)
    # opt.eta = 0.001
    # θ .+= randn(length(θ)) * 0.1

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
    end

    # Estimate likelihood
    flow = construct_flow(θ)
    td = transformed(base, flow)
    x_samples = generate_samples(5_000)
    @show - mean(logpdf(td, x_samples))
end

# in case we haven't done the hyper-optimization
learning_rate, seed = isdefined(Main, :ho) ? (first(minimum(ho)), seed) : (0.002223, 3)
println(learning_rate, "\t", seed)
rng = Random.MersenneTwister(seed)
θ = randn(rng, num_params_per_layer * num_layers)

diff_result = DiffResults.GradientResult(θ)
nlls = []

opt = Optimise.ADAM(learning_rate)
# opt.eta = 0.001
# θ .+= randn(length(θ)) * 0.1

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

        # base_samples = rand(td.dist, BATCH_SIZE)
        # nf_samples = hcat([td.transform(base_samples[:, i]) for i = 1:size(base_samples, 2)]...)
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
    plot(p1, p2, layout = grid(2, 1, hieghts = [2 * 750 / 3, 750 / 3]), size = (500, 750))
end

gif(anim, "../figures/nf_banana_figures/nf_banana_training.gif", fps=10)

# Ice-cream posterior
parlour1 = [5.0, -5.0]
parlour2 = [2.5, 2.5]

# x_samples = generate_samples(BATCH_SIZE)
x_samples = rand(td, 5000)
scatter(x_samples[1, :], x_samples[2, :], label = "x", markerstrokewidth = 0, color = :orange, alpha = 0.4)

scatter!(parlour1[1:1], parlour1[2:2], label = "Parlour #1", color = :red, markersize = 5)
scatter!(parlour2[1:1], parlour2[2:2], label = "Parlour #2", color = :blue, markersize = 5)

xlims!(-10.0, 30.0)
ylims!(-15.0, 15.0)


x_range = -10:0.05:30
y_range = -10:0.05:15
contour(x_range, y_range, (x, y) -> pdf(td, [x, y]))

# Using resulting flow, let's do some neat stuff
using Turing
@model napkin_model(x, ::Type{TV} = Vector{Float64}) where {TV} = begin
    locs = Vector{TV}(undef, length(x))
    
    for i ∈ eachindex(x)
        locs[i] ~ td.dist
        loc = td.transform(locs[i])
        
        d1 = exp(- norm(parlour1 - loc))
        d2 = exp(- norm(parlour2 - loc))

        πs = [d1 / (d1 + d2), d2 / (d1 + d2)]
        
        x[i] ~ Turing.Categorical(πs)
    end
end

# Example of the inner part of the model
locs = rand(td, 10)
d1 = exp.(- [norm(parlour1 - locs[:, i]) for i = 1:size(locs, 2)])
d2 = exp.(- [norm(parlour2 - locs[:, i]) for i = 1:size(locs, 2)])
ds = hcat(d1, d2)
πs = ds ./ sum(ds; dims = 2)

parlour1 .- locs

# Running the model with some data
fake_samples = [1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1]
num_fake_samples = length(fake_samples)
m = napkin_model(fake_samples)

num_mcmc_samples = 10_000
mcmc_warmup = 1_000
samples = sample(m, NUTS(mcmc_warmup, 0.65), num_mcmc_samples + mcmc_warmup; progress = true)

posterior_locs_samples = reshape(samples[:locs].value[:, :, 1], (:, num_fake_samples, 2))
x_samples = rand(td, 10_000)

# CHECK ONE OF THE CUSTOMERS

customer_idx = rand(1:num_fake_samples)
# Transform because samples will be untransformed
posterior_locs_idx = td.transform(posterior_locs_samples[:, customer_idx, :]')

posterior_locs_idx = td.transform(reshape(posterior_locs_samples, (:, 2))')

mean_locs = mean(posterior_locs_idx; dims = 2)
std_locs = std(posterior_locs_idx; dims = 2)

p1 = scatter(x_samples[1, :], x_samples[2, :], label = "locations", markerstrokewidth = 0, color = :orange, alpha = 0.3)

xlims!(-10.0, 30.0)
ylims!(-15.0, 15.0)

# scatter!(mean_locs[1:1], mean_locs[2:2], xerr = std_locs[1:1], yerr = std_locs[2:2], label = "")

histogram2d!(posterior_locs_idx[1, :], posterior_locs_idx[2, :], bins = 100, normed = true, alpha = 0.8, color = cgrad(:viridis))
title!("Posterior")

scatter!(parlour1[1:1], parlour1[2:2], label = "Parlour #1", color = :red, markersize = 5)
scatter!(parlour2[1:1], parlour2[2:2], label = "Parlour #2", color = :blue, markersize = 5)

savefig("../figures/nf_banana_figures/posterior_$(num_mcmc_samples)_$(mcmc_warmup).svg")
savefig("../figures/nf_banana_figures/posterior_$(num_mcmc_samples)_$(mcmc_warmup).png")

p2 = scatter(x_samples[1, :], x_samples[2, :], label = "locations", markerstrokewidth = 0, color = :orange, alpha = 0.3)

xlims!(-10.0, 30.0)
ylims!(-15.0, 15.0)

histogram2d!(x_samples[1, :], x_samples[2, :]; bins = 100, normed = true, alpha = 0.8, color = cgrad(:viridis))
title!("Prior")

scatter!(parlour1[1:1], parlour1[2:2], label = "Parlour #1", color = :red, markersize = 5)
scatter!(parlour2[1:1], parlour2[2:2], label = "Parlour #2", color = :blue, markersize = 5)

savefig("../figures/nf_banana_figures/prior_$(num_mcmc_samples)_$(mcmc_warmup).svg")
savefig("../figures/nf_banana_figures/prior_$(num_mcmc_samples)_$(mcmc_warmup).png")

plot(p1, p2, layout = (2, 1), size = (500, 1000))
savefig("../figures/nf_banana_figures/combined_$(num_mcmc_samples)_$(mcmc_warmup).svg")
savefig("../figures/nf_banana_figures/combined_$(num_mcmc_samples)_$(mcmc_warmup).png")


# SAVE THE STUFF
for customer_idx = 1:num_fake_samples
    # Transform because samples will be untransformed
    posterior_locs_idx = td.transform(posterior_locs_samples[:, customer_idx, :]')

    mean_locs = mean(posterior_locs_idx; dims = 2)
    std_locs = std(posterior_locs_idx; dims = 2)

    scatter(x_samples[1, :], x_samples[2, :], label = "x", markerstrokewidth = 0, color = :orange, alpha = 0.3)

    xlims!(-10.0, 30.0)
    ylims!(-15.0, 15.0)

    # scatter!(mean_locs[1:1], mean_locs[2:2], xerr = std_locs[1:1], yerr = std_locs[2:2], label = "")

    histogram2d!(posterior_locs_idx[1, :], posterior_locs_idx[2, :], bins = 100, normed = true, alpha = 0.8, color = cgrad(:viridis))

    scatter!(parlour1[1:1], parlour1[2:2], label = "Parlour #1", color = :red, markersize = 5)
    scatter!(parlour2[1:1], parlour2[2:2], label = "Parlour #2", color = :blue, markersize = 5)

    title!("Customer $customer_idx: Parlour #$(fake_samples[customer_idx])")
    
    savefig("../figures/nf_banana_figures/customer_$(num_mcmc_samples)_$(mcmc_warmup)_$customer_idx.svg")
    savefig("../figures/nf_banana_figures/customer_$(num_mcmc_samples)_$(mcmc_warmup)_$customer_idx.png")
end
