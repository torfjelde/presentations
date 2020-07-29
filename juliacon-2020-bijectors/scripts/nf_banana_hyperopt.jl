using Hyperopt

# learning_rate = 1e-3
ho = @hyperopt for  opt_idx = 10, learning_rate = LinRange(1e-6, 1e-1, 100), seed = [1, 2, 3, 4, 5]
    println(opt_idx, "\t", learning_rate, "\t", seed)

    # initialization
    rng = Random.MersenneTwister(seed)
    θ = randn(rng, num_params_per_layer * num_layers)
    
    diff_result = DiffResults.GradientResult(θ)
    nlls = []
    
    opt = Optimise.ADAM(learning_rate)

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
    end

    # Estimate likelihood
    flow = construct_flow(θ)
    td = transformed(base, flow)
    x_samples = generate_samples(5_000)
    @show - mean(logpdf(td, x_samples))
end

plot(ho)
learning_rate, seed = first(minimum(ho))

# Now we do a finer search over the `learning_rate` using Bayesina optimization
# with the "optimal" initialization obtained from the previous random search.
ho = @hyperopt for  opt_idx = 10, sampler = GPSampler(Min), learning_rate = LinRange(1e-6, 1e-2, 100)
    println(opt_idx, "\t", learning_rate, "\t", seed)

    # initialization
    global seed
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
    end

    # Estimate likelihood
    flow = construct_flow(θ)
    td = transformed(base, flow)
    x_samples = generate_samples(5_000)
    @show - mean(logpdf(td, x_samples))
end
