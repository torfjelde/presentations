using Plots, StatsPlots

"""
    quadrant_masks(x::AbstractMatrix)

Returns a tuple of masks `(ll_mask, ul_mask, ur_mask, lr_mask)`
where `l` and `u` in first position stands for lower and upper,
respectively. while `l` and `r` in the second position stands for
left and right, respectively.

# Examples
```julia-repl
julia> z1 = [-1., 1]; z2 = [1., -1.]; z3 = [-1., -1.];

julia> z = hcat(z1, z1, z2, z3)
2×4 Array{Float64,2}:
 -1.0  -1.0   1.0  -1.0
  1.0   1.0  -1.0  -1.0

julia> ll_mask, ul_mask, ur_mask, lr_mask = quadrant_masks(z)
(Bool[false, false, false, true], Bool[true, true, false, false], Bool[false, false, false, false], Bool[false, false, true, false])

julia> z[:, ll_mask] == reshape(z3, :, 1)
true

julia> z[:, ul_mask] == hcat(z1, z1)
true

julia> isempty(z[:, ur_mask])
true

julia> z[:, lr_mask] == reshape(z2, :, 1)
true
```
"""
function quadrant_masks(x::AbstractMatrix)
    ll_mask = [(x[1, i] < 0) & (x[2, i] < 0) for i = 1:size(x, 2)]
    lr_mask = [(x[1, i] ≥ 0) & (x[2, i] < 0) for i = 1:size(x, 2)]

    ul_mask = [(x[1, i] < 0) & (x[2, i] ≥ 0) for i = 1:size(x, 2)]
    ur_mask = [(x[1, i] ≥ 0) & (x[2, i] ≥ 0) for i = 1:size(x, 2)]

    return ll_mask, ul_mask, ur_mask, lr_mask
end

"""
    plot_samples(x::AbstractMatrix; masks = nothing, colors = [:red, :blue, :black, :green])

Plots the samples `x` using `masks` to color each sample according to the four masks.

If `masks === nothing`, `quadrant_masks` will be used to compute the masks.
"""
function plot_samples(x::AbstractMatrix; masks = nothing, colors = [:red, :blue, :black, :green])
    ll_mask, ul_mask, ur_mask, lr_mask = (masks === nothing) ? quadrant_masks(x) : masks
    
    p = scatter(x[1, ll_mask], x[2, ll_mask], markercolor = colors[1], label = "", markersize = 2, markerstrokewidth = 0)
    scatter!(x[1, ul_mask], x[2, ul_mask], markercolor = colors[2], label = "", markersize = 2, markerstrokewidth = 0)
    scatter!(x[1, ur_mask], x[2, ur_mask], markercolor = colors[3], label = "", markersize = 2, markerstrokewidth = 0)
    scatter!(x[1, lr_mask], x[2, lr_mask], markercolor = colors[4], label = "", markersize = 2, markerstrokewidth = 0)

    return p
end

"""
    function layerbylayer_plot(
        td::TransformedDistribution{<:Distribution, <:Composed},
        x::AbstractMatrix
    )

Plots the transformed samples `x` after each "layer", i.e. each `Bijector`
in the top-most composition.

In each plot, the points will be coloured according to which quadrant the
the untransformed point lies within. This makes use of `quadrant_masks` and
`plot_samples`.
"""
function layerbylayer_plot(
    td::TransformedDistribution{<:Distribution, <:Composed},
    x::AbstractMatrix
)
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

