using Bijectors
import Bijectors: logabsdetjac, forward

struct LeakyReLU{T} <: Bijector{1} # <= expects 1-dim input i.e. vector-input
    α::T
end

(b::LeakyReLU)(x::AbstractVecOrMat) = @. ifelse(x < 0.0, b.α * x, x)
(ib::Inversed{<:LeakyReLU})(y::AbstractVecOrMat) = @. ifelse(y < 0.0, inv(ib.orig.α) * y, y)

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

# Just some quick tests

b = LeakyReLU(Float32.(2))
z = ones(Float32, 2) - Float32.([0.0, 2.0])
y = b(z)
z_ = inv(b)(y)

@assert z_ == z

rv, logjac = forward(b, z)
logjac_forward = logabsdetjac(b, z)
logjac_inverse = logabsdetjacinv(b, y)

@assert logjac ≈ logjac_forward ≈ - logjac_inverse

b = LeakyReLU(Float32.(2))
z = ones(Float32, (2, 3)) - Float32.(repeat([0.0, 2.0]', 3)')
y = b(z)
z_ = inv(b)(y)

@assert z_ == z

rv, logjac = forward(b, z)
logjac_forward = logabsdetjac(b, z)
logjac_inverse = logabsdetjacinv(b, y)

@assert logjac ≈ logjac_forward ≈ - logjac_inverse
@assert all([y[:, i] == b(z[:, i]) for i = 1:size(z, 2)])
