using LinearAlgebra

using Bijectors
import Bijectors: logabsdetjac

struct MatrixPlusLowRankScale{TL, TV} <: Bijector{1}
    L::TL
    V::TV
end

(b::MatrixPlusLowRankScale)(x::AbstractVecOrMat) = b.L * x + b.V * (b.V' * x)
function (ib::Inversed{<:MatrixPlusLowRankScale})(y::AbstractVecOrMat)
    b = ib.orig
    r = size(b.V, 2)

    # return (b.L + b.V * b.V') \ y

    L = b.L
    L⁻¹ = inv(L)

    V = b.V'
    U = b.V
    C = Diagonal(ones(r))

    # "Woodbury matrix identity"
    B = (C + V * L⁻¹ * U)
    # @assert det(B) ≠ 0

    # @info size(inv(F))

    # F = (inv(B) * (V * (L⁻¹ * y)))
    # F = B \ (V * (L⁻¹ * y))
    # @info V * (L⁻¹ * y)
    # @info det(B)
    # @info norm(L⁻¹ * y)
    # @info norm((V * (L⁻¹ * y)))
    
    return L⁻¹ * y - (L⁻¹ * (U * (inv(B) * (V * (L⁻¹ * y)))))
end

function logabsdetjac(b::MatrixPlusLowRankScale, x::AbstractVecOrMat)    
    # "Matrix determinant lemma"
    logjac = first(logabsdet(1 .+ b.V' * inv(b.L) * b.V)) * det(b.L)
    if x isa AbstractVector
        return logjac
    else
        return logjac .* ones(size(x, 2))
    end
end

# T = Float64
# d = 1000; r = 50;
# L = LowerTriangular(randn(T, d, d))
# V_ = randn(T, d, r)
# b = MatrixPlusLowRankScale(L, V_) # 
# b⁻¹ = inv(b)

# x = ones(T, d)
# y = b(x)
# norm(x - b⁻¹(y))

# logabsdetjac(b, x)

# L⁻¹ = inv(L)
# @info L * L⁻¹

# V = V_'
# U = V_
# C = Diagonal(ones(r))

# # "Woodbury matrix identity"
# B = C + V * L⁻¹ * U
# @assert det(B) ≠ 0
# F = svd(B)
# inv(F)

# using BenchmarkTools
# @benchmark $b⁻¹($y)
