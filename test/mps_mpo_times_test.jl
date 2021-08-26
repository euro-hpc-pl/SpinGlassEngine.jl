using SpinGlassTensors
using TensorCast
using TensorOperations

function dot_by_hand(O::AbstractMPO, ψ::AbstractMPS)
    S = promote_type(eltype(ψ), eltype(O))
    T = typeof(ψ)
    ϕ = T.name.wrapper(S, length(ψ))

    for (i, (A, B)) ∈ enumerate(zip(O, ψ))
        BB = permutedims(B, (2, 1, 3))
        x, y, z = size(BB)
        BBB = reshape(BB, (x, y*z))

        a, b, c, d = size(A)
        AA = reshape(A, (a*b*c, d))

        C = reshape(AA * BBB, (a, b, c, y, z))
        CC = permutedims(C, (1, 4, 2, 3, 5))

        a, b, c, d, e = size(CC)
        ϕ[i] = reshape(CC, (a*b, c, d*e))

        #@reduce N[(x, a), σ, (y, b)] := sum(η) A[x, σ, y, η] * B[a, η, b]
        #ϕ[i] = N
    end
    ϕ
end

function dot_by_hand_tc(O::AbstractMPO, ψ::AbstractMPS)
    S = promote_type(eltype(ψ), eltype(O))
    T = typeof(ψ)
    ϕ = T.name.wrapper(S, length(ψ))
    for (i, (A, B)) ∈ enumerate(zip(O, ψ))
        @tensor AA[x, σ, y, a, b] := A[x, σ, y, η] * B[a, η, b]
        @cast B[(x, a), σ, (y, b)] := AA[x, σ, y, a, b]
        ϕ[i] = B
    end
    ϕ
end

function dot_by_hand_mm(O::AbstractMPO, ψ::AbstractMPS)
    S = promote_type(eltype(ψ), eltype(O))
    T = typeof(ψ)
    ϕ = T.name.wrapper(S, length(ψ))
    for (i, (A, B)) ∈ enumerate(zip(O, ψ))
        @matmul N[(x, a), σ, (y, b)] := sum(η) A[x, σ, y, η] * B[a, η, b]
        ϕ[i] = N
    end
    ϕ
end


D = 24
d = 2
sites = 1000
T = Float64

function test()
    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    @time z = dot(W, ψ)
    @time dot!(ψ, W)

    #@time x = dot_by_hand_tc(W, ψ)
    #@time f = dot_by_hand_mm(W, ψ)
    #@time y = dot_by_hand(W, ψ)

    #@test x ≈ y
end

test()


