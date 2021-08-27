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


function test2()

    D = 100
    L = rand(D, D)
    R = rand(D, D)
    M = rand(D, 2, D)

    @time begin
        @tensor M̃[x, σ, y] := L[x, β] * M[β, σ, α] * R[α, y] order = (α, β)
        @cast A[(x, σ), y] |= M̃[x, σ, y]
    end

    @time begin
        @matmul M̃[x, σ, α] := sum(β) L[x, β] * M[β, σ, α] 
        @matmul B[(x, σ), y] |= sum(α) M̃[x, σ, α] * R[α, y]
    end

    @time begin
        @tensor M̃[x, σ, α] := L[x, β] * M[β, σ, α] 
        @matmul C[(x, σ), y] |= sum(α) M̃[x, σ, α] * R[α, y]
    end
end


function test3()
    m=1
    D = 4000
    L = rand(D)
    O = rand(1, D)
    M = rand(D, 2, D)

    @time @matmul LL[x] := sum(α) L[α] * M[α, $m, x]
    K = @view M[:, m, :]
    println(size(O), size(K))
    @time LLL = O * K
end


function test4()
    D = 5000
    X = rand(D, D)

    @time @cast A[_, a, _, b] := X[a, b]

    a, b = size(X)
    @time B = reshape(X, (1, a, 1, b))
end


function test5()
    D = 50
    d = 30
    
    projs = (rand(d, D) for _ in 1:4)    
    loc_exp = ones(d)

    @time begin
        @cast A[i, _] := loc_exp[i]
        for pv ∈ projs 
            @cast A[σ, (c, γ)] |= A[σ, c] * pv[σ, γ] 
        end 
        BB = sum(A, dims=1)
    end


    @time begin
        @cast A[_, i] := loc_exp[i]
        for pv ∈ projs 
            @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ] 
            B = sum(A, dims=5)
        end 
    end

    #@test A ≈ B
end 

test5()


