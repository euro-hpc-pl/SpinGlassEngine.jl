@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 5
    T = Float64
    
    Dcut = 8 
    max_sweeps = 4
    tol = =1E-8

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    mps = Dict(i => A for (i, A) ∈ enumerate(ψ))
    mpo = Dict(i => O for (i, O) ∈ enumerate(W))

    ϕ = compress(W, ψ, Dcut, tol, max_sweeps, ψ₀)
end