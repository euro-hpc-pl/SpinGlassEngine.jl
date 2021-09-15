@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 20
    T = Float64
    
    Dcut = 8 
    max_sweeps = 30
    tol = 1E-20

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    ket = Dict(i => A for (i, A) ∈ enumerate(ψ))
    mpo = Dict(i => O for (i, O) ∈ enumerate(W))

    @time ψ_new = compress(W * ψ, Dcut, tol, 1000)

    bra = Dict(i => A for (i, A) ∈ enumerate(ψ_new))
    
    @time compress!(bra, mpo, ket, Dcut, tol, max_sweeps)

    ϕ = MPS(T, sites) 
    for i ∈ 1:sites ϕ[i] = bra[i] end

    @test norm(ψ_new) ≈ 1 
    @test norm(ϕ) ≈ 1

    @test ϕ * ψ_new ≈ 1
    @test ψ_new * ϕ ≈ 1

    ψ̃ = W * ψ

    println(ψ̃ * ψ_new)
    println(ψ̃ * ϕ)
    
end