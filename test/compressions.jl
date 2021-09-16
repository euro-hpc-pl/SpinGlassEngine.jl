@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 100
    T = Float64
    
    Dcut = 8
    max_sweeps = 30
    tol = 1E-10

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    ket = Dict(i => copy(A) for (i, A) ∈ enumerate(ψ))
    mpo = Dict(i => B for (i, B) ∈ enumerate(W))

    @testset "Two mps representations are compressed to the same state" begin 
        @time ψ̃  = compress(W * ψ, Dcut, tol, max_sweeps)

        bra = copy(ket)
        ϕ = MPS(T, sites) 
        for i ∈ 1:sites ϕ[i] = copy(ket[i]) end
        canonise!(ϕ, :right)
        bra = Dict(i => copy(A) for (i, A) ∈ enumerate(ϕ))

        #bra = Dict(i => copy(A) for (i, A) ∈ enumerate(ψ̃ ))
        @time compress!(bra, mpo, ket, Dcut, tol, max_sweeps)

        ϕ = MPS(T, sites) 
        for i ∈ 1:sites ϕ[i] = bra[i] end

        @test ψ̃ * ψ̃ ≈ ϕ * ϕ ≈ 1
        @test dot(ϕ, ψ̃ ) ≈ dot(ψ̃ , ϕ) ≈ 1 
    end
end