@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 5
    T = Float64
    
    Dcut = 8
    max_sweeps = 30
    tol = 1E-10

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    ket = Dict(i => A for (i, A) ∈ enumerate(ψ))
    mpo = Dict(i => O for (i, O) ∈ enumerate(W))

    @testset "Two mps representations are the same" begin
        for i ∈ 1:sites @test ψ[i] ≈ ket[i] end
    end

    @testset "Two mps representations are truncated to the same state" begin
        ψ̃ = copy(ψ)
        truncate!(ψ̃, :right, Dcut)
        @test ψ̃ * ψ̃ ≈ 1 

        truncate!(ket, :right, Dcut)
        @test dot(ket, ket) ≈ 1 
        
        @test dot(ψ̃, ket) ≈ 1 
        @test dot(ket, ψ̃) ≈ 1 
    end    

    #=
    @testset "Two mps representations are compressed to the same state" begin 
        @time ψ̃ = compress(W * ψ, Dcut, tol, max_sweeps)

        bra = Dict(i => A for (i, A) ∈ enumerate(ψ))
        canonise!(bra, :right)
        @time bra = compress(bra, mpo, ket, Dcut, tol, max_sweeps)

        @test dot(ψ̃, bra) ≈ 1 
    end
    =#
end