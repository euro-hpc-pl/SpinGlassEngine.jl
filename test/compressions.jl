function copy_mps(ψ::AbstractMPS)
    L = length(ψ)
    ϕ = MPS(eltype(ψ), L)
    for i ∈ 1:L ϕ[i] = copy(ψ[i]) end
    ϕ
end

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

    ket = Dict(i => copy(A) for (i, A) ∈ enumerate(ψ))
    mpo = Dict(i => B for (i, B) ∈ enumerate(W))

    @testset "Two mps representations are the same" begin
        for i ∈ 1:sites @test ψ[i] ≈ ket[i] end
    end

    #
    @testset "Two mps representations are truncated to the same state" begin
        ψ̃ = copy_mps(ψ)
        ϕ = copy_mps(ψ)
        
        @test ϕ == ψ̃
        @test norm(ϕ) ≈ norm(ψ̃)

        truncate!(ψ̃, :right, Dcut)
        truncate!(ϕ, :right, Dcut)
        
        @test norm(ϕ) ≈ norm(ψ̃)
        @test dot(ψ̃, ϕ) ≈ dot(ϕ, ψ̃) ≈ 1

        #ϕ = MPS(T, sites) 
        #for i ∈ 1:sites ϕ[i] = copy(ψ̃[i]) end
        
        #truncate!(ϕ, :right, Dcut)

        #for i ∈ 1:sites ket[i] = copy(ϕ[i]) end

        #truncate!(ket, :right, Dcut)

        #@test dot(ket, ket) ≈ ψ * ψ ≈ 1 

        #@test dot(ϕ, ϕ) ≈ ψ * ψ ≈ 1 
        #@test dot(ψ, ket) ≈ dot(ket, ψ) ≈ 1
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