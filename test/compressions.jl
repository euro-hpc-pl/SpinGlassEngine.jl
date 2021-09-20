function SpinGlassTensors.MPS(ket::Dict)
    L = length(ket)
    ϕ = MPS(eltype(ket[1]), L) 
    for i ∈ 1:L ϕ[i] = ket[i] end
    ϕ
end

Base.Dict(
    ϕ::SpinGlassTensors.AbstractTensorNetwork
) = Dict(i => A for (i, A) ∈ enumerate(ϕ))

#=
@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 100
    T = Float64
    
    Dcut = 8
    max_sweeps = 10
    tol = 1E-10

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    ket = Dict(ψ)
    mpo = Dict(W)

    @testset "Two mps representations are compressed to the same state" begin 
        @time χ = compress(W * ψ, Dcut, tol, max_sweeps)
        @time is_right_normalized(χ)

        #ϕ = copy(ψ)
        #canonise!(ϕ, :right)
        #bra = Dict(ϕ)

        bra = copy(Dict(χ))
        @time compress!(bra, mpo, ket, Dcut, tol, max_sweeps)

        ϕ = MPS(bra)
        @time is_right_normalized(ϕ)
        @test norm(χ) ≈ norm(ϕ) ≈ 1
        @test dot(ϕ, χ) ≈ dot(χ, ϕ) ≈ 1 
    end
end
=#

@testset "Contraction" begin
    D = 3
    d = 2
    sites = 5
    T = Float64
    
    #Dcut = 8
    #max_sweeps = 10
    #tol = 1E-10

    M = randn(MPS{T}, sites, D, d)
    W = randn(MPS{T}, sites, D, d)
    ψ = Mps(Dict(M))
    ϕ = Mps(Dict(W))

    @testset "dot products" begin
        @testset "is equal to itself" begin
            @test dot(ψ, ψ) ≈ dot(ψ, ψ)
        end
    
        @testset "change of arguments results in conjugation" begin
            @test dot(ψ, ϕ) ≈ conj(dot(ϕ, ψ))
        end
    
        @testset "norm is 2-norm" begin
            @test norm(ψ) ≈ sqrt(abs(dot(ψ, ψ)))
        end
    
        @testset "renormalizations" begin
            ψ.ket[ψ.sites_ket[end]] *= 1/norm(ψ)
            @test dot(ψ, ψ) ≈ 1
    
            ϕ.ket[ψ.sites_ket[1]] *= 1/norm(ϕ)
            @test dot(ϕ, ϕ) ≈ 1
        end
    
    end 
end

@testset "Canonisation" begin

    D = 16
    Dcut = 8
    
    d = 2
    sites = 100
    
    T = Float64
    
    M = randn(MPS{T}, sites, D, d)
    W = randn(MPS{T}, sites, D, d)
    ψ = Mps(Dict(M))
    ϕ = Mps(Dict(W))
    
    
    @testset "Canonisation (left)" begin
        a = norm(ψ)
        b = canonise(ψ, :left, Dcut)
        @test a ≈ b
        #@test is_left_normalized(ψ)
        @test dot(ψ, ψ) ≈ 1
    end
    
    @testset "Canonisation (right)" begin
        a = norm(ϕ)
        b = canonise(ϕ, :right, Dcut)
        @test a ≈ b 
        #@test is_right_normalized(ϕ)
        @test dot(ϕ, ϕ) ≈ 1
    end
end