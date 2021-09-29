function SpinGlassTensors.MPS(ket::Mps)
    L = length(ket)
    ϕ = MPS(eltype(ket[1]), L) 
    for i ∈ 1:L ϕ[i] = ket[i] end
    ϕ
end

Base.Dict(
    ϕ::SpinGlassTensors.AbstractTensorNetwork
) = Mps(Dict(i => A for (i, A) ∈ enumerate(ϕ)))


@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 100
    T = Float64
    
    Dcut = 8
    max_sweeps = 100
    tol = 1E-10

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    ket = Mps(ψ)
    mpo = Mpo(W)

    @testset "Two mps representations are compressed to the same state" begin 
        χ = W * ψ
        @time overlap = compress!(χ, Dcut, tol, max_sweeps)
        @test is_left_normalized(χ)
        println(overlap)

        bra = Mps(χ)
        @time overlap = compress!(bra, mpo, ket, Dcut, tol, max_sweeps)
        println(overlap)
        
        ϕ = MPS(bra)
        @time is_right_normalized(ϕ)
        @test norm(χ) ≈ norm(ϕ) ≈ 1
    end
end