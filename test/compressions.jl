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

    # ψ₀
    @time ψ_new = compress(W * ψ, Dcut, tol, 1000)

    bra = Dict(i => A for (i, A) ∈ enumerate(ψ_new))
    
    @time compress!(bra, mpo, ket, Dcut, tol, max_sweeps)

    M = randn(MPS{T}, sites, D, d)
    for i ∈ 1:sites
        M[i] = bra[i]
    end

    println(ψ_new * ψ_new)
    println(M * M)
    println(M * ψ_new)
    println(ψ_new * M)

    xx = W * ψ

    println(xx * ψ_new)
    println(xx * M)
    

    # println(ψ_new)
    # println(M)

end