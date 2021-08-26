@testset "Tensor and tensor_size gives correct size of tensor for the simplest possible system of four spins" begin

    # Model's parameters
    J12 = -1.0
    J13 = -0.5
    J14 = -0.1
    h1 = 0.5
    h2 = 0.75
    h3 = 0.1
    h4 = 0.1

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 3) => J13,
             (1, 4) => J14,
             (1, 1) => h1,
             (2, 2) => h2,
             (3, 3) => h3,
             (4, 4) => h4,

    )

    # control parameters
    m, n = 2, 2
    t = 1
    β = 1.
    num_states = 4

    # read in pure Ising
    ig = ising_graph(D)

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t)), 
    )

    peps = FusedNetwork(m, n, fg, rotation(0), β=β)
    update_gauges!(peps, :rand)

    @testset "for site tensors" begin
        expected_traced = [(1, 1, 2, 2) (2, 1, 1, 1); (1, 2, 1, 1) (2, 1, 1, 1)]
        TA = tensor_assignment(peps, :site)
        for i ∈ 1:peps.nrows, k ∈ 1:peps.ncols
            A = tensor(peps, (i, k))
            @test size(A) == tensor_size(peps, (i, k)) == expected_traced[i,k]
            @test TA[i, k] == :site
        end
    end

    @testset "for connecting tensors" begin
        expected_connecting = [(2, 1, 2, 2) (1, 1, 1, 1); (1, 2, 2, 1) (1, 1, 1, 1)]
        TA = tensor_assignment(peps, :central_h)
        for i ∈ 1:peps.nrows, k ∈ 1:peps.ncols  
            A = tensor(peps, (i, k+1//2))
            @test size(A) == tensor_size(peps, (i, k+1//2)) == expected_connecting[i,k]
            @test TA[i, k+1//2] == :central_h
        end
    end

    @testset "for horizontal and vertical tensors" begin
        expected_vhcentral = [(1, 2, 1, 2) (1, 1, 1, 1)]
        TA = tensor_assignment(peps, :central_v)
        for i ∈ 1:peps.nrows-1, k ∈ 1:peps.ncols  
            A = tensor(peps, (i+1//2, k))
            @test size(A) == tensor_size(peps, (i+1//2, k)) == expected_vhcentral[i,k]
            @test TA[i+1//2, k] == :central_v
        end
    end

    @testset "for diagonal tensors" begin
        expected_dcentral = [(1, 2, 1, 2) (1, 1, 1, 1)]
        TA = tensor_assignment(peps, :central_d)
        for i ∈ 1:peps.nrows-1, k ∈ 1:peps.ncols-1  
            A = tensor(peps, (i+1//2, k+1//2))
            @test size(A) == tensor_size(peps, (i+1//2, k+1//2)) == expected_dcentral[i,k]
            @test TA[i+1//2, k+1//2] == :central_d
        end
    end

    @testset "for gauges" begin
        expected_gauges = [(1, 2, 1, 2) (1, 2, 1, 2) (1, 1, 1, 1)]
        TA = tensor_assignment(peps, :gauge_h)
        for i ∈ 1:peps.nrows-1, k ∈ 1:1//2:peps.ncols - 1, g ∈ [1//6, 2//6]
            A = tensor(peps, (i+g, k))
            @test size(A) == tensor_size(peps, (i+g, k)) == expected_gauges[Int(i), Int(k)]
            @test TA[i+g, k] == :gauge_h
        end
    end
end 