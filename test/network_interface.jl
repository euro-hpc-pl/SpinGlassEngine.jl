@testset "fuse_projectors correctly fuses projectors - 2x2 matrix." begin
    projectors = ([1 1; 0 1], [1 0; 1 1])
    expected_fused = [1 0; 0 1]
    expected_transitions = [[1 1; 0 1], [1 0; 1 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end

@testset "fuse_projectors correctly fuses projectors - another 2x2 matrix." begin
    projectors = ([1 1; 0 0], [1 0; 0 1])
    expected_fused = [1 0; 0 1]
    expected_transitions = [[1 1; 0 0], [1 0; 0 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end

@testset "fuse_projectors correctly fuses projectors - 3x3 matrix." begin
    projectors = ([1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1])
    expected_fused = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions = [[1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end

@testset "fuse_projectors correctly fuses projectors - another 3x3 matrix." begin
    projectors = ([1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1])
    expected_fused = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions = [[1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end

@testset "build_tensor_with_fusing correctly builds tensors with fusing" begin
    # Model's parameters
    J12 = -1.0
    h1 = 0.5
    h2 = 0.75
   
    # dict to be read
    D = Dict((1, 2) => J12,
            (1, 1) => h1,
            (2, 2) => h2,
    )
   
    # control parameters
    m, n = 1, 2
    L = 2
    β = 1.
    num_states = 4
   
    # read in pure Ising
    ig = ising_graph(D)
   
    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict((1, 1) => 2, (1, 2) => 2),
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2)), # treat it as a grid with 1 spin cells
    )
   
    # set parameters to contract exactely
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    transforms = (LatticeTransformation((1, 2, 3, 4), true), LatticeTransformation((4, 3, 2, 1), true))
    A = []
    expected_values1 = [1.6487212707001282, 0.0, 0.0, 0.6065306597126334]
    expected_values2 = [2.117000016612675 0.0; 0.0 0.4723665527410147]
    for transform ∈ transforms
        peps = PEPSNetwork(m, n, fg, transform, β=β)
        for v in vertices(fg)
            # expected output: exp(-β * local_energy)
            # for first vertex (1,1) we have exp(-[-0.5 0.5]) = [1.6487 0.6065]
            # for second vertex (1,2) we have exp(-[-0.75 0.75]) = [2.117 0.4723]
            # contract it with fused projectors 
            push!(A, build_tensor_with_fusing(peps, v))
        end
    end
    @test values(A[1]) ≈ values(A[3])
    for i in length(expected_values1)
        @test values(A[1][i]) ≈ expected_values1[i]
    end

    @test values(A[2]) ≈ values(A[4])
    for i in size(expected_values2)[1]
        @test values(A[1][i]) ≈ expected_values1[i]
    end
end