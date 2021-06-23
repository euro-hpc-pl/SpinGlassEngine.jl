@testset "fuse_projectors correctly fuses projectors." begin
    projectors1 = ([1 1; 0 1], [1 0; 1 1])
    expected_fused1 = [1 0; 0 1]
    expected_transitions1 = [[1 1; 0 1], [1 0; 1 1]]
    fused1, transitions1 = fuse_projectors(projectors1)
    @test expected_fused1 == fused1
    @test expected_transitions1 == values(transitions1)

    projectors2 = ([1 1; 0 0], [1 0; 0 1])
    expected_fused2 = [1 0; 0 1]
    expected_transitions2 = [[1 1; 0 0], [1 0; 0 1]]
    fused2, transitions2 = fuse_projectors(projectors2)
    @test expected_fused2 == fused2
    @test expected_transitions2 == values(transitions2)

    projectors3 = ([1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1])
    expected_fused3 = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions3 = [[1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1]]
    fused3, transitions3 = fuse_projectors(projectors3)
    @test expected_fused3 == fused3
    @test expected_transitions3 == values(transitions3)

    projectors4 = ([1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1])
    expected_fused4 = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions4 = [[1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1]]
    fused4, transitions4 = fuse_projectors(projectors4)
    @test expected_fused4 == fused4
    @test expected_transitions4 == values(transitions4)
end

@testset "build_tensor_with_fusing correctly builds tensors with fusing." begin
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

    transforms = (rotation(0), reflection(:x))
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
            tensor, _ , _ = build_tensor_with_fusing(peps, v)
            push!(A, tensor)
        end
    end
    @test values(A[1]) ≈ values(A[3])
    for i in length(expected_values1)
        @test values(A[1][i]) ≈ expected_values1[i]
    end

    @test values(A[2]) ≈ values(A[4])
    for i in size(expected_values2)[1]
        @test values(A[2][i]) ≈ expected_values2[i]
    end
end