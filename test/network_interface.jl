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

@testset "normalize_probability properly deals with negative probabilities" begin
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

    prob = [1.0, -1.0, 15.324, -2.2134, 0.123, 0.0]
    prob_new = [2.2134, 2.2134, 15.324, 2.2134, 2.2134, 2.2134]
    sum = 26.391
    expected_prob = [0.08386950096623849, 0.08386950096623849, 0.5806524951688076, 
                    0.08386950096623849, 0.08386950096623849, 0.08386950096623849]
    @test normalize_probability(peps, prob) ≈ expected_prob
end