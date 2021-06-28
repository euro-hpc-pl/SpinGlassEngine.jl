@testset "System of four spins" begin
    #
    # ----------------- Ising model ------------------
    #
    # E = -1.0 * s1 * s2 + 0.5 * s1 + 0.75 * s2
    #
    # states   -> [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    # energies -> [-2.25, 0.25, 0.75, 1.25]
    #
    # -------------------------------------------------
    #         Grid
    #     A1    |    A2
    #           |
    #       1 - | - 2
    #       | \     |    
    #       |  \    |    
    #       |   \   |    
    #       3 - | - 4
    # -------------------------------------------------

    # Model's parameters
    J12 = -1.0
    J13 = -1.0
    J34 = -0.5
    J24 = -0.6
    J14 = -1.0
    h1 = 0.5
    h2 = 0.75
    h3 = 0.0
    h4 = 0.0

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 3) => J13,
             (3, 4) => J34,
             (2, 4) => J24,
             (1, 4) => J14,
             (1, 1) => h1,
             (2, 2) => h2,
             (3, 3) => h3,
             (4, 4) => h4,
    )

    # control parameters
    m, n = 2, 2
    L = 4
    β = 1.
    num_states = 8
    T = Float64

    # read in pure Ising
    ig = ising_graph(D)

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict((1, 1) => 2, (1, 2) => 2, (2, 1) => 2, (2, 2) => 2), 
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2), 3 => (2, 1), 4 => (2, 2)), 
    )

    # set parameters to contract exactly
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )
    
    for transform ∈ all_lattice_transformations
        peps = PEPSNetwork(m, n, fg, transform, β=β)
    
        ψ = IdentityMPS()
    
        for i ∈ peps.nrows:-1:1
            ψ = MPO_with_fusing(T, peps, i) * ψ
            @test MPS(peps, i) ≈ ψ
        end
    end
end