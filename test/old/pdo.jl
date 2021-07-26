@testset "Minimal instance" begin
    #
    # ----------------- Ising model ------------------
    #
    # -------------------------------------------------
    #         Grid
    #      
    #       1 - | - 2 - | - 3
    #       | \     | \     |
    #       |   \   |   \   |
    #       |     \ |     \ | 
    #       4 - | - 5 - | - 6
    # -------------------------------------------------

    # Model's parameters
    J12 = -1.0
    J14 = -0.3
    J15 = -0.9
    J23 = -0.2
    J25 = -0.3
    J26 = -0.1
    J36 = -0.7
    J45 = -0.5
    J56 = -1.0
    h1 = 0.1
    h2 = 0.0
    h3 = 0.2
    h4 = 0.8
    h5 = 0.3
    h6 = 0.6

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 4) => J14,
             (1, 5) => J15,
             (2, 3) => J23,
             (2, 5) => J25,
             (2, 6) => J26,
             (3, 6) => J36,
             (4, 5) => J45,
             (5, 6) => J56,
             (1, 1) => h1,
             (2, 2) => h2,
             (3, 3) => h3,
             (4, 4) => h4,
             (5, 5) => h5,
             (6, 6) => h6,
    )

    # control parameters
    m, n = 2, 3
    L = 6
    β = 1.
    num_states = 20
    T = Float64

    # read in pure Ising
    ig = ising_graph(D)

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict((1, 1) => 2, (1, 2) => 2, (1, 3) => 2, (2, 1) => 2, (2, 2) => 2, (2, 3) => 2), 
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2), 3 => (1, 3), 4 => (2, 1), 5 => (2, 2), 6 => (2, 3)), 
    )

    # set parameters to contract exactly
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    sol = empty_solution()

    for transform ∈ rotation.([0])
        peps = PegasusNetwork(m, n, fg, transform, β=β)

        # solve the problem using B & B
        sol = low_energy_spectrum(peps, num_states)
        #prob = conditional_probability(peps, [1,1,1,1])#sol.states[1])
        println(sol.energies)
        println(sol.states)
        println(sol.probabilities)
        #println(prob)
    end
end