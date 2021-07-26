@testset "System of six spins with diagonal edges" begin
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
    J12 = 1.0
    J14 = 0.3
    J15 = 0.9
    J23 = 0.2
    J25 = 0.3
    J26 = 0.1
    J36 = 0.7
    J45 = 0.5
    J56 = 1.0
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
    num_states = 12
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

    states = collect.(all_states(rank_vec(ig)))
    ρ = exp.(-β .* energy.(states, Ref(ig)))
    ρ = ρ ./ sum(ρ)

    for transform ∈ rotation.([0, 180])
        peps = PegasusNetwork(m, n, fg, transform, β=β)
    
        I = IdentityMPS()
        for (i, s) in enumerate(states)
            s = (s .+ 3) .÷ 2 # -1 ==> 1, 1 ==> 2
            configuration = Dict((1, 1) => s[1], (1, 2) => s[2], (1, 3) => s[3], (2, 1) => s[4], (2, 2) => s[5], (2, 3) => s[6])
            Z1 = MPO_with_fusing(T, peps, 1)
            Z2 = MPO_with_fusing(T, peps, 2)
            ψ1 = MPO_with_fusing(T, peps, 1, configuration)
            ψ2 = MPO_with_fusing(T, peps, 2, configuration)
            prob = dot(I,ψ1*ψ2*I)/dot(I,Z1*Z2*I)

            @test prob ≈ ρ[i]
        end
    end
end