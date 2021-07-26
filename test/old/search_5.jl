@testset "System of six spins with diagonal edges" begin
    #
    # ----------------- Ising model ------------------
    #
    # -------------------------------------------------
    #         Grid
    #      
    #       1 - | - 2 
    #       | \     | 
    #       |   \   | 
    #       |     \ | 
    #       3 - | - 4
    #       | \     | 
    #       |   \   |  
    #       |     \ | 
    #       5 - | - 6

    # -------------------------------------------------

    # Model's parameters
    J12 = 1.0
    J14 = 0.1
    J13 = 0.1
    J23 = 0.1
    J24 = 0.5
    J34 = 0.1
    J35 = 0.3
    J36 = 0.1
    J45 = 0.1
    J46 = 0.1
    J56 = 0.7

    h1 = 0.1
    h2 = 0.0
    h3 = 0.2
    h4 = 0.8
    h5 = 0.3
    h6 = 0.6

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 4) => J14,
             (1, 3) => J13,
             (2, 3) => J23,
             (2, 4) => J24,
             (3, 4) => J34,
             (3, 5) => J35,
             (3, 6) => J36,
             (4, 5) => J45,
             (4, 6) => J46,
             (5, 6) => J56,
             (1, 1) => h1,
             (2, 2) => h2,
             (3, 3) => h3,
             (4, 4) => h4,
             (5, 5) => h5,
             (6, 6) => h6,
    )

    # control parameters
    m, n = 3, 2
    L = 6
    β = 1.
    num_states = 12
    T = Float64

    # read in pure Ising
    ig = ising_graph(D)

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict((1, 1) => 2, (1, 2) => 2, (2, 1) => 2, (2, 2) => 2, (3, 1) => 2, (3, 2) => 2), 
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2), 3 => (2, 1), 4 => (2, 2), 5 => (3, 1), 6 => (3, 2)), 
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
        peps = NNNNetwork(m, n, fg, transform, β=β)
    
        I = IdentityMPS()
        for (i, s) in enumerate(states)
            s = (s .+ 3) .÷ 2 # -1 ==> 1, 1 ==> 2
            configuration = Dict((1, 1) => s[1], (1, 2) => s[2], (2, 1) => s[3], (2, 2) => s[4], (3, 1) => s[5], (3, 2) => s[6])
            Z1 = MPO_with_fusing(T, peps, 1)
            Z2 = MPO_with_fusing(T, peps, 2)
            Z3 = MPO_with_fusing(T, peps, 3)
            ψ1 = MPO_with_fusing(T, peps, 1, configuration)
            ψ2 = MPO_with_fusing(T, peps, 2, configuration)
            ψ3 = MPO_with_fusing(T, peps, 3, configuration)

            prob = dot(I,ψ1*ψ2*ψ3*I)/dot(I,Z1*Z2*Z3*I)

            @test prob ≈ ρ[i]
        end
    end
end