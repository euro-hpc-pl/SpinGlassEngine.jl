@testset "Simplest possible system of fours spins with diagonal edges" begin
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
    # -------------------------------------------------

    # Model's parameters
    J12 = -1.0
    J13 = -1.0
    J14 = -0.5
    J24 = -0.6
    J34 = -1.0
    h1 = 0.5
    h2 = 0.75
    h3 = 0.0
    h4 = 0.0

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 3) => J13,
             (1, 4) => J14,
             (2, 4) => J24,
             (3, 4) => J34,
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
    
    ϱ = gibbs_tensor(ig, β)
    println("Gibbs tensor ", ϱ)
    for transform ∈ all_lattice_transformations
        peps = PegasusNetwork(m, n, fg, transform, β=β)
        
        #ψ = IdentityMPS()
    
        for i ∈ peps.nrows:-1:1
            Z = MPO_with_fusing(T, peps, i)
            println("i ", i)
            println("partition function ", Z)
            a = MPO_with_fusing(T, peps, i, Dict((1, 1) => 1, (1, 2) => 1, (2, 1) => 1, (2, 2) => 1))
            println("a ", a)
            display(a)
            #println("prob ", a/partition_function)
        #    ψ = MPO_with_fusing(T, peps, i) * ψ
        #    @test MPS(peps, i) ≈ ψ
        end
    end
end