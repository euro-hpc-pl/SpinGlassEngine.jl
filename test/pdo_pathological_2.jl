@testset "Pathological instance with one diagonal" begin
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t
    num_states = 10


    # control_params = Dict(
    #     "bond_dim" => typemax(Int),
    #     "var_tol" => 1E-8,
    #     "sweeps" => 4.
    # )

    instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_dd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        #cluster_assignment_rule=super_square_lattice((m, n, t))
        cluster_assignment_rule=Dict(1 => (1, 1), 2 => (1, 1), 3 => (1, 1), 4 => (1, 2), 5 => (1, 2), 6 => (1, 2),
                                    7 => (1, 3), 8 => (1, 3), 9 => (1, 3), 10 => (1, 4), 11 => (1, 4), 12 => (1, 4),
                                    13 => (2, 1), 14 => (2, 1), 15 => (2, 1), 16 => (2, 2), 17 => (2, 2), 18 => (2, 2),
                                    19 => (2, 3), 20 => (2, 3), 21 => (2, 3), 22 => (2, 4), 23 => (2, 4), 24 => (2, 4),
                                    25 => (3, 1), 26 => (3, 1), 27 => (3, 1), 28 => (3, 2), 29 => (3, 2), 30 => (3, 2),
                                    31 => (3, 3), 32 => (3, 3), 33 => (3, 3), 34 => (3, 4), 35 => (3, 4), 36 => (3, 4)), 
    )

    for transform ∈ rotation.([0])
        #peps = PegasusNetwork(m, n, fg, transform, β=β)
        peps = NNNNetwork(m, n, fg, transform, β=β)

        sol = low_energy_spectrum(peps, num_states)
        println(sol.energies)
        println(sol.states)
        println(sol.probabilities)
        # solve the problem using B & B
        # sol = low_energy_spectrum(peps, num_states)
    end
end