@testset "Pathological Pegasus instance works" begin

    # is this correct?
    expected_energies = [-23.301855000000000, -23.221513000000002, -23.002799000000003,
                         -22.922457000000005, -22.664197000000005, -22.583855000000000,
                         -22.546913000000000, -22.530343000000002, -22.474668999999999, 
                         -22.466571000000002
                        ]

    num_states = length(expected_energies)

    m = 3
    n = 4
    t = 3

    #m = 2
    #n = 2
    #t = 24

    β = 1.

    L = n * m * t
    states_to_keep = 20

    control_params = Dict(
         "bond_dim" => typemax(Int),
         "var_tol" => 1E-8,
         "sweeps" => 4.
    )

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"
    #instance = "$(@__DIR__)/instances/pegasus_droplets/2_2_3_00.txt"


    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t)) 
    )

    for transform ∈ rotation.([0])
        peps = FusedNetwork(m, n, fg, transform, β=β)
        sol = low_energy_spectrum(peps, states_to_keep)#, merge_branches(peps, 1.0))

        #@test sol.energies[1:num_states] ≈ expected_energies

        println(sol.energies)
        println(sol.states)
        println(sol.probabilities)
    end
end