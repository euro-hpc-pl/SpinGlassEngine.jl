@testset "Pathological Pegasus instance works" begin

    num_states = length(expected_energies)

    m = 2
    n = 2
    t = 24

    β = 1.

    L = n * m * t
    states_to_keep = 20

    control_params = Dict(
         "bond_dim" => typemax(Int),
         "var_tol" => 1E-8,
         "sweeps" => 4.
    )

    instance = "$(@__DIR__)/instances/pegasus_droplets/2_2_3_00.txt"


    ig = ising_graph(instance)

    
    #fg = factor_graph(
    #    ig,
    #    spectrum=full_spectrum,
    #    cluster_assignment_rule=super_square_lattice((m, n, t)) 
    #)

    #for transform ∈ rotation.([0])
    #    peps = FusedNetwork(m, n, fg, transform, β=β)
    #    sol = low_energy_spectrum(peps, states_to_keep)#, merge_branches(peps, 1.0))

    #    println(sol.energies)
    #    println(sol.states)
    #    println(sol.probabilities)
    #end
end