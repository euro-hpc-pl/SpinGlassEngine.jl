@testset "Pathological instance" begin
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t
    num_states = 20

    control_params = Dict(
         "bond_dim" => typemax(Int),
         "var_tol" => 1E-8,
         "sweeps" => 4.
    )

    instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_dd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t)) 
    )

    for transform ∈ rotation.([0])
        peps = FusedNetwork(m, n, fg, transform, β=β)

        sol = low_energy_spectrum(peps, num_states)
        println(sol.energies)
        println(sol.states)
        println(sol.probabilities)
    end
end