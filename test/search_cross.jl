@testset "Pegasus-like instance has the correct ground state energy" begin

    ground_energy = -23.301855

    m = 3
    n = 4
    t = 3
    
    β = 1.

    schedule = 1.

    L = n * m * t
    states_to_keep = 20

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t)) 
    )

    for transform ∈ all_lattice_transformations
        peps = FusedNetwork(m, n, fg, transform, β=β)
        update_gauges!(peps, :rand)
        sol = low_energy_spectrum(peps, states_to_keep)#, merge_branches(peps, 1.0))
        #println(sol.energies)
        @test first(sol.energies) ≈ ground_energy
    end
end