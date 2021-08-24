@testset "Chimera 2048 instance has the correct low energy spectrum" begin
    m = 16 
    n = 16
    t = 8

    β = 1.

    L = n * m * t
    num_states = 1

    instance = "$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    for transform ∈ all_lattice_transformations
        peps = PEPSNetwork(m, n, fg, transform, β=β)
        update_gauges!(peps, :rand)
        sol = low_energy_spectrum(peps, num_states)
    end
end