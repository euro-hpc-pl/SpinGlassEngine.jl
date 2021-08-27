using Memoize
@testset "Chimera 2048 instance has the correct low energy spectrum" begin
    m = 8 
    n = 8
    t = 8

    β = 1.

    L = n * m * t
    num_states = 5

    #instance = "$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"
    instance = "$(@__DIR__)/instances/chimera_droplets/512power/001.txt"
    #instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    #for transform ∈ all_lattice_transformations
    for transform ∈ rotation.([0])
        peps = PEPSNetwork(m, n, fg, transform, β=β, bond_dim=32)

        # @time x = mps(peps, 1)

        #update_gauges!(peps, :rand)
        @time sol = low_energy_spectrum(peps, num_states)#, merge_branches(peps, 1.0))
        #println(sol.energies[1:1])
    end
end