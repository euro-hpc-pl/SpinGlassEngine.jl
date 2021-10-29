using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench()
    m = 16 
    n = 16
    t = 8

    β = 3.
    bond_dim = 32

    L = n * m * t
    num_states = 100

    instance = "$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    for Layout ∈ (EnergyGauges, )
        #for transform ∈ all_lattice_transformations
        for transform ∈ rotation.([0])
            peps = PEPSNetwork{Square{Layout}}(m, n, fg, transform, β, bond_dim)
            update_gauges!(peps, :rand)
            @time sol = low_energy_spectrum(peps, num_states, merge_branches(peps))
            println(sol.energies[1:1])
        end
    end
end

bench()
