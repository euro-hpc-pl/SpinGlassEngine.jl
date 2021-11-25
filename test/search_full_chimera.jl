using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench()
    m = 16
    n = 16
    t = 8
    L = n * m * t

    β = 3.
    bond_dim = 32

    δp = 1E-2
    num_states = 1000

    instance = "$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, δp)

    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Sparse, Dense)
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), transform ∈ rotation.([0])
            println((Strategy, Sparsity, Layout, transform))

            network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
            ctr = MpsContractor{Strategy}(network, [β], params)

            @time sol = low_energy_spectrum(ctr, search_params, merge_branches(network))

            println(sol.energies[1:1])
        end
    end
end

bench()
