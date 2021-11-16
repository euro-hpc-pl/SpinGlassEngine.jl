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

    params = MpsParameters(bond_dim, 1E-8, 4)

    for Strategy ∈ (Basic,)
        for Sparsity ∈ (Sparse, Dense)
            for Layout ∈ (EnergyGauges, )
                for transform ∈ rotation.([0])
                    println((Strategy, Sparsity, Layout, transform))

                    network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                    ctr = MpsContractor{Strategy}(network, [β], params)

                    @time sol = low_energy_spectrum(ctr, num_states, merge_branches(network))

                    println(sol.energies[1:1])
                end
            end
        end
    end
end

bench()
