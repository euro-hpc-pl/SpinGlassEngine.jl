using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench()
    m = 16
    n = 16
    t = 8
    L = n * m * t

    ground_energy = -3336.773383

    β = 3.0
    bond_dim = 16
    δp = 1E-2
    num_states = 100

    fg = factor_graph(
        ising_graph("$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"),
        #spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    for Strategy ∈ (SVDTruncate, MPSAnnealing), Sparsity ∈ (Sparse, Dense)
        for Layout ∈ (EnergyGauges, ), transform ∈ rotation.([0])
            println((Strategy, Sparsity, Layout, transform))

            @time network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
            @time ctr = MpsContractor{Strategy}(network, [β/8, β/4, β/2, β], params)

            @time sol = low_energy_spectrum(ctr, search_params, merge_branches(network))

            @test sol.energies[begin] ≈ ground_energy
            println(sol.energies[begin])
        end
    end
end

bench()
