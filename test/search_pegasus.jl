using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Memoize

function bench(instance::String)
    m = 2
    n = 2
    t = 24

    L = n * m * t
    max_cl_states = 2^10

    β = 3.0
    bond_dim = 64
    δp = 1E-2
    num_states = 100

    @time ig = ising_graph(instance)

    @time fg = factor_graph(
        ig,
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
        for Layout ∈ (EnergyGauges, ), transform ∈ rotation.([0])
            println((Strategy, Sparsity, Layout, transform))

            @time network = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg, transform)
            @time ctr = MpsContractor{Strategy}(network, [β/8, β/4, β/2, β], params)

            @time sol = low_energy_spectrum(ctr, search_params, merge_branches(network))

            println(sol.energies[begin])
            clear_cache()
        end
    end
end

bench("$(@__DIR__)/instances/pegasus_droplets/2_2_3_00.txt")
