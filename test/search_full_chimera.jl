using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench(instance::String)
    m = 16
    n = 16
    t = 8

    L = n * m * t
    max_cl_states = 2^(t-0)

    ground_energy = -3336.773383

    β = 3.0
    bond_dim = 32
    δp = 1E-3
    num_states = 1000

    @time fg = factor_graph(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense,)
        for Layout ∈ (EnergyGauges, ), transform ∈ rotation.([0])
            net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
            ctr = MpsContractor{Strategy}(net, [β/8, β/4, β/2, β], params)
            sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
            @test sol.energies[begin] ≈ ground_energy
            push!(energies, sol.energies)
            clear_memoize_cache()
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/chimera_droplets/2048power/001.txt")
