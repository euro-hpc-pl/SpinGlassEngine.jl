# using SpinGlassExhaustive

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#      brute_force(ig, :GPU, num_states=num_states)
# end

function bench(instance::String)
    m = 4
    n = 4
    t = 24

    max_cl_states = 2^8

    β = 2.5
    bond_dim = 16
    δp = 1E-4
    num_states = 1000

    @time ig = ising_graph(instance)
    @time fg = factor_graph(
        ig,
        max_cl_states,
        spectrum=brute_force, #_gpu, # rm _gpu to use CPU
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Float64[]
    for Strategy ∈ (MPSAnnealing, ), Sparsity ∈ (Dense, )
        for Layout ∈ (GaugesEnergy, EnergyGauges, EngGaugesEng), transform ∈ all_lattice_transformations
            println((Strategy, Sparsity, Layout, transform))

            @time network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
            @time ctr = MpsContractor{Strategy}(network, [β/8, β/4, β/2, β], params)

            @time sol_peps = low_energy_spectrum(ctr, search_params, merge_branches(network))

            push!(energies, sol_peps.energies[begin])
            clear_memoize_cache()
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/pegasus_nondiag/4x4.txt")
