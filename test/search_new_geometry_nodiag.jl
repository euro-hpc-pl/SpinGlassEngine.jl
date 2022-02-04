# using SpinGlassExhaustive
using TensorOperations

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end

function bench(instance::String)
    m = 2
    n = 2
    t = 1
    L = n * m * t * 2 * 4

    max_cl_states = 2^2

    β = 1.0
    bond_dim = 16
    δp = 1e-4
    num_states = 100

    ig = ising_graph(instance)
    fg = factor_graph(
        ig,
        # max_cl_states,
        spectrum=full_spectrum, #_gpu, # rm _gpu to use CPU
        cluster_assignment_rule=pegasus_lattice((m, n, t))
    )

    fg2 = factor_graph(
        ig,
        # max_cl_states,
        spectrum=full_spectrum, #_gpu, # rm _gpu to use CPU
        cluster_assignment_rule=super_square_lattice((m, n, 8))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, MPSAnnealing), Sparsity ∈ (Dense, )
        for tran ∈ rotation.([180]), Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            net = PEPSNetwork{Pegasus, Sparsity}(m, n, fg, tran)
            net2 = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg2, tran)

            ctr = MpsContractor{Strategy}(net, [β/8, β/4, β/2, β], params)
            ctr2 = MpsContractor{Strategy}(net2, [β/8, β/4, β/2, β], params)

            sol = low_energy_spectrum(ctr, search_params)#, merge_branches(net))
            sol2 = low_energy_spectrum(ctr2, search_params)#, merge_branches(net2))

            @test sol.energies ≈ sol2.energies
            #@test sol.states == sol2.states

            push!(energies, sol.energies)
            clear_memoize_cache()
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/pegasus_nondiag/2x2x1.txt")
