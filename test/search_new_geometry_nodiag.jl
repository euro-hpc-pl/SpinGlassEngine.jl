# using SpinGlassExhaustive
using TensorOperations

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end

function bench(instance::String)
    m = 3
    n = 2
    t = 1
    L = n * m * t * 2 * 4

    max_cl_states = 2^8

    β = 2.0
    bond_dim = 16
    δp = 1E-4
    num_states = 1000

    @time ig = ising_graph(instance)
    @time fg = factor_graph(
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
    energies = Float64[]
    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
        for transform ∈ rotation.([180])
            println((Strategy, Sparsity, transform))
          
            @time network = PEPSNetwork{Pegasus, Sparsity}(m, n, fg, transform)
            
            network2 = PEPSNetwork{Square{EnergyGauges}, Sparsity}(m, n, fg2, transform)

            # println(network.ncols)
            # println(network.nrows)

            #

            for ii in 1:3, jj in 1:2
            
                to = tensor(network2, PEPSNode(ii, jj), β)
                tl = tensor(network2, PEPSNode(ii, jj - 1//2), β)
                tu = tensor(network2, PEPSNode(ii - 1//2, jj), β)
                @tensor too[k, l, m, n] := tl[k, x] * tu[l, y] * to[x, y, m, n]
                
                tn = tensor(network, PEPSNode(ii, jj), β)
                # # println(aa)
                
                println("---------------")
                println(tn ./ too)
                
            end
            # println(tn ./ too)
            # println("---------------")
            
            #println(get_prop(network2.factor_graph, (2, 2), :spectrum).states)
            @time ctr = MpsContractor{Strategy}(network, [β/8, β/4, β/2, β], params)
            @time sol_peps = low_energy_spectrum(ctr, search_params, merge_branches(network))
            # push!(energies, sol_peps.energies[begin])
            #clear_cache()
        end
    end
    #@test all(e -> e ≈ first(energies), energies)
    #println(energies)
end

bench("$(@__DIR__)/instances/pegasus_nondiag/3x2x1.txt")
