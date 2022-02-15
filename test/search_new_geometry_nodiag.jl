# using SpinGlassExhaustive
using TensorOperations

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end

function bench(instance::String)
    m = 2
    n = 2
    t = 1

    max_cl_states = 2^2

    β = 1
    bond_dim = 64
    δp = 1e-10
    num_states = 10

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
    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
        for tran ∈ rotation.([180]), Layout ∈ (GaugesEnergy, )
            net = PEPSNetwork{Pegasus, Sparsity}(m, n, fg, tran)
            net2 = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg2, tran)

            ctr = MpsContractor{Strategy}(net, [β/8, β/4, β/2, β], params)
            ctr2 = MpsContractor{Strategy}(net2, [β/8, β/4, β/2, β], params)

            sol = low_energy_spectrum(ctr, search_params)#, merge_branches(ctr))
            sol2 = low_energy_spectrum(ctr2, search_params)#, merge_branches(ctr2))

            ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
            @test sol.energies ≈ energy.(Ref(ig), ig_states)
            fg_states = decode_state.(Ref(net), sol.states)
            @test sol.energies ≈ energy.(Ref(fg), fg_states)

            #@test sol.energies ≈ sol2.energies
            #@test sol.states == sol2.states

            norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
            exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
            #for (a, b, c, d) in zip(norm_prob, sol.energies, sol.states, exct_prob)
            #    println(a ./ d, " ", b, " ", c)
            #end
            # @test norm_prob ≈ exct_prob
            push!(energies, sol.energies)

            ψ1 = mps(ctr, 2, 4)
            ψ1_top = mps_top(ctr, 1, 4)

            ψ2 = mps(ctr2, 2, 4)
            ψ2_top = mps_top(ctr2, 1, 4)
            println("overlap = ", ψ1 * ψ2)
            println("overlap = ", ψ1_top * ψ2_top)
            # println(size(ψ1[1]))

            clear_memoize_cache()
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/pegasus_nondiag/2x2x1.txt")
