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

    β = 2.0
    bond_dim = 16
    δp = 1e-4
    num_states = 1024

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
    energies = Float64[]
    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
        for transform ∈ rotation.([180])
            println((Strategy, Sparsity, transform))

            net = PEPSNetwork{Pegasus, Sparsity}(m, n, fg, transform)
            net2 = PEPSNetwork{Square{EnergyGauges}, Sparsity}(m, n, fg2, transform)
            #net = PEPSNetwork{SquareStar{EnergyGauges}, Sparsity}(m, n, fg2, transform)

            ctr2 = MpsContractor{Strategy}(net2, [β/8, β/4, β/2, β], params)
            sol_peps2 = low_energy_spectrum(ctr2, search_params) #merge_branches(network2))

            ig_states2 = decode_factor_graph_state.(Ref(fg2), sol_peps2.states)
            @test sol_peps2.energies ≈ energy.(Ref(ig), ig_states2)
            clear_memoize_cache()

            println("---------- switching to new geometry -------------- ")
            ctr = MpsContractor{Strategy}(net, [β/8, β/4, β/2, β], params)
            sol_peps = low_energy_spectrum(ctr, search_params) #, merge_branches(network))

            ig_states = decode_factor_graph_state.(Ref(fg), sol_peps.states)
            fg_states = decode_state.(Ref(net), sol_peps.states)
            @test energy.(Ref(fg), fg_states) ≈ energy.(Ref(ig), ig_states)
            #@test sort(energy.(Ref(fg), fg_states))[1:100] ≈ sol_peps2.energies[1:100]
            @test sort(energy.(Ref(fg), fg_states))[1:1] ≈ sort(sol_peps.energies)[1:1]

            #norm_prob = exp.(sol_peps.probabilities .- sol_peps.probabilities[1])
            #@test norm_prob ≈ exp.(-β .* (sol_peps.energies .- sol_peps.energies[1]))

            push!(energies, sol_peps.energies[begin])
            clear_memoize_cache()

        end
    end

    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/pegasus_nondiag/2x2x1.txt")
