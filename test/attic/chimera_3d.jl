using SpinGlassEngine

# function brute_force_gpu(ig::IsingGraph; num_states::Int)
#     brute_force(ig, :GPU, num_states=num_states)
# end


function run_test(instance, m, n, t, tran)
    #β = 1
    BETAS = collect(0.1:0.1:2)

    bond_dim = 64
    δp = 1e-10
    num_states = 512
    for β in BETAS
        ig = ising_graph(instance)

        cl_h = clustered_hamiltonian(
            ig,
            spectrum=full_spectrum, #_gpu, # rm _gpu to use CPU
            #cluster_assignment_rule=pegasus_lattice((m, n, t))
            cluster_assignment_rule=super_square_lattice((m, n, t))
        )

        params = MpsParameters(bond_dim, 1E-8, 10)
        search_params = SearchParameters(num_states, δp)

        # Solve using PEPS search
        energies = Vector{Float64}[]
        Strategy = MPSAnnealing # SVDTruncate
        Sparsity = Dense
        # tran = rotation(0)
        Layout = EnergyGauges
        Gauge = NoUpdate

        #net = PEPSNetwork{SquareDoubleNode{Layout}, Sparsity}(m, n, cl_h, tran)
        net = PEPSNetwork{SquareSingleNode{Layout}, Sparsity}(m, n, cl_h, tran)
    
        ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params; onGPU=onGPU)

        sol, s = low_energy_spectrum(ctr, search_params , merge_branches(ctr))

        ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol.states)
        @test sol.energies ≈ energy.(Ref(ig), ig_states)
        cl_h_states = decode_state.(Ref(net), sol.states)
        @test sol.energies ≈ energy.(Ref(cl_h), cl_h_states)

        norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
        exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))

        lZ = -β * (sol.energies[1]) - sol.probabilities[1]
        push!(energies, sol.energies)
        println("beta ", β)
        println("eng ",sol.energies[1])
        println("Z ",lZ)
        println("--------")
        clear_memoize_cache()
    end

    clear_memoize_cache()
end

instance = "$(@__DIR__)/../instances/chimera_3d/6x6x6.txt"
m, n, t = 6, 6, 6
for tran ∈ [rotation(0),] #all_lattice_transformations
    run_test(instance, m, n, t, tran)
end
