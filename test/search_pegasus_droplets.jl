using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

using SpinGlassExhaustive

function bench(instance::String)
    m, n, t = 3, 3, 3

    # max_cl_states = 2^(t-0)

    ground_energy = -469.0

    β = 0.5
    bond_dim = 8
    dE = 3.0
    δp = exp(-β * dE)
    num_states = 500
    all_betas = [β/8, β/4, β/2, β]

    cl_h = clustered_hamiltonian(
        ising_graph(instance),
        # max_cl_states,
        spectrum=my_brute_force,
        cluster_assignment_rule=pegasus_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
    search_params = SearchParameters(num_states, δp)

    energies = Vector{Float64}[]
    for Strategy ∈ (Zipper, ), Sparsity ∈ (Sparse, )
        for Gauge ∈ (NoUpdate, )
            for Layout ∈ (GaugesEnergy,), transform ∈ all_lattice_transformations[[1]]
                net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparsity}(m, n, cl_h, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, all_betas, :graduate_truncate, params; onGPU=onGPU)
                # sol1, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit, NoDroplets()))
                sol1, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit, SingleLayerDroplets(0.01, 20, :hamming)))

                sol2 = unpack_droplets(sol1, β)
                ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol2.states)
                # cl_h_states = decode_state.(Ref(net), sol2.states)

                # @test sol1.energies[begin] ≈ ground_energy
                # @test sol2.energies[begin] ≈ ground_energy
                # push!(energies, sol1.energies)

                # for sol ∈ (sol1, sol2)
                #     ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol.states)
                #     @test sol.energies ≈ energy.(Ref(ising_graph(instance)), ig_states)

                #     cl_h_states = decode_state.(Ref(net), sol.states)
                #     @test sol.energies ≈ energy.(Ref(cl_h), cl_h_states)

                #     norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                #     @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))
                # end
                clear_memoize_cache()
            end
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/pegasus_random/P4/CBFM-P/SpinGlass/001_sg.txt")
