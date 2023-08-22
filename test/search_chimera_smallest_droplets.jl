@testset "Smallest chimera pathological instance has the correct spectrum for all transformations" begin
    exact_energies = [-2.6, -1.1, -0.6, -0.4, -0.4, 1.1, 1.9, 2.1]

    m, n, t = 3, 1, 1
    L = n * m * t

    β = 1.0
    bond_dim = 16
    num_states = 2^8
    instance = "$(@__DIR__)/instances/pathological/chim_$(n)_$(m)_$(t).txt"

    ig = ising_graph(instance)
    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)
    Gauge = NoUpdate

    energies = Vector{Float64}[]
    for Strategy ∈ (Zipper,), Sparsity ∈ (Dense,)
        for Layout ∈ (EnergyGauges,)
            for transform ∈ all_lattice_transformations[[1]]
                net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params; onGPU=onGPU)

                sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit, SingleLayerDropletsHamming(2.2, 10)))

                @test sol.energies ≈ exact_energies[[1]]

                sol2 = unpack_droplets_hamming(sol, β)
                println(sol.droplets[1])
                @test length(sol.droplets[1]) == 4
                @test sol2.energies ≈ exact_energies[1:5]

                ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)
                fg_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(fg), fg_states)
                ig_states2 = decode_factor_graph_state.(Ref(fg), sol2.states)
                @test sol2.energies ≈ energy.(Ref(ig), ig_states2)
                fg_states2 = decode_state.(Ref(net), sol2.states)
                @test sol2.energies ≈ energy.(Ref(fg), fg_states2)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))
                norm_prob2 = exp.(sol2.probabilities .- sol2.probabilities[1])
                @test norm_prob2 ≈ exp.(-β .* (sol2.energies .- sol2.energies[1]))

                # push!(energies, sol.energies)
                clear_memoize_cache()
            end
        end
    end
end
