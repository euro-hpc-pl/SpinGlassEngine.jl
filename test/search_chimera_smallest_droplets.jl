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
            for transform ∈ all_lattice_transformations
                net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params; onGPU=onGPU)

                sol1, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit, SingleLayerDroplets(2.2, 1, :hamming)))

                @test sol1.energies ≈ exact_energies[[1]]
                sol2 = unpack_droplets(sol1, β)
                @test sol2.energies ≈ exact_energies[1:5]

                for sol ∈ (sol1, sol2)
                    ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
                    @test sol.energies ≈ energy.(Ref(ig), ig_states)

                    fg_states = decode_state.(Ref(net), sol.states)
                    @test sol.energies ≈ energy.(Ref(fg), fg_states)

                    norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                    @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))
                end
                clear_memoize_cache()
            end
        end
    end
end
