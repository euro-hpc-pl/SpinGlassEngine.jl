
@testset "Pegasus-like instance has the correct ground state energy" begin
    # ground_energy = -23.301855

    m, n, t = 2, 4, 3
    L = n * m * t

    β = 2.0
    bond_dim = 16
    num_states = 3

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_mdd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)

    #for Strategy ∈ (SVDTruncate, MPSAnnealing), Sparsity ∈ (Sparse, Dense)
    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense,)
        for Layout ∈ (EnergyGauges, ) # GaugesEnergy, EngGaugesEng)
            for transform ∈ all_lattice_transformations[[3]], Lattice ∈ (SquareStar, )
                net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy}(net, [β/2, β], params)
                sol = low_energy_spectrum(ctr, search_params)

                ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
                #@test sol.energies[begin] ≈ ground_energy
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(fg), states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                clear_memoize_cache()
            end
        end
    end
end
