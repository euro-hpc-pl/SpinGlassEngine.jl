
@testset "Pegasus-like (smallest cross-square-star) instance has the correct solution" begin
    m, n, t = 2, 3, 1
    L = n * m * t

    β = 1.0
    bond_dim = 16
    num_states = 22

    instance = "$(@__DIR__)/instances/pathological/cross_3_2.txt"

    ig = ising_graph(instance)
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(; bd = bond_dim, ϵ = 1E-8, sw = 4)
    search_params = SearchParameters(; max_states = num_states, cut_off_prob = 0.0)

    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            for transform ∈ all_lattice_transformations
                net = PEPSNetwork{KingSingleNode{Layout},Sparsity,Float64}(
                    m,
                    n,
                    potts_h,
                    transform,
                )
                ctr = MpsContractor{Strategy,NoUpdate,Float64}(
                    net,
                    params;
                    onGPU = onGPU,
                    βs = [β / 8.0, β / 4.0, β / 2.0, β],
                    graduate_truncation = :graduate_truncate,
                )
                sol, s = low_energy_spectrum(ctr, search_params)

                ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                potts_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                clear_memoize_cache()
            end
        end
    end
end
