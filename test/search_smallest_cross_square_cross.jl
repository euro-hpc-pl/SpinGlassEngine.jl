
@testset "Pegasus-like (smallest cross-square-star) instance has the correct solution" begin
    m, n, t = 2, 3, 1
    L = n * m * t

    β = 1.0
    bond_dim = 16
    num_states = 22

    instance = "$(@__DIR__)/instances/pathological/cross_3_2.txt"

    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)

    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            for transform ∈ all_lattice_transformations
                net = PEPSNetwork{SquareCrossSingleNode{Layout},Sparsity,Float64}(
                    m,
                    n,
                    cl_h,
                    transform,
                )
                ctr = MpsContractor{Strategy,NoUpdate,Float64}(
                    net,
                    [β / 8.0, β / 4.0, β / 2.0, β],
                    :graduate_truncate,
                    params;
                    onGPU = onGPU,
                )
                sol, s = low_energy_spectrum(ctr, search_params)

                ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                cl_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(cl_h), cl_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                clear_memoize_cache()
            end
        end
    end
end
