@testset "Chimera-like (pathological) instance has the correct energy spectrum for all heuristics" begin
    m, n, t = 3, 4, 3

    β = 1.0
    bond_dim = 16
    num_states = 22
    INDβ = [1, 2, 3]

    # energies
    exact_energies =
    [
        -16.4, -16.4, -16.4, -16.4, -16.1, -16.1, -16.1, -16.1, -15.9,
        -15.9, -15.9, -15.9, -15.9, -15.9, -15.6, -15.6, -15.6, -15.6,
        -15.6, -15.6, -15.4, -15.4
    ]

    # degenerate cl_h solutions
    exact_states =
    [   # E =-16.4
        [
        [1, 4, 5, 1, 2, 2, 1, 1, 1, 4, 2, 1], [1, 4, 7, 1, 2, 2, 1, 1, 1, 4, 2, 1],
        [1, 4, 5, 1, 2, 2, 1, 1, 1, 4, 6, 1], [1, 4, 7, 1, 2, 2, 1, 1, 1, 4, 6, 1]
        ],
        # E =-16.1
        [
        [2, 5, 4, 1, 1, 3, 1, 1, 1, 5, 7, 1], [2, 5, 2, 1, 1, 3, 1, 1, 1, 5, 3, 1],
        [2, 5, 4, 1, 1, 3, 1, 1, 1, 5, 3, 1], [2, 5, 2, 1, 1, 3, 1, 1, 1, 5, 7, 1]
        ],
        # E = -15.9
        [
        [1, 4, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1], [1, 4, 3, 1, 2, 2, 1, 1, 1, 4, 2, 1],
        [1, 4, 6, 1, 2, 2, 1, 1, 1, 4, 2, 1], [1, 4, 3, 1, 2, 2, 1, 1, 1, 4, 6, 1],
        [1, 4, 1, 1, 2, 2, 1, 1, 1, 4, 6, 1], [1, 4, 6, 1, 2, 2, 1, 1, 1, 4, 6, 1]
        ],
        # E = -15.6
        [
        [2, 5, 3, 1, 1, 3, 1, 1, 1, 5, 3, 1], [2, 5, 3, 1, 1, 3, 1, 1, 1, 5, 7, 1],
        [2, 5, 8, 1, 1, 3, 1, 1, 1, 5, 3, 1], [2, 5, 6, 1, 1, 3, 1, 1, 1, 5, 7, 1],
        [2, 5, 6, 1, 1, 3, 1, 1, 1, 5, 3, 1], [2, 5, 8, 1, 1, 3, 1, 1, 1, 5, 7, 1]
        ],
        # E = -15.4
        [
        [1, 4, 7, 1, 2, 2, 1, 1, 1, 2, 6, 1], [1, 4, 5, 1, 2, 2, 1, 1, 1, 2, 6, 1]
        ],
    ]

    deg = Dict(
        1 => 1, 2 => 1, 3 => 1, 4 => 1,
        #
        5 => 2, 6 => 2, 7 => 2, 8 => 2,
        #
        9 => 3, 10 => 3, 11 => 3, 12 => 3, 13 => 3, 14 => 3,
        #
        15 => 4, 16 => 4, 17 => 4, 18 => 4, 19 => 4, 20 => 4,
        #
        21 => 5, 22 => 5,
    )

    ig = ising_graph("$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt")
    cl_h = clustered_hamiltonian(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)
    Gauge = GaugeStrategy

    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (EngGaugesEng, GaugesEnergy, EnergyGauges)
            for Lattice ∈ (SquareSingleNode, ), transform ∈ all_lattice_transformations

                net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, cl_h, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/6., β/3., β/2., β], :graduate_truncate, params; onGPU=onGPU)
                update_gauges!(ctr, m, INDβ, Val(:up))

                sol, s = low_energy_spectrum(ctr, search_params)

                @test sol.energies ≈ exact_energies

                ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                cl_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(cl_h), cl_h_states)

                for (i, σ) ∈ enumerate(sol.states) @test σ ∈ exact_states[deg[i]] end

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                push!(energies, sol.energies)
                clear_memoize_cache()
            end
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end
