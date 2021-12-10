@testset "Chimera-like instance has the correct low energy spectrum" begin
    m = 3
    n = 4
    t = 3
    L = n * m * t

    β = 1.

    bond_dim = 16
    num_states = 22

    # energies
    exact_energies =
    [
        -16.4, -16.4, -16.4, -16.4, -16.1, -16.1, -16.1, -16.1, -15.9,
        -15.9, -15.9, -15.9, -15.9, -15.9, -15.6, -15.6, -15.6, -15.6,
        -15.6, -15.6, -15.4, -15.4
    ]

    # degenerate fg solutions
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

    instance = "$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)

    for Strategy ∈ (SVDTruncate, MPSAnnealing), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            for transform ∈ all_lattice_transformations

                network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                contractor = MpsContractor{Strategy}(network, [β/8., β/4., β/2., β], params)
                sol = low_energy_spectrum(contractor, search_params)

                @test sol.energies ≈ exact_energies
                for (i, σ) ∈ enumerate(sol.states) @test σ ∈ exact_states[deg[i]] end
                clear_cache()
            end
        end
    end
end
