@testset "Pegasus-like instance has the correct ground state energy" begin

    ground_energy = -23.301855

    m = 3
    n = 4
    t = 3
    L = n * m * t

    β = 1.
    num_states = 22

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters()
    search_params = SearchParameters(num_states, 0.0)

    for Strategy ∈ (SVDTruncate,), Sparsity ∈ (Dense,)
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            for transform ∈ all_lattice_transformations

                network = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg, transform)
                contractor = MpsContractor{Strategy}(network, [β], params)
                sol = low_energy_spectrum(contractor, search_params)

                @test first(sol.energies) ≈ ground_energy
            end
        end
    end
end
