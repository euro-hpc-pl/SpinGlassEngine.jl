
@testset "Pegasus-like instance has the correct ground state energy" begin
    ground_energy = -23.301855

    m = 3
    n = 4
    t = 3
    L = n * m * t

    β = 2.
    bond_dim = 16
    num_states = 22

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)

    for Strategy ∈ (SVDTruncate, MPSAnnealing), Sparsity ∈ (Sparse, Dense)
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            for transform ∈ all_lattice_transformations
                net = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy}(net, [β/2, β], params)
                sol = low_energy_spectrum(ctr, search_params)

                ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
                @test sol.energies[begin] ≈ ground_energy
                @test sol.energies ≈ energy.(Ref(ig), ig_states)
                clear_memoize_cache()
            end
        end
    end
end
