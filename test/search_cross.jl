
function _energy(ig::IsingGraph, fg, fg_state::Vector{Int})
    ig_state = decode_factor_graph_state(fg, fg_state)
    en = 0.0
    for (i, σ) ∈ ig_state
        en += get_prop(ig, i, :h) * σ
        for (j, η) ∈ ig_state
            if has_edge(ig, i, j)
                J = get_prop(ig, i, j, :J)
            elseif has_edge(ig, j, i)
                J = get_prop(ig, j, i, :J)
            else
                J = 0.0
            end
            en += σ * J * η / 2.0
        end
    end
    en
end

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
    J = couplings(ig) + Diagonal(biases(ig))

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

                network = PEPSNetwork{SquareStar{Layout}, Sparsity}(m, n, fg, transform)
                contractor = MpsContractor{Strategy}(network, [β/2, β], params)
                sol = low_energy_spectrum(contractor, search_params)
                @test sol.energies[begin] ≈ ground_energy

                @test sol.energies ≈ _energy.(Ref(ig), Ref(fg), sol.states)
                clear_memoize_cache()
            end
        end
    end
end
