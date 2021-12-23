
function to_state(fg_state::Dict{Int, Int})
    ig_state = zeros(Int, maximum(keys(fg_state)))
    for (i, σ) ∈ fg_state @inbounds ig_state[i] = σ end
    ig_state
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
                #@test sol.energies ≈ energy.(Ref(J), Ref(fg), sol.states)
                spins = decode_factor_graph_state(fg, sol.states[1])
                println(to_state(spins))
                clear_memoize_cache()
            end
        end
    end
end
