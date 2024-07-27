
@testset "Pegasus-like (cross-square-star) instance has the correct ground state energy" begin
    m, n, t = 2, 4, 3
    L = n * m * t

    β = 3.0
    bond_dim = 16
    num_states = 128

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_mdd.txt"

    ig = ising_graph(instance)
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(; bd = bond_dim, ϵ = 1E-8, sw = 4)
    search_params = SearchParameters(; max_states = num_states, cut_off_prob = 0.0)
    Gauge = NoUpdate

    energies = Vector{Float64}[]
    for Strategy ∈ (MPSAnnealing, Zipper, SVDTruncate), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (GaugesEnergy, EngGaugesEng, EnergyGauges)  #
            for transform ∈ all_lattice_transformations, Lattice ∈ (KingSingleNode,)
                net = PEPSNetwork{Lattice{Layout},Sparsity,Float64}(m, n, potts_h, transform)
                ctr = MpsContractor{Strategy,Gauge,Float64}(
                    net,
                    params;
                    onGPU = onGPU,
                    βs = [β / 2, β],
                    graduate_truncation = :graduate_truncate,
                )
                sol, s = low_energy_spectrum(ctr, search_params)

                ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                potts_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                push!(energies, sol.energies)
                clear_memoize_cache()
            end
        end
    end
end
