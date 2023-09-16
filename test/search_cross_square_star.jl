
@testset "Pegasus-like (cross-square-star) instance has the correct ground state energy" begin
    m, n, t = 2, 4, 3
    L = n * m * t

    β = 3.0
    bond_dim = 16
    num_states = 128

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_mdd.txt"

    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)
    Gauge = NoUpdate

    energies = Vector{Float64}[]
    for Strategy ∈ (MPSAnnealing, Zipper, SVDTruncate), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (GaugesEnergy, EngGaugesEng, EnergyGauges,)  #
            for transform ∈ all_lattice_transformations, Lattice ∈ (SquareStar, )
                net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, cl_h, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/2, β], :graduate_truncate, params; onGPU=onGPU)
                sol, s = low_energy_spectrum(ctr, search_params)

                ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                cl_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(cl_h), cl_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                push!(energies, sol.energies)
                clear_memoize_cache()
            end
        end
    end
end
