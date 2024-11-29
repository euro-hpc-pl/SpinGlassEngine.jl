EXACT_ENERGIES = [
    -14.355045, -14.355045, -14.355045, -14.355045, -14.355045, -14.355045, -14.355045,
    -14.355045, -14.167247, -14.167247, -14.167247, -14.167247, -14.167247, -14.167247,
    -14.167247, -14.167247, -13.921539, -13.921539, -13.921539, -13.921539, -13.921539,
    -13.921539, -13.921539, -13.921539, -13.733741, -13.733741, -13.733741, -13.733741,
    -13.733741, -13.733741, -13.733741, -13.733741, -13.503905, -13.503905, -13.503905,
    -13.503905, -13.503905, -13.503905, -13.503905, -13.503905, -13.413319, -13.413319,
    -13.413319, -13.413319, -13.413319, -13.413319, -13.413319, -13.413319, -13.369439,
    -13.369439, -13.369439, -13.369439, -13.369439, -13.369439, -13.369439, -13.369439,
    -13.343137, -13.343137, -13.343137, -13.343137, -13.343137, -13.343137, -13.343137,
    -13.343137, -13.324895, -13.324895, -13.324895, -13.324895, -13.324895, -13.324895,
    -13.324895, -13.324895, -13.225521, -13.225521, -13.225521, -13.225521, -13.225521,
    -13.225521, -13.225521, -13.225521, -13.168059, -13.168059, -13.168059, -13.168059,
    -13.168059, -13.168059, -13.168059, -13.168059, -13.155339, -13.155339, -13.155339,
    -13.155339, -13.155339, -13.155339, -13.155339, -13.155339, -13.137097, -13.137097,
    -13.137097, -13.137097,
]

@testset "Pegasus-like (cross-square-star) instance has the correct ground state energy" begin
    m, n, t = 2, 4, 3
    L = n * m * t

    β = 3.0
    bond_dim = 16
    num_states = 128 #length(EXACT_ENERGIES)

    instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_mdd.txt"

    ig = ising_graph(instance)
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = 0.0)
    Gauge = NoUpdate

    energies = Vector{Float64}[]
    for Strategy ∈ (Zipper, ), Sparsity ∈ (Sparse, )
    #for Strategy ∈ (Zipper, SVDTruncate), Sparsity ∈  (Dense, Sparse)
        #for Layout ∈ (GaugesEnergy, EngGaugesEng, EnergyGauges)  #
        for Layout ∈ (GaugesEnergy, )
            #for transform ∈ all_lattice_transformations, Lattice ∈ (KingSingleNode,)
            for transform ∈ all_lattice_transformations, Lattice ∈ (KingSingleNode, )
                net =
                    PEPSNetwork{Lattice{Layout},Sparsity,Float64}(m, n, potts_h, transform)
                ctr = MpsContractor{Strategy,Gauge,Float64}(
                    net,
                    params;
                    onGPU = onGPU,
                    beta = β,
                    graduate_truncation = true,
                )
                sol, s = low_energy_spectrum(ctr, search_params)

                k = 3
                @test EXACT_ENERGIES[1:k] ≈ sol.energies[1:k]

#=
                ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                potts_h_states = decode_state.(Ref(net), sol.states)
                #@test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                #@test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                #push!(energies, sol.energies[1:1])
=#
                clear_memoize_cache()
            end
        end
    end
end
