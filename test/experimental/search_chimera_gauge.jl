@testset "Smallest chimera pathological instance has the correct spectrum for all transformations" begin

    m, n, t = 8, 8, 8
    L = n * m * t

    β = 2.0
    BOND_DIM = 32
    MAX_STATES = 500
    MAX_SWEEPS = 10
    VAR_TOL = 1E-8
    TOL_SVD = 1E-16
    DE = 16.0
    δp = 1E-5 * exp(-β * DE)
    instance = "$(@__DIR__)/instances/chimera_droplets/512power/001.txt"
    INDβ = [1, 2, 3]
    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )

    params = MpsParameters{Float64}(BOND_DIM, VAR_TOL, MAX_SWEEPS, TOL_SVD)
    search_params = SearchParameters(MAX_STATES, δp)
    Gauge = GaugeStrategy

    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, Zipper), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (GaugesEnergy,)
            for transform ∈ all_lattice_transformations
                net = PEPSNetwork{SquareSingleNode{Layout},Sparsity,Float64}(
                    m,
                    n,
                    cl_h,
                    transform,
                )
                ctr = MpsContractor{Strategy,Gauge,Float64}(
                    net,
                    params;
                    onGPU = onGPU,
                    βs = [β / 6, β / 3, β / 2, β],
                    graduate_truncation = :graduate_truncate,
                )
                update_gauges!(ctr, m, INDβ, Val(:up))
                sol, s = low_energy_spectrum(ctr, search_params)
                #sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

                ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)
                cl_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(cl_h), cl_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                # println( maximum(abs.(norm_prob ./ exp.(-β .* (sol.energies .- sol.energies[1]))) .- 1 ))
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))   # test up to 1e-5
                push!(energies, sol.energies[1:Int(ceil(MAX_STATES / 4))])
                clear_memoize_cache()
            end
        end
    end
    println(energies)
    @test all(e -> e ≈ first(energies), energies)
end
