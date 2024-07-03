using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

@testset "Pegasus instance has the correct spectrum for all transformations" begin

    m, n, t = 1, 1, 3

    β = 1.0
    BOND_DIM = 2
    MAX_STATES = 500
    MAX_SWEEPS = 2
    VAR_TOL = 1E-16
    TOL_SVD = 1E-16
    DE = 16.0
    δp = 1E-5 * exp(-β * DE)
    ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/minimal.txt")
    INDβ = [1, 2, 3]
    cl_h = clustered_hamiltonian(
        ig,
        spectrum = my_brute_force,
        cluster_assignment_rule = pegasus_lattice((m, n, t)),
    )

    params = MpsParameters(BOND_DIM, VAR_TOL, MAX_SWEEPS, TOL_SVD)
    search_params = SearchParameters(MAX_STATES, δp)
    Gauge = GaugeStrategy
    Strategy = Zipper
    Sparsity = Sparse
    Layout = GaugesEnergy

    energies = Vector{Float64}[]

    for transform ∈ all_lattice_transformations
        net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity}(m, n, cl_h, transform)
        ctr = MpsContractor{Strategy,Gauge}(
            net,
            [β / 6, β / 3, β / 2, β],
            :graduate_truncate,
            params;
            onGPU = onGPU,
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
    #println(energies)
    @test all(e -> e ≈ first(energies), energies)
end
