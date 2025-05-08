using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using SpinGlassExhaustive

function bench()
    m, n, t = 3, 2, 3

    β = 0.5
    bond_dim = 8
    dE = 3.0
    δp = exp(-β * dE)
    num_states = 1024
    cs = 2^20
    iter = 2
    betahalf = true

    inst_filename = "$(m)x$(n).txt"
    RESULTS_FOLDER = "$(@__DIR__)/../test/instances/pegasus_random"
    instance_path = "$RESULTS_FOLDER/$inst_filename"

    potts_h = potts_hamiltonian(
        ising_graph(instance_path),
        spectrum = full_spectrum,
        cluster_assignment_rule = pegasus_lattice((m, n, t)),
    )
    potts_h = truncate_potts_hamiltonian(potts_h, 0.5, cs, RESULTS_FOLDER, inst_filename; tol=1e-6, iter=iter)

    params = MpsParameters{Float64}(;
        bond_dim = bond_dim,
        var_tol = 1E-8,
        num_sweeps = 4,
        tol_SVD = 1E-16,
    )
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = δp)

    energies = Float64[]
    for Strategy ∈ (Zipper,), Sparsity ∈ (Sparse,)
        for Gauge ∈ (GaugeStrategy,)
            for Layout ∈ (GaugesEnergy,), transform ∈ all_lattice_transformations[[1]]
                net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity,Float64}(m, n, potts_h, transform)
                ctr = MpsContractor{Strategy,Gauge,Float64}(
                    net, params;
                    onGPU = true,
                    beta = β,
                    graduate_truncation = true,
                )

                overlaps_path = "$(@__DIR__)/overlaps$(m)x$(n).txt"
                update_gauges!(ctr, m, Val(:up), overlaps_path, betahalf)

                sol, s = low_energy_spectrum(
                    ctr,
                    search_params,
                    merge_branches(ctr; merge_prob = :none),
                )

                ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                @test sol.energies ≈ SpinGlassNetworks.energy.(Ref(ising_graph(instance_path)), ig_states)

                potts_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ SpinGlassNetworks.energy.(Ref(potts_h), potts_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                push!(energies, sol.energies[1])
                clear_memoize_cache()
            end
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench()
