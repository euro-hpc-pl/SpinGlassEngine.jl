using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench(instance::String)
    m, n, t = 16, 16, 8

    max_cl_states = 2^(t-0)

    graduate_truncation = true
    ground_energy = -3336.773383

    β = 3.0
    bond_dim = 32
    dE = 3.0
    δp = exp(-β * dE)
    num_states = 500
    all_betas = [β/8, β/4, β/2, β]

    fg = factor_graph(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 10, 1E-16)
    search_params = SearchParameters(num_states, δp)

    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
        for Gauge ∈ (NoUpdate, GaugeStrategy, GaugeStrategyWithBalancing)
            for Layout ∈ (GaugesEnergy,), transform ∈ all_lattice_transformations
                net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncation, params)
                indβ = [3,]

                update_gauges!(ctr, m, indβ, Val(:up))
                @allocated sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit))
                #println("statistics ", maximum(values(ctr.statistics)))
                println("prob ", sol.probabilities[begin])
                println("largest discarded prob ", sol.largest_discarded_probability)
                #println("states ", sol.states)
                #println("degeneracy ", sol.degeneracy)
                println("energy ", sol.energies[begin])
                #@test sol.energies[begin] ≈ ground_energy
                push!(energies, sol.energies)
                clear_memoize_cache()
            end
        end
    end
    @assert all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/chimera_droplets/2048power/001.txt")
