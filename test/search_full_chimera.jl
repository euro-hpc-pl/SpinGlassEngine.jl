using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench(instance::String)
    m = 8
    n = 8
    t = 8

    L = n * m * t
    max_cl_states = 2^(t-0)

    graduate_truncation = true
    #ground_energy = -3336.773383 # for chimera 2048
    #ground_energy = -1881.226667 # for chimera 1152
    #ground_energy = -846.960013 # for chimera 512

    β = 2.0
    bond_dim = 32
    dE = 3.0
    δp = exp(-β * dE)
    num_states = 500

    @time fg = factor_graph(
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
            for Layout ∈ (GaugesEnergy,), transform ∈ all_lattice_transformations #rotation.([0]) 
                net = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], graduate_truncation, params)
                #ctr = MpsContractor{Strategy}(net, [β/6, β/3, β/2, β], params)
                indβ = [3,]

                #=
                if Gauge!= NoUpdate
                    for j in indβ
                        for i ∈ 1:m-1
                            ψ_top = mps_top(ctr, i, j)
                            ψ_bot = mps(ctr, i+1, j)
                            overlap_old = ψ_top * ψ_bot
                            overlap_new = update_gauges!(ctr, i, j)
                            #println(overlap_old, "  ", overlap_new)
                        end
                        #println("------------------")
                    end
                end
                =#
                update_gauges!(ctr, m, indβ, Val(:right))
                @allocated sol = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit))
                #println("statistics ", maximum(values(ctr.statistics)))
                println("prob ", sol.probabilities[begin])
                println("largest discarded prob ", sol.largest_discarded_probability)
                #println("states ", sol.states)
                #println("degeneracy ", sol.degeneracy)
                println("energy ", sol.energies[begin])
                #@test sol.energies[begin] ≈ ground_energy
                push!(energies, sol.energies)
                #clear_memoize_cache()
            end
        end
    end
    #@test all(e -> e ≈ first(energies), energies)
end

#bench("$(@__DIR__)/instances/chimera_droplets/2048power/001.txt")
bench("$(@__DIR__)/instances/chimera_droplets/512power/001.txt")
#bench("$(@__DIR__)/instances/J124/C=16_J124/002.txt")
